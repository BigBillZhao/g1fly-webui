/**
 * webui/js/sim/loop.mjs — 50 Hz 策略主循环 + 指标 (sim2sim/run.py::run_episode 的 JS 孪生)。
 *
 * 纯计算, 不碰 DOM: node 的对齐验收 (tools/align.mjs) 与浏览器台 (js/app) 共用同一份。
 * 每个策略步: 组 obs -> 策略 -> clip/scale/offset -> 散射到 SDK 序 -> decimation 个 mj_step -> 指标。
 * 指令来自指令表 (t 升序, 同 run.py::cmd_at)。浏览器用 `advance(wallDt)` 的固定步长累积器驱动。
 */

import { buildModel, resetToSpawn } from './mjload.mjs';
import { HeightScanner, ProprioBuilder, projectedGravity, quatToRPY } from './obs.mjs';
import { DepthPipeline } from './depth.mjs';

export class SimLoop {
  /** @param {object} mj WASM 模块; @param {object} opts {robot, terrain, contract, policy, meta} */
  /**
   * @param opts.renderDepth 可选 `(loop) => Float32Array(render_h*render_w)` 原始 z-depth (米, 行 0 = 上边);
   *   学生策略必需 (浏览器里是 three.js 的深度相机, node 里只有对照工具会喂录好的帧)。
   * @param opts.depthDelay 深度历史的固定延迟帧 (默认 0; python 台每集随机抽, 对照时两边都钉 0)。
   */
  constructor(mj, { robot, terrain, contract, policy, renderDepth = null, depthDelay = 0 }) {
    this.mj = mj;
    this.contract = contract;
    this.terrain = terrain;
    this.policy = policy;
    this.sim = buildModel(mj, robot, terrain, contract);
    this.model = this.sim.model;
    this.data = this.sim.data;
    this.ids = this.sim.ids;

    this.stepDt = contract.timing.step_dt;
    this.decimation = contract.timing.decimation;
    this.scanInterval = Math.max(1, Math.round(contract.height_scan.update_s / this.stepDt));
    this.actionIds = contract.action.ids_isaac;
    this.actionScale = contract.action.scale;
    this.actionOffset = contract.action.offset;
    this.actionClip = contract.action.clip;
    this.i2s = contract.joint.isaac_to_sdk;
    this.defaultIsaac = contract.joint.default_pos_isaac;
    this.holdIsaac = contract.physics.arm_hold === 'default' ? contract.joint.default_pos_isaac : contract.joint.hold_pos_isaac;

    this.prop = new ProprioBuilder(contract);
    this.scanner = new HeightScanner(contract, terrain);
    this.usesDepth = typeof policy.wantsDepth === 'function' && policy.kind !== 'teacher';
    this.renderDepth = renderDepth;
    this.depthPipeline = this.usesDepth ? new DepthPipeline(contract.depth, { delay: depthDelay }) : null;
    this.lastDepth = null;        // 归一化后的策略输入 (小窗画它)
    this.lastRawDepth = null;     // 渲染出的原始 z-depth (小窗第二格)
    this.scanPred = null;         // 学生 actor 顺带解出的 terrain_scan (面板用, 不进环)
    this.action = new Float32Array(this.actionIds.length);
    this.targetIsaac = new Float64Array(29);
    this.targetSdk = new Float64Array(29);
    this.qIsaac = new Float64Array(29);
    this.dqIsaac = new Float64Array(29);
    this.grav = new Float64Array(3);
    this.velBuf = new mj.DoubleBuffer(6);
    this.pos = new Float64Array(3);
    this.mat = new Float64Array(9);
    this.quat = new Float64Array(4);
    this.segments = [[0, [0, 0, 0]]];
    this.cmd = new Float64Array(3);
    this.scan = null;
    this.onStep = null;      // (k, t, {qpos, action}) => void
    this.reset();
  }

  setCommand(vx, vy, wz) { this.segments = [[0, [vx, vy, wz]]]; }

  setSchedule(segs) {
    const s = segs.slice().sort((a, b) => a[0] - b[0]);
    if (!s.length || s[0][0] > 0) s.unshift([0, [0, 0, 0]]);
    this.segments = s;
  }

  cmdAt(t) {
    let c = this.segments[0][1];
    for (const [t0, v] of this.segments) if (t >= t0) c = v;
    return c;
  }

  reset() {
    resetToSpawn(this.mj, this.sim, this.contract);
    this.policy.reset?.();
    this.depthPipeline?.reset();
    this.lastDepth = this.lastRawDepth = this.scanPred = null;
    this.action.fill(0);
    this.scan = null;
    this.k = 0;
    this.t = 0;
    this.acc = 0;
    this.fell = false;
    this.fallStep = null;
    this.startXY = [this.data.xpos[3 * this.ids.base], this.data.xpos[3 * this.ids.base + 1]];
    this.endXYatFall = null;
    this.velErrSum = 0; this.tauSum = 0; this.aliveSteps = 0;
    this.expectedDist = 0;
    this.maxX = this.startXY[0];
    this.torsoH = [];
    this.lastVel = [0, 0, 0, 0, 0, 0];
    this.physicsMs = 0; this.policyMs = 0;
  }

  /** 世界位姿快照 (渲染/指标共用)。 */
  _readBase() {
    const d = this.data, b = this.ids.base;
    this.pos[0] = d.xpos[3 * b]; this.pos[1] = d.xpos[3 * b + 1]; this.pos[2] = d.xpos[3 * b + 2];
    for (let i = 0; i < 4; i++) this.quat[i] = d.xquat[4 * b + i];
  }

  _objectVelocity(objType, id, local) {
    this.mj.mj_objectVelocity(this.model, this.data, objType, id, this.velBuf, local);
    return this.velBuf.GetView();
  }

  /** 一个 50 Hz 策略步 (obs -> 策略 -> decimation 个物理子步 -> 指标)。 */
  async step() {
    const mj = this.mj, model = this.model, data = this.data, c = this.contract;
    const t = this.k * this.stepDt;
    this.t = t;
    const cmd = this.cmdAt(t);
    this.cmd[0] = cmd[0]; this.cmd[1] = cmd[1]; this.cmd[2] = cmd[2];
    this.expectedDist += Math.hypot(cmd[0], cmd[1]) * this.stepDt;

    // ---- 观测 ----
    this._readBase();
    const vel = this._objectVelocity(mj.mjtObj.mjOBJ_BODY.value, this.ids.base, 1);   // 局部: [ang(3), lin(3)]
    projectedGravity(this.quat[0], this.quat[1], this.quat[2], this.quat[3], this.grav);
    for (let i = 0; i < 29; i++) {
      const s = this.i2s[i];
      this.qIsaac[i] = data.qpos[7 + s] - this.defaultIsaac[i];
      this.dqIsaac[i] = data.qvel[6 + s];
    }
    const p = this.prop, off = p.offsets, raw = p.raw;
    raw.set(this.grav, off.projected_gravity);
    raw[off.base_ang_vel] = vel[0]; raw[off.base_ang_vel + 1] = vel[1]; raw[off.base_ang_vel + 2] = vel[2];
    raw.set(this.cmd, off.velocity_commands);
    raw.set(this.qIsaac, off.joint_pos_rel);
    raw.set(this.dqIsaac, off.joint_vel_rel);
    raw.set(this.action, off.last_action);
    const proprio = p.finish();

    // ---- 策略 ----
    const tp = (typeof performance !== 'undefined' ? performance : Date).now();
    const b = this.ids.torso;
    if (this.scan === null || this.k % this.scanInterval === 0) {
      const tp2 = [data.xpos[3 * b], data.xpos[3 * b + 1], data.xpos[3 * b + 2]];
      for (let i = 0; i < 9; i++) this.mat[i] = data.xmat[9 * b + i];
      this.scan = this.scanner.scan(tp2, this.mat);       // 教师的输入 / 学生只用来画真值点云
    }
    let act;
    if (this.usesDepth) {
      let depthIn = null;
      if (this.policy.wantsDepth()) {                      // 先问再渲染 (与 run.py / policy.py 同一个谓词)
        if (!this.renderDepth) throw new Error(`${this.policy.kind} 策略需要深度源: new SimLoop(..., {renderDepth})`);
        const raw = this.renderDepth(this);
        this.lastRawDepth = raw;
        depthIn = this.depthPipeline.process(raw);
        this.lastDepth = depthIn;
      }
      act = await this.policy.act(proprio, depthIn);
      this.scanPred = this.policy.lastScan ?? null;
    } else {
      act = await this.policy.act(proprio, this.scan);
    }
    this.action.set(act);
    this.policyMs = (typeof performance !== 'undefined' ? performance : Date).now() - tp;

    // ---- 动作 -> PD 目标 ----
    this.targetIsaac.set(this.holdIsaac);
    for (let j = 0; j < this.actionIds.length; j++) {
      const a = this.action[j], lo = this.actionClip[j][0], hi = this.actionClip[j][1];
      const cl = a < lo ? lo : (a > hi ? hi : a);
      this.targetIsaac[this.actionIds[j]] = cl * this.actionScale[j] + this.actionOffset[j];
    }
    for (let i = 0; i < 29; i++) this.targetSdk[this.i2s[i]] = this.targetIsaac[i];

    // ---- 物理 ----
    const th = (typeof performance !== 'undefined' ? performance : Date).now();
    for (let s = 0; s < this.decimation; s++) {
      data.ctrl.set(this.targetSdk);
      mj.mj_step(model, data);
    }
    this.physicsMs = (typeof performance !== 'undefined' ? performance : Date).now() - th;

    // ---- 指标 (口径 = run.py) ----
    this._readBase();
    const ground = this.terrain.heightAt(this.pos[0], this.pos[1]);
    const h = this.pos[2] - ground;
    const [roll, pitch] = quatToRPY(this.quat[0], this.quat[1], this.quat[2], this.quat[3]);
    if (!this.fell) { this.torsoH.push(h); this.maxX = Math.max(this.maxX, this.pos[0]); }
    if (!this.fell && (h < c.fall.height || Math.abs(roll) > c.fall.tilt || Math.abs(pitch) > c.fall.tilt || !Number.isFinite(h))) {
      this.fell = true;
      this.fallStep = this.k;
      this.endXYatFall = [this.pos[0], this.pos[1]];
    }
    if (!this.fell) {
      const v = this._objectVelocity(mj.mjtObj.mjOBJ_BODY.value, this.ids.base, 1);
      this.lastVel = [v[0], v[1], v[2], v[3], v[4], v[5]];
      this.velErrSum += Math.hypot(v[3] - cmd[0], v[4] - cmd[1]) + 0.5 * Math.abs(v[2] - cmd[2]);
      let tau = 0;
      for (let i = 0; i < model.nu; i++) tau += Math.abs(data.actuator_force[i]);
      this.tauSum += tau / model.nu;
      this.aliveSteps++;
    } else {
      const v = this._objectVelocity(mj.mjtObj.mjOBJ_BODY.value, this.ids.base, 1);
      this.lastVel = [v[0], v[1], v[2], v[3], v[4], v[5]];
    }
    if (this.onStep) this.onStep(this.k, t, this);
    this.k++;
    return this;
  }

  /** 浏览器: 用墙钟增量驱动固定步长 (最多 maxSteps 步/帧, 落后太多就丢帧不追)。 */
  async advance(wallDt, maxSteps = 4) {
    this.acc += Math.min(wallDt, 0.25);
    let n = 0;
    while (this.acc >= this.stepDt && n < maxSteps) { this.acc -= this.stepDt; await this.step(); n++; }
    if (this.acc > this.stepDt * maxSteps) this.acc = 0;
    return n;
  }

  metrics() {
    const end = this.endXYatFall ?? [this.pos[0], this.pos[1]];
    const dist = Math.hypot(end[0] - this.startXY[0], end[1] - this.startXY[1]);
    const alive = Math.max(this.aliveSteps, 1);
    const success = !this.fell && (this.expectedDist > 1e-6 ? dist >= 0.7 * this.expectedDist : true);
    let hmin = null, hsum = 0;                       // 不用 Math.min(...arr): 长集会炸栈
    for (const h of this.torsoH) { if (hmin === null || h < hmin) hmin = h; hsum += h; }
    const hmean = this.torsoH.length ? hsum / this.torsoH.length : null;
    return {
      success, fell: this.fell,
      time_to_fall_s: this.fallStep === null ? null : +((this.fallStep + 1) * this.stepDt).toFixed(3),
      distance_xy: +dist.toFixed(4), expected_distance: +this.expectedDist.toFixed(4),
      progress_x: +(this.maxX - this.startXY[0]).toFixed(4),
      vel_err: +(this.velErrSum / alive).toFixed(4), mean_abs_tau: +(this.tauSum / alive).toFixed(3),
      base_height_above_ground: hmin === null ? null : { min: +hmin.toFixed(3), mean: +hmean.toFixed(3) },
      steps: this.k, alive_steps: this.aliveSteps, step_dt: this.stepDt, sim_dt: this.model.opt.timestep,
      terrain: this.terrain.spec, seconds: +(this.k * this.stepDt).toFixed(3),
      total_mass_kg: +this._totalMass().toFixed(3),
    };
  }

  _totalMass() {
    let m = 0;
    for (let i = 0; i < this.model.nbody; i++) m += this.model.body_mass[i];
    return m;
  }

  dispose() {
    this.velBuf.delete();
    this.sim.dispose();
  }
}
