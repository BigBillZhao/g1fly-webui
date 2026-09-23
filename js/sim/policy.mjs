/**
 * webui/js/sim/policy.mjs — onnxruntime-web 策略封装 (契约 webui_bundle_v1 §3; python 侧 = sim2sim/policy.py)。
 *
 * kind 分支 (以 meta.json 的 kind / ONNX 元数据为准, 不硬编码维度):
 *   teacher    obs[1, 82+187] -> actions[1, 15]                                                    [W1]
 *   student    policy_depth(depth[1,C,36,32], proprio[1,82], hidden_in[L,1,512]) -> depth_memory[1,512], hidden_out
 *              policy_actor(proprio, depth_memory) -> actions[1,15], terrain_scan[1,187]            [W2]
 *   bottleneck policy.onnx(proprio[1,P], depth[1,C,36,32] (盲走无), h_b[1,1,H_b], h_v[1,1,H_v])
 *              -> actions[1,15], h_b_out, h_v_out, z[1,K] (z 可选, 老导出没有)                        [W3]
 *
 * 深度节拍与隐状态口径照 `sim2sim/policy.py` 的 `_DepthPacedPolicy` / `StudentPolicy`:
 * `depth_interval = round(1 / (update_hz * step_dt))`, `wantsDepth()` 由调用方 (loop.mjs) 先问再渲染,
 * 策略内部用同一个谓词, 两边不会错拍; 隐状态外置, reset 置零。
 */

let ORT = null;

/**
 * ort-web 只初始化一次。wasmPath = 放 ort-wasm-*.wasm 的目录 URL (node 里给 file:// 前缀)。
 * @param opts.wasmBinary 可选: 已经拿到手的 ort-wasm-simd-threaded.wasm 字节 (`.gz` 旁车解压来的, 见 main.mjs)。
 *        `env.wasm.wasmBinary` 给了就不再 fetch .wasm, 但 `.mjs` 胶水仍然从 wasmPaths 取 (trap #4)。
 */
export async function initOrt(ortUrl, wasmPath, { numThreads = 1, simd = true, wasmBinary = null } = {}) {
  if (ORT) return ORT;
  const mod = await import(ortUrl);
  const ort = mod.default ?? mod;
  ort.env.wasm.numThreads = numThreads;
  ort.env.wasm.simd = simd;
  ort.env.wasm.wasmPaths = wasmPath;
  if (wasmBinary) ort.env.wasm.wasmBinary = wasmBinary;
  ort.env.logLevel = 'error';
  ORT = ort;
  return ort;
}

export function ortVersion() { return ORT ? ORT.env.versions : null; }

class TeacherPolicy {
  constructor(sess, meta) {
    this.kind = 'teacher';
    this.sess = sess;
    this.meta = meta;
    this.inputName = sess.inputNames[0];
    this.outputName = sess.outputNames[0];
    const shape = meta.io.inputs[this.inputName];
    this.obsDim = Number(shape[shape.length - 1]);
    this.obs = new Float32Array(this.obsDim);
    this.actionDim = Number((meta.io.outputs[this.outputName] ?? [1, 15]).slice(-1)[0]);
    this.lastScan = null;
  }

  reset() {}

  wantsDepth() { return false; }

  /** @returns {Promise<Float32Array>} 原始动作 (未裁剪) */
  async act(proprio, terrain) {
    if (proprio.length + terrain.length !== this.obsDim) {
      throw new Error(`teacher obs dim ${this.obsDim} != proprio ${proprio.length} + terrain ${terrain.length}`);
    }
    this.obs.set(proprio, 0);
    this.obs.set(terrain, proprio.length);
    const feeds = {};
    feeds[this.inputName] = new ORT.Tensor('float32', this.obs, [1, this.obsDim]);
    const out = await this.sess.run(feeds);
    return out[this.outputName].data;
  }
}

class StudentPolicy {
  /** @param sessions {enc, actor}; @param depthInterval 每几个策略步渲染一帧深度 */
  constructor(sessions, meta, depthInterval) {
    this.kind = 'student';
    this.enc = sessions.enc;
    this.actor = sessions.actor;
    this.meta = meta;
    const encIn = meta.io.depth.inputs, actIn = meta.io.actor.inputs;
    this.depthShape = encIn.depth.map(Number);            // [1, C, H, W]
    this.hiddenShape = encIn.hidden_in.map(Number);       // [L, 1, Hd]
    this.proprioDim = Number(actIn.proprio[actIn.proprio.length - 1]);
    this.memDim = Number(this.hiddenShape[this.hiddenShape.length - 1]);
    this.actionDim = Number(meta.io.actor.outputs.actions.slice(-1)[0]);
    this.depthInterval = Math.max(1, depthInterval | 0);
    this.depthNumel = this.depthShape.reduce((a, b) => a * b, 1);
    this.lastScan = null;                                  // actor 顺带吐的 terrain_scan[1,187] (面板用)
    this.reset();
  }

  reset() {
    this.hidden = new Float32Array(this.hiddenShape.reduce((a, b) => a * b, 1));
    this.depthMemory = new Float32Array(this.memDim);
    this.stepCount = 0;
    this.lastScan = null;
  }

  wantsDepth() { return this.stepCount % this.depthInterval === 0; }

  /** @param depth 归一化后的 (1, C, H, W) 展平帧; 非相机步给 null */
  async act(proprio, depth) {
    if (proprio.length !== this.proprioDim) throw new Error(`student proprio ${proprio.length} != ${this.proprioDim}`);
    if (this.wantsDepth()) {
      if (!depth) throw new Error('这一步需要深度帧 (wantsDepth() 为真)');
      if (depth.length !== this.depthNumel) throw new Error(`深度帧 ${depth.length} != ${this.depthShape}`);
      const feeds = {
        depth: new ORT.Tensor('float32', Float32Array.from(depth), this.depthShape),
        proprio: new ORT.Tensor('float32', Float32Array.from(proprio), [1, this.proprioDim]),
        hidden_in: new ORT.Tensor('float32', this.hidden, this.hiddenShape),
      };
      const out = await this.enc.run(feeds);
      this.depthMemory = out.depth_memory.data;
      this.hidden = out.hidden_out.data;
    }
    this.stepCount++;
    const out = await this.actor.run({
      proprio: new ORT.Tensor('float32', Float32Array.from(proprio), [1, this.proprioDim]),
      depth_memory: new ORT.Tensor('float32', this.depthMemory, [1, this.memDim]),
    });
    this.lastScan = out.terrain_scan ? out.terrain_scan.data : null;
    return out.actions.data;
  }
}

/**
 * Arm H/T/S 瓶颈学生 (`sim2sim/policy.py::BottleneckPolicy` 的孪生)。
 * 与双文件学生的**关键区别**: 编码器在同一张图里, 每个策略步都跑 —— 相机慢半拍时不是跳步, 而是
 * **持住上一帧深度重喂** (IsaacLab 的 depth_image 观测项每步返回历史里最后压入的那帧, 同口径)。
 * 维度一律按名字从 ONNX 元数据取, 不在这里写死 (contracts/policy_onnx_v1.md)。
 */
class BottleneckPolicy {
  constructor(sess, meta, depthInterval) {
    this.kind = 'bottleneck';
    this.sess = sess;
    this.meta = meta;
    const ins = meta.io.inputs, outs = meta.io.outputs;
    for (const n of ['proprio', 'h_b', 'h_v']) {
      if (!ins[n]) throw new Error(`${meta.name} 不是瓶颈导出: 缺输入 ${n} (有 ${Object.keys(ins)})`);
    }
    this.hasDepth = !!ins.depth;
    this.hasZ = !!outs.z;
    this.proprioDim = Number(ins.proprio[ins.proprio.length - 1]);
    this.hbShape = ins.h_b.map(Number);
    this.hvShape = ins.h_v.map(Number);
    this.depthShape = this.hasDepth ? ins.depth.map(Number) : null;
    this.depthNumel = this.hasDepth ? this.depthShape.reduce((a, b) => a * b, 1) : 0;
    this.actionDim = Number(outs.actions[outs.actions.length - 1]);
    this.zDim = this.hasZ ? Number(outs.z[outs.z.length - 1]) : 0;
    this.hvDim = this.hvShape[this.hvShape.length - 1];
    this.outputNames = ['actions', 'h_b_out', 'h_v_out'].concat(this.hasZ ? ['z'] : []);
    this.depthInterval = Math.max(1, depthInterval | 0);
    this.brain = meta.brain ?? null;
    this.lastScan = null;
    this.reset();
  }

  reset() {
    this.h_b = new Float32Array(this.hbShape.reduce((a, b) => a * b, 1));
    this.h_v = new Float32Array(this.hvShape.reduce((a, b) => a * b, 1));
    this.depthLast = null;
    this.lastZ = null;
    this.stepCount = 0;
  }

  /** 只有带深度的瓶颈才要新帧, 且只在相机节拍上 (其余步持住上一帧)。 */
  wantsDepth() { return this.hasDepth && this.stepCount % this.depthInterval === 0; }

  async act(proprio, depth) {
    if (proprio.length !== this.proprioDim) throw new Error(`bottleneck proprio ${proprio.length} != ${this.proprioDim}`);
    const feeds = {
      proprio: new ORT.Tensor('float32', Float32Array.from(proprio), [1, this.proprioDim]),
      h_b: new ORT.Tensor('float32', this.h_b, this.hbShape),
      h_v: new ORT.Tensor('float32', this.h_v, this.hvShape),
    };
    if (this.hasDepth) {
      if (depth) {
        if (depth.length !== this.depthNumel) throw new Error(`深度帧 ${depth.length} != ${this.depthShape}`);
        this.depthLast = Float32Array.from(depth);
      }
      if (!this.depthLast) throw new Error('第一个策略步就需要一帧深度');
      feeds.depth = new ORT.Tensor('float32', this.depthLast, this.depthShape);
    } else if (depth) {
      throw new Error(`${this.meta.name} 是盲走瓶颈 (没有 depth 输入), 却收到了深度帧`);
    }
    const out = await this.sess.run(feeds);
    this.h_b = out.h_b_out.data;
    this.h_v = out.h_v_out.data;
    this.lastZ = this.hasZ ? out.z.data : null;
    this.stepCount++;
    return out.actions.data;
  }
}

/** meta + onnx 字节 -> 策略对象。 */
export async function createPolicy(bundle, { executionProviders = ['wasm'] } = {}) {
  if (!ORT) throw new Error('先 initOrt()');
  const { meta, onnx, contract } = bundle;
  const opts = { executionProviders, graphOptimizationLevel: 'all' };
  if (meta.kind === 'teacher') {
    const sess = await ORT.InferenceSession.create(onnx.actor, opts);
    return new TeacherPolicy(sess, meta);
  }
  if (meta.kind === 'student') {
    const [enc, actor] = await Promise.all([
      ORT.InferenceSession.create(onnx.depth, opts),
      ORT.InferenceSession.create(onnx.actor, opts),
    ]);
    const hz = meta.depth_hz ?? contract?.depth?.update_hz ?? 50;
    const stepDt = contract?.timing?.step_dt ?? 0.02;
    return new StudentPolicy({ enc, actor }, meta, Math.round(1 / (hz * stepDt)));
  }
  if (meta.kind === 'bottleneck') {
    const sess = await ORT.InferenceSession.create(onnx.actor, opts);
    const hz = meta.depth_hz ?? contract?.depth?.update_hz ?? 50;
    const stepDt = contract?.timing?.step_dt ?? 0.02;
    return new BottleneckPolicy(sess, meta, Math.round(1 / (hz * stepDt)));
  }
  throw new Error(`未知 kind: ${meta.kind}`);
}
