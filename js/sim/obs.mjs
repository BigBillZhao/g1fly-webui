/**
 * webui/js/sim/obs.mjs — 观测组装 (contracts/obs_action_v1.md; python 侧 = sim2sim/run.py 的 raw/parts 段)。
 *
 * proprio 82 = [projected_gravity 3, base_ang_vel 3 (x0.2), velocity_commands 3, joint_pos_rel 29 (Isaac 序),
 *               joint_vel_rel 29 (x0.05), last_action 15]，每项先 clip +-100 再乘 scale (IsaacLab: noise->clip->scale)。
 * terrain 187 = 高程扫描: 挂 torso_link, yaw 对齐, 1.6x1.0 m @0.1 (x 最快 = IsaacLab GridPattern "xy"),
 *               值 = torso_z - hit_z - offset, clip +-1, 未命中 -1。
 *               python 台用 mj_ray 打 hfield 三角面; 这里对预烘高度做双线性 (量化差见 README)。
 */

export function projectedGravity(w, x, y, z, out) {
  out[0] = 2 * (-z * x + w * y);
  out[1] = -2 * (z * y + w * x);
  out[2] = 1 - 2 * (w * w + z * z);
  return out;
}

export function quatToRPY(w, x, y, z) {
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const s = Math.max(-1, Math.min(1, 2 * (w * y - z * x)));
  const pitch = Math.asin(s);
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [roll, pitch, yaw];
}

/** IsaacLab height_scan 的双线性孪生 (sim2sim/mjscene.py::HeightScanner)。 */
export class HeightScanner {
  constructor(contract, terrain) {
    const hs = contract.height_scan;
    this.offset = hs.offset;
    this.clip = hs.clip;
    this.terrain = terrain;
    const nx = Math.round(hs.size[0] / hs.res) + 1;      // 17
    const ny = Math.round(hs.size[1] / hs.res) + 1;      // 11
    this.n = nx * ny;
    this.gx = new Float64Array(this.n);
    this.gy = new Float64Array(this.n);
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {                   // x 最快
        this.gx[iy * nx + ix] = -hs.size[0] / 2 + ix * hs.res;
        this.gy[iy * nx + ix] = -hs.size[1] / 2 + iy * hs.res;
      }
    }
    this.out = new Float32Array(this.n);
    this.hits = new Float64Array(this.n * 3);            // 世界坐标命中点 (给 webui 画扫描网格)
  }

  /** @param pos torso 世界位置 (3), @param mat torso 旋转矩阵 (行主序 9) */
  scan(pos, mat) {
    const yaw = Math.atan2(mat[3], mat[0]);              // R[1,0], R[0,0]
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const t = this.terrain, o = this.out, lo = -this.clip, hi = this.clip;
    for (let i = 0; i < this.n; i++) {
      const x = pos[0] + c * this.gx[i] - s * this.gy[i];
      const y = pos[1] + s * this.gx[i] + c * this.gy[i];
      const hz = t.heightAt(x, y);
      let v = Number.isFinite(hz) ? pos[2] - hz - this.offset : -Infinity;   // 未命中 = python 的 hit_z=+inf
      o[i] = v < lo ? lo : (v > hi ? hi : v);
      this.hits[3 * i] = x; this.hits[3 * i + 1] = y; this.hits[3 * i + 2] = Number.isFinite(hz) ? hz : pos[2] - this.offset - this.clip;
    }
    return o;
  }
}

/** proprio 组装器: 预展平 scale, 每步零分配。 */
export class ProprioBuilder {
  constructor(contract) {
    const p = contract.proprio;
    this.terms = p.terms;
    this.clip = p.clip;
    this.dim = p.dim;
    this.scale = new Float64Array(this.dim);
    this.offsets = {};
    let k = 0;
    for (const t of this.terms) {
      const s = p.scales[t];
      this.offsets[t] = k;
      for (let i = 0; i < s.length; i++) this.scale[k++] = s[i];
    }
    if (k !== this.dim) throw new Error(`proprio dim ${this.dim} != 展平后的 ${k}`);
    this.out = new Float32Array(this.dim);
    this.raw = new Float64Array(this.dim);
  }

  /** raw 各项按 contract.proprio.terms 的顺序写进 this.raw 后调用。 */
  finish() {
    const c = this.clip, r = this.raw, o = this.out, s = this.scale;
    for (let i = 0; i < r.length; i++) {
      const v = r[i] < -c ? -c : (r[i] > c ? c : r[i]);
      o[i] = v * s[i];
    }
    return o;
  }
}
