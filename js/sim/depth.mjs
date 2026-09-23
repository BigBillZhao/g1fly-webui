/**
 * webui/js/sim/depth.mjs — 深度预处理 (`sim2sim/depth_preproc.py` 的 JS 移植, 逐步同序)。
 *
 * 原始 z-depth (render_h × render_w, 米, 行 0 = 图像上边) -> 策略输入 (1, C, H, W) ∈ [−0.5, 0.5]:
 *   裁 (up, down, left, right)=(0,0,16,16) -> 36×32
 *   nan/+inf -> max_depth, −inf -> 0, clamp [0, max]
 *   (训练噪声这里不做, 见文末)
 *   d < invalid_below(0.19) -> max_depth
 *   高斯模糊 k=3 σ=1 (replicate padding)
 *   clamp [0, max]  ->  d/max − 0.5
 *   历史长 2, stride 1, 输出 1 帧, 每集固定延迟 delay -> buf[len−1−delay]
 *
 * 口径正本 `contracts/depth_frame_v1.md`; python 侧与 zenbot_lab 的 torch 实现逐位一致 (check_depth_preproc.py),
 * 本文件与 python 侧的逐位对照在 `webui/tools/depth_check.mjs` (实测见 webui/README.md §对齐)。
 * **训练时噪声 (距离噪声 σ0.03 / 视差伪影 p=0.001) 不移植**: 随机数流跨语言不可能一致, python 台默认也关。
 */

export function gaussianKernel2d(k, sigma) {
  const r = (k - 1) / 2, k1 = new Float64Array(k);
  let s = 0;
  for (let i = 0; i < k; i++) { const x = i - r; k1[i] = Math.exp(-(x * x) / (2 * sigma * sigma)); s += k1[i]; }
  for (let i = 0; i < k; i++) k1[i] /= s;
  const out = new Float64Array(k * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) out[i * k + j] = k1[i] * k1[j];
  return out;
}

/** (H, W) float, replicate padding == numpy np.pad(mode='edge') + 互相关。 */
export function gaussianBlur(img, H, W, k, sigma, out) {
  if (k === 1) { out.set(img); return out; }
  if (k <= 0 || k % 2 === 0) throw new Error('kernel_size 必须是正奇数');
  const r = (k - 1) / 2, ker = gaussianKernel2d(k, sigma);
  out.fill(0);
  for (let dy = 0; dy < k; dy++) {
    for (let dx = 0; dx < k; dx++) {
      const w = ker[dy * k + dx];
      if (w === 0) continue;
      for (let y = 0; y < H; y++) {
        const sy = Math.min(Math.max(y + dy - r, 0), H - 1);   // edge padding
        for (let x = 0; x < W; x++) {
          const sx = Math.min(Math.max(x + dx - r, 0), W - 1);
          out[y * W + x] += w * img[sy * W + sx];
        }
      }
    }
  }
  return out;
}

export class DepthPipeline {
  /** @param cfg contract.json 的 `depth` 节; @param delay 固定延迟帧 (python 台每集随机抽, 对照时钉死) */
  constructor(cfg, { delay = 0 } = {}) {
    this.cfg = cfg;
    const [up, down, left, right] = cfg.crop;
    this.renderW = cfg.width; this.renderH = cfg.height;
    this.W = cfg.width - left - right;
    this.H = cfg.height - up - down;
    this.crop = { up, down, left, right };
    this.maxDepth = cfg.max_depth;
    this.invalidBelow = cfg.invalid_below;
    this.blurK = cfg.blur_k; this.blurSigma = cfg.blur_sigma;
    this.historyLen = cfg.history_len;
    this.outputFrames = 1; this.stride = 1;
    this.delay = delay;
    const need = (this.outputFrames - 1) * this.stride + 1 + delay;
    if (need > this.historyLen) throw new Error(`history_len ${this.historyLen} 太短, 需要 ${need}`);
    this.n = this.H * this.W;
    this._cropped = new Float32Array(this.n);
    this._blurred = new Float64Array(this.n);
    this.buf = new Float32Array(this.historyLen * this.n);
    this.out = new Float32Array(this.outputFrames * this.n);
    this.lastNorm = new Float32Array(this.n);   // 最新一帧 (小窗画它)
    this.lastRawCrop = new Float32Array(this.n);
    this.reset();
  }

  reset() { this.buf.fill(0); this.initialized = false; }

  /** @param raw Float32Array(render_h*render_w) 行主序, 行 0 = 上边, 单位米 */
  process(raw) {
    if (raw.length !== this.renderH * this.renderW) {
      throw new Error(`原始深度 ${raw.length} != render ${this.renderH}x${this.renderW}`);
    }
    const { up, left } = this.crop, W = this.W, H = this.H, max = this.maxDepth;
    const c = this._cropped;
    for (let y = 0; y < H; y++) {
      const sy = y + up;
      for (let x = 0; x < W; x++) {
        let v = raw[sy * this.renderW + (x + left)];
        if (Number.isNaN(v)) v = max;                     // nan_to_num(nan=max, posinf=max, neginf=0)
        else if (v === Infinity) v = max;
        else if (v === -Infinity) v = 0;
        if (v < 0) v = 0; else if (v > max) v = max;      // clip [0, max]
        if (this.invalidBelow > 0 && v < this.invalidBelow) v = max;   // 近场无效
        c[y * W + x] = v;
      }
    }
    const b = gaussianBlur(c, H, W, this.blurK, this.blurSigma, this._blurred);
    const norm = this.lastNorm;
    for (let i = 0; i < this.n; i++) {
      let v = b[i];
      if (v < 0) v = 0; else if (v > max) v = max;
      norm[i] = Math.fround(v) / max - 0.5;
    }
    this.lastRawCrop.set(c);
    return this._push(norm);
  }

  _push(norm) {
    const n = this.n, L = this.historyLen;
    if (this.initialized) {
      this.buf.copyWithin(0, n);                  // buf[:-1] = buf[1:]
      this.buf.set(norm, (L - 1) * n);
    } else {
      for (let i = 0; i < L; i++) this.buf.set(norm, i * n);
      this.initialized = true;
    }
    const idx = L - 1 - this.delay;               // output_frames=1, stride=1
    this.out.set(this.buf.subarray(idx * n, (idx + 1) * n));
    return this.out;
  }
}
