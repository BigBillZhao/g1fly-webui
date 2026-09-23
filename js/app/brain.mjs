/**
 * webui/js/app/brain.mjs — 左面板: 果蝇神经元点云 + module 聚合条 + DN 通道热条。
 *
 * 数据 = `data/brain/<graph>_neurons.json` (契约 webui_bundle_v1 §4, 由 g1fly/connectome/export_webui.py 出):
 *   `i` = `h_v_out[0,0,i]` 的下标 (= EIRateRNN 单元序 = graphs/<g>.nodes.feather 行序, 三者已核过逐行同)
 *   `xyz` = MaleCNS somaLocation 居中后的 µm; `ports.MN` = 读出单元, `ports.DN` = 下行输入
 * **n / module 数 / MN 数一律从这份 json 读, 不写死** (t1_w5 = 3,609 / 37 / 135;
 * core3_w5 = 9,156 / 39 / 381 —— 同一份代码两张图, 09-23 worker #6 换图时核过)。
 * 着色 = `ramp(clip(h_v[i] / h_v_clip, 0, 1))` 灰 → 橙红 (契约 §4); DN 热条 = `z` (K=dn_dim)。
 *
 * 独立的 WebGLRenderer (与右侧主场景各用各的上下文), 每策略步只改颜色缓冲, 不重建几何
 * (9,156 点同样是静态 position buffer, 每步只 needsUpdate 颜色)。
 * `drawZBar` 是**模块级函数**: `core=gru` 的对照策略没有点云/module 条, 但仍然有 z, 热条照画 (open ⑰)。
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/** 灰 → 橙红 (t ∈ [0,1]); t=0 是暗灰, 保证"没激活"也看得见结构。 */
function ramp(t, out, o) {
  const g = 0.26 + 0.10 * t;
  out[o] = g + t * (1.0 - g);
  out[o + 1] = g + t * (0.42 - g);
  out[o + 2] = g + t * (0.13 - g);
}

function discTexture(ring) {
  const S = 64, c = document.createElement('canvas');
  c.width = c.height = S;
  const x = c.getContext('2d');
  x.clearRect(0, 0, S, S);
  if (ring) {
    x.strokeStyle = '#fff'; x.lineWidth = 9;
    x.beginPath(); x.arc(S / 2, S / 2, S / 2 - 7, 0, Math.PI * 2); x.stroke();
  } else {
    x.fillStyle = '#fff';
    x.beginPath(); x.arc(S / 2, S / 2, S / 2 - 2, 0, Math.PI * 2); x.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.needsUpdate = true;
  return t;
}

export class BrainPanel {
  /** @param host 放 3D 的 div; @param bars module 聚合条 canvas; @param dn DN 热条 canvas; @param info 文字行 */
  constructor(host, bars, dn, info) {
    this.host = host; this.barsCv = bars; this.dnCv = dn; this.infoEl = info;
    this.data = null; this.ready = false;
  }

  /** 载入神经元表并建点云 (只做一次; 换策略只改 clip)。 */
  async load(data) {
    this.data = data;
    const n = data.n, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let k = 0; k < n; k++) {
      const p = data.neurons[k].xyz;
      pos[3 * k] = p[0]; pos[3 * k + 1] = p[1]; pos[3 * k + 2] = p[2];
      box.expandByPoint(v.set(p[0], p[1], p[2]));
      ramp(0, col, 3 * k);
    }
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() / 2;
    for (let k = 0; k < n; k++) { pos[3 * k] -= center.x; pos[3 * k + 1] -= center.y; pos[3 * k + 2] -= center.z; }
    this.pos = pos; this.col = col; this.n = n; this.radius = radius;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x12181f);
    this.camera = new THREE.PerspectiveCamera(42, 1, radius * 0.01, radius * 20);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0, -radius * 2.4, radius * 0.6);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.host.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.points = new THREE.Points(geo, new THREE.PointsMaterial({
      size: radius * 0.022, vertexColors: true, map: discTexture(false), transparent: true, alphaTest: 0.35,
      sizeAttenuation: true, depthWrite: true,
    }));
    this.scene.add(this.points);

    // 读出单元 (MN, 张数按 ports.MN: t1_w5 135 / core3_w5 381) 单独一层: 大一圈 + 圆环描边
    const mn = (data.ports && data.ports.MN) ? data.ports.MN : [];
    this.mnIdx = mn;
    if (mn.length) {
      const mp = new Float32Array(mn.length * 3), mc = new Float32Array(mn.length * 3);
      for (let k = 0; k < mn.length; k++) {
        const i = mn[k];
        mp[3 * k] = pos[3 * i]; mp[3 * k + 1] = pos[3 * i + 1]; mp[3 * k + 2] = pos[3 * i + 2];
        mc[3 * k] = 0.35; mc[3 * k + 1] = 0.85; mc[3 * k + 2] = 1.0;
      }
      const g2 = new THREE.BufferGeometry();
      g2.setAttribute('position', new THREE.BufferAttribute(mp, 3));
      g2.setAttribute('color', new THREE.BufferAttribute(mc, 3));
      this.mnCol = mc;
      this.mnPoints = new THREE.Points(g2, new THREE.PointsMaterial({
        size: radius * 0.055, vertexColors: true, map: discTexture(true), transparent: true, alphaTest: 0.2,
        sizeAttenuation: true, depthWrite: false,
      }));
      this.scene.add(this.mnPoints);
    }

    // module 聚合: 每个 module 的下标表 (张数按 data.modules: t1_w5 37 / core3_w5 39)
    this.modules = (data.modules || []).map((m) => ({ ...m, idx: [] }));
    const byName = new Map(this.modules.map((m, k) => [m.name, k]));
    for (let k = 0; k < n; k++) {
      const m = byName.get(data.neurons[k].module);
      if (m !== undefined) this.modules[m].idx.push(k);
    }
    this.moduleVals = new Float32Array(this.modules.length);

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = radius * 0.02;
    this.mouse = new THREE.Vector2();
    this._onMove = (e) => this._pick(e);
    this.renderer.domElement.addEventListener('pointermove', this._onMove);
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this.resize();
    this.ready = true;
    return { n, modules: this.modules.length, mn: mn.length };
  }

  resize() {
    if (!this.renderer) return;
    const w = this.host.clientWidth || 280, h = this.host.clientHeight || 220;
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  _pick(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    this.mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hit = this.raycaster.intersectObject(this.points, false)[0];
    if (!hit) return;
    const nd = this.data.neurons[hit.index];
    const act = this.lastHv ? this.lastHv[hit.index] : null;
    this.infoEl.textContent = `#${nd.i} ${nd.type ?? '?'} · ${nd.module} · ${nd.side ?? '?'} · ` +
      `${nd.sign > 0 ? '兴奋' : (nd.sign < 0 ? '抑制' : '符号未知')}` +
      (this.mnIdx.includes(nd.i) ? ' · MN 读出' : '') + (act !== null ? ` · h_v=${act.toFixed(4)}` : '');
  }

  /** 每策略步: h_v -> 点色 + module 条。 */
  update(h_v, clip) {
    if (!this.ready || !h_v) return;
    this.lastHv = h_v;
    const inv = 1 / Math.max(clip, 1e-6), col = this.col, n = Math.min(this.n, h_v.length);
    for (let i = 0; i < n; i++) {
      let t = h_v[i] * inv;
      t = t < 0 ? 0 : (t > 1 ? 1 : t);
      ramp(t, col, 3 * i);
    }
    this.points.geometry.attributes.color.needsUpdate = true;
    if (this.mnPoints) {
      for (let k = 0; k < this.mnIdx.length; k++) {
        let t = h_v[this.mnIdx[k]] * inv;
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        this.mnCol[3 * k] = 0.35 + 0.65 * t; this.mnCol[3 * k + 1] = 0.85 - 0.43 * t; this.mnCol[3 * k + 2] = 1.0 - 0.87 * t;
      }
      this.mnPoints.geometry.attributes.color.needsUpdate = true;
    }
    for (let m = 0; m < this.modules.length; m++) {
      const idx = this.modules[m].idx;
      let s = 0;
      for (const i of idx) s += h_v[i];
      this.moduleVals[m] = idx.length ? s / idx.length : 0;
    }
    this._drawBars(clip);
  }

  _drawBars(clip) {
    const cv = this.barsCv, ctx = cv.getContext('2d');
    const n = this.modules.length, w = cv.width / n;
    ctx.fillStyle = '#12181f'; ctx.fillRect(0, 0, cv.width, cv.height);
    for (let m = 0; m < n; m++) {
      let t = this.moduleVals[m] / Math.max(clip, 1e-6);
      t = t < 0 ? 0 : (t > 1 ? 1 : t);
      const h = Math.max(1, t * cv.height);
      const c = [0, 0, 0];
      ramp(t, c, 0);
      ctx.fillStyle = `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`;
      ctx.fillRect(m * w, cv.height - h, Math.max(1, w - 1), h);
    }
  }

  /** DN 瓶颈向量 z (K 维) 的热条; 没有 z 就画空。 */
  drawZ(z, clip) { drawZBar(this.dnCv, z, clip); }

  render() {
    if (!this.ready) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    if (!this.ready) return;
    window.removeEventListener('resize', this._onResize);
    this.renderer.domElement.removeEventListener('pointermove', this._onMove);
    this.points.geometry.dispose(); this.points.material.dispose();
    if (this.mnPoints) { this.mnPoints.geometry.dispose(); this.mnPoints.material.dispose(); }
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.ready = false;
  }
}

/**
 * DN 瓶颈向量 `z` (K 维) 的热条 —— **与点云面板解耦**, 单独一个 canvas 就能画。
 * `core=gru` 的瓶颈对照策略没有 connectome 点云 (graph_name=null), 但它一样有 z 输出,
 * 对照组要看得见自己的 z ⇒ main.mjs 在没有 BrainPanel 时直接调这个 (open ⑰, 09-23 worker #6)。
 * 着色规则与点云同: `ramp(clip(z[i] / z_clip, 0, 1))`。z 为 null 只清底色。
 */
export function drawZBar(cv, z, clip) {
  if (!cv) return;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#12181f'; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!z || !z.length) return;
  const n = z.length, w = cv.width / n, inv = 1 / Math.max(clip, 1e-6);
  for (let i = 0; i < n; i++) {
    let t = z[i] * inv;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    const c = [0, 0, 0];
    ramp(t, c, 0);
    ctx.fillStyle = `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`;
    ctx.fillRect(i * w, 0, Math.max(1, w), cv.height);
  }
}
