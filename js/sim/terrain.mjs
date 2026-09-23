/**
 * webui/js/sim/terrain.mjs — 预烘 hfield 地形的 JS 侧读取 (build.py 出的 <id>.json + <id>.f32)。
 *
 * 三种几何表示 (跟 sim2sim 09ebb5d 的 `Terrain.to_spec()`, 契约 §1.2):
 *   `repr: "boxes"` — slope/stairs/steps 现在是**解析几何**: 一个底 plane + 若干 box (斜坡是绕 +y 转的 box)。
 *     `heightAt` 是 `_surfaces`/`_height_from_geoms` 的逐行移植 (底 plane 打底, box 取顶面, 斜坡取倾斜顶面)。
 * 另两种 (92a23ba 起):
 *   `repr: "plane"` — 完全平坦的地形是**真 plane geom**, 不是平的 hfield。MuJoCo 的球-hfield 碰撞把每格
 *     三角化, 即使高度全 0, 5 mm 足球落在格边也会同时命中相邻三角面 (一球最多 6 接触, 法向偏竖直最多 53°),
 *     凭空生侧向冲量, ≥0.8 m/s 就摔 (sim2sim README §5.7)。plane 每球恰好 1 接触 |nz|=1。
 *     plane 是**无限**的 ⇒ `heightAt` 处处返回常数 (没有"场外"; 否则机器人走出记账网格会被判成摔)。
 *   `repr: "hfield"` — 其余地形照旧: <id>.f32 + hfield 元数据。
 * `heightAt` 是 `Terrain.height_at` 的逐行移植 (双线性, 场外 -> -inf); 高程扫描与摔倒判据都吃它。
 * MuJoCo 的 hfield 碰撞/射线走的是**三角化**表面, 双线性只在单元内部与三角面有 <= 半个格差;
 * 0.025 m 格上实测 (align.mjs --scan-check) 见 webui/README.md。
 */

export class Terrain {
  constructor(meta, heights) {
    this.meta = meta;
    this.repr = meta.repr ?? (meta.hfield ? 'hfield' : 'plane');
    this.geoms = (this.repr === 'hfield') ? null : (meta.geoms ?? []);
    if (this.geoms) this._surfaces(this.geoms);
    this.h = heights;                 // Float32Array, row-major (row = y, col = x), 绝对世界 z (plane 时为 null)
    this.nrow = meta.nrow;
    this.ncol = meta.ncol;
    this.res = meta.res;              // == python Terrain.res = xs[1] - xs[0] (原样带 fp 误差, 保证同算术)
    this.x0 = meta.x0;
    this.y0 = meta.y0;
    this.isPlane = this.repr === 'plane';
    this.isAnalytic = this.repr === 'plane' || this.repr === 'boxes';
    this.heightConst = meta.height_const ?? (meta.z_range ? meta.z_range[0] : 0);
    this.zmin = meta.hfield ? meta.hfield.zmin : this.heightConst;
    this.elev = meta.hfield ? meta.hfield.elev : 1;
    this.startXY = meta.start_xy;
  }

  get id() { return this.meta.id; }
  get spec() { return this.meta.spec; }

  /** 解析几何 (`_surfaces` 的口径) 按 geoms 算, 底 plane 打底 ⇒ 处处有定义; hfield 走双线性, 场外 -Infinity。 */
  heightAt(x, y) {
    if (this.geoms) {
      let z = this._base === null ? -Infinity : this._base;
      for (const b of this._boxes) {
        if (x >= b[0] && x <= b[1] && y >= b[2] && y <= b[3] && b[4] > z) z = b[4];
      }
      for (const r of this._ramps) {
        if (x >= r[0] && x <= r[1] && y >= r[2] && y <= r[3]) {
          const zz = r[5] + r[6] * (x - r[4]);
          if (zz > z) z = zz;
        }
      }
      return z;
    }
    const fx = (x - this.x0) / this.res;
    const fy = (y - this.y0) / this.res;
    if (!(fx >= 0 && fx <= this.ncol - 1 && fy >= 0 && fy <= this.nrow - 1)) return -Infinity;
    let ix = Math.floor(fx); if (ix > this.ncol - 2) ix = this.ncol - 2; if (ix < 0) ix = 0;
    let iy = Math.floor(fy); if (iy > this.nrow - 2) iy = this.nrow - 2; if (iy < 0) iy = 0;
    let tx = fx - ix; if (tx < 0) tx = 0; else if (tx > 1) tx = 1;
    let ty = fy - iy; if (ty < 0) ty = 0; else if (ty > 1) ty = 1;
    const h = this.h, nc = this.ncol, a = iy * nc + ix, b = a + nc;
    return h[a] * (1 - tx) * (1 - ty) + h[a + 1] * tx * (1 - ty) + h[b] * (1 - tx) * ty + h[b + 1] * tx * ty;
  }

  /** `sim2sim/terrains.py::_surfaces` 的移植: geoms -> (底 plane z, box 顶面表, 斜坡表)。 */
  _surfaces(geoms) {
    this._base = null;
    this._boxes = [];
    this._ramps = [];
    for (const g of geoms) {
      const [px, py, pz] = g.pos;
      if (g.type === 'plane') { this._base = this._base === null ? pz : Math.max(this._base, pz); continue; }
      const [hx, hy, hz] = g.size;
      const phi = (g.euler ?? [0, 0, 0])[1];          // 本仓只用绕 +y 的斜坡
      if (Math.abs(phi) < 1e-12) {
        this._boxes.push([px - hx, px + hx, py - hy, py + hy, pz + hz]);
      } else {
        const c = Math.cos(phi), sn = Math.sin(phi);
        const tcx = px + sn * hz, tcz = pz + c * hz;   // 顶面中心 = pos + R_y(phi)·(0,0,hz)
        const halfSpan = hx * c;                        // 顶面在 x 上的投影
        const slope = -sn / c;                          // 顶面的 dz/dx
        this._ramps.push([tcx - Math.abs(halfSpan), tcx + Math.abs(halfSpan), py - hy, py + hy, tcx, tcz, slope]);
      }
    }
  }

  /** 拿 height_at_ref 自检 (契约 §1.2: python 侧采样点, 容差 1e-9)。 */
  checkHeightRef(tol = 1e-9) {
    const ref = this.meta.height_at_ref ?? [];
    let max = 0, worst = null;
    for (const [x, y, h] of ref) {
      const d = Math.abs(this.heightAt(x, y) - h);
      if (d > max) { max = d; worst = { x, y, js: this.heightAt(x, y), py: h }; }
    }
    return { n: ref.length, max, worst, pass: ref.length > 0 && max <= tol };
  }

  /** 台阶级 (sim2sim/terrains.py stair_level_at); 非楼梯地形返回 null。 */
  stairLevelAt(x, y = 0.0) {
    const st = this.meta.stairs;
    if (!st || !st.length) return null;
    const r = this.meta.name.startsWith('pyramid') ? Math.hypot(x - this.startXY[0], y - this.startXY[1]) : x;
    for (const [x0, x1, lvl] of st) if (x0 <= r && r < x1) return lvl;
    return null;
  }

  get maxStairLevel() { return this.meta.max_stair_level; }

  /** MJCF 资产片段 (只有 hfield 需要): build.py 留的 @HFIELD@ 占位。 */
  assetXml() {
    if (!this.meta.hfield) return '';
    const s = this.meta.hfield.size;
    return `<hfield name="${this.meta.hfield.name}" nrow="${this.nrow}" ncol="${this.ncol}" size="${s[0]} ${s[1]} ${s[2]} ${s[3]}"/>`;
  }

  /** MJCF geom 片段: **按 `meta.geoms` 列表建** (plane / hfield / 以后的 box 都走这里)。 */
  geomXml() {
    const list = this.meta.geoms ?? [];
    if (!list.length) throw new Error(`地形 ${this.meta.id} 没有 geoms 列表 (数据包太旧, 重跑 webui/build.py)`);
    return list.map((g) => {
      const a = [`name="${g.name ?? 'terrain'}"`, `type="${g.type}"`];
      if (g.type === 'hfield') a.push(`hfield="${g.hfield ?? 'terrain'}"`);
      // box: size = 半边长, euler = 弧度 XYZ (MuJoCo 约定; MJCF 默认 compiler angle=radian, 本仓 MJCF 就是)
      if (g.pos) a.push(`pos="${g.pos.join(' ')}"`);
      // hfield geom 的尺寸在 <hfield> 资产上, geom 上再给 size 会报 "attribute 'size' has too much data"
      if (g.size && g.type !== 'hfield') a.push(`size="${g.size.join(' ')}"`);
      if (g.euler && g.euler.some((v) => v !== 0)) a.push(`euler="${g.euler.join(' ')}"`);
      const common = this.meta.geom ?? {};             // 公共属性: 所有地形 geom 共用一套 (契约 §1.2)
      const pick = (k) => (g[k] !== undefined ? g[k] : common[k]);
      if (pick('group') !== undefined) a.push(`group="${pick('group')}"`);
      if (pick('condim') !== undefined) a.push(`condim="${pick('condim')}"`);
      if (pick('friction')) a.push(`friction="${pick('friction').join(' ')}"`);
      if (pick('priority') !== undefined) a.push(`priority="${pick('priority')}"`);
      if (pick('rgba')) a.push(`rgba="${pick('rgba').join(' ')}"`);
      return `<geom ${a.join(' ')}/>`;
    }).join('\n  ');
  }

  /** 把绝对高度归一化写进 model.hfield_data (MuJoCo 要 [0,1] × elev + pos.z); plane 没有这一步。 */
  fillHfield(hfieldData) {
    if (this.isPlane) return;
    if (hfieldData.length !== this.nrow * this.ncol) throw new Error(`hfield_data 长度 ${hfieldData.length} != ${this.nrow * this.ncol}`);
    const inv = 1.0 / this.elev, z0 = this.zmin, h = this.h;
    for (let i = 0; i < h.length; i++) hfieldData[i] = (h[i] - z0) * inv;
  }
}
