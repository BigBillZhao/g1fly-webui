/**
 * webui/js/app/viewer.mjs — three.js 场景 (机器人 + hfield 地形 + 跟随/自由相机)。
 *
 * 只吃 MuJoCo 的模型/数据数组, 不反过来驱动物理 (物理在 js/sim/loop.mjs, 与 node 对齐验收共用一份)。
 * 机器人网格直接从 `model.mesh_vert/mesh_face/mesh_normal` 建 BufferGeometry (参考
 * ~/mujoco_web_viz_g1/.../src/mujoco-viewer.ts), 每帧用 `data.geom_xpos/geom_xmat` 刷矩阵 —— 比
 * `mjv_updateScene` + 每帧 `geoms.get(i).delete()` 省掉一堆 embind 垃圾。
 * 地形不走 MuJoCo (绑定里 hfield 落 default 分支 = 空几何), 直接用预烘高度自己建网格。
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const GEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };

export class Viewer {
  constructor(host) {
    this.host = host;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1419);
    this.scene.fog = new THREE.Fog(0x0f1419, 14, 40);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.02, 200);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-2.2, -2.6, 1.8);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    host.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0.8);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.follow = true;
    this._lastTarget = new THREE.Vector3(0, 0, 0.8);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(3, -4, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x9fc8ff, 0.35);
    fill.position.set(-4, 3, 2);
    this.scene.add(fill);

    this.robotRoot = new THREE.Group();
    this.scene.add(this.robotRoot);
    this.terrainMesh = null;
    this.geomMeshes = [];       // [{gid, mesh}]
    this.scanPoints = null;
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this.resize();
  }

  resize() {
    const w = this.host.clientWidth || 640, h = this.host.clientHeight || 480;
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  /** hfield 地形网格。**stride 必须是 1**: 深度相机也看这块网格, 抽稀会让渲染出的深度与 MuJoCo 的
   *  hfield 几何差半格 (实测 stride=2 时策略输入 max Δ 6.7e-3 vs stride=1 的 1.5e-3)。 */
  buildTerrain(terrain, stride = 1) {
    this.disposeTerrain();
    if (terrain.repr === 'boxes') return this._buildGeoms(terrain);
    if (terrain.isPlane) return this._buildPlane(terrain);
    const { nrow, ncol, res, x0, y0 } = terrain;
    const nx = Math.floor((ncol - 1) / stride) + 1, ny = Math.floor((nrow - 1) / stride) + 1;
    const pos = new Float32Array(nx * ny * 3);
    const col = new Float32Array(nx * ny * 3);
    let zmin = Infinity, zmax = -Infinity;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const r = j * stride, c = i * stride, k = (j * nx + i) * 3;
        const z = terrain.h[r * ncol + c];
        pos[k] = x0 + c * res; pos[k + 1] = y0 + r * res; pos[k + 2] = z;
        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
      }
    }
    const span = Math.max(zmax - zmin, 1e-3);
    for (let n = 0; n < nx * ny; n++) {
      const t = (pos[n * 3 + 2] - zmin) / span;                 // 低=青灰, 高=暖灰
      col[n * 3] = 0.34 + 0.40 * t; col[n * 3 + 1] = 0.38 + 0.34 * t; col[n * 3 + 2] = 0.44 + 0.20 * t;
    }
    const idx = new Uint32Array((nx - 1) * (ny - 1) * 6);
    let p = 0;
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;   // 绕序要让法线朝 +z, 否则地形全黑
        idx[p++] = a; idx[p++] = b; idx[p++] = c;
        idx[p++] = b; idx[p++] = d; idx[p++] = c;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    this.terrainMesh = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 4, flatShading: false }));
    this.terrainMesh.layers.enable(1);            // 深度相机的可见集 (layer 1) = 地形 + 腿部 visual mesh
    this.scene.add(this.terrainMesh);
    // 网格线用粗得多的一套 (0.4 m): 0.05 m 的线框在 700x240 格上只剩摩尔纹
    const gw = Math.max(1, Math.round(0.4 / (res * stride))) * stride;
    this.terrainWire = new THREE.LineSegments(this._gridLines(terrain, gw), new THREE.LineBasicMaterial({ color: 0x3c4c5c, transparent: true, opacity: 0.45 }));
    this.scene.add(this.terrainWire);
    return { nx, ny, verts: nx * ny };
  }

  _gridLines(terrain, stride) {
    const { nrow, ncol, res, x0, y0 } = terrain;
    const pts = [];
    const z = (r, c) => terrain.h[r * ncol + c] + 0.004;
    for (let r = 0; r < nrow; r += stride) {
      for (let c = 0; c + stride < ncol; c += stride) {
        pts.push(x0 + c * res, y0 + r * res, z(r, c), x0 + (c + stride) * res, y0 + r * res, z(r, c + stride));
      }
    }
    for (let c = 0; c < ncol; c += stride) {
      for (let r = 0; r + stride < nrow; r += stride) {
        pts.push(x0 + c * res, y0 + r * res, z(r, c), x0 + c * res, y0 + (r + stride) * res, z(r + stride, c));
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3));
    return g;
  }

  /** 解析几何地形 (底 plane + 若干 box/斜坡 box): 照 geoms 建 three 的 Mesh, 与 MuJoCo 同一套数
   *  (box size = 半边长 ⇒ BoxGeometry 要 ×2; euler 是弧度, three 的 Euler 也是弧度, 直接给)。 */
  _buildGeoms(terrain) {
    const group = new THREE.Group();
    const mat = new THREE.MeshPhongMaterial({ color: 0x8e979f, shininess: 6, flatShading: true });
    let zmin = Infinity, zmax = -Infinity;
    for (const g of terrain.geoms) {
      if (g.type === 'plane') { this._buildPlane(terrain, -6, 32, -10, 10, g.pos[2]); continue; }
      const [hx, hy, hz] = g.size;
      const m = new THREE.Mesh(new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz), mat);
      m.position.set(g.pos[0], g.pos[1], g.pos[2]);
      const e = g.euler ?? [0, 0, 0];
      m.rotation.set(e[0], e[1], e[2]);                 // 弧度, 与 MuJoCo euler 同
      m.layers.enable(1);                                // 深度相机看得见
      group.add(m);
      zmin = Math.min(zmin, g.pos[2] - hz); zmax = Math.max(zmax, g.pos[2] + hz);
    }
    this.terrainBoxes = group;
    this.scene.add(group);
    return { geoms: terrain.geoms.length, z: [zmin, zmax] };
  }

  /** 真 plane 地形 (无限): 画一块够大的方地 + 网格线 (物理上是 MuJoCo 的 plane geom, 这里只是好看)。 */
  _buildPlane(terrain, x0 = -6, x1 = 32, y0 = -10, y1 = 10, zOverride = null) {
    const z = zOverride === null ? terrain.heightConst : zOverride;
    const pos = new Float32Array([x0, y0, z, x1, y0, z, x0, y1, z, x1, y1, z]);
    const col = new Float32Array(12).fill(0);
    for (let i = 0; i < 4; i++) { col[3 * i] = 0.44; col[3 * i + 1] = 0.47; col[3 * i + 2] = 0.52; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array([0, 1, 2, 1, 3, 2]), 1));
    g.computeVertexNormals();
    this.terrainMesh = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 4 }));
    this.terrainMesh.layers.enable(1);
    this.scene.add(this.terrainMesh);
    const pts = [];
    for (let x = Math.ceil(x0); x <= x1; x++) pts.push(x, y0, z + 0.004, x, y1, z + 0.004);
    for (let y = Math.ceil(y0); y <= y1; y++) pts.push(x0, y, z + 0.004, x1, y, z + 0.004);
    const gl = new THREE.BufferGeometry();
    gl.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3));
    this.terrainWire = new THREE.LineSegments(gl, new THREE.LineBasicMaterial({ color: 0x3c4c5c, transparent: true, opacity: 0.45 }));
    this.scene.add(this.terrainWire);
    return { nx: 2, ny: 2, verts: 4, plane: true };
  }

  disposeTerrain() {
    for (const m of [this.terrainMesh, this.terrainWire]) {
      if (m) { this.scene.remove(m); m.geometry.dispose(); m.material.dispose(); }
    }   // terrainMesh 与 wire 现在各有自己的 geometry, 分别 dispose
    if (this.terrainBoxes) {
      for (const c of this.terrainBoxes.children) c.geometry.dispose();
      this.terrainBoxes.children[0]?.material?.dispose();
      this.scene.remove(this.terrainBoxes);
      this.terrainBoxes = null;
    }
    this.terrainMesh = this.terrainWire = null;
  }

  /** 机器人: 只画视觉 geom (group 1/2), 碰撞 geom (3) 与地形 (0) 不画。 */
  buildRobot(mj, model) {
    this.disposeRobot();
    for (let g = 0; g < model.ngeom; g++) {
      const grp = model.geom_group[g];
      if (grp !== 1 && grp !== 2) continue;
      const geometry = this._geometryFor(model, g);
      if (!geometry) continue;
      const mat = new THREE.MeshPhongMaterial({
        color: new THREE.Color(model.geom_rgba[4 * g], model.geom_rgba[4 * g + 1], model.geom_rgba[4 * g + 2]),
        shininess: 28, specular: 0x222222,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.matrixAutoUpdate = false;
      if (grp === 1) mesh.layers.enable(1);       // group 1 = 左右腿 6 link 的 visual (训练侧 mesh_prim_paths 同集)
      this.robotRoot.add(mesh);
      this.geomMeshes.push({ gid: g, mesh });
    }
    return this.geomMeshes.length;
  }

  _geometryFor(model, g) {
    const t = model.geom_type[g], s = [model.geom_size[3 * g], model.geom_size[3 * g + 1], model.geom_size[3 * g + 2]];
    switch (t) {
      case GEOM.MESH: {
        const mid = model.geom_dataid[g];
        if (mid < 0) return null;
        const va = model.mesh_vertadr[mid], vn = model.mesh_vertnum[mid];
        const fa = model.mesh_faceadr[mid], fn = model.mesh_facenum[mid];
        const verts = new Float32Array(vn * 3);
        for (let i = 0; i < verts.length; i++) verts[i] = model.mesh_vert[va * 3 + i];
        const idx = new Uint32Array(fn * 3);
        for (let i = 0; i < idx.length; i++) idx[i] = model.mesh_face[fa * 3 + i];
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geo.setIndex(new THREE.BufferAttribute(idx, 1));
        const na = model.mesh_normaladr ? model.mesh_normaladr[mid] : -1;
        const nn = model.mesh_normalnum ? model.mesh_normalnum[mid] : 0;
        if (na >= 0 && nn === vn) {
          const nor = new Float32Array(vn * 3);
          for (let i = 0; i < nor.length; i++) nor[i] = model.mesh_normal[na * 3 + i];
          geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
        } else {
          geo.computeVertexNormals();
        }
        return geo;
      }
      case GEOM.SPHERE: return new THREE.SphereGeometry(s[0], 20, 12);
      case GEOM.BOX: return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      case GEOM.CAPSULE: { const c = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16); c.rotateX(Math.PI / 2); return c; }
      case GEOM.CYLINDER: { const c = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 20); c.rotateX(Math.PI / 2); return c; }
      case GEOM.ELLIPSOID: { const e = new THREE.SphereGeometry(1, 20, 12); e.scale(s[0], s[1], s[2]); return e; }
      default: return null;     // PLANE / HFIELD 自己画
    }
  }

  disposeRobot() {
    for (const { mesh } of this.geomMeshes) { this.robotRoot.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
    this.geomMeshes.length = 0;
  }

  /** 每帧: MuJoCo geom 位姿 -> three 矩阵。 */
  sync(data) {
    for (const { gid, mesh } of this.geomMeshes) {
      const p = 3 * gid, m = 9 * gid;
      mesh.matrix.set(
        data.geom_xmat[m], data.geom_xmat[m + 1], data.geom_xmat[m + 2], data.geom_xpos[p],
        data.geom_xmat[m + 3], data.geom_xmat[m + 4], data.geom_xmat[m + 5], data.geom_xpos[p + 1],
        data.geom_xmat[m + 6], data.geom_xmat[m + 7], data.geom_xmat[m + 8], data.geom_xpos[p + 2],
        0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
    }
  }

  /** 高程扫描命中点 (187) 的小点云。 */
  syncScan(scanner) {
    if (!this.scanPoints) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(scanner.n * 3), 3));
      this.scanPoints = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xff9f43, size: 0.035 }));
      this.scene.add(this.scanPoints);
    }
    const arr = this.scanPoints.geometry.attributes.position.array;
    for (let i = 0; i < scanner.n * 3; i++) arr[i] = scanner.hits[i];
    this.scanPoints.geometry.attributes.position.needsUpdate = true;
  }

  setFollowTarget(x, y, z) {
    if (!this.follow) return;
    const dx = x - this._lastTarget.x, dy = y - this._lastTarget.y, dz = z - this._lastTarget.z;
    this.camera.position.x += dx; this.camera.position.y += dy; this.camera.position.z += dz;
    this.controls.target.set(x, y, z);
    this._lastTarget.set(x, y, z);
  }

  resetCamera(x, y, z) {
    this._lastTarget.set(x, y, z);
    this.controls.target.set(x, y, z);
    this.camera.position.set(x - 2.2, y - 2.6, z + 1.0);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    window.removeEventListener('resize', this._onResize);
    this.disposeRobot();
    this.disposeTerrain();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
