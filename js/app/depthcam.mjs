/**
 * webui/js/app/depthcam.mjs — 头部 D435 深度相机 (three.js), 出与 python 台 `mjscene.DepthRig` 同口径的
 * **线性 z-depth** (米, `distance_to_image_plane`), 供 js/sim/depth.mjs 预处理后喂学生策略。
 *
 * 口径 (contracts/depth_frame_v1.md + sim2sim/README.md §3.3):
 *   外参 torso_link + pos_b, ROS 四元数 -> MuJoCo 相机系 R_mj = R_ros·diag(1,−1,−1) (= three 的相机系: x 右 / y 上 / −z 前)
 *   内参 64×36, fx/fy/cx/cy (D435i ROI 推的非对称主点) -> 自己拼投影矩阵 (three 的 PerspectiveCamera 是对称的, 不能用)
 *        left=−cx·n/fx  right=(W−cx)·n/fx  top=cy·n/fy  bottom=−(H−cy)·n/fy   (像素中心 u+0.5 与 MuJoCo 光线模型同)
 *   可见集 = 地形 + 左右腿 6 link 的 visual mesh (自遮挡) —— 用 three 的 layer 1 圈定 (viewer 建网格时打的标)
 *   深度值 = 透视校正插值的 −viewZ (线性, 不是 gl_FragCoord.z), 写进 float RT 后读回; 空像素 = far (2.5 m 外, 预处理里会被压到 max)
 */

import * as THREE from 'three';

const DEPTH_LAYER = 1;

const VERT = `
varying float vViewZ;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewZ = -mv.z;                       // 相机前方为正; varying 是透视校正插值 ⇒ 逐片元精确
  gl_Position = projectionMatrix * mv;
}`;
const FRAG = `
varying float vViewZ;
void main() { gl_FragColor = vec4(vViewZ, 0.0, 0.0, 1.0); }`;

function quatToMat3(w, x, y, z) {
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

export class DepthCamera {
  constructor(renderer, scene, cfg, { near = 0.05, far = 20.0 } = {}) {
    this.renderer = renderer;
    this.scene = scene;
    this.cfg = cfg;
    this.W = cfg.width; this.H = cfg.height;
    this.near = near; this.far = far;

    this.camera = new THREE.PerspectiveCamera();
    this.camera.matrixAutoUpdate = false;
    this.camera.layers.set(DEPTH_LAYER);
    const n = near;
    const left = -cfg.cx * n / cfg.fx, right = (this.W - cfg.cx) * n / cfg.fx;
    const top = cfg.cy * n / cfg.fy, bottom = -(this.H - cfg.cy) * n / cfg.fy;
    this.camera.projectionMatrix.makePerspective(left, right, top, bottom, near, far);
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    this.frustum = { left, right, top, bottom };

    this.material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide });
    this.target = new THREE.WebGLRenderTarget(this.W, this.H, {
      type: THREE.FloatType, format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true, stencilBuffer: false,
    });
    this.pixels = new Float32Array(this.W * this.H * 4);
    this.depth = new Float32Array(this.W * this.H);      // 行 0 = 图像上边 (与 MuJoCo Renderer 同)

    // 相机外参 (torso 体坐标): R_mj = R_ros · diag(1,−1,−1)
    const q = cfg.quat_ros_b;
    const nq = Math.hypot(q[0], q[1], q[2], q[3]);
    const R = quatToMat3(q[0] / nq, q[1] / nq, q[2] / nq, q[3] / nq);
    this.R_mj_b = [R[0], -R[1], -R[2], R[3], -R[4], -R[5], R[6], -R[7], -R[8]];   // 列 1,2 取反
    this.pos_b = cfg.pos_b;
    this._m = new THREE.Matrix4();
  }

  /** 从 MuJoCo 的 torso 位姿摆相机。 */
  updatePose(data, torsoId) {
    const p = [data.xpos[3 * torsoId], data.xpos[3 * torsoId + 1], data.xpos[3 * torsoId + 2]];
    const T = [];
    for (let i = 0; i < 9; i++) T.push(data.xmat[9 * torsoId + i]);          // 行主序
    const Rc = new Array(9);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += T[3 * r + k] * this.R_mj_b[3 * k + c];
        Rc[3 * r + c] = s;
      }
    }
    const t = [p[0], p[1], p[2]];
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) t[r] += T[3 * r + k] * this.pos_b[k];
    this._m.set(Rc[0], Rc[1], Rc[2], t[0], Rc[3], Rc[4], Rc[5], t[1], Rc[6], Rc[7], Rc[8], t[2], 0, 0, 0, 1);
    this.camera.matrix.copy(this._m);
    this.camera.matrixWorld.copy(this._m);
    this.camera.matrixWorldInverse.copy(this._m).invert();
    return this;
  }

  /** 渲染一帧 -> Float32Array(H*W) 线性 z-depth (米), 行 0 = 上边。 */
  render() {
    const r = this.renderer, prevTarget = r.getRenderTarget(), prevOverride = this.scene.overrideMaterial;
    const prevClear = r.getClearColor(new THREE.Color()), prevAlpha = r.getClearAlpha();
    this.scene.overrideMaterial = this.material;
    r.setRenderTarget(this.target);
    r.setClearColor(0x000000, 1);          // 空像素 = 0 -> 读回时换成 far (MuJoCo 那边是 zfar, 预处理都压到 max_depth)
    r.clear(true, true, false);
    r.render(this.scene, this.camera);
    r.readRenderTargetPixels(this.target, 0, 0, this.W, this.H, this.pixels);
    r.setRenderTarget(prevTarget);
    r.setClearColor(prevClear, prevAlpha);
    this.scene.overrideMaterial = prevOverride;
    const W = this.W, H = this.H, out = this.depth;
    for (let y = 0; y < H; y++) {
      const src = (H - 1 - y) * W;         // GL 读回是下->上, MuJoCo 的图是上->下
      for (let x = 0; x < W; x++) {
        const v = this.pixels[(src + x) * 4];
        // 空像素 (清成 0) 与**近裁剪面以内**的片元都算"没东西": MuJoCo 把跨近平面的三角形裁掉 -> 背景 zfar,
        // 而 GL 这边裁出来的碎片 (以及跨相机的三角形那种 w 变号的垃圾插值) 会留下 <near 的值。
        // 这两种情形都出现在"机器人自己的腿贴上/穿过头部相机"时 (抬膝、摔倒), 不这么处理逐像素差会到 0.2 m。
        out[y * W + x] = v >= this.near ? v : this.far;
      }
    }
    return out;
  }

  dispose() { this.target.dispose(); this.material.dispose(); }
}

export { DEPTH_LAYER };
