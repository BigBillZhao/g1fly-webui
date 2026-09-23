/**
 * webui/js/sim/mjload.mjs — 载 MuJoCo WASM (@mujoco/mujoco 3.10.0) + 按 python 台的口径拼模型。
 *
 * 逐条对应 `sim2sim/mjscene.py::build_model` + `sim2sim/run.py::configure_actuators` (改了哪条就是脱钩):
 *   1. XML: meshdir 已由 build.py 改成 "meshes"; @HFIELD@ 换成 hfield 资产 (plane 地形为空),
 *      @TERRAIN_GEOM@ 换成**地形 geoms 列表** (plane / hfield / 以后的 box 都只是列表里的一条)
 *   2. geom 分组与碰撞掩码: 视觉 geom -> group 1 (腿) / 2 (其余), 碰撞 geom -> group 3 + contype 1 / conaffinity 0
 *      (训练 enabled_self_collisions=False); 地形 geom -> group 0 + contype 1 / conaffinity 1
 *   3. hfield_data <- 预烘高度 (归一化)
 *   4. 全 geom solref = (0.005, 1) + solimp = (0.99, 0.999, 1e-4)   (sim2sim README trap #14 与 #26:
 *      默认接触太软 —— solref 决定时间常数, solimp 决定阻抗斜坡, 少一个 0.8 m/s 就摔)
 *   5. 29 个 motor 改成位置伺服 = IsaacLab ImplicitActuator (gaintype FIXED / biastype AFFINE / forcerange = URDF effort)
 *   6. 骨盆 +1 kg (+1e-4 惯量) 后 mj_setConst   (README trap #16: Isaac 的无质量 base 链被 PhysX 赋 1 kg)
 *   7. opt.timestep = 0.001 (decimation 20 @ 50 Hz 策略), integrator 沿用 MJCF 的 implicitfast
 *   8. 出生 z: 默认位姿下最低足球触地 + 2 mm
 *
 * WASM 绑定坑 (3.10, 见 webui/README.md trap ledger): embind 对象要 .delete(); 出参用 DoubleBuffer/IntBuffer
 * + .GetView(); 绑定里没有 mjr_* (无渲染器, 画面交给 three.js)。
 */

export const GROUP_TERRAIN = 0, GROUP_LEG_VIS = 1, GROUP_OTHER_VIS = 2, GROUP_COLLISION = 3;
export const LEG_LINKS = new Set(['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
  .flatMap((j) => [`left_${j}_link`, `right_${j}_link`]));
export const FOOT_BODIES = ['left_ankle_roll_link', 'right_ankle_roll_link'];
export const FOOT_SITES = ['left_foot', 'right_foot'];

/**
 * 载入 WASM 模块 (url = vendor/mujoco/mujoco.js 的 URL 或路径)。
 * @param opts.wasmBinary 可选: 已经拿到手的 mujoco.wasm 字节 (例如从 `.gz` 旁车解压出来的, 见 main.mjs)。
 *        Emscripten 的 `Module.wasmBinary` 一旦给了就不会再自己去 fetch .wasm。
 */
export async function loadMujoco(url, { wasmBinary = null } = {}) {
  const mod = await import(url);
  const factory = mod.default ?? mod;
  return await (wasmBinary ? factory({ wasmBinary }) : factory());
}

/** 占位替换: build.py 在 </asset> / </worldbody> 前留的注释。 */
export function injectTerrain(xml, terrain) {
  const out = xml.replace('<!-- @HFIELD@ -->', terrain.assetXml()).replace('<!-- @TERRAIN_GEOM@ -->', terrain.geomXml());
  if (out === xml) throw new Error('MJCF 里找不到 @HFIELD@ / @TERRAIN_GEOM@ 占位 (build.py 版本不对?)');
  return out;
}

const nameOf = (mj, model, type, id) => mj.mj_id2name(model, type, id) ?? '';

/**
 * @param {object} mj      loadMujoco() 的返回
 * @param {object} robot   {xml, meshes: Map<name, Uint8Array>}
 * @param {Terrain} terrain
 * @param {object} contract  policies/<id>/contract.json
 * @returns {{model, data, ids, spawn, dispose}}
 */
export function buildModel(mj, robot, terrain, contract) {
  const xml = injectTerrain(robot.xml, terrain);
  const vfs = new mj.MjVFS();
  try {
    vfs.addBuffer('model.xml', new Uint8Array(new TextEncoder().encode(xml)));
    for (const [f, bytes] of robot.meshes) vfs.addBuffer('meshes/' + f, bytes);
    var model = mj.MjModel.from_xml_path('model.xml', vfs);
  } finally {
    vfs.delete();
  }
  const OBJ = mj.mjtObj;
  const bodyName = (id) => nameOf(mj, model, OBJ.mjOBJ_BODY.value, id);

  // --- 2. 分组 + 碰撞掩码 ---
  // 地形可能不止一个 geom (plane / hfield / 以后的 box 列表), 名字来自 meta.geoms
  const terrainNames = (terrain.meta.geoms ?? []).map((g) => g.name ?? 'terrain');
  const terrainGeoms = terrainNames.map((n) => mj.mj_name2id(model, OBJ.mjOBJ_GEOM.value, n)).filter((i) => i >= 0);
  const terrainSet = new Set(terrainGeoms);
  const terrainGeom = terrainGeoms[0];
  for (let g = 0; g < model.ngeom; g++) {
    if (terrainSet.has(g)) continue;
    const b = bodyName(model.geom_bodyid[g]);
    if (b === 'world' || b === '') continue;
    const visual = model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0;
    if (visual) {
      model.geom_group[g] = LEG_LINKS.has(b) ? GROUP_LEG_VIS : GROUP_OTHER_VIS;
    } else {
      model.geom_group[g] = GROUP_COLLISION;
      if (!contract.physics.self_collision) { model.geom_contype[g] = 1; model.geom_conaffinity[g] = 0; }
    }
  }
  for (const g of terrainGeoms) {
    model.geom_group[g] = GROUP_TERRAIN;
    model.geom_contype[g] = 1;
    model.geom_conaffinity[g] = 1;
  }

  // --- 3. hfield 高度 (只有 repr=hfield 才有这一步; plane/boxes 是解析几何) ---
  if (terrain.repr === 'hfield') terrain.fillHfield(model.hfield_data);

  // --- 4. 接触刚度 (solref) + 阻抗斜坡 (solimp) ---
  const [tc, dr] = contract.physics.solref;
  for (let g = 0; g < model.ngeom; g++) { model.geom_solref[2 * g] = tc; model.geom_solref[2 * g + 1] = dr; }
  const simp = contract.physics.solimp;          // (d0, d1, width): python 台 09-21 才加 (0.8 m/s 缝隙的另一半)
  if (simp && simp.length) {
    const NIMP = model.geom_solimp.length / model.ngeom;    // MuJoCo: 5 个一组
    for (let g = 0; g < model.ngeom; g++) {
      for (let k = 0; k < Math.min(simp.length, NIMP); k++) model.geom_solimp[NIMP * g + k] = simp[k];
    }
  }

  // --- 5. 执行器 = IsaacLab ImplicitActuator ---
  const kp = contract.joint.kp_sdk, kd = contract.joint.kd_sdk;
  const lim = contract.physics.torque_limit === 'train' ? contract.joint.effort_sdk_train
    : contract.physics.torque_limit === 'none' ? new Array(29).fill(1e6) : contract.joint.effort_sdk_urdf;
  const NG = model.actuator_gainprm.length / model.nu, NB = model.actuator_biasprm.length / model.nu;
  for (let i = 0; i < model.nu; i++) {
    model.actuator_gaintype[i] = 0;                       // mjGAIN_FIXED
    for (let j = 0; j < NG; j++) model.actuator_gainprm[i * NG + j] = 0;
    for (let j = 0; j < NB; j++) model.actuator_biasprm[i * NB + j] = 0;
    model.actuator_forcerange[2 * i] = -lim[i];      // forcelimited 在 MJCF 里钉死 (绑定读不了 mjtByte, trap #2)
    model.actuator_forcerange[2 * i + 1] = lim[i];
    model.actuator_biastype[i] = 1;                       // mjBIAS_AFFINE
    model.actuator_gainprm[i * NG + 0] = kp[i];
    model.actuator_biasprm[i * NB + 1] = -kp[i];
    model.actuator_biasprm[i * NB + 2] = -kd[i];
  }

  // --- 6. 骨盆配重 ---
  const pelvis = mj.mj_name2id(model, OBJ.mjOBJ_BODY.value, 'pelvis');
  if (contract.physics.extra_base_mass) {
    model.body_mass[pelvis] += contract.physics.extra_base_mass;
    for (let k = 0; k < 3; k++) model.body_inertia[3 * pelvis + k] += 1e-4;
    const tmp = new mj.MjData(model);
    mj.mj_setConst(model, tmp);
    tmp.delete();
  }

  // --- 7. 步长 ---
  model.opt.timestep = contract.timing.sim_dt;

  const ids = {
    base: pelvis,
    torso: mj.mj_name2id(model, OBJ.mjOBJ_BODY.value, 'torso_link'),
    terrainGeom, terrainGeoms,
    footBodies: FOOT_BODIES.map((n) => mj.mj_name2id(model, OBJ.mjOBJ_BODY.value, n)),
    footSites: FOOT_SITES.map((n) => mj.mj_name2id(model, OBJ.mjOBJ_SITE.value, n)),
  };

  // --- 8. 出生高度 (最低足球贴地 + 2 mm) ---
  const data = new mj.MjData(model);
  mj.mj_resetData(model, data);
  const [x0, y0] = terrain.startXY;
  data.qpos[0] = x0; data.qpos[1] = y0; data.qpos[2] = 1.0;
  data.qpos[3] = 1; data.qpos[4] = 0; data.qpos[5] = 0; data.qpos[6] = 0;
  const dflt = contract.joint.default_pos_sdk;
  for (let i = 0; i < 29; i++) data.qpos[7 + i] = dflt[i];
  mj.mj_forward(model, data);
  let lowest = Infinity;
  const footSet = new Set(ids.footBodies);
  for (let g = 0; g < model.ngeom; g++) {
    if (footSet.has(model.geom_bodyid[g]) && model.geom_group[g] === GROUP_COLLISION) {
      lowest = Math.min(lowest, data.geom_xpos[3 * g + 2] - model.geom_rbound[g]);
    }
  }
  const ground = terrain.heightAt(x0, y0);
  const spawn = [x0, y0, 1.0 - lowest + ground + 0.002];

  return {
    model, data, ids, spawn,
    dispose() { data.delete(); model.delete(); },
  };
}

/** reset 到默认位姿 (== python: mj_resetDataKeyframe(key "default") + mj_forward)。 */
export function resetToSpawn(mj, sim, contract) {
  const { model, data, spawn } = sim;
  mj.mj_resetData(model, data);
  data.qpos[0] = spawn[0]; data.qpos[1] = spawn[1]; data.qpos[2] = spawn[2];
  data.qpos[3] = 1; data.qpos[4] = 0; data.qpos[5] = 0; data.qpos[6] = 0;
  const dflt = contract.joint.default_pos_sdk;
  for (let i = 0; i < 29; i++) data.qpos[7 + i] = dflt[i];
  data.time = 0.0;
  mj.mj_forward(model, data);
}
