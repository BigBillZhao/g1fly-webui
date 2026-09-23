/**
 * webui/js/app/main.mjs — 页面装配: 载数据包 -> 起 SimLoop (js/sim, 与 node 对齐验收同一份) -> three.js 画面 + 键盘 + HUD。
 *
 * 键盘: W/S vx · A/D vy · Q/E wz · 空格停 · R 重置 · P 暂停 · C 相机跟随/自由。
 * 左面板 (脑) 与右下深度小窗是 W2/W3 的占位。
 */

import { GZIP_SUPPORTED, fetchMaybeGz, httpIO, loadIndex, loadPolicyBundle, loadRobot, loadTerrainData } from '../sim/bundle.mjs';
import { loadMujoco } from '../sim/mjload.mjs';
import { Terrain } from '../sim/terrain.mjs';
import { SimLoop } from '../sim/loop.mjs';
import { createPolicy, initOrt, ortVersion } from '../sim/policy.mjs';
import { Viewer } from './viewer.mjs';
import { DepthCamera } from './depthcam.mjs';
import { BrainPanel, drawZBar } from './brain.mjs';

const $ = (id) => document.getElementById(id);
const base = document.baseURI;
const CMD_STEP = { vx: 0.1, vy: 0.1, wz: 0.1 };
// 首屏进度: vendor 的两个 wasm 由运行时自己取 (拿不到分块进度), 按固定权重记; 数据包走 io 的回调按字节记。
const VENDOR_BYTES = { mujoco: 10.1e6, ort: 14.2e6 };
const DEFAULT_POLICY = 'T_core3_w2';        // 演示台的主角 (09-23 起): **全神经元版** —— MaleCNS VNC 的
                                            // 全部 DN→腿前运动→腿 MN, core3_w5 **n=9,156** 个神经元的 EI 核在驱动腿。
                                            // 两段式 + softplus ⇒ 脑下行通路是活的 (z 不恒 0, DN 热条会亮;
                                            // 深度消融 stairs_up 真/空 9.67 → 7.19 m, Δ −2.48), Isaac eval 0.82 (README §4.9)。
                                            // 第二默认 = `T_graph_w2` (t1_w5 3,609, eval 0.87); 第一波 `T_graph` 留着当
                                            // 负结果对照 (z≡0, 热条全暗, §4.7)。
                                            // 代价: 首屏 EI ONNX 329 MB (gz 11.9 MB 下行) + 浏览器里 ~1.4 GB 常驻内存,
                                            // WASM 9.5 ms/步 (20 ms 预算的 47%)。想省就 ?policy=T_graph_w2 / R2_student_E_v1
const DEFAULT_TERRAIN = 'flat';
const P = { done: 0, total: 0, wire: 0 };        // done/total = 解压后字节 (进度条口径); wire = 真正下行的字节

const S = {
  mj: null, io: null, index: null, robot: null, viewer: null, loop: null,
  policyCache: new Map(), terrainCache: new Map(), depthCam: null, brain: null, brainGraph: null, brainData: new Map(),
  cmd: [0, 0, 0], running: true, fps: 0, stepsPerFrame: 0, err: null,
};

function progressTotal(index, policyBytes) {
  return VENDOR_BYTES.mujoco + VENDOR_BYTES.ort + 1.3e6 /* three */ +
    (index?.robot?.mesh_bytes ?? 20e6) + 0.7e6 /* 一块地形 */ + (policyBytes ?? 11e6);
}

function progress(add = 0, msg = null) {
  P.done = Math.min(P.done + add, P.total || Infinity);
  const pct = P.total ? Math.min(100, 100 * P.done / P.total) : 0;
  const fill = $('boot-fill'), by = $('boot-bytes');
  if (fill) fill.style.width = pct.toFixed(1) + '%';
  // 进度按**解压后**字节 (数据包里的 bytes 口径); 括号里是真正下行的字节 (有 .gz 旁车时小得多)
  const wire = (S.io ? S.io.stats.wire : 0) + P.wire;
  const tail = wire && wire < P.done * 0.97 ? ` (下行 ${(wire / 1048576).toFixed(1)} MB)` : '';
  if (by) by.textContent = `${(P.done / 1048576).toFixed(1)} / ${(P.total / 1048576).toFixed(1)} MB${tail}` + (msg ? ` · ${msg}` : '');
}

/** vendor 的两个 wasm: 优先取 `.gz` 旁车 (Pages 不会给 application/wasm 压缩), 失败就让运行时自己去 fetch。 */
async function vendorWasm(rel, label) {
  try {
    const r = await fetchMaybeGz(new URL(rel, base).href, true);
    P.wire += r.wire;
    console.log(`[g1fly] ${label}: ${(r.bytes.byteLength / 1e6).toFixed(1)} MB` +
                (r.gz ? ` (下行 ${(r.wire / 1e6).toFixed(1)} MB, .gz 旁车)` : ' (原文件)'));
    return r.bytes;
  } catch (e) {
    console.warn(`[g1fly] ${label} 预取失败, 交回运行时自己 fetch:`, e);
    return null;                                  // 兜底: 让 emscripten / ort 自己去拉 .wasm
  }
}

const T0 = performance.now();
function boot(msg) {
  const b = $('boot');
  if (b && !b.hidden) b.firstElementChild.textContent = msg;
  console.log(`[g1fly] +${(performance.now() - T0).toFixed(0)} ms ${msg}`);
}

/** 切地形/策略要 5–14 s (MJCF 重编译 + 17 万顶点的地形网格): 不给反馈的话看起来像卡死。 */
function busy(msg) {
  const b = $('boot');
  if (!b) return;
  if (msg) { b.firstElementChild.textContent = msg; b.hidden = false; b.style.background = 'rgba(15,20,25,.92)'; }
  else b.hidden = true;
}

function fail(e) {
  S.err = e;
  console.error(e);
  const pre = $('boot-err');
  if (pre && !$('boot').hidden) { pre.hidden = false; pre.textContent = String(e && e.stack || e); }
  $('foot-err').textContent = String(e && e.message || e);
}

async function main() {
  window.__g1 = S;          // 浏览器里调试/验证用: __g1.loop / __g1.viewer / __g1.cmd
  window.addEventListener('error', (ev) => fail(ev.error || ev.message));
  window.addEventListener('unhandledrejection', (ev) => fail(ev.reason));

  P.total = progressTotal(null, null);
  boot('载入 MuJoCo WASM (3.10.0)…');
  const mjBin = await vendorWasm('vendor/mujoco/mujoco.wasm', 'mujoco.wasm');
  S.mj = await loadMujoco(new URL('vendor/mujoco/mujoco.js', base).href, { wasmBinary: mjBin });
  progress(VENDOR_BYTES.mujoco + 1.3e6, 'mujoco wasm');
  boot('载入 onnxruntime-web (1.30.0, WASM SIMD)…');
  const ortBin = await vendorWasm('vendor/ort/ort-wasm-simd-threaded.wasm', 'ort wasm');
  await initOrt(new URL('vendor/ort/ort.wasm.bundle.min.mjs', base).href, new URL('vendor/ort/', base).href,
                { wasmBinary: ortBin });
  progress(VENDOR_BYTES.ort, 'onnxruntime wasm');

  S.io = httpIO(new URL('data/', base).href, (n) => progress(n));
  boot('读数据包 index.json…');
  S.index = await loadIndex(S.io);
  // Pages 包只带 .gz 旁车 (省仓库体积) ⇒ 没有 DecompressionStream 的老浏览器在这儿就要说清楚, 别让它去 404 一堆文件
  if (S.index.gz && S.index.gz.only && !GZIP_SUPPORTED) {
    throw new Error('这个部署只带了 gzip 旁车, 但你的浏览器没有 DecompressionStream。' +
                    '请用 Chrome/Edge 80+ · Firefox 113+ · Safari 16.4+ 打开 ' +
                    '(This deployment ships gzip sidecars only and needs DecompressionStream support).');
  }
  const q0 = new URLSearchParams(location.search);
  const wantP0 = S.index.policies.find((p) => p.name === q0.get('policy')) ?? S.index.policies.find((p) => p.name === DEFAULT_POLICY) ?? S.index.policies[0];
  P.total = progressTotal(S.index, wantP0?.bytes);
  boot(`载入机器人网格 (${S.index.robot.meshes.length} 个 STL, ${(S.index.robot.mesh_bytes / 1e6).toFixed(1)} MB)…`);
  S.robot = await loadRobot(S.io, S.index);

  const selT = $('sel-terrain'), selP = $('sel-policy');
  for (const t of S.index.terrains) selT.add(new Option(t.spec, t.id));
  // note = 数据包给的小字 (例: 随机权重的演示模型), 有就挂在名字后面, 别让人拿演示当策略看
  for (const p of S.index.policies) selP.add(new Option(p.note ? `${p.name} (${p.kind} · ${p.note})` : `${p.name} (${p.kind})`, p.name));
  // ?policy=…&terrain=… 可以直接开在指定组合上 (分享链接 / 取证时省掉切换那一趟)
  const q = new URLSearchParams(location.search);
  const wantT = (q.get('terrain') || '').replace(':', '_');
  const wantP = q.get('policy') || '';
  // 默认: 真会走路的 R2 学生 + flat (其余策略点选再拉, 不预取)
  selT.value = S.index.terrains.some((t) => t.id === wantT) ? wantT
    : (S.index.terrains.some((t) => t.id === DEFAULT_TERRAIN) ? DEFAULT_TERRAIN : S.index.terrains[0].id);
  selP.value = S.index.policies.some((p) => p.name === wantP) ? wantP
    : (S.index.policies.some((p) => p.name === DEFAULT_POLICY) ? DEFAULT_POLICY : (S.index.policies[0]?.name ?? ''));

  S.viewer = new Viewer($('canvas-host'));
  await rebuild(selT.value, selP.value);
  $('boot').hidden = true;

  selT.onchange = () => rebuild(selT.value, selP.value).catch(fail);
  selP.onchange = () => rebuild(selT.value, selP.value).catch(fail);
  $('btn-reset').onclick = () => doReset();
  $('btn-pause').onclick = () => togglePause();
  $('btn-cam').onclick = () => toggleCam();
  $('btn-help').onclick = () => toggleHelp(true);
  $('help-close').onclick = () => toggleHelp(false);
  $('help').onclick = (e) => { if (e.target === $('help')) toggleHelp(false); };
  document.addEventListener('keydown', onKey);

  $('foot-info').textContent = `mujoco-wasm ${S.mj.mj_versionString ? S.mj.mj_versionString() : S.mj.mj_version()} · ` +
    `ort-web ${ortVersion()?.web} · three r160 · 数据包 ${S.index.built}`;
  nextFrame(tick);
}

async function getTerrain(id) {
  if (!S.terrainCache.has(id)) {
    const { meta, heights } = await loadTerrainData(S.io, id);
    S.terrainCache.set(id, new Terrain(meta, heights));
  }
  return S.terrainCache.get(id);
}

async function getPolicy(name) {
  if (!S.policyCache.has(name)) {
    const bundle = await loadPolicyBundle(S.io, name);
    bundle.policy = await createPolicy(bundle);
    S.policyCache.set(name, bundle);
  }
  return S.policyCache.get(name);
}

/**
 * 脑面板: 只有 kind=bottleneck 且 brain.graph_name 非空 (core=ei_*) 才有点云 + module 条;
 * `core=gru` 的对照策略没有图, 但**仍然有 z** ⇒ 只隐点云与 module 条, DN z 热条照画 (open ⑰)。
 * **换图要重建点云** (t1_w5 3,609 ↔ core3_w5 9,156): 点数/module 数/MN 数全从神经元表读, 一个都不写死。
 */
async function setupBrain(bundle) {
  const b = bundle.meta.brain ?? null;
  const graph = b && b.graph_name;
  $('brain-none').style.display = graph ? 'none' : '';
  $('brain-host').style.display = graph ? '' : 'none';
  $('brain-info').style.display = graph ? '' : 'none';
  $('brain-mod-h').style.display = graph ? '' : 'none';
  $('brain-modules').style.display = graph ? '' : 'none';
  S.brainClip = b?.h_v_clip ?? 1.0;
  S.zClip = b?.z_clip ?? S.brainClip;
  S.hasZ = !!(bundle.meta.io?.outputs?.z);
  $('brain-ndn').textContent = S.hasZ ? (bundle.meta.brain?.dn_dim ?? '?')
    : (bundle.meta.kind === 'bottleneck' ? '无 (导出里没有 z)' : '—');
  if (!graph) {
    S.brain?.dispose(); S.brain = null; S.brainGraph = null;
    $('brain-title').textContent = b ? `${b.core} (无 connectome 图, 只有 z)` : '—';
    $('brain-nmod').textContent = '—';
    drawZBar($('brain-dn'), null, 1);                     // 清底色; 第一帧 z 由 HUD 那边画上去
    return;
  }
  if (!S.brainData.has(graph)) {
    boot(`载入神经元表 ${graph}…`);
    S.brainData.set(graph, await S.io.json(bundle.meta.brain.neurons_file ?? `brain/${graph}_neurons.json`));
  }
  if (!S.brain || S.brainGraph !== graph) {               // 换了图就整块重建 (点云是静态 buffer, 尺寸不能复用)
    S.brain?.dispose();
    S.brain = new BrainPanel($('brain-host'), $('brain-modules'), $('brain-dn'), $('brain-info'));
    const info = await S.brain.load(S.brainData.get(graph));
    S.brainGraph = graph;
    $('brain-nmod').textContent = info.modules;
    $('brain-title').textContent = `${graph} (${b.core}) · ${info.n} 点 · MN ${info.mn}`;
    console.log(`[g1fly] 脑面板: ${graph} ${info.n} 点 / ${info.modules} module / MN ${info.mn}`);
  } else {
    $('brain-title').textContent = `${graph} (${b.core}) · ${S.brain.n} 点 · MN ${S.brain.mnIdx.length}`;
  }
}

async function rebuild(terrainId, policyName) {
  busy(`切换到 ${terrainId} × ${policyName} … (要重编译 MJCF + 重建地形网格, 约 5–15 s)`);
  boot(`拼场景: ${terrainId} × ${policyName} — 取地形…`);
  const terrain = await getTerrain(terrainId);
  boot(`拼场景: 取策略 ${policyName}…`);
  const bundle = await getPolicy(policyName);
  if (S.loop) S.loop.dispose();
  boot('拼场景: 编译 MJCF (35 网格 + hfield)…');
  const usesDepth = bundle.meta.kind !== 'teacher';
  if (usesDepth && !S.depthCam) S.depthCam = new DepthCamera(S.viewer.renderer, S.viewer.scene, bundle.contract.depth);
  const renderDepth = usesDepth ? (loop) => {
    S.viewer.sync(loop.data);                       // 深度相机吃的是 three 场景, 先把位姿刷进去
    S.depthCam.updatePose(loop.data, loop.ids.torso);
    return S.depthCam.render();
  } : null;
  S.loop = new SimLoop(S.mj, { robot: S.robot, terrain, contract: bundle.contract, policy: bundle.policy, renderDepth,
    depthDelay: bundle.contract.depth?.delay_range?.[0] ?? 0 });   // 与 ref_traj.py 的默认一致 (口径不能两边不同)
  boot('拼场景: 建 three.js 网格…');
  S.loop.setCommand(...S.cmd);
  S.viewer.buildTerrain(terrain);
  S.viewer.buildRobot(S.mj, S.loop.model);
  S.viewer.sync(S.loop.data);
  S.viewer.resetCamera(S.loop.sim.spawn[0], S.loop.sim.spawn[1], S.loop.sim.spawn[2]);
  await setupBrain(bundle);
  S.limits = bundle.contract.cmd_limits || {};
  S.kind = bundle.meta.kind;
  $('depth-none').style.display = usesDepth ? 'none' : '';
  for (const id of ['depth-policy', 'depth-raw']) $(id).parentElement.style.display = usesDepth ? '' : 'none';
  $('mem-note').textContent = usesDepth ? 'policy_depth 的 GRU 输出 (512), 喂给 actor'
    : '教师策略没有深度记忆 (这条是 W2 的学生才有)';
  $('scan-note').textContent = usesDepth ? '橙框内 = actor 解出的 terrain_scan (学生的"想象"), 点云是真值'
    : '17×11 @0.1 m, 挂 torso, yaw 对齐 (近=亮)';
  busy(null);
}

function doReset() {
  if (!S.loop) return;
  S.loop.reset();
  S.viewer.sync(S.loop.data);
  S.viewer.resetCamera(S.loop.sim.spawn[0], S.loop.sim.spawn[1], S.loop.sim.spawn[2]);
}

function togglePause() { S.running = !S.running; $('btn-pause').textContent = S.running ? '暂停 (P)' : '继续 (P)'; }
function toggleHelp(on) { $('help').hidden = on === undefined ? !$('help').hidden : !on; }
function toggleCam() {
  S.viewer.follow = !S.viewer.follow;
  $('btn-cam').textContent = S.viewer.follow ? '相机: 跟随 (C)' : '相机: 自由 (C)';
}

function clampCmd() {
  const L = S.limits || {};
  const cl = (v, r) => (r ? Math.max(r[0], Math.min(r[1], v)) : v);
  S.cmd[0] = +cl(S.cmd[0], L.lin_vel_x).toFixed(3);
  S.cmd[1] = +cl(S.cmd[1], L.lin_vel_y).toFixed(3);
  S.cmd[2] = +cl(S.cmd[2], L.ang_vel_z).toFixed(3);
  if (S.loop) S.loop.setCommand(...S.cmd);
}

function onKey(e) {
  if (e.target && /input|select|textarea/i.test(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  let hit = true;
  if (k === 'w') S.cmd[0] += CMD_STEP.vx;
  else if (k === 's') S.cmd[0] -= CMD_STEP.vx;
  else if (k === 'a') S.cmd[1] += CMD_STEP.vy;
  else if (k === 'd') S.cmd[1] -= CMD_STEP.vy;
  else if (k === 'q') S.cmd[2] += CMD_STEP.wz;
  else if (k === 'e') S.cmd[2] -= CMD_STEP.wz;
  else if (k === ' ') S.cmd = [0, 0, 0];
  else if (k === 'r') doReset();
  else if (k === 'p') togglePause();
  else if (k === 'c') toggleCam();
  else if (k === 'h' || k === '?' || k === '/') toggleHelp();
  else if (k === 'escape') toggleHelp(false);
  else hit = false;
  if (hit) { e.preventDefault(); clampCmd(); }
}

let lastT = performance.now(), fpsAcc = 0, fpsN = 0, hudT = 0;

// 窗口不可见时 rAF 会被浏览器停掉 (5090D 上 Chrome 窗口本来就不可见: document.visibilityState='hidden'
// ⇒ 一帧都不跑, 截图只能拿到旧帧, webui/README.md trap #6)。退回 setTimeout 让台子照常推进。
const nextFrame = (fn) => (document.hidden ? setTimeout(() => fn(performance.now()), 16) : requestAnimationFrame(fn));

async function tick() {
  const now = performance.now();
  const dt = Math.min((now - lastT) / 1000, 0.2);
  lastT = now;
  try {
    if (S.running && S.loop) S.stepsPerFrame = await S.loop.advance(dt, 4);
    if (S.loop) {
      S.viewer.sync(S.loop.data);
      S.viewer.syncScan(S.loop.scanner);
      if (S.brain?.ready) S.brain.render();
      const b = S.loop.ids.base, d = S.loop.data;
      S.viewer.setFollowTarget(d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]);
    }
    S.viewer.render();
  } catch (e) { fail(e); return; }
  fpsAcc += dt; fpsN++;
  if (now - hudT > 150) { hudT = now; updateHud(fpsN / Math.max(fpsAcc, 1e-6)); fpsAcc = 0; fpsN = 0; }
  nextFrame(tick);
}

function updateHud(fps) {
  const L = S.loop;
  if (!L) return;
  const v = L.lastVel;
  const dist = Math.hypot(L.pos[0] - L.startXY[0], L.pos[1] - L.startXY[1]);
  $('hud-cmd').textContent = S.cmd.map((x) => x.toFixed(2)).join(' / ');
  $('hud-vel').textContent = `${v[3].toFixed(2)} / ${v[4].toFixed(2)} / ${v[2].toFixed(2)}`;
  $('hud-t').textContent = `${(L.k * L.stepDt).toFixed(2)} s / ${dist.toFixed(2)} m`;
  const stepMs = L.physicsMs + L.policyMs;
  $('hud-ms').textContent = `${stepMs.toFixed(2)} ms  (物理 ${L.physicsMs.toFixed(2)} + 策略 ${L.policyMs.toFixed(2)}) · ${(20 / Math.max(stepMs, 1e-6)).toFixed(0)}x 实时`;
  $('hud-fps').textContent = `${fps.toFixed(0)} fps · ${S.stepsPerFrame} 步/帧` + (document.hidden ? ' (窗口不可见, 被浏览器节流)' : '');
  const st = $('hud-state');
  st.textContent = S.err ? '出错 (见页脚)' : (!S.running ? '暂停' : (L.fell ? `摔倒 @ ${L.metrics().time_to_fall_s} s (R 重置)` : '走着'));
  st.className = S.err || L.fell ? 'bad' : 'ok';
  drawScan(L.scanPred ?? L.scan);
  drawDepth(L);
  drawMemory(L);
  if (S.brain?.ready) {                    // h_v -> 点色 + module 条; z -> DN 热条 (每次 HUD 刷新时画, ~7 Hz)
    S.brain.update(L.policy.h_v, S.brainClip);
    S.brain.drawZ(L.policy.lastZ, S.zClip);
  } else if (S.hasZ) {                     // gru 核没有点云, 但 z 照画 (open ⑰)
    drawZBar($('brain-dn'), L.policy.lastZ, S.zClip);
  }
}

/** 右下小窗: 策略输入 (归一化, 32×36) + 原始深度 (64×36)。 */
function drawDepth(L) {
  if (!L.usesDepth) return;
  const cfg = L.contract.depth;
  const W = cfg.width - cfg.crop[2] - cfg.crop[3], H = cfg.height - cfg.crop[0] - cfg.crop[1];
  if (L.lastDepth) blit($('depth-policy'), L.lastDepth, W, H, (v) => 1 - (v + 0.5));        // [-0.5,0.5] 近=亮
  if (L.lastRawDepth) {
    const cv = $('depth-raw');
    blit(cv, L.lastRawDepth, cfg.width, cfg.height, (v) => 1 - Math.min(v / cfg.max_depth, 1));
    const ctx = cv.getContext('2d');          // 框出真正进策略的那 32 列 (左右各裁 16)
    ctx.strokeStyle = '#ff9f43'; ctx.lineWidth = 1;
    ctx.strokeRect(cv.width * cfg.crop[2] / cfg.width, 0.5,
      cv.width * (cfg.width - cfg.crop[2] - cfg.crop[3]) / cfg.width, cv.height - 1);
  }
}

function blit(cv, arr, W, H, toGray) {
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const g = Math.round(255 * Math.max(0, Math.min(1, toGray(arr[i]))));
    img.data[4 * i] = g; img.data[4 * i + 1] = g; img.data[4 * i + 2] = g; img.data[4 * i + 3] = 255;
  }
  const tmp = blit._c || (blit._c = document.createElement('canvas'));
  tmp.width = W; tmp.height = H;
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
}

/** 左面板: depth_memory 512 维热条 (学生), 教师/无深度时画空。 */
function drawMemory(L) {
  const cv = $('mem-canvas'), ctx = cv.getContext('2d');
  const mem = L.policy.depthMemory;
  ctx.fillStyle = '#12181f'; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!mem) return;
  const n = mem.length, cols = 32, rows = Math.ceil(n / cols);
  const cw = cv.width / cols, ch = cv.height / rows;
  let m = 1e-6;
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(mem[i]));
  for (let i = 0; i < n; i++) {
    const t = Math.abs(mem[i]) / m;
    ctx.fillStyle = mem[i] >= 0 ? `rgb(${Math.round(40 + 215 * t)},${Math.round(50 + 109 * t)},${Math.round(60 - 5 * t)})`
      : `rgb(${Math.round(40 + 30 * t)},${Math.round(50 + 80 * t)},${Math.round(60 + 160 * t)})`;
    ctx.fillRect((i % cols) * cw, Math.floor(i / cols) * ch, Math.ceil(cw), Math.ceil(ch));
  }
}

function drawScan(scan) {
  const cv = $('scan-canvas');
  if (!cv || !scan) return;
  const ctx = cv.getContext('2d');
  const nx = 17, ny = 11, cw = cv.width / nx, ch = cv.height / ny;
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const v = scan[iy * nx + ix];                       // torso_z - hit_z - 0.75, clip +-1
      const t = Math.max(0, Math.min(1, (1 - v) / 2));    // 地面越高 (v 越小) 越亮
      ctx.fillStyle = `rgb(${Math.round(40 + 200 * t)},${Math.round(50 + 130 * t)},${Math.round(60 + 40 * t)})`;
      ctx.fillRect(ix * cw, (ny - 1 - iy) * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  ctx.strokeStyle = '#ff9f43'; ctx.strokeRect(8 * cw, 5 * ch, cw, ch);   // 机器人所在格
}

main().catch(fail);
