/**
 * webui/js/app/check_depth.mjs — W2 第三把尺子 (只能在浏览器里跑: node 没有 WebGL)。
 *
 *   webui/check_depth.html?ref=<data/checks 下的名字>&policy=<策略名>
 *
 * 把 python 台 (`webui/tools/ref_traj.py --record-depth`) 录下的每一帧 qpos 摆进同一个 MuJoCo 模型,
 * 用 three.js 深度相机渲染, 与参照逐像素比:
 *   A. 原始 z-depth (米) vs `mjscene.DepthRig` 的光栅深度      —— 验收 max Δ < 1e-3 (契约 §6)
 *   B. 预处理后的策略输入 vs python 的                          —— 端到端 (渲染 + depth.mjs)
 * 结果写 DOM + console (`window.__depthcheck` 也留一份)。
 */

import * as THREE from 'three';
import { httpIO, loadIndex, loadPolicyBundle, loadRobot, loadTerrainData } from '../sim/bundle.mjs';
import { buildModel, loadMujoco } from '../sim/mjload.mjs';
import { Terrain } from '../sim/terrain.mjs';
import { DepthPipeline } from '../sim/depth.mjs';
import { Viewer } from './viewer.mjs';
import { DepthCamera } from './depthcam.mjs';

const $ = (id) => document.getElementById(id);
const base = document.baseURI;
const log = (m) => { $('log').textContent += '\n' + m; console.log('[depth-check] ' + m); };

function stats(a, b, mask) {
  let max = 0, sum = 0, n = 0, arg = -1;
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask(a[i], b[i])) continue;
    const d = Math.abs(a[i] - b[i]);
    if (d > max) { max = d; arg = i; }
    sum += d; n++;
  }
  return { max, mean: n ? sum / n : 0, n, argmax: arg };
}

/** 裁判: MuJoCo 自己的解析光线 (mj_ray, 同一份 hfield/mesh 几何) —— 三方比 three.js / python 光栅 / 光线。 */
function rayDepths(mj, model, data, cam, cfg, pixels) {
  const M = cam._m.elements;
  const R = [M[0], M[4], M[8], M[1], M[5], M[9], M[2], M[6], M[10]];
  const pos = [M[12], M[13], M[14]];
  const geomid = new mj.IntBuffer(1);
  const gg = new Uint8Array(6); gg[0] = 1; gg[1] = 1;     // group 0 地形 + 1 腿 visual = DepthRig 的可见集
  const out = [];
  for (const [c, r] of pixels) {
    const x = (c + 0.5 - cfg.cx) / cfg.fx, y = (cfg.cy - (r + 0.5)) / cfg.fy;   // 像素中心, 同 MuJoCo 光线模型
    const nrm = Math.hypot(x, y, 1);
    const dw = [0, 0, 0];
    for (let i = 0; i < 3; i++) dw[i] = (R[3 * i] * x + R[3 * i + 1] * y + R[3 * i + 2] * -1) / nrm;
    const dist = mj.mj_ray(model, data, pos, dw, gg, true, -1, geomid, null);
    out.push({ c, r, z: dist >= 0 ? dist / nrm : null });
  }
  geomid.delete();
  return out;
}

function drawGray(cv, arr, W, H, lo, hi, scale = 4) {
  cv.width = W * scale; cv.height = H * scale;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(W * scale, H * scale);
  for (let y = 0; y < H * scale; y++) {
    for (let x = 0; x < W * scale; x++) {
      const v = arr[Math.floor(y / scale) * W + Math.floor(x / scale)];
      const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      const g = Math.round(255 * (1 - t));                    // 近 = 亮
      const i = (y * W * scale + x) * 4;
      img.data[i] = g; img.data[i + 1] = g; img.data[i + 2] = g; img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

async function main() {
  const q = new URLSearchParams(location.search);
  const refName = q.get('ref') || 'student_depth';
  const io = httpIO(new URL('data/', base).href);
  const index = await loadIndex(io);
  const policyName = q.get('policy') || (index.policies.find((p) => p.kind === 'student')?.name ?? index.policies[0].name);
  log(`参照 data/checks/${refName}.json · 策略 ${policyName}`);
  const ref = await io.json(`checks/${refName}.json`);
  const bundle = await loadPolicyBundle(io, policyName);
  const cfg = bundle.contract.depth;
  log(`参照字段: ${ref.frames[0].pre_now ? 'pre_now (当前帧)' : 'pre (含延迟, 老参照)'} · 延迟 ${ref.depth.delay} 帧 · 相机 ${ref.depth.update_hz} Hz`);
  log(`${ref.n} 帧, 地形 ${ref.terrain}, 渲染 ${cfg.width}×${cfg.height}, 裁后 ${cfg.width - cfg.crop[2] - cfg.crop[3]}×${cfg.height - cfg.crop[0] - cfg.crop[1]}`);

  const mj = await loadMujoco(new URL('vendor/mujoco/mujoco.js', base).href);
  const robot = await loadRobot(io, index);
  const tid = ref.terrain.replace(':', '_');
  const { meta: tmeta, heights } = await loadTerrainData(io, tid);
  const terrain = new Terrain(tmeta, heights);
  const sim = buildModel(mj, robot, terrain, bundle.contract);
  const { model, data } = sim;

  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;left:-10000px;width:640px;height:400px';
  document.body.appendChild(host);
  const viewer = new Viewer(host);
  const stride = parseInt(q.get('stride') || '1', 10);      // 地形网格抽稀: 深度对照必须 1 (= MuJoCo hfield 的 0.025 m 格)
  const tinfo = viewer.buildTerrain(terrain, stride);
  log(`地形网格 stride=${stride} (${tinfo.nx}×${tinfo.ny} = ${tinfo.verts} 顶点)`);
  viewer.buildRobot(mj, model);
  const cam = new DepthCamera(viewer.renderer, viewer.scene, cfg);
  const gl = viewer.renderer.getContext();
  log(`WebGL: ${gl.getParameter(gl.VERSION)} · float RT: ${!!viewer.renderer.extensions.get('EXT_color_buffer_float')}`);

  const pipe = new DepthPipeline(cfg, { delay: ref.depth.delay });
  window.__dc = { mj, model, data, sim, viewer, cam, terrain, ref, cfg, pipe, THREE };   // 探针用
  const rows = [];
  let rawMax = 0, rawMean = 0, preMax = 0, preMean = 0, worst = null;
  let rayMax = 0, rayPyMax = 0, upMax = 0, upPreMax = 0, upN = 0;
  const probe = [];                       // 8×8 采样点 (给 mj_ray 裁判用; 逐像素打光线太慢)
  for (let r = 2; r < cfg.height; r += 5) for (let c = 2; c < cfg.width; c += 8) probe.push([c, r]);
  for (const f of ref.frames) {
    for (let i = 0; i < 36; i++) data.qpos[i] = f.qpos[i];
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    mj.mj_forward(model, data);
    viewer.sync(data);
    cam.updatePose(data, sim.ids.torso);
    const raw = cam.render();
    // **策略可见带**内才算数: 两边都超过 max_depth (2.5 m) 的像素, 预处理会一起压成 max_depth,
    // 差多少对策略都没有意义 —— 地平线上那种"一边擦到地一边擦过去"的掠射像素就属于这类 (实测单像素差 16 m)。
    const band = cfg.max_depth + 0.05;
    const sr = stats(raw, f.raw, (a, b) => Math.min(a, b) < band);
    const srAll = stats(raw, f.raw, (a, b) => b < 5.0);   // 旧口径 (只按 python 侧 <5 m), 留着对照
    pipe.reset();                       // 新管线首帧填满历史 ⇒ 输出即当前帧
    const pre = pipe.process(raw);
    // 比 `pre_now` (当前帧): 参照是隔帧录的, delay>0 时重建不了历史, 拿策略真吃的 `pre` 比会假报错
    // (R2 的 delay=1 实测被这条坑过: 策略输入 mean 从 1.3e-2 掉到 1e-4 级)。延迟语义由 depth_check.mjs ③ 单独验。
    const sp = stats(pre, f.pre_now ?? f.pre);
    const rays = rayDepths(mj, model, data, cam, cfg, probe);
    let rmax = 0, rpmax = 0;
    for (const { c, r, z } of rays) {
      if (z === null || z > 5) continue;
      rmax = Math.max(rmax, Math.abs(raw[r * cfg.width + c] - z));
      rpmax = Math.max(rpmax, Math.abs(f.raw[r * cfg.width + c] - z));
    }
    rayMax = Math.max(rayMax, rmax); rayPyMax = Math.max(rayPyMax, rpmax);
    const upright = f.k <= (ref.upright_until ?? 60);     // 摔倒后相机埋进自己腿里, 两边都是垃圾 (预处理会压成 max)
    if (upright) { upMax = Math.max(upMax, sr.max); upPreMax = Math.max(upPreMax, sp.max); upN++; }
    rows.push({ k: f.k, raw_max: sr.max, raw_mean: sr.mean, n: sr.n, raw_max_all: srAll.max,
                pre_max: sp.max, pre_mean: sp.mean, ray_three: rmax, ray_py: rpmax, upright });
    rawMax = Math.max(rawMax, sr.max); rawMean += sr.mean / ref.frames.length;
    preMax = Math.max(preMax, sp.max); preMean += sp.mean / ref.frames.length;
    if (sr.max >= rawMax) worst = { k: f.k, px: sr.argmax, js: raw[sr.argmax], py: f.raw[sr.argmax] };
    if (rows.length <= 3) {                                // 前 3 帧留图 (人眼看一眼像不像)
      const W = cfg.width, H = cfg.height;
      const fig = document.createElement('figure');
      const c1 = document.createElement('canvas'), c2 = document.createElement('canvas'), c3 = document.createElement('canvas');
      drawGray(c1, raw, W, H, 0, 2.5); drawGray(c2, Float32Array.from(f.raw), W, H, 0, 2.5);
      const diff = new Float32Array(W * H);
      for (let i = 0; i < diff.length; i++) diff[i] = Math.min(Math.abs(raw[i] - f.raw[i]), 0.05);
      drawGray(c3, diff, W, H, 0.05, 0);                   // 反色: 差越大越亮
      fig.append(c1, c2, c3);
      const cap = document.createElement('figcaption');
      cap.textContent = `k=${f.k} 左: three.js · 中: python · 右: |Δ| (0–5 cm)`;
      fig.appendChild(cap);
      $('imgs').appendChild(fig);
    }
  }
  const res = {
    ref: refName, policy: policyName, frames: ref.frames.length, stride,
    raw: { max: rawMax, mean: rawMean, pass: rawMax < 1e-3, worst },
    pre: { max: preMax, mean: preMean },
    upright: { frames: upN, raw_max: upMax, pre_max: upPreMax },
    ray: { three_vs_ray_max: rayMax, python_vs_ray_max: rayPyMax,
           note: 'mj_ray = MuJoCo 自己的解析几何; three.js 与它一致 ⇒ 差值在 python 的光栅侧' },
    rows, camera: cam.frustum,
  };
  window.__depthcheck = res;
  // 判据 (README §4.4): 逐像素对着**另一套光栅器**比 max 是不可达的 (轮廓像素注定跳变),
  // 所以三条一起看 —— 对解析几何的 max / 对 python 光栅的 mean / 策略输入的 mean。
  const verdicts = [
    ['three.js vs mj_ray 解析几何 (m)', rayMax, null, rayMax < 1e-3, 'max < 1e-3'],
    ['three.js vs python 光栅, 策略可见带内 (m)', rawMax, rawMean, rawMean < 1e-3, 'mean < 1e-3'],
    ['策略输入 (归一化)', preMax, preMean, preMean < 1e-3, 'mean < 1e-3'],
  ];
  res.verdict_pass = verdicts.every((v) => v[3]);
  $('out').innerHTML = '<table><tr><th>量</th><th>max</th><th>mean</th><th>判据</th><th>判定</th></tr>' +
    verdicts.map(([name, mx, mn, ok, rule]) => `<tr><td>${name}</td><td>${mx.toExponential(3)}</td>` +
      `<td>${mn === null ? '-' : mn.toExponential(3)}</td><td>${rule}</td>` +
      `<td class="${ok ? 'ok' : 'bad'}">${ok ? 'PASS' : 'FAIL'}</td></tr>`).join('') +
    `<tr><td>(契约 v1 原判据: 对 python 光栅 max &lt; 1e-3)</td><td>${rawMax.toExponential(3)}</td><td>-</td>` +
    `<td>max &lt; 1e-3</td><td class="${res.raw.pass ? 'ok' : 'bad'}">${res.raw.pass ? 'PASS' : '不可达, 见 README §4.4'}</td></tr></table>`;
  log(`原始深度 (策略可见带 ≤${cfg.max_depth} m 内): max Δ ${rawMax.toExponential(3)} m, mean ${rawMean.toExponential(3)} m`);
  log(`  其中直立帧 (${upN}/${ref.frames.length}): 原始 max ${upMax.toExponential(3)} m · 策略输入 max ${upPreMax.toExponential(3)}`);
  log(`裁判 mj_ray (解析几何): three.js vs ray max ${rayMax.toExponential(3)} m · python 光栅 vs ray max ${rayPyMax.toExponential(3)} m`);
  log(`策略输入 max Δ ${preMax.toExponential(3)}, mean ${preMean.toExponential(3)}`);
}

main().catch((e) => { log('ERROR ' + (e && e.stack || e)); console.error(e); });
