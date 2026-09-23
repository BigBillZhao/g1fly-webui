/**
 * webui/js/app/bench.mjs — 在浏览器里成集跑分 (学生策略只能在这儿跑: node 没有 WebGL, 深度渲染做不了)。
 *
 *   webui/bench.html?policy=<名字>&terrains=flat,stairs_up:0.10,slope_up:10&seconds=20&cmd=0.5,0,0
 *
 * 指标口径 = `sim2sim/run.py` (success/fell/t_fall/dist/vel_err/|tau|), 与 `webui/tools/ref_traj.py` 的
 * python 台一一对照; 结果写 DOM + `window.__bench` (取证用 javascript_tool 读)。
 */

import { httpIO, loadIndex, loadPolicyBundle, loadRobot, loadTerrainData } from '../sim/bundle.mjs';
import { loadMujoco } from '../sim/mjload.mjs';
import { Terrain } from '../sim/terrain.mjs';
import { SimLoop } from '../sim/loop.mjs';
import { createPolicy, initOrt } from '../sim/policy.mjs';
import { Viewer } from './viewer.mjs';
import { DepthCamera } from './depthcam.mjs';

const $ = (id) => document.getElementById(id);
const base = document.baseURI;
const log = (m) => { $('log').textContent += '\n' + m; console.log('[bench] ' + m); };

async function main() {
  const q = new URLSearchParams(location.search);
  const seconds = parseFloat(q.get('seconds') || '20');
  const cmd = (q.get('cmd') || '0.5,0,0').split(',').map(Number);
  const io = httpIO(new URL('data/', base).href);
  const index = await loadIndex(io);
  const policyName = q.get('policy') || index.policies[0].name;
  const terrains = (q.get('terrains') || index.terrains.map((t) => t.spec).join(',')).split(',');

  const mj = await loadMujoco(new URL('vendor/mujoco/mujoco.js', base).href);
  await initOrt(new URL('vendor/ort/ort.wasm.bundle.min.mjs', base).href, new URL('vendor/ort/', base).href);
  const robot = await loadRobot(io, index);
  const bundle = await loadPolicyBundle(io, policyName);
  const policy = await createPolicy(bundle);
  log(`策略 ${policyName} (${bundle.meta.kind}) · 指令 ${cmd} · ${seconds} s/集 · ${terrains.length} 个地形`);

  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;left:-10000px;width:640px;height:400px';
  document.body.appendChild(host);
  const viewer = new Viewer(host);
  const usesDepth = bundle.meta.kind !== 'teacher';
  const depthCam = usesDepth ? new DepthCamera(viewer.renderer, viewer.scene, bundle.contract.depth) : null;

  const rows = [];
  for (const spec of terrains) {
    const { meta: tmeta, heights } = await loadTerrainData(io, spec.replace(':', '_'));
    const terrain = new Terrain(tmeta, heights);
    viewer.buildTerrain(terrain);
    const renderDepth = usesDepth ? (loop) => {
      viewer.sync(loop.data);
      depthCam.updatePose(loop.data, loop.ids.torso);
      return depthCam.render();
    } : null;
    const loop = new SimLoop(mj, { robot, terrain, contract: bundle.contract, policy, renderDepth,
      depthDelay: parseInt(q.get('depth_delay') ?? String(bundle.contract.depth?.delay_range?.[0] ?? 0), 10) });
    if (usesDepth) viewer.buildRobot(mj, loop.model);
    loop.setCommand(...cmd);
    loop.reset();
    const n = Math.round(seconds / loop.stepDt);
    const t0 = performance.now();
    let phys = 0, pol = 0;
    for (let k = 0; k < n; k++) { await loop.step(); phys += loop.physicsMs; pol += loop.policyMs; }
    const wall = performance.now() - t0;
    const m = loop.metrics();
    rows.push({ terrain: spec, ...m, ms_per_step: +(wall / n).toFixed(2), physics_ms: +(phys / n).toFixed(2),
                policy_ms: +(pol / n).toFixed(2), realtime: +(n * loop.stepDt / (wall / 1000)).toFixed(1) });
    log(`${spec.padEnd(18)} success=${m.success} fell=${m.fell} t_fall=${m.time_to_fall_s} dist=${m.distance_xy} ` +
      `vel_err=${m.vel_err} |tau|=${m.mean_abs_tau} | ${(wall / n).toFixed(2)} ms/步 (物理 ${(phys / n).toFixed(2)} + 策略 ${(pol / n).toFixed(2)})`);
    loop.dispose();
  }
  window.__bench = { policy: policyName, kind: bundle.meta.kind, cmd, seconds, rows };
  $('out').innerHTML = '<table><tr><th>地形</th><th>success</th><th>fell</th><th>t_fall</th><th>dist</th>' +
    '<th>vel_err</th><th>|tau|</th><th>ms/步</th><th>×实时</th></tr>' +
    rows.map((r) => `<tr><td>${r.terrain}</td><td class="${r.success ? 'ok' : 'bad'}">${r.success}</td><td>${r.fell}</td>` +
      `<td>${r.time_to_fall_s ?? '-'}</td><td>${r.distance_xy}</td><td>${r.vel_err}</td><td>${r.mean_abs_tau}</td>` +
      `<td>${r.ms_per_step}</td><td>${r.realtime}</td></tr>`).join('') + '</table>';
  log('完成 (window.__bench)');
}

main().catch((e) => { log('ERROR ' + (e && e.stack || e)); console.error(e); });
