# g1fly — fly-brain × Unitree G1, in the browser

A browser-only demo of a 29-DoF humanoid walking under five different control policies — physics
([MuJoCo](https://mujoco.org/) compiled to WebAssembly), the policies themselves (ONNX via
onnxruntime-web) and the rendering (three.js) all run **inside your tab**. There is no backend:
this is a static site plus a pre-baked data bundle.

Everything here is **simulation only. None of it has ever run on real hardware.**

## The policies

| name | what it is |
|---|---|
| `R1_teacher_E_v1` | the *teacher*: a plain MLP that gets a privileged 187-point height scan of the ground |
| `R2_student_E_v1` | the *depth student*: no height scan — it sees a 64×36 head-mounted depth image (rendered in WebGL) and distills the teacher |
| **`T_core3_w2`** | the *all-neuron bottleneck student* — the recurrent motor core is a **connectome-derived graph of 9,156 fly neurons**: *every* descending neuron, leg pre-motor neuron and leg motor neuron of the Drosophila ventral nerve cord, wired from the MaleCNS connectome, reading out through 381 motor neurons, driven from above through a 128-channel descending bottleneck `z`. **Default on load.** |
| `T_graph_w2` | the same recipe on the smaller **T1-only** graph (3,609 neurons, 135 motor neurons) — the second default, and much lighter to load |
| `T_graph` | the T1 architecture, *first wave*. Kept as a control: its descending pathway died during training (see below) |
| `H_k128_w2` | control with the fly graph swapped for a plain 256-unit GRU — same bottleneck, same training recipe, no connectome (so: no neuron point cloud, but the `z` strip is still drawn) |

`T_core3_w2` is the point of the whole thing. The left panel shows its 9,156 neurons coloured by
firing rate, aggregated into 39 anatomical modules, with the 128 descending channels `z` as a heat
strip underneath. Switching to `T_graph_w2` swaps the whole point cloud for the 3,609-neuron
T1 graph, so you can see what the extra 5,547 neurons buy.

**What the all-neuron version costs.** Its recurrent weight matrix is 9,156 × 9,156 dense floats:
a **329 MB** ONNX file, which gzips to **11.9 MB** over the wire (the matrix is 99.5 % zeros — only
387,472 of the 84 M entries are real synaptic connections) and then needs about **1.4 GB** of WASM
heap once onnxruntime unpacks it. Inference costs **9.5 ms per 50 Hz control step** in WebAssembly
(6.4× the FLOPs of the T1 graph for 3.9× the time), against a 20 ms budget — so it runs in real
time, but it is a heavy page. On a memory-constrained machine open `?policy=T_graph_w2` instead:
same recipe, same live brain, 58 MB / 2.4 ms / ~320 MB of heap.

| | `T_core3_w2` (all neurons) | `T_graph_w2` (T1 only) |
|---|---|---|
| neurons in the motor core | **9,156** (39 modules, 381 motor neurons) | 3,609 (37 modules, 135 motor neurons) |
| ONNX / over the wire (gzip) | 329 MB / **11.9 MB** | 58 MB / **8.4 MB** |
| WASM inference, 1 thread | **9.5 ms** / step (47 % of budget) | 2.4 ms / step (12 %) |
| resident memory (WASM heap) | ~1.4 GB | ~0.32 GB |
| success rate in the training simulator | 0.82 | 0.87 |
| blanking the depth camera (0.10 m staircase) | 9.7 m → 7.2 m | 11.9 m → 5.9 m |

**Why there are two waves.** In the first wave (`T_graph`) the descending-neuron bottleneck
collapsed: `z = relu(dn_head(h_b))` came out identically zero, so the depth/brain pathway sent
*nothing* down to the motor core — the legs were driven by the fly graph's own recurrent dynamics
alone, walking blind. The demo shows that honestly (all-dark `z` strip) rather than rescaling it to
look alive. The second wave fixes it with a two-stage recipe (warm-start the motor core from the
first wave, then train with a `softplus` bottleneck that cannot saturate to zero): `z` now carries
signal, the heat strip lights up, and blanking the depth camera really does hurt — on a 0.10 m
staircase the robot goes 11.9 m with depth and 5.9 m without (9.7 m → 7.2 m for the all-neuron
`T_core3_w2`). Flip between `T_core3_w2` / `T_graph_w2` and `T_graph` in the policy dropdown to see
the difference on the same panel.

Everything here is a research artifact, not a product: the numbers above come from simulation
rollouts, and the first wave is kept precisely because a negative result is worth showing.

## Controls

`W`/`S` forward speed · `A`/`D` strafe · `Q`/`E` turn · `Space` stop · `R` reset · `P` pause ·
`C` camera follow/free · `H` help. Mouse: drag to orbit, wheel to zoom, right-drag to pan.
Pick the terrain and the policy in the top-right corner (switching recompiles the MuJoCo model,
which takes 5–15 s). You can also deep-link: `?policy=T_core3_w2&terrain=stairs_up:0.10`.

## Browser requirements

Desktop Chrome / Edge / Firefox / Safari, reasonably recent:

- **WebGL2** (rendering, and the student's depth camera),
- **WebAssembly with SIMD** (MuJoCo + onnxruntime),
- **`DecompressionStream`** — Chrome/Edge 80+, Firefox 113+, Safari 16.4+.

That last one matters: GitHub Pages does not compress `.wasm` / `.onnx` / `.stl`, so this site
ships **gzip sidecars** (`<file>.gz`) and unpacks them in the page. First load pulls ~29 MB
over the wire and expands to ~389 MB in memory; after that it is cached. It is a heavy page —
about 1400 MB of WASM heap for the all-neuron fly graph — so a desktop is strongly recommended.
If your machine is tight on memory, `?policy=T_graph_w2` is the same demo with the smaller
3,609-neuron graph (~320 MB of heap).

## Credits & licenses

| what | who | license |
|---|---|---|
| MuJoCo 3.10.0 (physics, official `@mujoco/mujoco` WASM build) | Google DeepMind | Apache-2.0 |
| onnxruntime-web 1.30.0 (policy inference) | Microsoft | MIT |
| three.js r160 (rendering) | three.js authors | MIT |
| Unitree G1 meshes + URDF-derived MJCF | Unitree Robotics (`unitree_ros`) | BSD-3-Clause |
| MaleCNS v1.0 neuron coordinates / connectivity | Janelia FlyEM & collaborators | CC-BY 4.0 |
| policy weights, data bundle, and this JS demo stand | BigBill Zhao (personal, non-commercial) | see below |

**Licensing of my own parts** (confirmed 2026-09-22): the demo code in this repository is under
the **MIT License** (`LICENSE`); the trained policy weights (`data/policies/*/policy*.onnx*`) and the
derived data bundle are under **CC BY-NC 4.0** (`LICENSE-WEIGHTS.md`, non-commercial, attribution).
This is a personal hobby project and is not affiliated with, nor endorsed by, any employer.

---

## 中文一段

这是「果蝇脑 × 宇树 G1」的浏览器演示台: 物理 (MuJoCo WASM)、策略 (onnxruntime-web)、画面 (three.js)
**全部跑在你这个标签页里**, 没有后端。默认加载的是**全神经元瓶颈学生 `T_core3_w2`** —— 它的循环运动核是
**9,156 个真实果蝇神经元**的连接组 (MaleCNS 腹神经索里**全部**的下行神经元 + 腿前运动神经元 + 腿运动
神经元), 经 381 个运动神经元读出, 上面挂一条 128 路的下行瓶颈 `z`; 另外几个是同配方的 T1 小图版
(`T_graph_w2`, 3,609 神经元, 机器吃不消就切它)、特权教师 (`R1`)、深度学生 (`R2`)、同架构的第一波
(`T_graph`) 与把果蝇图换成普通 GRU 的对照 (`H_k128_w2`)。**第一波 `T_graph` 的 DN 热条全暗不是画错了**:
那次训练里下行瓶颈 `z = relu(dn_head(h_b))` 塌成了恒零, 腿是果蝇图自己的动力学在盲走; 第二波用两段式
+ softplus 修好了, 热条真的亮起来, 而且把深度相机蒙上真的会变差 (0.10 m 楼梯: T1 版 11.9 → 5.9 m,
全神经元版 9.7 → 7.2 m)。负结果照实留着, 你可以在下拉框里来回切着看。**全部是仿真, 没有上过真机。**
代价说在前面: 全神经元版的循环权重是 9,156² 稠密矩阵 = 329 MB 的 ONNX (99.5% 是 0, gzip 后下行只有
11.9 MB), 在浏览器里要 ~1.4 GB 内存, 单步 9.5 ms (20 ms 预算的 47%)。键位: `W/S` 前后 · `A/D` 横移 ·
`Q/E` 转向 · `空格` 停 · `R` 重置 · `P` 暂停 · `C` 相机 · `H` 帮助。首屏约 29 MB (gzip 旁车,
页面内解压), 请用较新的桌面浏览器。
