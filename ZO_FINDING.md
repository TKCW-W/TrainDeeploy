<!-- SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna ; SPDX-License-Identifier: MIT -->
# ZO (MeZO) on-device fine-tuning on TrainDeeploy — port + smoke test

**Branch:** `feat/zo-support` (from `feat/BNFRozen_OptionB`) · **Dates:** 2026-08-04 → 2026-08-05
**Purpose:** port the zeroth-order (MeZO) on-device fine-tuning path into TrainDeeploy for SpeechNet, and validate device-vs-reference bit-exactly, starting from a single step / single `n_accum`.
**Companion docs:** gap analysis `../Deeploy/ZO_DEEPLOY_STUDY.md` · Onnx4Deeploy export `../Onnx4Deeploy/ZO_PORT_FINDINGS.md` · recipe `SilentWear/exp18_zo_faithful_sim/FINDINGS.md`.

> **STATUS: ✅ RESOLVED.** The single-step / single-`n_accum` ZO smoke test passes **bit-exact** on device, perturbing **all 22 trainable parameters** (Conv **and** BN γ/β **and** fc — see §2). Device `loss_plus = 0.016488`, `loss_minus = 0.069274` **== reference** (diff ~4e-7, `Errors: 0/2`), ~35.6M cycles/step, runner works end-to-end.

---

## 1. Result at a glance

| Milestone | State |
|---|---|
| Study `Deeploy@zo-support` + gap analysis | ✅ (`../Deeploy/ZO_DEEPLOY_STUDY.md`) |
| Port FP32 `PerturbRademacher` op + kernel; RNG bit-exact (isolated) | ✅ `0/6144` errors |
| Runtime perturb controls (sign / seed-base / eps-override) | ✅ `ZORuntime.h` |
| Native ZO-graph emission (Onnx4Deeploy `feat/ZO`); 22/22 weight buffers shared | ✅ |
| ZO device-loop harness `deeploymezotest.c` (±ε → g_proj → in-place update) | ✅ |
| Full pipeline codegen → build → GVSoC on RISC-V | ✅ |
| **Single-step / `n_accum`=1, all 22 params (Conv+BN+fc), on-device == reference** | ✅ **bit-exact** |
| `tile_seed_offset` across tiles (perturb tensor that tiles) | ⚠️ **not triggered** for SpeechNet (all single-tile); **open in general** (§4) |
| Multi-step / `n_accum` > 1 → full recipe | ⏳ next (functional; full-accuracy stays in the PyTorch exp18 sim) |
| Strip debug instrumentation (ZTRACE / BN_DEBUG / loss-bit dumps) | ⏳ cleanup |

## 2. Reproduction

Run **inside the `traindeeploy` container** (repo mounted at `/app/ETH/TrainDeeploy`; Onnx4Deeploy exports at `/app/ETH/Onnx4Deeploy`). **Kill any orphan `gvsoc_launcher` by explicit PID first** (§9).

**The passing single-step / single-`n_accum` smoke test (full SpeechNet, all 22 trainable params):**
```bash
cd /app/ETH/TrainDeeploy/DeeployTest
python3 deeployMezoRunner_tiled_siracusa.py \
  -t              Tests/Models/Training/SpeechNet/speechnet_zo_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_zo_update \
  --n-steps 1 --n-accum 1 --num-data-inputs 2 \
  --eps 0.01 --lr 3e-6 --q 1 --seed 0 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8
```
Expected device output: `[loss+ 0] computed=0.016488 ref=0.016488 diff=0.000000` · `[loss- 0] computed=0.069274 ref=0.069274 diff=0.000000` · `Errors: 0 out of 2` · `BENCH train_cycles=35636028 opt_cycles=140177`.

**Fixtures**
- Smoke-test fixture: `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_zo_{train,update}/` — `network.onnx` + `inputs.npz` + `outputs.npz`. The **22 perturbed weights** in `speechnet_zo_train` = **all trainable params**: 10 conv (`blocks_X_0_weight/bias`) + 10 BN γ/β (`blocks_X_1_weight/bias`) + 2 fc (`fc_weight/bias`). Reference losses live in `outputs.npz['loss_plus'|'loss_minus']` (= `0.016488` / `0.069274`), computed by `_zo_pytorch_reference` (PyTorch, `model.eval()` → frozen BN; verified == ORT).
- These fixtures are produced by **Onnx4Deeploy `feat/ZO`** native emission (source export dir `onnx/model/speechnet_zo_native_s1`).
- **Isolation-ladder fixtures** (built during §5–§6, for re-running the bisection) live under `/app/ETH/Onnx4Deeploy/onnx/model/{fc,conv,mp,b2,sn0,sn1,snc,bn,m4}_zo_{train,update}/`, each with its own numpy/ORT reference in `outputs.npz`. Run any of them by pointing `-t …/<name>_zo_train --optimizer-dir …/<name>_zo_update`.
- Logs from all runs: `DeeployTest/experiments/zo_smoke/logs/` (final: `speechnet_full_fix.log`).

**Reading a device loss** (harness prints raw bits, FP-free — see §5.1): `struct.unpack('<f', struct.pack('<I', int(bits,16)))` on the `lp_bits=0x…` / `lm_bits=0x…` markers.

## 3. The ZO pipeline as shipped

A **separate ZO pipeline that reuses the BP training machinery**, one CMake branch over from BP. No new update kernel: the ZO weight update is `ApplyRademacherPerturbation` applied **in place** with `epsilon = −lr·g_proj` (θ ← θ − lr·g_proj·z).

**Two graphs (from Onnx4Deeploy `feat/ZO`, native — structurally BP-forward + perturb nodes, minus backward):**
- `network_zo_train.onnx`: per-weight `PerturbRademacher` → `Conv` / `BatchNormInternal`(frozen) / `Relu` / `MaxPool` / `GAP` / `Gemm` → canonical 2-output `SoftmaxCrossEntropyLoss` `[loss[], log_prob[B,K]]`. The 22 weights are **INITIALIZERS (baked constants)** — graph inputs are only `[input, label]`; each perturb reads a **constant** base weight (loaded once by `InitTrainingNetwork`). *This is why `_foldLayoutIntoPerturb`, which requires a constant, fired here — and why disabling it (§5.2) fixed the bug.* No `mezo` domain; logits `value_info` added (ORT can't infer through the custom Perturb op).
- `network_zo_update.onnx`: the 22 weights as **graph INPUTS** (writable, so the perturb can update them) → `PerturbRademacher` in place → `*_updated` outputs.
- **Weight-buffer sharing (hybrid → in-place update):** because the two graphs differ (train = constants, update = inputs), the harness redirects `zo_update`'s 22 weight-input buffers onto `zo_train`'s 22 weight-constant buffers by name (`_patch_shared_buffers`) so the in-place update lands on the training weights for the next step. The **single-step** smoke test reads `loss_plus`/`loss_minus` *before* the update, so it validates the **forward** regardless; the cross-step **update propagation** (this aliasing actually feeding step-1) is only exercised by a multi-step run (§8).
- `create_training_test_data_zo` packs weight arrays + `input`/`label` + ZO meta (`eps`, `seed`, `lr`, `q`, `n_accum`) into `inputs.npz`; `outputs.npz` carries `loss_plus`/`loss_minus` (+ updated weights + `log_prob`) as the reference.

**Device-loop harness** `DeeployTest/Platforms/Siracusa/src/deeploymezotest.c` — the analog of BP's `deeploytraintest.c`, selected by `option(MEZO_TRAINING)` (parallel to `TRAINING`). The loop:
```
for update_step:  seed_base = update_step·ZO_Q
  for accum_step:
    perturbation_sign=1 → RunTrainingNetwork → L+ ;  =0 → RunTrainingNetwork → L−   (same seed_base)
    AccumulateZODiffOnCluster:  acc += (L+ − L−)
  ComputeZOUpdateCoeffOnCluster:  g_proj = acc/(2·ε·N_ACCUM);  perturb_eps_override = −lr·g_proj
  RunOptimizerNetwork (zo_update, in place, same seed_base)      ← θ ← θ − lr·g_proj·z
compare stored_loss_plus/minus vs testLossPlusRef/MinusRef  (on cluster)
```
All floating-point math runs **on the cluster** (the FC has no FPU — see §5.1); the FC only moves raw bytes.

**Runtime perturb controls** `TargetLibraries/PULPOpen/{src/ZORuntime.c,inc/kernel/ZORuntime.h}` — mutable extern globals with neutral defaults: `perturbation_sign` (±ε via `dir`), `perturb_seed_base` (added onto each node's baked seed → per-step seeds), `perturb_eps_override` + `perturb_eps_use_override` (runtime magnitude → the update coefficient). The C main drives them per pass.

**BN must use frozen stats** — SpeechNet ZO trains BN γ/β but keeps **frozen** running stats. `add_mezo_cmake_flags` emits `-DBN_FROZEN_STATS=ON` (`g_bn_frozen_stats=1`), so `BatchNormInternal` normalizes with running stats, matching the reference. (Note: the 22 perturbs **include** BN γ/β — the *stats* are frozen, the *affine* is trained/perturbed.)

**Python/runner wiring** (all `-- QW`, BP path untouched): `deeployMezoRunner{,_tiled_siracusa}.py`, `run_zo_codegen`/`resolve_zo_update_dir`/`add_mezo_cmake_flags` (`trainingUtils.py`), `generateZOTestOutputsHeader` (`codeGenerateTraining.py`, emits `loss_plus`/`loss_minus` refs), `testMVPTraining.py --zo`, `mezo` flag in `core/{config,execution}.py`, `Platforms/Siracusa/CMakeLists.txt` + `DeeployTest/CMakeLists.txt` MEZO branch. Needs `--l1 128000` (frozen-BN full-spatial tiles; BP frozen-BN needs it too).

## 4. Ported perturb primitive + RNG bit-exactness (and the tile-offset open item)

`zo-support` ships a complete perturb **primitive** but no training runner (see the study doc). Ported the FP32 Rademacher slice (all `-- QW`):
- **Kernel** `TargetLibraries/PULPOpen/src/RandomNoise.c` (+ `inc/kernel/RandomNoise.h`; the shipped file only — not `_base.c`/`_unrolled.c`, which diverge).
- **Op wiring** (mirrors SGD/InPlaceAccumulatorV2): `PerturbRademacherParser`, `PerturbRademacherLayer`, `PerturbZOChecker`, `PULPPerturbRademacherBindings` + `…TilingReadyBindings`, `PerturbTileConstraint`, `PerturbRademacherMapper` + `'PerturbRademacher'` in `PULPMapping`.

**Isolated RNG gate (bit-exact):** a single-node `PerturbRademacher` graph (`[128,48]`, `seed=42,eps=0.01,idx=0`) vs the device-faithful reference RNG → **`0 / 6144` errors**. Confirms LCG scramble (`·1664525+1013904223`), xorshift32, 8-core partition, packed 32-bit LSB-first Rademacher, per-core seed `scramble(base + 8·idx + core)`.

**⚠️ Open item — `tile_seed_offset` across tiles.** The device template computes `chunk_seed = ((baked_seed + perturb_seed_base) + 8·node_id + core_id) ^ (tile_seed_offset · 0x9E3779B1)`, where `tile_seed_offset` is the per-tile global element offset (decorrelates RNG across tiles). The reference `_perturb_rademacher` is **untiled** and assumes `tile_seed_offset = 0`.

> **Note — this is *not* "we run untiled".** We use the **tiled** runner and the full SpeechNet **is** tiled: tiling is decided **per-tensor** by size vs the L1 budget (128 KB). The large **activations** tile (e.g. conv0 output `1×8×14×701` = 314 KB → `numTiles=4`, runs tile-by-tile); the small **perturbed weights** don't (largest = `blocks_4_0_weight` 28.6 KB < L1 → `numTiles=1`). `tile_seed_offset` lives only in the **Perturb** kernel — it only matters for tiles *of the weight being perturbed*. Activation tiling (conv/BN/pool) splits deterministic compute, never touches the perturb seed, and is numerically exact regardless of tile count (verified bit-exact in the §6 ladder).

- **SpeechNet: not triggered.** All 22 **perturbed weights** are single-tile — verified in the shipped generated code: every perturb uses `^ (0 * 0x9E3779B1u)` and `numTiles = {0,1}` (even while activation ops run at `numTiles=4`). Hence device z == reference z, and the smoke test is bit-exact.
- **General case: OPEN.** If a **perturbed weight** is itself large enough to **tile** (`numTiles > 1`), each tile's non-zero `tile_seed_offset` makes the device z diverge from the untiled reference → loss mismatch. Not exercised by SpeechNet, but must be resolved before deploying a model with a large perturbed weight. **Fix options:** (a) make `_perturb_rademacher` model the per-tile offset (mirror the device seeding exactly), or (b) guarantee/verify perturbed weights stay single-tile (assert `numTiles==1` per perturb at codegen).

## 5. Root causes & fixes (the debugging payoff)

The pipeline built and ran, but the full SpeechNet loss mismatched the reference (device systematically over-confident: `loss_plus` 0.0157 vs 0.0165, `loss_minus` 0.0478 vs 0.0693). Two independent bugs, found via the isolation method in §6.

### 5.1 FP on the FPU-less Fabric Controller → illegal-instruction trap
GAP9/Siracusa's **Fabric Controller (runs `main()`) has no FPU**; only the cluster cores do. The harness read the loss into `float` locals on the FC (`float lp; … stored_loss[mb]=lp;`) → the compiler emitted `flw`/`fsw` → **illegal instruction** → the FC jumped to `pos_illegal_instr` (PC `0x1c00808c`, a `j`-to-self) and **spun forever**. This looked for a long time like a "~500× slowdown / hang" — GVSoC stdout is buffered until `main()` returns, so a killed run shows nothing; `--trace=fc/insn` revealed the tight loop at `pos_illegal_instr`. (`printf("%f")` is another instance of the same trap.)
**Fix:** the FC handles the loss only as raw `uint32_t` bits (`memcpy`, no float ops), passing bits to the cluster helpers `AccumulateZODiffOnCluster` / `ComputeZOUpdateCoeffOnCluster`, which reinterpret and do all FP. Matches BP's `memcpy`-direct pattern. → the loop runs fast (~35.6M cyc/step) and completes.

### 5.2 `_foldLayoutIntoPerturb` reordered the Rademacher z (the correctness bug)
The ported fold pushed the NCHW→NHWC layout permutation into each perturb's **constant base conv-weight** at compile time (to skip a runtime `Transpose`). Consequence: the perturb kernel generated its Rademacher `z` over the **transposed (NHWC) flat buffer**, while the reference (and the ZO algorithm) perturb in **logical NCHW order**. For `in_ch=1` weights the transpose leaves the flat order unchanged → correct; for **`in_ch>1`** conv weights (conv1–4) the order changes → `z` lands on different elements → the device's perturbed weight differs from the reference (verified: ndiff 990–3600 of 2048–7168) → wrong `g_proj` → wrong loss (worse in the steep −ε direction).
**Fix (`Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py`):** **disabled the fold** — the perturb now runs on the base weight in **logical NCHW order** (z matches the reference) followed by a runtime `Transpose` (`_appendTranspose`) that converts the already-correctly-perturbed weight to NHWC. This is **exactly how the BP/training path handles conv weights**, so the fold was never necessary — and its stated "~100× slowdown" rationale was really bug 5.1.

## 6. How it was found — the isolation ladder (supervisor's method)

*Verify small / no-tiling first, then add one layer or one perturbed weight at a time*, comparing the device loss bits to a fresh ORT reference. Every rung was bit-exact until the bug triggered:

| Test (fixture) | Perturbed | Before fix | After fix |
|---|---|---|---|
| fc only, no tiling (`fc`) | fc | ✅ | — |
| conv0 + frozen-BN0, tiled (`conv`) | conv0 | ✅ | — |
| + MaxPool, tiled (`mp`) | conv0 | ✅ | — |
| 2 blocks (`b2`) | conv0 | ✅ | — |
| full 5-block forward (`sn0`) | conv0 | ✅ | — |
| BN0 γ/β (`bn`) | BN affine | ✅ | — |
| conv0 + BN0, 4 perturbs (`m4`) | 4 | ✅ | — |
| all 5 conv weights (`snc`) | 10 | ❌ | ✅ |
| **conv1 only (`sn1`)** | conv1 | ❌ | ✅ |
| **full SpeechNet** | 22 (Conv+BN+fc) | ❌ | ✅ **bit-exact** |

The **conv0-passes / conv1-fails** contrast (in_ch=1 vs in_ch=8) pinned 5.2 immediately; a host-side Python check confirmed the fold reorders `z` for every multi-channel conv weight. **ZO mechanics independently verified correct:** weights match; z bit-exact (where the fold doesn't reorder); ±ε sign; scalar accumulation; `g_proj = −lr·(L+−L−)/(2ε·N)`; in-place `zo_update`; cluster-side comparison.

## 7. Approaches tried and superseded (so they aren't re-tread)

- **Downstream graph band-aid** (`zo_graph_prep.py`): a preprocessing pass patching the inference-style ZO graphs to pass the training frontend. **Superseded** by **native ZO-graph emission** in Onnx4Deeploy `feat/ZO` (§3): graphs are now structurally BP-native and flow through the existing codegen with **zero** prep, and the weight-buffer sharing works via the hybrid redirection described in §3 (`zo_train` constants ← `zo_update` inputs, 22/22). `zo_graph_prep.py` no longer used. *(Note: an earlier native-pivot variant made `zo_train` weights graph* inputs *too; the shipped fixture keeps them as* initializers *— see §3.)*
- **Inference-path detour** ("run the ZO forward through `deeploytest.c`, fix `BatchNormInternal` on the inference path"): explored because a single perturbed forward is cheap, but abandoned. `BatchNormInternal` is a training op; on the NHWC inference path it needs NHWC↔NCHW transposes and gave garbage (a `block0→block1` "forward bug" that was an **inference-path artifact**, not the real issue). The shipped path is the **training/device-loop** harness (§3), where the forward is correct.
- **Mis-diagnosed "~500× slowdown"**: earlier attributed to weights-as-inputs / fold-failing / runtime-transpose. **Wrong** — the real cause was the FP trap (5.1). The runtime `Transpose` is cheap and is the correct, shipped approach (5.2).

## 8. Remaining work
1. **Scale up:** `n_accum` > 1 (one update step) → a few multi-step correctness runs. GVSoC is far too slow for the full exp18 recipe (200 epochs) — on-device tests validate **functional correctness on tiny step counts**; full-accuracy training stays in the PyTorch **exp18** sim (ZO 87.36 ≈ BP 86.11 ≈ paper fold-3 87.64).
2. **Resolve the `tile_seed_offset` open item (§4)** before any model with a perturbed tensor large enough to tile: model the per-tile offset in `_perturb_rademacher`, or assert single-tile per perturb.
3. **Strip debug instrumentation** (all `-- QW`): `ZTRACE` phase markers + loss-bit dumps in `deeploymezotest.c`; `g_bn_debug` in `BatchNorm.c`; `BN_DEBUG` passthroughs. Keep `g_bn_frozen_stats` / `BN_FROZEN_STATS` (functional).
4. Optionally re-add a **correct** layout fold (perturb `z` indexed by the pre-transpose element position) if the runtime `Transpose` cost ever matters — it doesn't for SpeechNet (tiny weights).

## 9. GVSoC operational gotchas (see memory `gvsoc_kill_by_pid`)
- **`killall`/`pkill -f` do NOT reliably kill `gvsoc_launcher`.** Orphans survive and starve new sims via CPU contention → runs look "hung". **Kill by explicit PID** (`pgrep -f gvsoc_launcher | xargs -r kill -9`), verify 0 remaining before each run; also kill orphaned `gmake`/`gapy`/`cc1` build chains. A stray `gvsoc_launcher` once held **14 GB** of deleted trace files open, filling the disk.
- **Device stdout flushes only at `main()` return** → a killed run shows an empty log. Use `fflush`/`ZTRACE` (immediate flush) or GVSoC `--trace=fc/insn|cluster/pe0/insn|cluster/dma` for live/where-stuck diagnosis (map PCs with `llvm-objdump`). Trace output is huge — bound it (`tail -c`) to avoid filling the disk.
- Launch long runs via `nohup … &` inside a foreground `docker exec` (the `run_in_background`+`docker exec`+redirect combo drops the log); poll the on-disk log.
