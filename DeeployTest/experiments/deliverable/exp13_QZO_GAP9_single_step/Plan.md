# exp13 — Port on-device QZO simulation to GAP9 (single-step smoke test)

Branch `feat/QZO`. Goal: run the exact QZO (quantized MeZO) on-device flow we validated on
Siracusa, but with the target hardware wired to **GAP9** — the platform we will extend with the
NE16 accelerator next. Deliverable = a single-step QZO smoke test on GVSoC/GAP9, the GAP9 analog
of exp10 (Siracusa single-step).

## What "port to GAP9" actually involves (it is more than a platform string)

The runner takes a `default_platform` argument, so the entry point is a one-liner. But the QZO
graph uses three training-only ops — `RQSPerturbRademacher` (int8 weight/bias perturbation),
`PerturbRademacher` (fp32 BN/fc perturbation) and `BatchNormInternal` (frozen-stat BN) — plus
`GlobalAveragePool`, and **none of those were registered in the GAP9 Deeploy target** (they lived
only in PULPOpen, which is what the DeeployTest "Siracusa" platform maps to). So the port has two
layers:

1. **DeeployTest harness** (per-platform directory) — runner, `deeploymezotest.c`, CMake.
2. **Deeploy compiler target** (`Deeploy/Targets/GAP9/Platform.py`) — the op→kernel mapping that
   decides how each ONNX node is lowered.

The port is low-risk because GAP9 was *already* reusing PULPOpen pieces (Quant/Dequant, Conv1D)
and — critically — **`TargetLibraries/GAP9/CMakeLists.txt` globs `../PULPOpen/src/**`**, so the
QZO kernels (`RandomNoiseQuant.c`, `BatchNorm.c`, `ZORuntime.c`) are already compiled into GAP9's
`deeploylib`. No C kernel had to be moved; the port only had to *register* the ops.

## Step-by-step (files changed)

### 1. Runner — `DeeployTest/deeployMezoRunner_tiled_GAP9.py` (new)
Thin entry mirroring `deeployMezoRunner_tiled_siracusa.py`, calling
`main(tiling_enabled=True, default_platform='GAP9')`. The driver
(`testUtils/deeployMezoRunner.py`) already threads `default_platform` through code generation
(`platformMapping.py` maps `"GAP9" → GAP9Platform()`) and the build
(`TEST_GAP9/build_<worker>`), so nothing else in the driver changed.

### 2. On-device harness — `Platforms/GAP9/src/deeploymezotest.c` (new)
Byte-for-byte copy of the Siracusa MeZO harness (only the cosmetic banner string changed to
"GAP9"). Its includes are all platform-portable — `pmsis.h`, `dory_mem.h`, `CycleCounter.h`, the
generated `TrainingNetwork.h`/`OptimizerNetwork.h`, and `kernel/ZORuntime.h` (on the deeploylib
include path) — and GAP9 is also a PULP/pmsis cluster, so no source change was needed.

### 3. Build — `Platforms/GAP9/CMakeLists.txt` (extended)
Added the `MEZO_TRAINING` branch ported from Siracusa: it selects `src/deeploymezotest.c`, links
`training_network` + `optimizer_network` + `deeploylib`, and passes the ZO compile-time defines
(`N_TRAIN_STEPS`, `N_ACCUM_STEPS`, `TRAINING_NUM_DATA_INPUTS`, `ZO_EPS`, `ZO_LR`, `ZO_Q`,
`ZO_SEED`) plus the `DUMP_WEIGHTS` / `DUMP_STEP_LO/HI` / `BN_FROZEN_STATS` passthroughs.
GAP9-specific lines were preserved: `sdk.config`, `add_gvsoc_emulation("gap9.evk")`,
`add_board_deployment("gap9.evk")`, `POWER_MEASUREMENT`, the warning waivers and
`--print-memory-usage`. One generalization: the warning-waiver `target_compile_options(network …)`
now targets `${NETWORK_LIB}` so it applies to `training_network` in the MeZO build.

### 4. Deeploy target — `Deeploy/Targets/GAP9/Platform.py` (extended)
Registered the missing ops in `GAP9Mapping`, reusing the PULPOpen tiling-ready bindings (kernels
already in GAP9 `deeploylib`):
- `BatchNormInternal` → `PULPBatchNormInternalTilingReadyBindings`
- `PerturbRademacher` → `PULPPerturbRademacherTilingReadyBindings` (fp32 ZO perturb)
- `RQSPerturbRademacher` → `PULPRQSPerturbRademacherTilingReadyBindings` (int8 ZO perturb)
- `GlobalAveragePool` → `PULPGlobalAveragePool2DTilingReadyBindings` (SpeechNet forward)

`Constant` needs no mapper (folded during lowering — PULPOpen has no `Constant` mapper either).

## How the missing ops were found (no guessing, one build per discovery avoided)
`_opdiff.py` loads both ZO ONNX graphs and diffs every `op_type` against `GAP9Mapping.keys()`:
- `speechnet_qzo12_train/network.onnx` (66 nodes): missing `Constant` (folded) + `GlobalAveragePool`.
- `speechnet_qzo12_update/network.onnx` (22 nodes): **nothing missing** — the perturb ops added in
  step 4 fully cover the update graph.

## Smoke test (the deliverable)
Fixture = `speechnet_qzo12_train` / `speechnet_qzo12_update` (exp12 fresh pooled-99.99 calibration,
lr 1e-5). One update step, `n_accum 4`, ε 0.01, seed 42, `BN_FROZEN_STATS=ON`, tiled, 8 cores,
GAP9 L1 128 KB / L2 1.5 MB. Pass criterion = build succeeds, GVSoC runs the two graphs, and the
per-pair device losses match the host reference within the ZO tolerance (same criterion as the
Siracusa smoke tests). Reproduction command is in `Findings.md`.
