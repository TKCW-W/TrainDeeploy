# exp13 — Port on-device QZO simulation to GAP9 — Findings

Date: **2026-09-10** · Branch `feat/QZO` · Fixture = `speechnet_qzo12_train` / `speechnet_qzo12_update`
(exp12 fresh pooled-99.99 calibration, lr 1e-5). Container: `traindeeploy` (deeploy:traindeeploy-dev).

## Headline

The QZO (quantized MeZO) on-device flow is **ported to the GAP9 target and validated end-to-end
through code generation**: with `deeployMezoRunner_tiled_GAP9.py`, the SpeechNet ZO train + update
graphs map onto GAP9, **tile successfully**, and generate valid, GAP9-targeted, 8-core-PULP C code
(`TrainingNetwork.c` 1.14 MB, `OptimizerNetwork.c` 284 KB) with all QZO kernels emitted.

The final GVSoC **execution** step is blocked by one environment fact: **the GAP9 SDK
(`GAP_SDK_HOME`) is not installed in this container** — per `README_GAP9.md` it is "not publicly
available" and requires a dedicated `deeploy-gap9` container built with SSH access to private
GreenWaves repos. This is a provisioning step (private credentials the user controls), **not a
defect in the port**. Everything up to the point where the GAP9 SDK is required is done and proven.

## What "port to GAP9" required (it was more than a platform string)

The runner takes a `default_platform` argument, so the entry point is a one-liner — but the QZO
graph uses training-only ops (`RQSPerturbRademacher`, `PerturbRademacher`, `BatchNormInternal`) plus
`GlobalAveragePool` that were **not registered in the GAP9 Deeploy target** (they lived only in
PULPOpen, which is what the DeeployTest "Siracusa" platform actually maps to). Registering them was
low-risk because `TargetLibraries/GAP9/CMakeLists.txt` already globs `../PULPOpen/src/**`, so the
QZO **kernels were already compiled into GAP9's `deeploylib`** — only the Python-side op→kernel
bindings were missing. See `Plan.md` for the full step-by-step.

## Files changed (the port)

| # | File | Change |
|---|---|---|
| 1 | `DeeployTest/deeployMezoRunner_tiled_GAP9.py` (new) | thin runner: `main(tiling_enabled=True, default_platform='GAP9')` |
| 2 | `DeeployTest/Platforms/GAP9/src/deeploymezotest.c` (new) | copy of the Siracusa MeZO harness (only the banner string differs) |
| 3 | `DeeployTest/Platforms/GAP9/CMakeLists.txt` | added the `MEZO_TRAINING` branch (harness, `training_network`+`optimizer_network` link, ZO defines, `DUMP_WEIGHTS`/`BN_FROZEN_STATS` passthroughs); preserved GAP9 `sdk.config`, `gap9.evk` gvsoc/board, warning waivers; generalized the waiver to `${NETWORK_LIB}` |
| 4 | `Deeploy/Targets/GAP9/Platform.py` | registered `BatchNormInternal`, `PerturbRademacher`, `RQSPerturbRademacher`, `GlobalAveragePool` in `GAP9Mapping` (reusing the PULPOpen tiling-ready bindings); switched Quant/Dequant to the tiling-ready bindings |

## Real bugs fixed along the way (latent GAP9-target defects, exposed by the QZO graph)

1. **GAP9 forward `SoftmaxCrossEntropyLoss` binding declared only one output type**
   (`Deeploy/Targets/GAP9/Bindings.py`). SCE has **two** outputs (loss + `log_prob`); with one
   declared output type the second buffer never gets `_type` annotated →
   `AttributeError: 'GAP9VariableBuffer' object has no attribute '_type'` at
   `CrossEntropyLoss / log_prob / output_1`. Fixed to `[float32_t, float32_t]`, matching PULPOpen.
   Only the two-output SCE in the training graph triggers it, so GAP9 BP/inference never hit it.

2. **GAP9 Quant/Dequant used the untiled `BasicQuant/DequantBindings`**
   (`Deeploy/Targets/GAP9/Platform.py`). A tiled run needs a `tileConstraint` →
   `AttributeError: 'PULPQuantTemplate' object has no attribute 'tileConstraint'`. Switched to
   `PULPQuant/DequantTilingReadyBindings` (`UnaryTileConstraint`), matching `PULPMapping`.

3. **`DeeployTest/CMakeLists.txt` GAP9 block only handled `if(TRAINING)`, not `MEZO_TRAINING`**
   (lines ~86, ~108). In a MeZO build the `else()` ran `target_compile_options(network …)` but
   `network` is not built (we build `training_network`+`optimizer_network`) →
   "Cannot specify compile options for target network which is not built." Changed both to
   `if(TRAINING OR MEZO_TRAINING)`.

## How the missing ops were found (method, not guesswork)

`_opdiff.py` loads both ZO ONNX graphs and diffs every `op_type` against `GAP9Mapping.keys()`:
- `speechnet_qzo12_train/network.onnx` (66 nodes): missing `Constant` (folded, no mapper needed) + `GlobalAveragePool`.
- `speechnet_qzo12_update/network.onnx` (22 nodes): **nothing missing** after the perturb ops were registered.

This turned "one slow build per missing op" into a single up-front check. (`_import_check.py` verifies
`GAP9Platform` imports and the new mapping entries resolve.)

## Evidence the port works through codegen (no GAP9 SDK needed for this part)

A full `deeployMezoRunner_tiled_GAP9.py` run (single step, n_accum 4, ε 0.01, seed 42,
`BN_FROZEN_STATS=ON`, tiled, 8 cores, GAP9 L1 128 KB / L2 1.5 MB) completes **frontend →
lowering → binding → tiling → codegen** and emits GAP9 C. The QZO kernels are actually in the
generated code (`fixture/emitted_kernels.txt`, headers in `fixture/`):

- `OptimizerNetwork.c` (ZO update): `ApplySequentialRademacherPerturbation`×12,
  `ApplyRademacherPerturbation`×12 (fp32 perturb), `ApplyPerturbQuantRademacher_i32`×5 (int32 bias),
  `ApplyPerturbQuantRademacher_CHW`×5 (int8 per-channel conv weights) = 12 fp32 + 10 int8 perturbs.
- `TrainingNetwork.c` (ZO forward): `PULP_BatchNormInternal_fp32`×5, plus tiled cluster-fork kernels
  for GlobalAveragePool / Conv / Gemm / RequantShift / Quant / Dequant.

The run then fails **only** at the GAP9 CMake/SDK stage:
`include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)` → `/utils/cmake/setup.cmake` (GAP_SDK_HOME
empty), and `gap9_gvsoc.cmake` `FATAL_ERROR "Environment variable GAP_SDK_HOME not set"`. Full log:
`logs/smoke_run.log`.

## What remains: run the smoke test in the GAP9 container

The GAP9 SDK is provisioned only in the dedicated GAP9 image
(`ghcr.io/pulp-platform/deeploy-gap9:latest`, or built locally via
`cd Container && make deeploy-gap9` with an SSH key for the private GAP SDK repo; SDK pinned in
`Makefile` to v5.21.1 commit `1796873…`). Inside that container, the **exact** smoke-test command is:

```bash
# kill any orphan gvsoc first
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9; rm -rf DeeployTest/TEST_GAP9

cd DeeployTest
python3 deeployMezoRunner_tiled_GAP9.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo12_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo12_update \
  --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
  --l1 128000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON
```

Pass criterion (same as the Siracusa smoke tests): GVSoC runs both graphs and the per-pair device
losses match the host reference within the ZO tolerance. Because the generated C reuses the exact
PULP kernels already validated bit-exact on Siracusa (exp12), and GAP9 links the same kernel sources,
the expectation is a clean pass; only the GAP9 SDK build/run environment is untested here.

## Reproduction of the codegen validation (in THIS container, no SDK)

```bash
docker exec traindeeploy bash -lc 'cd /app/ETH/TrainDeeploy && \
  python3 DeeployTest/experiments/deliverable/exp13_QZO_GAP9_single_step/_opdiff.py'      # op coverage
docker exec traindeeploy bash -lc 'cd /app/ETH/TrainDeeploy && \
  python3 DeeployTest/experiments/deliverable/exp13_QZO_GAP9_single_step/_import_check.py' # mapping resolves
# then the runner command above (reaches codegen, stops at the GAP_SDK_HOME include)
```
