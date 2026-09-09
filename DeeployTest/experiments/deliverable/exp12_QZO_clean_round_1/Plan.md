# exp12 — QZO clean round-1: full pipeline, from scratch, ffast-math on

Date: 2026-09-08 · Branch `feat/QZO`. A clean-room re-run of the whole quantized-ZO pipeline with
**no reused artifacts** — every fixture, weight dump and number is regenerated here — to verify
the pipeline end to end, and to check that the fast default build (ffast-math, fused fp) is
**learning-neutral**: it gives up bit-exactness but should keep the same accuracy.

Fold 3, subject S01, vocalized. Fine-tune on session-3 batch 1 (54 windows, 30%, seed 42),
evaluate on the whole session-3 batch 2 (180 windows). lr 1e-5, ε 0.01, n_accum 4, 200 epochs
(= 2700 update steps), Rademacher seed 42.

---

## What the QZO flow does now (plain-language walkthrough)

There are five parts. Two live in `Onnx4Deeploy` (host: build the graphs and the references),
three in `TrainDeeploy` (device: compile and run on the simulated chip). Paths are given for each
step, with the fixes we made along the way.

### Part 1 — Onnx4Deeploy builds the fixture (graphs + calibration + reference)

Command: `Onnx4Deeploy.py -mode q-zo-train` → `export_zo_training(quant=True)`
(`onnx4deeploy/core/base_exporter.py:529`) → `_export_qzo_training` (`base_exporter.py:655`),
which is the orchestrator. Step by step:

1. **Build the quantized model and load the checkpoint.** `create_brevitas_model`
   (`models/speechnet_exporter.py:125`) builds `QuantSpeechNetDeploy`
   (`models/pytorch_models/speechnet/speechnet_quant.py`): each block is int8 conv → fp32
   BatchNorm (kept *unfolded* so its γ/β stay trainable) → ReLU → MaxPool; the classifier is a
   quantized Linear. Weights loaded from `leave_one_session_out_fold_3.pt`.

2. **Pick the quantization scales.**
   - *Weight scales are data-free:* per output channel, `s_w[c] = max|W[c]| / 127` (Brevitas
     `Int8WeightPerChannelFloat`). No data needed.
   - *Activation scales need data — this is "calibration".* We use the **pooled-99.99** scales
     computed once on the pretraining distribution (sessions 1+2, all 1800 windows): pool the
     absolute activations at each site over all windows and take the 99.99th percentile, then
     `s_x = threshold / 128`. These thresholds are stored in a small JSON and **baked in at
     export**: `base_exporter.py` reads `QZO_POOLED_THRESHOLDS` and replaces each activation
     quantizer's scale with a frozen constant (a `ConstThreshold` module), asserting per site
     that `proxy.scale() == threshold/128`. Without the env var it falls back to the old default
     (a short `calibration_mode` pass on a few windows) — kept, not deleted.
     *(Why: the default calibration was a per-batch percentile ≈ the batch max, on target-session
     data; the pooled version is a real percentile over the pretraining data. Study:
     `Onnx4Deeploy/QZO_exp/exp_calibration/Findings.md`.)*

3. **Export to an integer ONNX graph.** `exportBrevitas` (DeepQuant) writes the fake-quant model
   as an ONNX where every quantized tensor sits behind a QCDQ chain; `create_quant_pipeline`
   (`core/optimization_passes.py`) folds the per-tensor ones; **`build_int8_forward`**
   (`transform/qzo_weight_integerize.py:264`) finishes the per-channel conv/fc weights, rewiring
   each Conv to read an int8 weight directly and building the matching per-channel RequantShift.
   The fc head is turned into a **float** Linear here (dequantized weight, its input-quant
   bypassed) — this is the QMCUNetZO "int8 conv + fp32 head" convention.

4. **Inject the perturbation ops.** `build_qzo_train_graph` / `build_qzo_update_graph`
   (`transform/qzo_transform.py:53,153`) add a `RQSPerturbRademacher` (int8 weight / int32 bias)
   or `PerturbRademacher` (fp32 BN/fc) before each parameter's consumer, keyed by a per-parameter
   node id so the train and update graphs regenerate the *same* random `z`.

5. **Compute the reference and save the fixture.** The multi-step QZO reference sim (inside
   `_export_qzo_training`) replays the training with `run_onnx_graph` and writes:
   `network_zo_train.onnx`, `network_zo_update.onnx`, `inputs.npz` (the 54 windows in graph-input
   order), `outputs.npz` (reference losses + final weights).

**Fixes in Part 1:** (a) inputs saved as positional `arr_NNNN` in graph-input order (was
name-sorted → wrong buffer mapping); (b) 4-D calibration window shape (a 3-D shape silently fell
back to random calibration → loss ~9); (c) `build_int8_forward` RequantShift `mul`/`add` use the
**per-layer** `s_in/s_out` traced from the graph, not a uniform 1/128 (the uniform assumption
was only valid before per-layer calibration — commit `bc147f6`); (d) the requant rounding `+div/2`
is baked into the variable bias `add` so host and device round the same way (commit in
`qzo_weight_integerize.py`); (e) pooled-99.99 baking via `QZO_POOLED_THRESHOLDS` (commit
`142d742`); (f) the reference sim rebuilds the ±ε graphs once and only patches the seed per step
(≈15 h → a few hours), and can cover the whole round to match the float-ZO convention (commit
`8566ec0`).

### Part 2 — how the calibration scales actually reach the graph nodes

No scale is looked up by name from a side file; everything is by graph connectivity.

- **Weight/bias scales → RequantShift constants.** `trace_weight_qcdq`
  (`transform/qzo_weight_integerize.py:62`) starts at a conv's own weight input and walks back the
  dequant chain to the scale. Then the post-conv RequantShift gets `mul[c] = round(s_in·s_w[c]/
  s_out · 2^16)` and the int32 bias becomes the RequantShift `add[c] = round(bias/s_out·2^16) +
  div/2` — per output channel, computed from that layer's own traced `s_in`, `s_out`, `s_w`.
- **Activation scales → Quant node attribute.** After the pooled thresholds are frozen (Part 1
  step 2), each `Quant` node carries `scale = threshold/128` as an attribute (verifiable in the
  exported graph: the first Quant's scale is 22.296875 = 2854/128 for fold 3).
- **Perturbation size → RQSPerturb mul.** The ε perturbation is `round(ε/s_w[c] · 2^15)` per
  channel, so it is a whole number of LSBs on each channel (3–6 LSB at ε=0.01).

### Part 3 — the faithful PyTorch + Brevitas model (the accuracy reference)

This is what tells us the accuracy to expect; it is NOT the device — it is a clean PyTorch
forward on the same quantized network, run entirely on the host.

- Build `QuantSpeechNetDeploy` (Brevitas fake-quant), freeze the weight scales (a `ConstScale`
  module) and the activation scales at the same pooled-99.99 thresholds, then make the fc head
  **float exactly as the device does** (`to_float_fc`: dequantize-then-requantize the fc weight
  as the init, bypass the fc input-quant, treat fc weight+bias as float ZO parameters).
- Run MeZO by hand: antithetic forwards at θ±εz, `g=(L+−L−)/(2ε·n_accum)`, and the **direct int8
  update** `w_int += round(−lr·g·z/s_w)` for the quantized weights (fp32 params updated in
  float). lr 1e-5.
- Script: `Onnx4Deeploy/QZO_exp/.../run_fc_float_ref.py` / `run_incremental_fcfloat.py`.

**Why fc-float here matters (a fix):** an earlier PyTorch sim quantized fc as int8 and reported
90.00%; that is not what the device runs (the device fc is float). The device-faithful fc-float
version gives **≈88.89%** on round-1 batch 2 — that is the correct reference number.

### Part 4 — the "device host": how the reference losses are computed

The reference the device is checked against is NOT PyTorch. It is `run_onnx_graph`
(`onnx4deeploy/utils/onnx_node_implementations.py:1037`), a pure-Python executor that runs the
*exact exported integer graph* with device-faithful arithmetic. Each op mirrors the device kernel
bit-for-bit in fp32: Quant multiplies by `1/scale` in fp32 (not fp64 divide); RequantShift
truncates; BatchNorm uses the device's per-channel order; GlobalAveragePool is a sequential sum;
Gemm is the device's 6-way unrolled accumulation; SoftmaxCrossEntropy (`:846`) uses the device's
max → sequential sum-of-exp → log order. This is why "bit-exact" is even reachable.

**Fixes in Part 4:** the op-by-op mirrors above were added in exp11 (originals kept as
`# [exp11-orig]`). The one residual left is the SCE `expf`/`logf` (numpy/glibc vs device newlib) —
a ~1-ulp difference that is the sole remaining non-bit-match (proven in `exp11_bitexact_SCE` and
`exp11_bitexact_SCE_verif`). It does not affect accuracy.

### Part 5 — the device run (compile + simulate)

- `deeployMezoRunner_tiled_siracusa.py` → codegen (`testMVPTraining.py`,
  `codeGenerateTraining.py`) → C → CMake → GVSoC. Every graph input is typed from the ONNX
  `elem_type` (not inferred from values).
- The training loop is in `Platforms/Siracusa/src/deeploymezotest.c`: for each update step, run
  the +ε forward then the −ε forward per mini-batch (loss = SCE output), form `g` and
  `coeff=−lr·g` on the cluster, then apply the update in place via `RunOptimizerNetwork`
  (the zo_update graph), reusing the same seed so `z` matches the probes.
- The perturbation kernel is `TargetLibraries/PULPOpen/src/RandomNoiseQuant.c`
  (`ApplyPerturbQuantRademacher_CHW`/`_i32`): `noise = (z·mul + rounding) >> S`, then clamp — with
  a runtime `eps_scale` so the same kernel does both the ±ε probe and the `−lr·g` update.
- Quant/Dequant use the shipped **8-core parallel** PULP templates with **fp32-cast** scale
  constants (`Deeploy/Targets/PULPOpen/Templates/{Quant,Dequant}Template.py`, wired in
  `Bindings.py`).

**Fixes in Part 5:** the five bit-exactness bugs (positional npz order; Quant passes `1/scale`;
4-D calib shape; graph-input types from `elem_type`; RQSPerturb per-channel index `/`, not `%`);
`BN_FROZEN_STATS=ON` for the frozen-stat BatchNorm; the Quant/Dequant kernels were running a
single-core generic template with double-precision soft-float — rewired to the parallel templates
and cast to fp32 (**~45× faster; the QZO step went from 10.8× slower to 4.3× faster than float
ZO** — commit `21b2910`, doc `exp10_QZO_single_step_profiling/QZO_Latency_Assurance.md`).
Optional strict-fp32 build (`DEEPLOY_STRICT_FP32`) exists for full bit-exactness; **exp12 does NOT
use it** (see below).

---

## What exp12 runs (clean, no reused artifacts)

1. **Reference accuracy (PyTorch, Part 3):** fc-float device-faithful sim, fold-3 round 1, lr 1e-5
   → batch-2 balanced accuracy. Expect ≈88.89%.
2. **Export the fixture (Part 1):** `-mode q-zo-train`, 54 seed-42 windows, pooled-99.99 baked.
   Verify: 54/54 windows byte-identical to the sim draw; leading Quant scale 22.296875.
3. **Device round-1 (Part 5), ffast-math ON:** 2700 steps, lr 1e-5, ε 0.01, n_accum 4, seed 42,
   `BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON` and **no `DEEPLOY_STRICT_FP32`** — the fast fused-fp
   default. We accept it will not be bit-exact vs the host reference; the question is whether it
   is **learning-neutral** (same accuracy as strict/exp11's 89.44% and the 88.89% reference).
4. **Device eval (Part 5 + Part 1):** extract the dumped weights, rebuild the quantized inference
   fixture via Onnx4Deeploy (`build_int8_forward` + inject the trained weights, offset handled),
   run the untiled device inference over all 180 batch-2 windows (ffast-math, `BN_FROZEN_STATS`),
   compute balanced accuracy.

**Success:** device round-1 batch-2 balanced accuracy ≈ 88–89% (matching the PyTorch reference and
the strict exp11 result) ⇒ the pipeline is correct end to end and ffast-math is learning-neutral,
so we keep the speed.

## Files

`Plan.md` (this) · `Findings.md` (after) · `results/` (accuracy JSONs, carry weights) ·
`fixture/` (exported graphs + inputs/outputs) · `logs/` (export, device round, eval — gzipped).
Reproduction commands appended to Findings as executed.
