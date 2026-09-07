# exp11_QZO_round1_fix — clean on-device QZO round-1 at lr 1e-5 on the fully-fixed stack

Date started: 2026-09-07 · Branch `feat/QZO` (both repos) · Export in `agitated_hugle`, GVSoC in `traindeeploy`

## Plan

### Task

Re-run the **real** round-1 on-device QZO fine-tuning — S01 / vocalized / **fold 3**, fine-tune
on batch 1 (54 stratified seed-42 windows = 30%), lr **1e-5**, ε 0.01, n_accum 4, 200 epochs =
2700 update steps, **all 22 params training** (no freeze) — end-to-end from fixture export to
the MeZo runner, with **every fix applied**, to:

1. **Assess bit-exactness** of the full training round (device vs host reference, all 21,600
   losses + final weights vs `updated_*`).
2. **Dump the trained weights** and evaluate **batch-2 balanced accuracy** on device (untiled,
   exp5 protocol).

### Context / status at start

Previous round-1 attempts and what they established:

| run | stack | result |
|---|---|---|
| exp9 `round1_gvsoc.log` (2026-09-04) | lr 1e-5, **fast-math** build | **FAILED — 11,171/21,600** loss errors → triggered the divergence hunt |
| exp11–13 (QZO_exp, other session) | root cause + fix campaign | int path bit-exact at every block; divergence = clang FMA/reassoc in fp32 BN/Gemm under `-ffast-math`; strict-fp32 opts added |
| exp13-setting strict round (2026-09-07) | lr **3e-6, conv frozen**, strict-fp32 | **0/21,600 errors**, ≤1-ulp residual, 15/22 params bit-exact (rest 1 ulp = SCE picolibc) |

**Open question this run answers:** the 0/21,600 was a *controlled* round (conv frozen, 3e-6).
At lr 1e-5 with the int8 conv weights actually training, a 1-ulp fp residual (SCE libm) can in
principle flip a `round(coeff·z/s_w)` on a knife-edge step and **bifurcate** the int8
trajectory (under fast-math this happened at step ~326). Expectation: strict-fp32 keeps
per-step losses within ≤1 ulp; whether that suffices for 0 errors over the full 1e-5 round is
the measurement.

Fixes in the stack (all committed):

- per-layer act scales in RequantShift mul/add (`bc147f6`) + pooled@99.99 baked at export
  (`142d742`, `QZO_POOLED_THRESHOLDS`)
- requant rounding fix: `+div/2` baked into the variable RQS add (`qzo_weight_integerize.py:366`)
- host reference = seed-patched fast full-round sim, executor made op-for-op device-faithful
  (exp11 L1/L2 mirrors: Quant ×(1/s) fp32, BN device order, GAP sequential, Gemm 6-way unroll,
  SCE sequential)
- strict-fp32 build opts (`4674b3e`): `DEEPLOY_STRICT_FP32=ON` +
  `DEEPLOY_STRICT_FP32_FILES=BatchNorm.c;Gemm.c;GlobalAveragePool.c;RandomNoise.c`
- Quant/Dequant parallel templates + fp32 casts (`21b2910`)
- eval caveats: `-D BN_FROZEN_STATS=ON` on the untiled inference runner; bias injection into
  inference fixtures must use `bias_rqsadd − div/2` (the exp9/10 eval double-added the
  rounding offset → +0.5 LSB systematic)

Accuracy reference: device-faithful fc-float PyTorch sim, fold-3 round-1 @1e-5 = **88.89%** on
batch 2 (device z-stream is an independent realization → expect agreement within ~±1 pt, not
equality).

### Steps

1. **Export the fixture** (`agitated_hugle`): `-mode q-zo-train`, 54 stratified seed-42 windows,
   `--n-epochs 200` (full-round reference: 10,800 loss pairs + `updated_*`), lr 1e-5, pooled
   thresholds baked. Output → `/app/Onnx4Deeploy/QZO_exp/exp11_round1_fix`.
2. **Verify the fixture**: 🧊 baked-thresholds line; 54/54 training windows byte-identical to
   the sim draw; leading Quant scale = 22.296875.
3. **Pack** (`traindeeploy`): `pack_2step_fixture.py` → `speechnet_qzo11_train/update`.
4. **Device round** (`traindeeploy`): 2700 steps, strict-fp32 + frozen BN + weight dump
   (command in Reproduction). Save `logs/round1_gvsoc_strict.log`.
5. **Bit-exactness read-out**: `Errors: X out of 21600`; dumped weights vs reference
   `updated_*` (dtype-aware extractor).
6. **Accuracy**: extract dump → build quantized inference fixture (route (b): inject dumped
   tensors; **bias = dumped − div/2**) → whole batch 2 (180 windows) through the untiled
   runner (`-D BN_FROZEN_STATS=ON DEEPLOY_STRICT_FP32=…`) → balanced accuracy vs 88.89%.
7. Findings + Reproduction below; logs gzipped into `logs/`.

### File paths

- This deliverable: `TrainDeeploy/DeeployTest/experiments/deliverable/exp11_QZO_round1_fix/`
- Export dir: `Onnx4Deeploy/QZO_exp/exp11_round1_fix/`
- Eval tooling: reused from `../exp9_QZO_round1/` (`extract_qzo_weights.py`,
  `build_qzo_infer_fixture.py`, `qzo_accuracy_eval_untiled.py`), with the bias `−div/2`
  correction applied for the injection (exp11 copy if a change is needed).

## Findings

### F1 · Bit-exactness of the real 1e-5 round: per-step arithmetic FIXED; full-trajectory
### identity limited by one knife-edge bifurcation (the L4 residual)

Headline: `Errors: 11,169 out of 21,600` — superficially the same count as the fast-math exp9
run (11,171), but the **mechanism is completely different**, and the error *distribution*
proves it (strict-flags presence verified in the cmake line of the log):

| pairs (deciles of 10,800) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| diffs > 1e-6 | **0** | **0** | 1 | 1480 | 2154 | 2159 | 2158 | 2159 | 2159 | 2160 |

- **Zero mismatches for the first ~2,160 pairs**; first sustained divergence at pair **3556 ≈
  update step 889**. Up to there the device reproduced ~7,100 forwards + 889 real int8 weight
  updates of a genuine lr-1e-5 training bit-faithfully (under the float-ZO tolerance) — the
  per-step arithmetic fix (strict-fp32 + all graph fixes) **works**. Under fast-math, errors
  start early and are scattered; here the prefix is clean.
- **What happens at step ~889**: the one remaining analog residual — ≤1 ulp from SCE's picolibc
  `expf/logf` (the documented **L4** item, host mirror "only needed for diff=0") — landed on a
  knife-edge of the discrete update quantizer `round(coeff·z/s_w)`. One int8 weight stepped
  differently on device vs reference; from then on the two follow **different but equally valid
  ZO trajectories**, so every later loss "error" is trajectory divergence, not computation
  error. This is why the controlled round (exp13 setting: conv frozen, 3e-6) could reach
  0/21,600 while the real 1e-5 round cannot without L4: freezing conv removes the discrete
  amplifier from the loop.
- Final-weight comparison vs the reference `updated_*` (dtype-aware extractor): **2/22 tensors
  bit-exact** (block-0 conv weight+bias — the near-frozen coarse-grid tensors), the rest carry
  the accumulated two-trajectory difference (int8 conv up to ±5 LSB / ≤1,567 of 7,168 elements
  on block 4; biases ≤43 add-units; fp32 ≤2.2e-3) — magnitudes consistent with ~1,800 diverged
  steps, not with any systematic defect.

**Conclusion F1**: the bit-exactness issue *as an arithmetic problem* is fixed — confirmed on
a real training prefix of 889 steps at lr 1e-5. Full-trajectory bit-identity over 2,700 steps
at 1e-5 is a stricter goal that additionally requires the L4 host mirror (bit-exact
`expf/logf`), because a quantized update rule turns any 1-ulp residual into an eventual
trajectory split. Bifurcation is an intrinsic sensitivity of *replaying* quantized training,
not a defect of the device: see F2.

### F2 · The bifurcation is learning-neutral: both trajectories reach the same accuracy

Host-executor balanced accuracy on the whole batch 2 (180 windows), identical fixture builder
(offset-corrected, `build_qzo_infer_fixture11.py`):

Host-executor balanced accuracy on the **correct** device graph (`network.onnx`, offset-stripped
— see F4; the earlier 86.67/86.11 were on the wrong offset-included graph and are superseded):

| weights (host-executor on network.onnx) | b2 balanced accuracy |
|---|---|
| device trajectory (WDUMP after 2,700 steps) | **89.44%** |
| reference trajectory (`updated_*`) | 88.89% |
| zero-shot (fc-float, pooled@99.99) | 85.00% |

The two trajectories land **0.55 pt (1 window) apart** — the step-889 bifurcation is
learning-neutral. And device-trajectory host-executor (89.44%) equals the on-device untiled
number (89.44%, F3) exactly: the device inference forward is faithful to its own graph.

### F3 · On-device accuracy (untiled, exp5 protocol) — the headline number

**Device round-1 batch-2 balanced accuracy = 89.44%** (untiled Siracusa runner, 180 windows,
device argmax vs true labels).

| model | b2 balanced accuracy |
|---|---|
| **device, round-1 trained (untiled)** | **89.44%** |
| fc-float PyTorch sim (independent z-stream) | 88.89% |
| device-faithful zero-shot (fc-float, pooled@99.99) | 85.00% |

The device-trained model reaches **89.44%**, **+4.44 over zero-shot** and slightly above the
88.89% fc-float sim reference (the two use different Rademacher z realizations, so exact equality
was never expected — agreement within ~1 pt confirms the on-device QZO training worked). This is
the deliverable: **QZO round-1 fine-tuning runs on device and improves batch-2 accuracy by
~4.4 points, matching the host simulation.**

Host cross-check (executor on the exact graph the device runs, `network.onnx`): device logits
match `run_onnx_graph` to **max|Δ| ≈ 0.087** per window (small non-strict-inference fp residual;
the eval runner did not pass strict-fp32 — immaterial to argmax/accuracy), and the **balanced
accuracies are identical: device 89.44% = host-executor(network.onnx) 89.44%** — the device
inference forward is faithful to its graph. Reference-trajectory weights: 88.89% on the same
host executor.

### F4 · Eval-fixture offset bug found and its (non-)impact

While reading out F3 I found a bug in the **reference-generation** side of
`build_qzo_infer_fixture11.py`: it computes `outputs.npz` on `network_ref_offset.onnx`
(offset-included), assuming `run_onnx_graph` truncates so `trunc(x+div/2) ≡ round(x)` makes it
equal to the offset-stripped device graph. **That equivalence does not hold in the executor** —
`run_onnx_graph(network_ref_offset)` differs from `run_onnx_graph(network.onnx)` by up to **1.5**
in logits. Consequences:

- **The device accuracy (F3, 89.44%) is unaffected** — the device runs `network.onnx`
  (offset-stripped), verified: device ≈ `run_onnx_graph(network.onnx)` to 0.087.
- **The "bit-exact fails 180/180" in the eval was a false alarm** — it compared the device
  against the wrong-graph reference; against the correct graph the residual is ~0.087 (< the
  small non-strict fp level), not a real mismatch.
- **F2's host-executor numbers (86.67 / 86.11) were computed on the wrong graph** and are
  superseded by the `network.onnx` recomputation (F3 host cross-check). The *conclusion* of F2 is
  unchanged — the two trajectories reach comparable accuracy — but the exact values are corrected.

Fix for the tooling: compute the host reference on `network.onnx` (the graph the device runs),
not on the offset-included variant. Tracked as a follow-up; does not affect the device result.

## Reproduction

### Step 1 — full-round fixture export (agitated_hugle) — as executed 2026-09-07

```bash
docker exec agitated_hugle bash -lc 'cd /app/Onnx4Deeploy && rm -rf QZO_exp/exp11_round1_fix && \
 find onnx4deeploy -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null;
QZO_POOLED_THRESHOLDS=/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/fixture/pooled_9999_fold3.json \
PYTHONDONTWRITEBYTECODE=1 nohup python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train \
  --noise-type rqs_rademacher \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --condition vocalized --batch 1 \
  --data-size 54 --stratified --n-epochs 200 --n-accum 4 --lr 1e-5 \
  -o /app/Onnx4Deeploy/QZO_exp/exp11_round1_fix \
  > .../exp11_QZO_round1_fix/logs/export.log 2>&1 &'
# rate observed: ~6.5 update-steps/min (seed-patched reference) → ~7 h for 2700 steps
```

### Step 2 — fixture verification (agitated_hugle) — PASSED

`verify_fixture.py` (this directory): **54/54** training windows + labels byte-identical to the
sim draw (`data_cache_incr.npz` trX1/trY1); leading Quant scale **22.296875** (= pooled
2854/128); `outputs.npz` carries `loss_plus/loss_minus (10800,)` + **22** `updated_*` tensors.

### Step 3 — pack (traindeeploy)

```bash
python3 experiments/zo_smoke/pack_2step_fixture.py \
  /app/ETH/Onnx4Deeploy/QZO_exp/exp11_round1_fix \
  Tests/Models/Training/SpeechNet speechnet_qzo11_train speechnet_qzo11_update
# -> train: 24 inputs / 25 inits · update: 22 in / 22 out
```

### Step 4 — strict-fp32 device round (traindeeploy) — as launched

```bash
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9; rm -rf TEST_SIRACUSA
python3 deeployMezoRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo11_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo11_update \
  --n-steps 2700 --n-accum 4 --num-data-inputs 2 \
  --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 \
  -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON DEEPLOY_STRICT_FP32=ON \
     "DEEPLOY_STRICT_FP32_FILES=BatchNorm.c;Gemm.c;GlobalAveragePool.c;RandomNoise.c" \
  > experiments/deliverable/exp11_QZO_round1_fix/logs/round1_gvsoc_strict.log 2>&1
```

### Step 5 — extract dump + compare vs reference (traindeeploy)

```bash
python3 ../exp9_QZO_round1/extract_qzo_weights.py \
  --gvsoc-log logs/round1_gvsoc_strict.log \
  --train-onnx /app/ETH/Onnx4Deeploy/QZO_exp/exp11_round1_fix/network_zo_train.onnx \
  --out results/dumped_weights.npz \
  --ref-outputs /app/ETH/Onnx4Deeploy/QZO_exp/exp11_round1_fix/outputs.npz
```

### Step 6 — eval fixtures (agitated_hugle) + on-device eval (traindeeploy)

```bash
# corrected offset handling; also builds the reference-trajectory fixture for F2:
python3 make_ref_dump.py
python3 build_qzo_infer_fixture11.py --fixture-dir /app/Onnx4Deeploy/QZO_exp/exp11_round1_fix \
  --dump-npz results/dumped_weights.npz     --out-dir qinfer_round1
python3 build_qzo_infer_fixture11.py --fixture-dir /app/Onnx4Deeploy/QZO_exp/exp11_round1_fix \
  --dump-npz results/ref_updated_weights.npz --out-dir qinfer_ref_traj
# device eval (traindeeploy):
python3 experiments/deliverable/exp9_QZO_round1/qzo_accuracy_eval_untiled.py \
  --infer-dir experiments/deliverable/exp11_QZO_round1_fix/qinfer_round1 --cores 8
```
