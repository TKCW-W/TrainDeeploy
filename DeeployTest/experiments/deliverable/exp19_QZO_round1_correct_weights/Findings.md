# exp19 — QZO (quantized zeroth-order) round-1 on-device fine-tuning, on the CORRECT pretrained weights

**Date:** 2026-09-15 · **Branch:** `feat/GAP9_w_NE16` (TrainDeeploy + Onnx4Deeploy)
**Cell:** S01 / session 3 / vocalized / fold 3 · train batch 1 → eval batch 2
**Supersedes:** [`exp12_QZO_clean_round_1`](../exp12_QZO_clean_round_1/Findings.md)
**Siblings:** [`exp17`](../exp17_BP_round1_correct_weights/Findings.md) (BP) · [`exp18`](../exp18_ZO_round1_correct_weights/Plan.md) (ZO)
**Plan:** [`Plan.md`](./Plan.md) · **Reproduction:** §5 · **Changes:** §6

---

## 1. Result — PASS

**On-device batch-2 balanced accuracy: 85.00 %** (180/180 windows evaluated, 0 failures).

| | batch-2 balanced acc |
|---|---|
| PyTorch fc-float **zero-shot** (fresh calibration, no FT) | 75.00 % |
| **device-trained, on-device untiled eval (ffast-math)** | **85.00 %** |
| device-trained, **host-executor** on the same weights | 85.56 % |
| PyTorch fc-float reference, device-faithful sim | 87.78 % |

- **+10.00 pp over zero-shot.**
- **Device vs host-executor = 0.56 pp = exactly one eval window.** This is the meaningful faithfulness
  check: the on-device inference forward agrees with its own graph's host execution up to a single
  window, despite inference logits **not** being bit-exact (§3.2).
- 2.78 pp below the PyTorch reference (5 eval windows) — see §4 for why that is expected here.

### 1.1 Against the reference and the sibling experiments

Same cell, and — verified byte-for-byte — the **same 54 fine-tuning windows and the same 180
evaluation windows** as exp17/exp18:

| | batch-2 acc | train cycles | notes |
|---|---|---|---|
| reference, no FT (base model) | 76.67 % | — | `artifacts_reference` `ft_summary.csv` |
| **exp19 QZO (this)** | **85.00 %** | **89.5 G** | int8 conv datapath, fc float |
| exp17 BP (first-order) | 87.78 % | ~75.8 G | |
| exp18 ZO (fp32 forward-only) | *pending* | ~385 G (exp5) | |
| reference, paper's own FT recipe | 88.89 % | — | Adam, 50 ep, 70 % data, BN updating |

**QZO is ~4.3× cheaper than fp32 ZO in cycles** (89.5 G vs exp5's 385.3 G) for the same 2700 steps,
which is the int8 datapath — the PULP cluster's 8-bit kernels do 4 MACs per SIMD instruction where
fp32 does one. Measured device throughput during the round matched this: 13.9 updates/min for QZO
against 3.6 for ZO (3.9×). Quantization buys **throughput**, not only memory.

---

## 2. What had to change beyond the checkpoint path — calibration

QZO is the one method of the three where swapping the base weights is not a path edit. The int8
activation thresholds are obtained by running **the pretrained network** over the 1800 fold-3
pretraining windows and taking a pooled 99.99th percentile per site. Different weights ⇒ different
intermediate activations ⇒ different thresholds ⇒ **different quantization scales**.

Calibration was therefore regenerated. The diff against exp12 confirms the mechanism exactly:

| site | exp12 (old weights) | exp19 (new) | Δ |
|---|---|---|---|
| `blocks.0.conv.input_quant` | 2854.0000 | 2854.0000 | **0.0 %** |
| `blocks.0.conv.output_quant` | 1698.0000 | 1833.0000 | +8.0 % |
| `blocks.1.conv.input_quant` | 23.1250 | 30.3906 | +31.4 % |
| `blocks.1.conv.output_quant` | 27.1406 | 42.5312 | +56.7 % |
| `blocks.2.conv.input_quant` | 11.8359 | 16.4369 | +38.9 % |
| `blocks.2.conv.output_quant` | 11.3644 | 18.9531 | **+66.8 %** |
| `blocks.3.conv.input_quant` | 14.1906 | 13.6938 | −3.5 % |
| `blocks.3.conv.output_quant` | 15.6250 | 21.3156 | +36.4 % |
| `blocks.4.conv.input_quant` | 7.3500 | 10.6641 | +45.1 % |
| `blocks.4.conv.output_quant` | 5.3820 | 6.9180 | +28.5 % |
| `fc_iq.act_quant` / `fc.input_quant` | 8.5431 | 12.6213 | +47.7 % |

**Exactly 1 of 12 sites is unchanged, and it is precisely the weight-independent one** —
`blocks.0.conv.input_quant` is computed from the raw EMG windows, which did not change. Every
downstream site moved. Reusing exp12's calibration would have quantized the new network against the
old network's dynamic range, with scale errors up to 67 %.

That the input threshold reproduces 2854.0000 **exactly** also re-confirms the calibration is
deterministic and the data path unchanged.

---

## 3. Bit-exactness over the FULL round — the measurement exp12 could not make

### 3.1 Training round

**`Errors: 9949 out of 21600`** (46.1 %).

> **This is the headline methodological improvement.** exp12 exported its reference with the default
> `--n-epochs 1`, giving `loss_plus`/`loss_minus` of length 52 — a 13-step, **104-forward** prefix.
> The device ran 2700 steps, so only the first 0.5 % of the round was ever compared, and exp12's own
> Findings flagged the rest as unmeasured. exp19 exports with `--n-epochs 200` (`n_batches=10800`),
> so all **21 600** forwards are checked and the error count is a property of the whole round — the
> same basis on which exp5/exp18 report `Errors: 0 out of 21600`.

**The divergence does not start at step 1 — it starts just before the halfway point.** exp12
predicted "ffast-math diverges from step ~1"; the full trace shows otherwise:

| stream | breaches | first breach | pre-onset max \|diff\| | median | p99 | max |
|---|---|---|---|---|---|---|
| `loss+` | 4987 / 10800 (46.2 %) | mini-batch **5127** (47.5 %) | **6.4e-05** | 1.87e-04 | 4.06e-01 | 1.06e+00 |
| `loss-` | 4961 / 10800 (45.9 %) | mini-batch **5111** (47.3 %) | **1.0e-06** | 2.04e-04 | 3.91e-01 | 9.26e-01 |

Breach rate by quarter of the round: **0 % → 6 % → 88 % → 91 %** (both streams, independently).

Mini-batch 5127 is **update step 1281 of 2700**. So the ffast-math device trajectory tracks the host
reference to within 6.4e-05 for the first ~1280 update steps, then separates and stays separated.
This is the same *cumulative-divergence* signature seen in BP's MaxPool tie-flips (exp17 §3.1): the
breach count is not per-step noise but "steps after the trajectories parted".

*Counting note:* an offline recount of the logged diffs gives 9948 with a strict `>` test. Exactly
one forward prints as `0.001000` at the log's 6-decimal precision while the device compared it as
`> TOL`; 9948 + 1 = **9949**, matching the device counter, which is authoritative.

### 3.2 Inference

`bit-exact fails: 116 / 180`. Per-window the error is `bitexact_err` 0 or 9 (all nine logits, or
none) — i.e. a window either matches the host reference exactly or all its logits drift.

**This is the accepted trade, not a failure.** The build is the ffast-math default (no
`DEEPLOY_STRICT_FP32`), which fuses and reorders floating-point operations. What matters is whether
it changes *predictions*, and it does not meaningfully: device 85.00 % vs host-executor 85.56 % on
identical weights — one window. exp12 established the same conclusion on the strict-fp32 build
(88.33 vs 89.44, also ~1 window), so the fast build remains learning-neutral.

This is a genuine difference from exp17 (BP) and exp18 (ZO), which are bit-exact on inference, and it
should not be presented as a pass.

---

## 4. Why 85.00 % sits below the 87.78 % PyTorch reference

Three contributions, none of which is a defect:

1. **Trajectory divergence** (§3.1). From update step ~1281 the device follows a different — but
   equally valid — ZO path than the host sim. exp12 showed the endpoint is learning-neutral, and the
   2.78 pp here (5 windows) is the same order as exp12's 1.11 pp (2 windows).
2. **Quantization of the forward.** The reference is `fc`-float but int8 in the conv stack; the
   device additionally carries the requant/rounding behaviour of the real kernels.
3. **Seed/trajectory noise.** Previously measured run-to-run spread on this pipeline is mean \|Δ\|
   4.59 pp, max 25.00 pp. A 2.78 pp gap is **inside** that, so it should not be quoted as a
   systematic penalty from a single run.

**A caveat about the zero-shot row.** exp12's results table labels its 85.00 % as
"*device* zero-shot (fc-float, fresh pooled-99.99)", but its own §"What was regenerated" shows that
number came from `results/pytorch_ref.json` — i.e. the **PyTorch** sim, not a device measurement.
exp19's 75.00 % is likewise the PyTorch fc-float zero-shot (`results/pytorch_ref.json`). **No device
zero-shot was measured for QZO in either experiment**, and the row is labelled accordingly in §1.
Measuring it would cost one extra untiled eval (~35 min) and is listed in §7.

---

## 5. Reproduction

```bash
cd TrainDeeploy/DeeployTest/experiments/deliverable/exp19_QZO_round1_correct_weights
bash scripts/run_round1.sh all      # or: 1 | 2 | 3 | 4 | 5
```

| phase | what | container | cost |
|---|---|---|---|
| 1 | fresh pooled@99.99 calibration + PyTorch fc-float reference | `agitated_hugle` | ~25 min |
| 2 | export `-mode q-zo-train` with the fresh thresholds, **`--n-epochs 200`** | `agitated_hugle` | ~3.2 h |
| 3 | pack + device QZO round, 2700 steps, ffast-math | `traindeeploy` | ~3.3 h |
| 4 | extract `[WDUMP s=2699]` → rebuild quantized inference fixture | both | ~5 min |
| 5 | untiled device inference over 180 batch-2 windows | `traindeeploy` | ~35 min |

**Fixed inputs**

| what | path |
|---|---|
| base checkpoint | `SilentWear/SilentWear/artifacts_reference/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt` |
| data | `SilentWear/SilentWear_data/data_raw_and_filt` |
| fresh thresholds | `fixture/pooled_9999_fold3_ref.json` |
| export dir | `Onnx4Deeploy/QZO_exp/exp19_ref_full` |
| device fixture | `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_qzo19_{train,update}` |

**Recipe** — `--noise-type rqs_rademacher`, lr **1e-5**, ε 0.01, q 1, seed 42, n_accum 4,
200 epochs → **2700 update steps**, 54 stratified FT windows (6/class, seed 42), Conv int8 /
fc float, `--l1 128000 --l2 2000000 --cores 8`, `-D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON`,
**ffast-math default (no `DEEPLOY_STRICT_FP32`)**.

**Expected output**

| phase | expect |
|---|---|
| 1 | `block0-in threshold=2854.0000 -> scale=22.296875`; `REFERENCE fc-float b2: 75.00 -> 87.78%` |
| 2 | `n_batches=10800`; `n_steps=2700 lr=1e-05 eps=0.01 seed=42`; leading Quant attr `scale = 22.296875` |
| 3 | `update 2700/2700`; `Errors: 9949 out of 21600`; 22 `[WDUMP s=2699]` tensors; `BENCH train_cycles_hi=20 train_cycles_lo=3612894973` |
| 4 | `[extract] last dump step s=2699: 22 tensors`; `[fixture] 180 windows; host-reference balanced accuracy = 85.56%` |
| 5 | `DONE: balanced=85.00%  bit-exact fails: 116/180` |

> ⚠️ **Do not run phase 5 beside another device simulation** unless you are on commit `93458a0` or
> later. See §6.2 — the eval harness used to wipe `TEST_SIRACUSA` and kill every GVSoC on the machine.

---

## 6. Changes made (with file paths)

### 6.1 New — the experiment
| path | what |
|---|---|
| `…/exp19_QZO_round1_correct_weights/Plan.md` | plan, recipe, the full-reference rationale (§2.1 there) |
| `…/Findings.md` | this file |
| `…/scripts/run_round1.sh` | all five phases, `bash run_round1.sh [1-5\|all]` |
| `…/run_ref_and_calib.py` | fresh calibration + PyTorch reference (adapted from exp12) |
| `…/fixture/pooled_9999_fold3_ref.json` | the regenerated pooled@99.99 thresholds |
| `…/results/pytorch_ref.json` | `{zero_shot: 75.0, after: 87.78}` |
| `…/results/dumped_weights.npz` | the 22 device-trained tensors (int8 conv, int32 rqs-add, fp32 BN/fc) |
| `…/results/eval_b2_results.json` | `{balanced_accuracy: 85.0, n_bitexact_fail: 116, n_done: 180}` |
| `…/qinfer/` | the rebuilt quantized inference fixture |
| `…/logs/*.log[.gz]` | full logs incl. the 21 600-forward loss trace and `[WDUMP]` blocks |

### 6.2 Fixed — `exp9_QZO_round1/qzo_accuracy_eval_untiled.py` (commit `93458a0`)
The harness opened with two **machine-wide** cleanup steps:

```python
if args.start == 0:
    subprocess.run(["rm", "-rf", "TEST_SIRACUSA"], check=False)
subprocess.run("pgrep -f '[g]vsoc_launcher' | xargs -r kill -9", shell=True, check=False)
```

The first deletes **every** worker's build dir and generation dir; the second kills **every**
`gvsoc_launcher` on the machine. Running this eval beside exp18's ZO device round **destroyed that
round at update 1566/2700** — ~7 h of simulation, no `BENCH`, no `[WDUMP]`, unrecoverable. From the
build system it surfaced only as a bare `gmake … Error 2`, which is what a killed simulator looks
like.

`PYTEST_XDIST_WORKER` did **not** protect it: that variable only selects which build dir the *runner*
writes to (`testUtils/deeployRunner.py:219`). It is no defence against a script that removes the whole
tree or kills by process name. Isolation has to hold on both sides.

Both steps are now scoped to the eval's own worker (originals commented in place):
the wipe targets `TEST_SIRACUSA/build_<worker>` plus this eval's own generation dir — all the
stale-CMake-cache problem ever required, since `TRAINING` is cached *per build dir* — and the kill
matches only GVSoC processes belonging to this eval's temp test name.

The BP/ZO harness (`experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py`)
neither wipes nor kills and was never a hazard.

### 6.3 Changed — `Onnx4Deeploy` (commit `90e308d`)
`QZO_exp/exp3_lr1e-5_stability/stability_lib.py` and `QZO_exp/exp_calibration/run_study.py` hard-coded
the `artifacts/` checkpoint. Both now read `QZO_ARTIFACTS_DIR`, defaulting to `artifacts_reference`;
set `QZO_ARTIFACTS_DIR=artifacts` to reproduce exp12.

---

## 7. Open items

1. **Device zero-shot for QZO** — never measured, in exp12 or here (§4). One untiled eval (~35 min)
   on the un-fine-tuned quantized fixture would complete the table honestly.
2. **Strict-fp32 comparison on the new weights.** exp11/exp12 showed ffast-math is learning-neutral on
   the *old* weights; a `DEEPLOY_STRICT_FP32` round here would confirm it holds, and would also test
   whether the step-1281 divergence onset is reproducible.
3. **Seeds.** Every number here is one seed. The gaps discussed in §4 are inside the measured noise
   floor (mean \|Δ\| 4.59 pp), so 2–3 seeds are needed before any of them is quoted as an effect.
4. Rounds 2–4 and subjects S02–S04, to match the reference table's shape.
