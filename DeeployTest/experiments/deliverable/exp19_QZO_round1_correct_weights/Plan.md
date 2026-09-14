# exp19 — QZO (quantized zeroth-order) round-1 on-device fine-tuning, on the CORRECT pretrained weights

**Date opened:** 2026-09-15 · **Branch:** `feat/GAP9_w_NE16` (TrainDeeploy + Onnx4Deeploy)
**Supersedes:** [`exp12_QZO_clean_round_1`](../exp12_QZO_clean_round_1/Findings.md) — same recipe, superseded base checkpoint
**Sibling:** [`exp17`](../exp17_BP_round1_correct_weights/Findings.md) (BP) · [`exp18`](../exp18_ZO_round1_correct_weights/Plan.md) (ZO)

---

## 1. Why — and why this one needs more than a path swap

Like exp17/exp18, the base checkpoint moves from `SilentWear/artifacts/` (Mac, does not match the
paper) to `artifacts_reference/` (lab machine, matches). But QZO has an extra dependency the other two
do not:

> **The activation/weight scales are calibrated *from the model*.** The int8 datapath's per-site
> thresholds come from running the *pretrained network* over the fold-3 pretraining windows and taking
> a pooled 99.99th percentile of each activation site. Different weights ⇒ different intermediate
> activations ⇒ **different thresholds ⇒ different quantization scales**. Reusing exp12's
> `pooled_9999_fold3_fresh.json` would quantize the new network against the old network's dynamic
> range. Calibration is therefore **regenerated from scratch** here.

Note which sites can and cannot move: the *input* threshold (`blocks.0.conv.input_quant`) is computed
from the raw EMG windows and is weight-independent, so it should reproduce exp12's 2854.0 exactly —
a useful self-check that the calibration is deterministic and the data path unchanged. Every
*downstream* site is weight-dependent and is expected to move.

## 2. What is held fixed (recipe, from exp12)

| knob | value |
|---|---|
| method | QZO — MeZO on the int8 datapath; `--noise-type rqs_rademacher` |
| quantization | Conv int8 (per-channel RequantShift), **fc float** (device-faithful) |
| calibration | pooled @ 99.99 over the 1800 fold-3 pretraining windows (sessions 1+2) — **REGENERATED** |
| lr | **1e-5** (QZO's own lr, not ZO's 3e-6) |
| ε / q / seed | 0.01 / 1 / 42 |
| epochs / steps | 200 epochs → **2700 update steps**, n_accum 4 |
| reference losses | **FULL round — all 21 600 forwards** (see §2.1; exp12 exported only 104) |
| FT data | 54 windows (30 %, 6/class, seed 42, `--stratified`) |
| build | **ffast-math default — NO `DEEPLOY_STRICT_FP32`** (exp12 showed it is learning-neutral) |
| memory | `--l1 128000 --l2 2000000`, 8 cores |
| device flags | `BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON` |
| cell | S01 / session 3 / vocalized / fold 3 · train batch 1 → eval batch 2 |

## 2.1 Deliberate deviation from exp12 — a FULL reference, not a 104-forward prefix

exp12 exported its reference with the default `--n-epochs 1`, producing `loss_plus`/`loss_minus` of
length 52 — a 13-step, 104-forward prefix. The device then ran 2700 steps, so **only the first 104 of
21 600 forwards were ever compared against the reference**, and exp12's own Findings flagged the rest as
unmeasured ("Expect a large count ... quantifying the accepted trade").

That is not good enough for a bit-exactness claim. exp19 exports with `--n-epochs 200`
(`n_batches=10800`), so the device checks **every one of the 21 600 forwards** and the reported error
count is a property of the whole round. This is the same basis on which exp5/exp18 report
`Errors: 0 out of 21600`, so the three methods become directly comparable.

Expect a **large** nonzero count here, unlike ZO: this build is ffast-math (no `DEEPLOY_STRICT_FP32`),
which reorders/fuses floating-point ops, so the device trajectory diverges from the host reference
early. exp12 established that this divergence is **learning-neutral** — the endpoint accuracy matched
the strict-fp32 build. The number is therefore a *characterisation*, not a pass/fail: the acceptance
criterion for QZO is accuracy (§4), not bit-exactness. Reporting it honestly is the point.

## 3. Phases

| # | phase | container | cost | gate |
|---|---|---|---|---|
| 1 | **Fresh calibration** (pooled @ 99.99) + device-faithful PyTorch fc-float reference | `agitated_hugle` | ~1 h | `blocks.0.conv.input_quant` must reproduce 2854.0 (weight-independent) |
| 1b | Diff the new thresholds against exp12's | host | s | downstream sites *should* differ; input site should not |
| 2 | Export `-mode q-zo-train` with the fresh thresholds, **`--n-epochs 200`** | `agitated_hugle` | ~1 h | `n_batches=10800`; leading Quant scale == threshold/128; 54/54 FT windows match the sim draw |
| 3 | Pack + device QZO round, 2700 steps, ffast-math | `traindeeploy` | **~12 h** | `Errors: N out of 21600` over the FULL round; `[WDUMP s=2699]` present |
| 4 | Extract dumped weights → rebuild quantized inference fixture | both | min | offset-corrected injection |
| 5 | Device untiled eval on batch 2 (180 windows) | `traindeeploy` | ~35 min | device == host-executor accuracy |

## 4. Acceptance criteria

1. Batch-2 balanced accuracy reported against: device zero-shot (fresh calibration), the fresh PyTorch
   fc-float reference, and the exp12 values (85.00 / 89.44 / 88.33 on the old weights).
2. **Device == host-executor** balanced accuracy on the same trained weights ⇒ the on-device inference
   forward is faithful to its graph.
3. Bit-exactness reported honestly: under ffast-math, inference logits are **not** bit-exact vs the host
   reference (exp12 saw `bit-exact fails: 110/180`) and that is the **accepted** trade — it does not
   change predictions. This is a real difference from exp17/exp18 and must not be presented as a pass.

## 5. Runtime

Same order as exp18 (2700 steps). exp18 measured 3.71 updates/min on this host → **≈12 h**. Run **concurrently with exp18** via a private build dir:
`PYTEST_XDIST_WORKER=qzo` → `TEST_SIRACUSA/build_qzo`.

## 6. Changes required outside the experiment dir

`Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability/stability_lib.py` and
`QZO_exp/exp_calibration/run_study.py` hard-coded the `artifacts/` checkpoint. Both now read
`QZO_ARTIFACTS_DIR` (default `artifacts_reference`), so exp12's superseded run stays reproducible with
`QZO_ARTIFACTS_DIR=artifacts`.
