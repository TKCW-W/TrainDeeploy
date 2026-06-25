# SpeechNet (SilentWear) MaxPool — Precision & Fine-Tuning Accuracy Findings

Model: SpeechNet, S01 / vocalized / session 3 (the held-out session of
`leave_one_session_out_fold_3`). 5× Conv–BN–ReLU–{MaxPool×3, Identity×2},
GlobalAvgPool → Linear(32→9). On-device training = Deeploy SGD, effective
batch 1, gradient-accumulator **sums** (so `--n-accum K` ⇒ effective LR ×K).

Two investigations are recorded here:
- **A.** Does a larger effective batch fix the device-vs-ORT precision drift? → **No.**
- **B.** Do the fine-tuned weights actually improve accuracy? → **Not with our config;
  yes (+8.33 pp held-out) with the faithful SilentWear recipe.** Precision was never
  the blocker — the fine-tuning configuration was.

---

## A. Effective-batch (`--n-accum`) precision experiment

The 90-step MaxPool fine-tuning run breaches the device-vs-ORT loss tolerance
`TOL = 1e-3` at ~step 36 (~2 epochs). Hypothesis: the breach is gradient noise,
so a larger effective batch (LR-compensated, since accumulation sums) should
delay/prevent it. Three runs, all 8 cores, 18 stratified batch-1 windows:

| run  | eff-batch | LR       | updates | epochs | breach rate | onset (epochs) | max diff |
|------|-----------|----------|---------|--------|-------------|----------------|----------|
| acc1 | 1         | 1e-3     | 90      | 5      | 52/90       | 2.00 (step 36) | 0.064    |
| acc2 | 2         | 5e-4     | 45      | 5      | 27/45       | 1.89 (step 17) | 0.043    |
| acc4 | 4         | 2.5e-4   | 90      | 20     | 58/90       | 7.11 (step 32) | 0.020    |

**Conclusion — larger effective batch does NOT fix the drift.** On the controlled
acc1-vs-acc2 comparison (both 5 epochs, batch the only variable) the onset
(~2 epochs) and breach frequency (~58–60%) are unchanged; only the max-diff
*magnitude* shrinks with batch. The drift is **MaxPool argmax discreteness**:
fp32 reduction-order differences (~1e-6) flip argmax ties once the weights have
moved ~2 epochs. Gradient averaging dampens the size of each flip's perturbation
but cannot change how often or when ties flip. The epoch-quantized appearance is
a sampling artifact (each window's loss-diff is measured once per epoch in fixed
cyclic order).

Logs: `speechnet_maxpool_90step_acc1_val.log`, `speechnet_maxpool_5ep_acc2_val.log`,
`speechnet_maxpool_90step_acc4_val.log`. Fixtures: `speechnet_train_maxpool_{5ep_acc2,90_acc4}`.

---

## B. Fine-tuning accuracy investigation (ORT simulation)

**Flow:** baseline = zero-shot accuracy on whole batch 1; fine-tune on 10% of
batch 1; evaluate the fine-tuned weights on whole batch 2 (held out). Goal: see
whether — before paying for on-device weight extraction + infer-graph
regeneration — the *simulation* already shows a positive accuracy message.

All evaluations use the validated PyTorch forward (reproduces the infer-fixture
ORT logits bit-for-bit). Eval sets are the balanced 180-window infer fixtures
(20/class). Zero-shot reproduces the known baselines exactly:
**batch1 = 70.56 %, batch2 = 78.33 %** (balanced = overall, 20/class).

### B.1 The 90-step reference weights give NO accuracy gain (same eval set)

Using the ORT reference final weights from `speechnet_train_maxpool_90/outputs.npz`
(acc1: 90 SGD steps, eff-batch 1, lr 1e-3, 18 batch-1 windows):

| eval set | zero-shot | FT acc1 (5 ep) | FT acc4 (20 ep) |
|----------|-----------|----------------|-----------------|
| whole batch 1 | 70.56 % | 70.56 % (+0.00) | 67.78 % (−2.78) |
| whole batch 2 (held-out) | 78.33 % | 77.78 % (−0.56) | 76.11 % (−2.22) |

- Flat-to-negative on **both** eval sets, even in ideal ORT (no precision error).
  ⇒ **precision drift is not the bottleneck** — the on-device weights can at best
  match this, not beat it.
- `max|Δ fc_weight|` = 0.0118 for **both** acc1 and acc4: acc4 is *not* "more
  training" — it is the same 90 updates with the same effective step (eff-batch 4,
  lr/4), just smoother. Per-step training loss never trends down (~2.1–2.5
  throughout), i.e. the FT objective itself does not converge.
- **Confound caught:** the tempting "70.56 % → 77.78 % = +7.22 pp" reads positive
  only because it compares *different* eval sets (batch1 vs batch2). Batch 2 is
  intrinsically easier — its zero-shot is already 78.33 %. Isolated, same-eval-set
  fine-tuning is −0.56 pp.

### B.2 Why our config could not learn — vs the SilentWear recipe

SilentWear's actual per-round fine-tuning (`offline_experiments/IV_inter_session_with_ft.py`,
`config/paper_ft_config.yaml`):

| knob | SilentWear FT | our experiment |
|------|---------------|----------------|
| optimizer | **Adam** (β 0.9/0.999), wd 1e-4 | plain SGD, no wd |
| LR / scheduler | 1e-3 / ReduceLROnPlateau (×0.1, patience 2) | 1e-3 / none |
| epochs | 50, early-stop patience 10 | fixed 5 |
| batch size | 32 | 1 (eff-batch) |
| rest downsample | rest 159 → 20/class (180 balanced) | **same** |
| data used | full balanced batch, 70/30 stratified → **126 train** | **10% stratified subsample → 18** (2/class) |
| validation | 54-window val (early-stop/scheduler) | none |

Both pipelines apply the *same* rest-only downsampling; the difference is that we
take a **10 % stratified subsample** of the balanced batch (2/class) and train with
plain SGD/eff-batch-1/no-early-stop. The dominant gap is **data fraction (10 % vs
100 %): 18 vs 126 training windows.**

### B.3 Faithful-recipe ceiling check → fine-tuning DOES help

Replicated SilentWear's recipe in PyTorch on the full balanced batch-1 (126 train /
54 val, stratified 70/30), Adam lr 1e-3 wd 1e-4, ReduceLROnPlateau, 50 epochs
(early-stopped at epoch 28, **116 updates**, val loss 0.317), then evaluated on
whole batch 2:

| | batch 1 (train side) | **batch 2 (held-out)** |
|---|----------------------|------------------------|
| zero-shot | 70.56 % | 78.33 % |
| **faithful FT** | 96.67 % | **86.67 %** |
| improvement | +26.11 pp | **+8.33 pp** |

**Conclusion.** Proper fine-tuning recovers a real **+8.33 pp on held-out batch 2**
(same eval set, apples-to-apples). The flat result in B.1 was caused by the
fine-tuning *configuration* (10 % data + plain SGD eff-batch-1 + no early-stop),
**not** the MaxPool precision drift.

### B.4 Recommended step choice for the on-device path

Anchor on data exposure, not raw update count (eff-batch-1 ≠ batch-32):
- **Use the full balanced batch (126–180 windows), not 18** — the single biggest deficiency.
- Faithful target ≈ **116 updates** (bs 32, ~28 epochs) for the +8.33 pp ceiling.
- Deeploy SGD eff-batch-1 analog: match epochs → `--data-size 180 --n-accum 1
  --n-steps ~3600` (≈20 epochs), or `--n-accum 32 --n-steps ~200 --lr 3e-5`
  (lr ÷32 because the accumulator sums) to mimic batch-32.

**Open tension:** meaningful FT needs ~25–30 epochs, but on-device precision drift
breaches `TOL` at ~2 epochs, and Deeploy runs SGD (not Adam). So an eff-batch-1
SGD run long enough to capture the gain will diverge from ORT substantially — and
may not reach the Adam ceiling. The next step is to measure the SGD eff-batch-1
attainable accuracy (full-batch, 10/20/30-epoch sweep) in simulation before
committing to the on-device path.

---

## Reproduce

```bash
# Batch-2 infer fixture (zero-shot logits + labels + identically-preprocessed windows)
docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py \
  -model SpeechNet -mode infer -o /app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights .../leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --batch 2 --condition vocalized"

# Eval = PyTorch SpeechNetDeploy forward (reproduces infer-fixture ORT logits).
# Zero-shot uses the pretrained checkpoint; FT injects outputs.npz weights
#   (key map: blocks_i_j_x -> blocks.i.j.x, fc_weight -> fc.weight).
# Faithful ceiling = upstream models.cnn_architectures.SpeechNet (batched, dropout 0.5),
#   Adam lr 1e-3 wd 1e-4, ReduceLROnPlateau, 50 ep, bs 32, balanced batch-1, 70/30 split.
```
