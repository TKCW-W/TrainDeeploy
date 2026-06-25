# Plan: On-Device Fine-Tuning that Meaningfully Improves Batch-2 Accuracy

## North-star goal
Find and validate a SpeechNet fine-tuning setup that, **run on-device (tiled GVSoC,
SGD, eff-batch-1)**, produces updated weights whose accuracy on **whole batch 2**
beats **batch 2's own zero-shot accuracy (78.33%)** by a meaningful margin —
*despite* the MaxPool argmax precision drift. The improvement must be measured with
the **actual on-device weights**, not the ORT reference.

Definition of done: `acc(device-fine-tuned weights, whole batch 2)` > **78.33%**
with margin (target ≥ **+3 pp**), reported as an honest held-out test (batch 2 is
never used to choose the setup).

---

## Fixed context (verified this session)

- **Model:** SpeechNet, S01 / vocalized / session 3 (held-out session of
  `leave_one_session_out_fold_3`). 5× Conv–BN–ReLU–{MaxPool×3, Identity×2} →
  GlobalAvgPool → Linear(32→9). Raw EMG (no normalization). ~16K params.
- **Zero-shot baselines (balanced = overall, 180 windows, 20/class):**
  batch 1 = **70.56%**, batch 2 = **78.33%**. Reproduced bit-exact by the PyTorch
  `SpeechNetDeploy` forward (validated against the infer-fixture ORT logits).
- **Reference ceiling (Adam, NOT what we ship):** faithful SilentWear recipe
  (Adam, wd 1e-4, ReduceLROnPlateau, 50 ep, bs 32, full 126-window 70/30 split)
  → **+8.33 pp** on batch 2 (86.67%). Reference only; on-device is SGD.
- **On-device training mechanics:** Deeploy SGD, **effective batch 1**,
  gradient accumulator **sums** ⇒ `--n-accum K` ≡ mean-gradient with effective
  LR ×K ⇒ compensate by `lr → lr/K`. LR is baked into the optimizer ONNX.
- **Precision drift:** device-vs-ORT loss diff breaches `TOL=1e-3` at ~2 epochs;
  root cause = MaxPool argmax tie-flips (fp32 ~1e-6 reduction-order). On the
  90-step run it looked **bounded/oscillatory** (max |diff| ≈ 0.064, non-monotonic),
  not exploding — to be confirmed at the chosen step count.

## Environment / tools

- **Onnx4Deeploy container** `agitated_hugle` — `/app/Onnx4Deeploy`, sees
  `/app/SilentWear`, `/app/TrainDeeploy`. Has torch 2.7 + ORT + h5py.
  Used for: fixture generation, ORT/PyTorch eval.
- **TrainDeeploy container** `traindeeploy` — `/app/ETH/TrainDeeploy`. Used for:
  the **tiled** training runner + GVSoC. **`rm -rf TEST_SIRACUSA` before every run.**
- **Runner:** `deeployTrainingRunner_tiled_siracusa.py` (ALWAYS tiled). Fixture
  registered in `test_siracusa_tiled_config.py` (`L2_SINGLEBUFFER_TRAINING_MODELS`,
  e.g. 128000). Per-step memory is one-window, so steps/data-size don't blow L2.
- **Pretrained:** `/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`
- **Data:** `/app/SilentWear/SilentWear_data/data_raw_and_filt` (S01/vocalized/sess_3_batch_{1,2}.h5, 319 windows each → 180 after rest-downsample to 20/class).
- **Batch-2 eval fixture (zero-shot logits + labels + windows):**
  `/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2` (input (180,1,14,700), label (180,)).
- **Batch-1 eval fixture:** `Tests/Models/speechnet_infer_original` (same shape).

## Locked decisions

1. **Optimizer:** SGD-matched to Deeploy (NOT Adam). Adam is reference-only.
2. **Selection metric:** a **batch-1 held-out val split** picks the step count;
   **batch 2 is the untouched test.** Never select on batch 2.
3. **Meaningful bar:** ≥ **+3 pp** on batch 2 over 78.33%, chosen with sim margin
   so on-device drift won't flip the sign.
4. **Conservatism:** prefer the **minimum** steps that clears the bar (less drift
   exposure, less GVSoC wall-clock).
5. **Search priority:** (1) steps/epochs, then (2) data % (10% → 20%).
6. **Data split:** partition batch-1 (180) into a fixed **val** subset and a
   **train pool**; sample FT training windows only from the train pool (no leak).
   10% data = 18 windows (2/class); 20% = 36 (4/class).

---

## Phases (the loop works these in order; update "Status" at the bottom each iteration)

### Phase 1 — Simulation search (find a candidate setup)
Goal: pick `(data%, n-steps/epochs, lr)` that clears the +3 pp bar on a clean test.
- Mirror Deeploy SGD exactly in PyTorch (full-model SGD, eff-batch-1 or n-accum
  sum with `lr/K`, BN in train mode, CrossEntropy). Controlled stratified
  train/val partition of batch-1.
- Grid (small first, ~6–8 configs): data {10%, 20%} × epochs {a few points} ×
  lr {1e-3, 5e-4}. For each: train, snapshot accuracy vs epoch on **batch-1-val**
  and on **batch-2**.
- **Select** the config + stopping epoch by best **batch-1-val** (robust plateau,
  not argmax); read off its **batch-2** number as the honest result.
- If nothing clears +3 pp at 10%, escalate to 20% (priority rule). Expect 10%
  (18 windows) may overfit.
- **Cross-check:** regenerate the winner as a real Onnx4Deeploy train graph and
  confirm its ORT-reference final-weights batch-2 accuracy ≈ the PyTorch search
  (predictiveness gate). Generation CLI:
  ```
  Onnx4Deeploy.py -model SpeechNet -mode train -o <out> --dataset silentwear \
    --data-path .../data_raw_and_filt --pretrained-weights .../fold_3.pt \
    --subject S01 --session 3 --batch 1 --condition vocalized \
    --stratified --data-size <N> --n-steps <S> --n-accum <K> --lr <lr>
  ```
- **Exit:** a chosen config with sim batch-2 ≥ 78.33% + margin, val-selected.

### Phase 2 — Drift study (is the chosen step count safe?)
Goal: confirm the precision drift at the chosen step count is **bounded**, not
exploding, so it won't erase the sim gain on-device.
- Classify the **per-step loss-diff trajectory** shape (already logged by the
  runner): fit growth — bounded/oscillatory vs monotonic/exponential. Start from
  existing acc1 (90-step) data; extend to the chosen step count.
- Decision-relevant metric is **final weight divergence → accuracy**, not the TOL
  breach: even if per-step loss breaches TOL, the run is fine if the **final**
  device weights stay close enough to ORT that batch-2 accuracy is preserved.
- If the trajectory looks unbounded near the chosen step count, **reduce steps**
  (conservatism rule) and re-check Phase 1 margin.
- **Exit:** a step ceiling we trust; loss-diff trajectory characterized.

### Phase 3 — Instrument the weight dump (in `deeploytraintest.c`)
Goal: extract the actual on-device updated weights, bit-exact.
- Insert after `run_optimizer_step()` (**line 376**, inside the outer
  `update_step` loop). The updated weights are in
  `DeeployNetwork_inputs[TRAINING_NUM_DATA_INPUTS + wi]`, read via
  `IS_L2(buf) ? memcpy : ram_read` into a temp L2 buffer.
- Dump each weight as **raw 32-bit words** (`printf(" %08x", word)`) — FPU-free
  (FC has no FPU), bit-exact. Marker e.g. `[WDUMP s=%u wi=%u n=%u] <hex...>`.
  Guard with `if (update_step==N_TRAIN_STEPS-1)` for final-only (per-step
  available by removing the guard). Gate behind a `-DDUMP_WEIGHTS` macro.
- **Validation gate:** on a tiny 1-step config, reconstruct the dumped weights in
  Python (`struct.unpack('<f', ...)`) and confirm they match the fixture's ORT
  `outputs.npz` to fp tolerance — this also fixes the `wi → tensor-name` mapping
  (wi order = `testInitWeights[]` = `inputs.npz` order). Only then trust the dump.
- **Exit:** a parser that turns a runner log into `outputs_ondevice.npz`,
  validated round-trip.

### Phase 4 — On-device run (final setup)
- `rm -rf TEST_SIRACUSA`; run the **tiled** runner with the chosen config and
  `-DDUMP_WEIGHTS`. Save the full log (loss convergence + numerical diffs +
  `[WDUMP]` lines).
- Parse → `outputs_ondevice.npz` (the actual device-fine-tuned weights).
- **Exit:** device weights extracted; loss/diff log saved for study.

### Phase 5 — Accuracy evaluation (the verdict)
- Regenerate the infer graph with the device weights injected (reuse the validated
  injection eval: map `wi`/`blocks_i_j_x → blocks.i.j.x`, `fc_weight → fc.weight`).
- Compute accuracy on **whole batch 2** with the device weights.
- Report the 3-way: **device-FT vs ORT-FT vs zero-shot (78.33%)**.
- **Success** = device-FT batch-2 > 78.33% with margin (≥ +3 pp target).
- If device-FT < ORT-FT (drift cost) but still > zero-shot → success with a noted
  drift penalty. If drift erased the gain → loop back to Phase 1/2 (fewer steps /
  more data) and retry.

---

## Deliverables
- `speechnet_finetune_<config>_train.log` — on-device loss convergence + diffs.
- `outputs_ondevice.npz` — actual device-fine-tuned weights.
- Updated findings (append to `SPEECHNET_FINETUNE_PRECISION_FINDINGS.md`):
  chosen setup, drift classification, device-vs-ORT-vs-zero-shot batch-2 table.

## Status (loop updates this)
- [ ] Phase 1 — sim search → candidate config: _TBD_
- [ ] Phase 2 — drift classified (bounded/explode), step ceiling: _TBD_
- [ ] Phase 3 — weight dump instrumented + round-trip validated
- [ ] Phase 4 — on-device run done, weights extracted
- [ ] Phase 5 — batch-2 accuracy vs zero-shot: _TBD_
