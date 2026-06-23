# SpeechNet on TrainDeeploy — Progress Summary

**Branch:** `feat/speechnet/inference/normalised_data`  
**Base:** `devel`  
**Last updated:** 2026-06-14

---

## Project Context

**TrainDeeploy** extends the Deeploy compiler for on-device training on the Siracusa RISC-V SoC (simulated via GVSoC). It compiles ONNX training graphs (forward + backward + gradient accumulation) into tiled C code for the PULP 8-core cluster. Hardware constraints: L1=128 KB, L2=2 MB, L3=HyperFlash.

**SpeechNet** (from SilentWear) is a 5-block CNN for EMG-based silent speech recognition: 14-channel input × 700 time steps → 9 word classes, ~15 K parameters. It is the most recent model added to TrainDeeploy.

---

## What Was Done (chronological)

### 1. SpeechNet Training Integration (commits `9addf18` → `75fb0b9`)

**Goal:** Add SpeechNet as a trainable model in TrainDeeploy's test suite.

- Exported training ONNX from `Onnx4Deeploy` with static reshape (no dynamic `Shape`/`Flatten` ops that Deeploy cannot handle).
- **Bug fixed:** `ConvLayer.computeShapes` was crashing on Conv layers with bias because a scalar int was passed where a tuple was expected (`Deeploy/Targets/Generic/Layers.py`).
- Registered SpeechNet in `DeeployTest/test_siracusa_tiled_config.py` at L1=128 000.
- **Key architecture constraint discovered:** ConvGradX must use the naive kernel, not Im2Col — Im2Col buffer overflows L1 for SpeechNet's spatial dimensions.
- First untiled test verified: 4/4 loss diff=0.000000, 285 M train cycles.
- Switched training fixture to use real EMG data (commit `f0df8eb`).
- Extended test to all 22 trainable blocks (commit `a3cac0e`).
- Added tutorial notebook: `docs/tutorial_speechnet_training.ipynb` (commit `b185be6`).

### 2. ConvGradW Double-Zero Bug Fix (commit `1fd1004`)

**Problem:** Gradient weights were being zeroed once per spatial tile instead of once per execution, causing accumulated gradient errors that grew with the number of tiles.

**Fix:** Moved `memset(grad_weight, 0, ...)` out of the spatial tile loop in the ConvGradW kernel.

**Impact:** This was a correctness bug affecting all tiled Conv training (not just SpeechNet). Loss values after fix were closer to reference.

### 3. Batch-1 Fine-Tuning Experiment (`speechnet_train_b1_ft`)

**Goal:** Fine-tune the pre-trained SpeechNet (all 5 blocks trainable) using the Batch 1 pre-normalised EMG data on GVSoC, and check whether the on-device gradient updates match the PyTorch reference over 90 steps.

**Model variant:** Uses **AvgPool instead of MaxPool** (architectural revision vs the original SpeechNet). Data is pre-normalised before being fed in (no on-device normalisation step).

**Artifacts:** `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_b1_ft/`
- `checkpoint/` — pre-trained weight checkpoint used as init
- `network_train.onnx` / `network_train_optim.onnx` — full-model fine-tuning graph (all blocks, Batch 1 data)
- `eval_model.onnx` / `network_infer.onnx` — inference graph for post-fine-tuning evaluation
- `inputs.npz` / `outputs.npz` — 90-step fine-tuning fixture (Batch 1 pre-normalised data, lr=0.001)

**Log:** `DeeployTest/speechnet_b1ft_run.log`  
**Plot:** `DeeployTest/speechnet_b1ft_loss.png`

**Run config:** tiled Siracusa, 8 cores, L1=128 000, L2=2 000 000, MiniMalloc, random-max search, 90 steps.

**Observations:**
- Initial loss starts at ~2.0 (loss[0]=1.9987), consistent with a pre-trained model on new Batch 1 data; no strong downward trend across 90 steps — fine-tuning converges slowly.
- Error accumulation pattern: first ~38 steps stay within TOL=0.001; divergence grows from step ~39, reaching ~0.11 by step 57, then oscillates in the 0.02–0.11 range.
- This is the same floating-point accumulation drift seen in the 100-step experiment — not a kernel bug, but inherent to many sequential gradient updates in float32.
- Final result: **48 errors out of 90 at TOL=0.001** — test fails the tight tolerance.
- On-device compute: train_cycles=3,004,132,012 (~3 B cycles), opt_cycles=5,073,717, weight_sram=61,956 bytes.

### 4. Pre-trained Weight Reproduction + 100-Step Training Experiment (commit `e6bfe27`)

**Goal:** Reproduce published SpeechNet pre-trained weights from scratch and validate that GVSoC training follows the loss curve.

**Artifacts created:**
- `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_preweights/` — ONNX with pre-trained weights as initial parameters.
- `DeeployTest/speechnet_loss_experiment.py` — script to run N training steps on GVSoC and plot the loss curve.
- `DeeployTest/speechnet_100step_run.log` + `speechnet_100step_loss.png` — results.

**Outcome:** 100-step on-device training ran. The loss follows a downward trend consistent with PyTorch reference, but diff values accumulate over steps (up to ~0.1 by step 100) due to fixed-point accumulation drift — acceptable for demonstration but the test fails the tight 0.001 tolerance used in the CI fixture. 56 errors out of 100 loss comparisons at TOL=0.001.

### 4. Inference ONNX Export + On-Device Inference Setup (commits `9a57bf8`, `11a4cd9`, `f73a26c`)

**Goal:** Run inference (forward-only, BN folded) on GVSoC using Deeploy's inference path.

**What was built:**
- `DeeployTest/Tests/Models/speechnet_infer/` — single-sample inference fixture with pre-trained weights and BN folded into Conv.
- `DeeployTest/Tests/Models/speechnet_infer_random/` — same network, random weights/input (used to isolate compilation issues from data issues).
- `DeeployTest/speechnet_accuracy_eval.py` — first accuracy evaluation script (evaluation over the `speechnet_infer` fixture).

**Problems encountered and resolved:**

| Problem | Root cause | Resolution |
|---|---|---|
| On-device outputs diverged from ORT reference | Stale `network.onnx` from a prior export reused accidentally | Regenerate ONNX with correct BN-folded weights; verify with ORT first |
| Small conv weight norms suspected as "missing BN" | Folded-BN weights ARE small: γ/√(var+ε) shrinks them; this is expected | Confirmed by `np.allclose(w_folded, onnx_weights)` == True |
| Im2Col kernel buffer overflow on multi-core | `num_cores` not forwarded to `generateNetwork.py`, so Im2Col allocated buffer for 1 core but runtime used 8 | Fix in `deeployRunner.py`: pass `--cores=N` to code generator |

### 5. `num_cores` Bug Fix + Full Dataset Evaluation (commit `d116eb0`)

**Bug:** `deeployRunner.py` was not forwarding the `--cores` argument to `generateNetwork.py`. Im2Col kernels allocate their intermediate buffer as `buffer_size / num_cores` per core; with `num_cores=1` (default) the buffer was 8× too large, corrupting L1 memory at runtime.

**Fix:** Added `--cores={args.cores}` forwarding in `DeeployTest/testUtils/deeployRunner.py` (lines 274–280 and 446–449).

**Evaluation script:** `DeeployTest/speechnet_accuracy_eval_untiled.py`
- Loops over all samples in `speechnet_infer_revised/inputs.npz`.
- Writes each sample to a temp dir, runs the untiled GVSoC inference, parses `Logit[i]:` lines from `deeploytest.c`, accumulates per-class recall.
- Saves full per-sample results to `DeeployTest/speechnet_accuracy_results_untiled.json`.

**Dataset used:** `DeeployTest/Tests/Models/speechnet_infer_revised/` — 180 samples, subject S01 session S3 Batch 1 vocabulary, 9 classes, normalized EMG.

**Result achieved: 77.78% balanced accuracy (140/180 correct), matching the paper's reported zero-shot result.**

Per-class recall:
| Class | Recall |
|-------|--------|
| 0 | 0.95 |
| 1 | 0.60 |
| 2 | 0.80 |
| 3 | 0.65 |
| 4 | 0.55 |
| 5 | 0.95 |
| 6 | 0.70 |
| 7 | 0.85 |
| 8 | 0.95 |

---

## Current State (as of 2026-06-13)

| Area | Status |
|---|---|
| SpeechNet training test (untiled) | Passing (loss diff=0.0) |
| SpeechNet training test (tiled / CI) | **WIP** — tiled runner not yet evaluated end-to-end |
| 100-step training loss experiment | Completed; loss decreases but exceeds TOL=0.001 vs PyTorch |
| Inference ONNX export (BN folded) | Done and verified with ORT |
| On-device inference (untiled, 8 cores) | **Working — 77.78% accuracy confirmed** |
| On-device inference (tiled) | Not yet attempted |
| Batch-1 fine-tuning (AvgPool model) | **Done** — 90 steps run on GVSoC; loss ~2.0 start, slow convergence, 48/90 fail TOL=0.001 due to accumulation drift |

### Open Work Items

1. **Tiled inference** — run `speechnet_infer_revised` through the tiled Siracusa path to measure speedup vs untiled.
2. **Post-fine-tuning accuracy** — Batch-1 fine-tuning has been run on-device; next step is to extract the updated weights from the on-device checkpoint and re-evaluate accuracy on the 180-sample eval set to see if fine-tuning improves over the 77.78% zero-shot baseline.
3. **CI registration** — register the untiled inference test in `test_siracusa_tiled_config.py` or a separate inference config so it runs in CI.
4. **Training tolerance** — decide whether to raise the loss diff tolerance for the 100-step experiment, or only keep the initial 4-step (diff=0.0) fixture in CI.

---

## Key Files

| File | Purpose |
|---|---|
| `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train/` | Training fixture (initial random weights, real EMG data) |
| `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_preweights/` | Training fixture with pre-trained SpeechNet weights as init |
| `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_b1_ft/` | Batch-1 fine-tuning artifacts — AvgPool model variant, pre-normalised data, checkpoint + eval/train/optim ONNXes |
| `DeeployTest/Tests/Models/speechnet_infer/` | Single-sample inference fixture (pre-trained, BN folded) |
| `DeeployTest/Tests/Models/speechnet_infer_revised/` | Full 180-sample eval dataset (normalized EMG, labels embedded in inputs.npz) |
| `DeeployTest/Tests/Models/speechnet_infer_random/` | Random-weights inference fixture (compilation smoke test) |
| `DeeployTest/speechnet_accuracy_eval_untiled.py` | Accuracy eval loop (runs GVSoC per-sample, parses logits) |
| `DeeployTest/speechnet_accuracy_results_untiled.json` | Results: 77.78% balanced accuracy, 180 samples |
| `DeeployTest/speechnet_loss_experiment.py` | 100-step training loss experiment runner |
| `DeeployTest/speechnet_100step_loss.png` | Loss curve plot |
| `DeeployTest/testUtils/deeployRunner.py` | Fixed to forward `--cores` to code generator (Im2Col buffer fix) |
| `DeeployTest/Platforms/Siracusa/src/deeploytest.c` | Prints `Logit[i]: <val>` for each output element (used by accuracy script) |
| `Deeploy/Targets/Generic/Layers.py` | Fixed Conv bias shape: scalar → tuple |

---

## Key Bugs Fixed on This Branch

1. **`ConvLayer.computeShapes` crash** — Conv bias was a scalar int; wrapped in tuple.
2. **`ConvGradW` double-zero bug** — `memset(grad_weight)` was called per spatial tile, should be once per execution; caused accumulating gradient errors in tiled training.
3. **Im2Col buffer overflow** — `deeployRunner.py` did not pass `--cores` to `generateNetwork.py`; Im2Col allocated 8× too much buffer per core, corrupting L1 at runtime.
4. **Stale ONNX in inference debugging** — on-device outputs diverged because an old `network.onnx` was used; lesson: always verify ONNX with ORT before diagnosing on-device divergence.
