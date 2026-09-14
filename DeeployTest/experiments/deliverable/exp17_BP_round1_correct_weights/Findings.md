# exp17 — BP round-1 on-device fine-tuning on the CORRECT (lab-machine) pretrained weights

**Date:** 2026-09-14 → 2026-09-15 · **Branch:** `feat/GAP9_w_NE16` (TrainDeeploy + Onnx4Deeploy)
**Cell:** S01 / session 3 / vocalized / fold 3 · train batch 1 → eval batch 2
**Supersedes:** [`exp4_BP_round1`](../exp4_BP_round1/FINDING.md) — same recipe, superseded base checkpoint
**Plan:** [`Plan.md`](./Plan.md) · **Reproduction:** §4 · **Changes:** §5

---

## 1. Result — PASS

**Batch-2 balanced accuracy: 87.78 %** (158/180), all 9 classes at 20 windows each.
All **180/180** eval windows bit-exact vs the ORT reference (`sim_errors=0`, every test `PASSED`), i.e. the
device-trained model's on-device inference equals its own reference inference. 0 parse failures.

### 1.1 In context

| | batch 2 balanced acc | Δ vs no-FT |
|---|---|---|
| no FT — reference base model on batch 2 | 76.67 % | — |
| **exp17 — on-device BP (recipe S2), reference weights** | **87.78 %** | **+11.11 pp** |
| reference — paper's own FT recipe | 88.89 % | +12.22 pp |
| exp4 — on-device BP, superseded Mac weights | 87.22 % | (not comparable — different base) |

On-device BP recovers **91 % of the paper recipe's fine-tuning gain** (11.11 / 12.22 pp) while using
substantially less of everything:

| | paper FT recipe | on-device BP (S2) |
|---|---|---|
| FT windows | 126 (70 % of batch) | **54 (30 %)** |
| optimizer | Adam, ReduceLROnPlateau | **SGD, static lr 3e-4, no momentum** |
| epochs | 50 (+ early stopping) | **40** |
| batch | 32 | **1** (n_accum 4, SUM) |
| BatchNorm | running stats **updated** | **frozen** at pretraining stats |

> **Relevance to the BN ablation.** The whole question driving `exp21`/`exp22` is whether batch-size-1
> BatchNorm is the main cause of the accuracy gap to the paper. On this cell the on-device arm — which is
> batch-1 **and** frozen-BN **and** 30 % data **and** plain SGD, all at once — lands **1.11 pp** below the
> paper recipe. That is a *bound*, not an ablation: it says the total penalty from all four deviations
> combined is small here, which makes "BN at batch 1 is the main cause of a large gap" hard to sustain for
> this cell. One cell, one seed — see §3.2 before generalising.

### 1.2 Reference table this is measured against

From the lab-machine reproduction that matches the paper
(`artifacts_reference/.../ft_config_0/ft_summary.csv`, S01/vocalized/fold 3) — see
[`results/reference_S01_fold3.txt`](./results/reference_S01_fold3.txt):

| eval batch | no-FT (base) | FT (paper recipe) |
|---|---|---|
| 1 | 72.22 | 72.22 |
| **2** | **76.67** | **88.89** ← this experiment's round |
| 3 | 72.78 | 82.78 |
| 4 | 80.56 | 88.89 |
| 5 | 72.78 | 87.78 |

### 1.3 Phase-1 gate

Before committing hours of simulation, the un-fine-tuned model was evaluated on batch 1:

| | |
|---|---|
| device zero-shot b1 | **72.22 %** (180/180, `sim_errors=0`) |
| reference `ft_summary.csv`, `num_prev_ft_rounds=0` | **72.22 %** |

Exact match ⇒ the checkpoint loads correctly and device inference is faithful. This gate is cheap
(~30 min) and is the only phase that can fail for a *setup* reason rather than a scientific one; it is
worth running first in every future round.

---

## 2. Why this experiment was needed — the base checkpoint changed

Every on-device experiment up to and including `exp4` started from
`SilentWear/artifacts/…/leave_one_session_out_fold_3.pt`, a reproduction run **on the Mac** that does
**not** match the paper. The lab-machine (GPU) reproduction **does**, and now lives at
`SilentWear/SilentWear/artifacts_reference/`. The two are not small perturbations of each other
([`results/ckpt_provenance.json`](./results/ckpt_provenance.json)):

| | Mac (`artifacts/`) | lab machine (`artifacts_reference/`) |
|---|---|---|
| epochs run | 30 | **47** |
| best epoch | 21 | **38** |
| best val loss | 0.4001 | **0.2639** |
| best val acc | 0.8972 | **0.9139** |
| tensors differing | — | **37 / 37** |
| `num_batches_tracked` delta | — | **+765** |

765 = 17 extra epochs × 45 batches/epoch (1440 train windows ÷ batch 32). The lab run simply trained 17
epochs longer before early stopping fired — a better-converged model, not a noise perturbation.

---

## 3. Secondary findings

### 3.1 Device-vs-ORT agreement is 26–82× tighter on the better-converged base model

Both runs are recipe-identical; only the base weights differ.

| | exp4 (Mac weights) | **exp17 (reference weights)** | ratio |
|---|---|---|---|
| loss breaches > TOL | 416 / 2160 | **1 / 2160** | 416× |
| median \|device − ORT\| | 1.64e-04 | **2.00e-06** | 82× |
| p99 | 9.51e-03 | **3.61e-04** | 26× |
| max | 3.49e-02 | **1.28e-03** | 27× |
| first breach ("drift onset") | step **133** | step **1209** | — |

exp17's single breach: `[loss 1209] computed=0.188445 ref=0.189725 diff=0.001280 TOL=0.001000`.

**Interpretation.** These breaches are not independent per-step noise. They are **cumulative divergence**:
once one MaxPool argmax tie-flips (device and ORT reduce in different orders, so a pooling window's max
can flip between two near-equal values), the device's weights permanently differ from ORT's and every
later step inherits the gap. The error count is therefore approximately *"how many steps after the first
flip exceed tolerance"* — exp4 had 2027 steps after its flip and breached 416; exp17 had 951 and
breached 1. The meaningful quantity is **when the first flip happens**, not the raw count.

**Caveat — do not over-read this.** n = 1 run per condition. The plausible mechanism is that a
better-converged model has larger margins between competing maxima inside a pooling window, making
near-ties rarer, but a single pair of runs cannot establish that. Treat it as a strong observation. It
does *not* change the acceptance criterion, which remains accuracy, not the per-step loss trace
(`BP_FLOW.md §B.5`).

### 3.2 Scope limits

- **One cell**: S01, fold 3, vocalized, round 1 only. Rounds 2–4 (b2→b5) and subjects S02–S04 are not run.
- **One seed**: the FT draw is seed 42, `--stratified`. Previously measured run-to-run spread on this
  pipeline was mean \|Δ\| 4.59 pp with a max of 25.00 pp, so a 1.11 pp difference between exp17 and the
  paper recipe is **well inside the noise floor** and must not be quoted as "on-device is 1.11 pp worse".
- exp4's 87.22 % and exp17's 87.78 % are **not** a controlled comparison of anything except the base
  checkpoint, and the 0.56 pp between them is far below that noise floor.

### 3.3 Carry-checkpoint integrity

[`results/carry_fullfrozen_b1_fold3_ref.pt`](./results/carry_fullfrozen_b1_fold3_ref.pt), verified against
the base:

| check | result |
|---|---|
| BN stat tensors (`running_*`, `num_batches_tracked`) that must stay frozen | 15 — **0 changed** ✅ |
| trainable tensors that must move | 22 — **0 unchanged** ✅ |
| `state_dict` key set preserved | ✅ |
| max weight movement | 4.738e-02 (`blocks_0_0_weight`) |

### 3.4 Latency — still unquotable

`BENCH train_cycles=1947274114 opt_cycles=30527702 weight_sram=61956`.
`train_cycles` is a **`uint32` that overflows** for a 540-step round (exp4 reported 1957961288 for the
same work), so the printed value is a wrapped remainder — **do not quote it**. `opt_cycles` (30.5 M) and
`weight_sram` (61 956 B) are within range and match exp4's 30 491 774 / 61 956, as expected for an
identical graph. The faithful figure remains ~130–140 M cycles/step × 540 ≈ 75.8 G cycles ≈ 3.4 min on
GAP9 @ 370 MHz. The ZO harness already prints a `uint64` hi/lo split; the BP harness could take the same
two-line fix.

**Wall-clock on this host** (GVSoC under x86_64 emulation on Apple Silicon): phase 3 ≈ 3 h 40 min
(20:09 → 23:49), phase 6 ≈ 33 min, phase 1 ≈ 39 min.

---

## 4. Reproduction

One script runs all six phases from the **host**, with both containers up:

```bash
cd TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights
bash scripts/run_round1.sh all      # or: 1 | 2 | 3 | 4 | 5 | 6
```

### 4.1 Fixed inputs

| what | path |
|---|---|
| base checkpoint | `SilentWear/SilentWear/artifacts_reference/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt` |
| data | `SilentWear/SilentWear_data/data_raw_and_filt` |
| carry checkpoint out | `exp17…/results/carry_fullfrozen_b1_fold3_ref.pt` |
| fixtures | `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_{train,optimizer,infer}_fullfrozen_b1_fold3_ref`, `speechnet_infer_fullfrozen_b0_fold3_ref` |

### 4.2 Recipe S2 — full-model, frozen-BN (unchanged from exp4)

full model (conv + BN γ/β + fc), no BN folding · frozen pretrained BN stats (`--bn-frozen-stats`,
`BN_FROZEN_STATS=ON`) · SGD, no momentum, no weight decay · lr 3e-4 static · n_accum 4 **SUM** ·
40 epochs · 54 FT windows (30 %, 6/class, seed 42, `--stratified`) · **540 device steps** ·
MaxPool argmax-mask · `--l1 128000 --l2 1500000` (GAP9's 1.5 MB L2) · 8 cores.

### 4.3 Phase commands

**Containers mount the same host tree at DIFFERENT prefixes** — `agitated_hugle`: `/app/TrainDeeploy`;
`traindeeploy`: `/app/ETH/TrainDeeploy`. See §5.2.

```bash
# phase 1 — zero-shot b1 gate (export in agitated_hugle, eval in traindeeploy)
docker exec agitated_hugle bash -lc 'cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b0_fold3_ref \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /app/SilentWear/SilentWear/artifacts_reference/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --batch 1 --condition vocalized'

# phase 2 — export TRAIN fixture (dir name MUST contain `_train`, or no `_optimizer` sibling is made)
docker exec agitated_hugle bash -lc 'cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights <base ckpt> --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified --maxpool-argmax-mask'

# phase 3 — device TRAIN, 540 steps (~3h40m here). RUN DETACHED -- see §5.1
docker exec -d traindeeploy bash -lc "
  pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
  cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
  nohup python3 -u deeployTrainingRunner_tiled_siracusa.py \
    -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref \
    --n-steps 540 --n-accum 4 --cores 8 \
    --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
    -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON > <logdir>/phase3_gvsoc_train_round1.log 2>&1 &"

# phase 4 — [WDUMP s=539] -> carry ckpt  (write to the BIND MOUNT, not /tmp -- see §5.2)
docker exec traindeeploy bash -lc 'cd /app/ETH/TrainDeeploy/DeeployTest/experiments/exp1/ondevice_sim_S01_fold3 && \
  python3 extract_device_weights.py --gvsoc-log <logdir>/phase3_gvsoc_train_round1.log \
    --base-ckpt <base ckpt> --out-carry <exp17>/results/carry_fullfrozen_b1_fold3_ref.pt'

# phase 5 — export INFER fixture for batch 2 with the carry weights   (as phase 1, --batch 2)
# phase 6 — device EVAL on batch 2                                     (as phase 1's eval, _b1_ dir)
```

### 4.4 Expected output

| phase | expect |
|---|---|
| 1 | `Balanced accuracy : 0.7222` |
| 2 | `n_batches=2160 … n_steps=540`, `Stratified split: 6 samples/class × 9 classes = 54 total` |
| 3 | `update 540/540`, `Errors: 1 out of 2160`, 22 `[WDUMP s=539 …]` lines. **Nonzero exit and a `FAILED` banner are EXPECTED** — see §3.1. |
| 4 | `parsed [WDUMP] last step s=539: 22 tensors (expected 22)` |
| 5 | `Balanced accuracy = 0.8778` |
| 6 | `Balanced accuracy : 0.8778`, `158/180`, 0 parse failures |

---

## 5. Changes made (with file paths)

No production code changed. exp17 is new experiment material plus three documentation fixes, each of
which had caused a silently-wrong or failed run.

### 5.1 New — the experiment
| path | what |
|---|---|
| `DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/Plan.md` | plan, recipe table, phase gates |
| `…/Findings.md` | this file |
| `…/scripts/run_round1.sh` | all six phases, `bash run_round1.sh [1-6\|all]` |
| `…/results/ckpt_provenance.json` | base-checkpoint provenance + Mac-vs-lab diff |
| `…/results/reference_S01_fold3.txt` | the reference table of §1.2 |
| `…/results/carry_fullfrozen_b1_fold3_ref.pt` | 22 device-trained tensors over the reference ckpt |
| `…/results/eval_b2_results.json` | phase-6 per-class + balanced accuracy |
| `…/logs/phase{1..6}*.log[.gz]` | full logs, incl. the 2160-step loss trace and `[WDUMP]` blocks |

### 5.2 Fixed — `TrainDeeploy/BP_FLOW.md`
Three bugs, each of which **failed silently or misleadingly**:

1. **§A.0 container paths.** The doc said `agitated_hugle` sees TrainDeeploy at `/app/ETH/TrainDeeploy`.
   It does not — its bind mount is `/Users/qiwenwu/ETH → /app`, so the tree is at `/app/TrainDeeploy`.
   (`traindeeploy` mounts at `/app/ETH`, hence the confusion.) Exporting with the wrong prefix does **not**
   error: it creates the path in the container's overlay filesystem, prints `✅ Export Complete!`, and
   yields a valid fixture that is invisible to the host and to the other container. Corrected, with an
   explicit warning box, and the `-o` paths in §A.1/§A.4 fixed.
   *Same trap applies to the carry checkpoint:* `/tmp/carry_….pt` is container-local, so phase 4 must
   write to the bind mount or phase 5 cannot read it.
2. **§A.2 `pgrep -f gvsoc_launcher`** matches its own shell's command line; piping to `kill -9` kills the
   shell (exit 137, empty log). Now bracketed: `pgrep -f "[g]vsoc_launcher"`.
3. **`rm -rf TEST_SIRACUSA` before ANY build-mode switch**, not just before training.
   `DeeployTest/CMakeLists.txt:9` branches on `if(TRAINING OR MEZO_TRAINING)` and CMake **caches** that
   variable in `TEST_SIRACUSA/build_master`. An inference run in a training-configured tree fails on every
   sample with `CMake Error … Cannot find source file TrainingNetwork.c`, surfaced only as
   `could not parse logits` — i.e. it looks like 180 model failures, not one build error.

Also updated: the §A.0 path table now points at `artifacts_reference/`, with a warning that results from
`artifacts/` are not comparable to the paper.

### 5.3 Operational note — long runs must be detached
Phase 3 was first launched via a foreground `docker exec` whose output was streamed to a host process.
That process was killed by host memory pressure at update 31/540; the simulation kept running inside the
container but its stdout went nowhere, making the run unrecoverable (phase 4 needs `[WDUMP]` from the
log). **Always launch with `docker exec -d` + `nohup` + `python3 -u`, redirecting to a log on the bind
mount**, so the container's own shell owns the file descriptor. Done this way, the run survived three
subsequent host-side reaps.

---

## 6. Next steps

1. **Rounds 2–4** (b2→b5) to complete the S01/fold-3 incremental chain — same script, swap the batch and
   `--pretrained-weights` per `BP_FLOW.md`'s per-round table. ≈4 h each on this host.
2. **Subjects S02–S04** for a 4-subject table comparable to §1.2.
3. **Re-point exp21** (`PyTorch_for_On_Device/exp21_BP_faithful`) at `artifacts_reference/` — its current
   numbers are measured against the superseded Mac reproduction.
4. **exp22 BN ablation** — frozen BN with *everything else the paper's* (batch 32, Adam 1e-3, 50 epochs,
   70 % split). §1.1 suggests the combined penalty is small on this cell, which makes the isolated BN
   term the interesting unknown.
5. Establish a **noise floor** (2–3 seeds) before quoting any of these differences: the previously
   measured run-to-run spread (mean \|Δ\| 4.59 pp, max 25.00 pp) exceeds every gap reported here.
