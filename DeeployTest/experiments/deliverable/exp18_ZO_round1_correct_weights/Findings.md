# exp18 — ZO (MeZO, forward-only) round-1 on-device fine-tuning, on the CORRECT pretrained weights

**Date:** 2026-09-15 → 2026-09-16 · **Branch:** `feat/GAP9_w_NE16`
**Cell:** S01 / session 3 / vocalized / fold 3 · train batch 1 → eval batch 2
**Supersedes:** [`exp5_ZO_round1`](../exp5_ZO_round1/FINDING.md) — same recipe, superseded base checkpoint
**Plan:** [`Plan.md`](./Plan.md) · **Siblings:** [`exp17`](../exp17_BP_round1_correct_weights/Findings.md) (BP) · [`exp19`](../exp19_QZO_round1_correct_weights/Findings.md) (QZO)

---

## 1. Result — PASS

**On-device batch-2 balanced accuracy: 90.00 %** (162/180). All **180/180** windows evaluated,
`sim_errors=0` on every one, 0 parse failures — the device-trained model's on-device inference is
bit-exact against its own ORT reference.

### 1.1 In context

| | batch-2 balanced acc | train cycles |
|---|---|---|
| no FT — reference base model | 76.67 % | — |
| exp19 QZO (int8) | 85.00 % | 89.5 G |
| exp17 BP (first-order) | 87.78 % | ~75.8 G |
| reference — paper's own FT recipe | 88.89 % | — |
| **exp18 ZO (this)** | **90.00 %** | **384.95 G** |

**Forward-only ZO is the most accurate on-device result we have on this cell, and it beats the
paper's own fine-tuning recipe** (+1.11 pp) — with no gradient buffers, no saved activations, and
about half the L2 peak of backprop (`exp5`: 0.79 MB vs 1.51 MB). It pays **4.3×** QZO's cycles for
that.

**+13.33 pp over no adaptation**, against the paper recipe's +12.22.

## 2. The host simulation predicted this exactly

`CleanUp/exp2_ZO_simulation` recorded **90.00 %** for this cell as a falsifiable prediction, written
down while exp18 was still at ~1300/2700 steps. The device returned **90.00 %**.

That is not a fit. exp2 was matched to the device on four axes, each verified rather than assumed:
the labelled draw (byte-identical to the exporter's), the cyclic window order, the MEAN accumulation
semantics, and the Rademacher direction stream — the last checked against the device's own logged
losses to **2.15e-06 after 999 replayed updates**.

Both device experiments now reproduce exactly from their host simulations:

| | device | matched host simulation |
|---|---|---|
| exp17 BP | 87.78 | **87.78** |
| **exp18 ZO** | **90.00** | **90.00** |

## 3. Bit-exactness

### 3.1 Training — 5 / 21 600, all marginal

`Errors: 5 out of 21600` (0.023 %) over the full 200-epoch round.

| | |
|---|---|
| median \|device − ORT\| | 3.40e-05 |
| mean | 9.13e-05 |
| p99 | 5.94e-04 |
| p99.9 | 8.74e-04 |
| **max** | **1.369e-03** = **1.37 × tolerance** |
| exactly 0 | 965 (4.5 %) |

The five breaches, all on the `loss-` pass and all in the **final quarter** of the trace
(count by quarter: 0 / 0 / 0 / 5):

```
[loss- 6483]  computed=2.919342  ref=2.918277  diff=0.001065   (1.07x TOL)
[loss- 9772]  computed=1.912172  ref=1.911079  diff=0.001093   (1.09x)
[loss- 10096] computed=2.154860  ref=2.153807  diff=0.001053   (1.05x)
[loss- 10265] computed=2.785082  ref=2.786452  diff=0.001369   (1.37x)
[loss- 10778] computed=3.853919  ref=3.854938  diff=0.001019   (1.02x)
```

> **A claim from exp5 needs weakening.** exp5 reported `Errors: 0 out of 21600` on the old weights,
> and this project has since described forward-only ZO as bit-exact on device *by construction*. The
> mechanism argument still holds — ZO has no MaxPool **backward**, so none of backprop's argmax
> tie-flips — but it does not guarantee zero. The accurate statement is **"essentially bit-exact: 5
> marginal breaches in 21 600, none exceeding 1.4× tolerance, p99.9 still below tolerance"**.

The contrast with BP is what makes the mechanism claim credible even so:

| | exp17 BP | **exp18 ZO** |
|---|---|---|
| breaches | 1 / 2160 (and 416 / 2160 on the old weights) | **5 / 21 600** |
| median | 2.00e-06 | 3.40e-05 |
| **max** | 1.28e-03 (and **3.49e-02** on the old weights = 35× TOL) | **1.37e-03** |

A genuine tie-flip moves the weights permanently and produces errors orders of magnitude past
tolerance, then persists. Nothing here does that.

### 3.2 Inference — exact

180/180 windows `PASSED` with `sim_errors=0`. Unlike QZO (exp19: 116/180 inference bit-exact fails
under ffast-math), the float ZO inference graph matches ORT on every window.

## 4. Latency

`BENCH train_cycles_hi=89 train_cycles_lo=2697639026 opt_cycles_hi=0 opt_cycles_lo=389289212`

| | |
|---|---|
| train | 89·2³² + 2 697 639 026 = **384.95 G cycles** ≈ **17.3 min @ 370 MHz** |
| optimizer | 389.3 M |

The ZO harness prints a **uint64** hi/lo split, so unlike BP's `BENCH` this figure is faithful and
quotable. It matches exp5's 384.92 G almost exactly — same graph, same step count, different weights
— which is a useful confirmation that nothing structural changed with the checkpoint swap.

**Per-method cost on this cell:** BP ~75.8 G · QZO 89.5 G · **ZO 385 G**. ZO buys its accuracy and
its halved memory with ~5× BP's cycles.

## 5. Carry-checkpoint integrity

`results/carry_zo_b1_fold3_ref.pt` — 22 device-trained tensors written over the reference checkpoint;
max weight movement **1.214e-02**; BN running statistics untouched (frozen, as the recipe requires).

## 6. Reproduction

```bash
cd TrainDeeploy/DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights
bash scripts/run_round1.sh all      # or: 1 | 2 | 3 | 4 | 5 | 6
```

Recipe, read back from exp5's own device log rather than from prose:

```
N_TRAIN_STEPS=2700  N_ACCUM_STEPS=4  DATA_INPUTS=2
ZO_EPS=0.010000  ZO_LR=0.000003000  ZO_Q=1  ZO_SEED=42
BN_FROZEN_STATS=ON  DUMP_WEIGHTS=ON
--l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc
--searchStrategy random-max --cores 8
```

Base weights `artifacts_reference/.../leave_one_session_out_fold_3.pt`; 54 stratified FT windows
(6/class, seed 42), byte-identical to exp17's and exp19's draws.

**Expected output**

| phase | expect |
|---|---|
| 1 | export: `n_batches=10800 n_accum=4 n_steps=2700 q=1 lr=3e-06 eps=0.01 seed=42` |
| 2 | pack: train 24 inputs, update 22 in / 22 out |
| 3 | `update 2700/2700`, `Errors: 5 out of 21600`, 22 `[WDUMP s=2699]`, `BENCH train_cycles_hi=89` |
| 4 | `parsed [WDUMP] last step s=2699: 22 tensors`; max movement 1.214e-02 |
| 5 | `Balanced accuracy = 0.9000` |
| 6 | `Balanced accuracy : 0.9000`, `162/180`, 0 parse failures |

> ⚠️ Launch phase 3 detached (`docker exec -d` + `nohup` + `python3 -u`, log on the bind mount) — a
> foreground `docker exec` streams stdout to a host process, and losing it loses the `[WDUMP]` block.
> See exp17 Findings §5.3. And do not start exp9's QZO eval harness beside a live device run unless
> you are on commit `93458a0` or later — see exp19 Findings §6.2.

## 7. Files

| path | what |
|---|---|
| `Plan.md`, `Findings.md` | plan and this file |
| `scripts/run_round1.sh` | all six phases |
| `results/carry_zo_b1_fold3_ref.pt` | the 22 device-trained tensors |
| `results/eval_b2_results.json` | `{balanced_accuracy: 0.90, n_evaluated: 180, n_failed: 0}` |
| `logs/phase3_gvsoc_zo_round1.log` | 2700-step round: 21 600-forward loss trace, `[WDUMP]`, `BENCH` |
| `logs/phase3_gvsoc_zo_round1_KILLED_at_1566.log` | the first attempt, destroyed by a concurrent eval — see `Plan.md` §7 |
| `logs/phase{1,2,4,5,6}*.log` | export, pack, extract, infer-export, device eval |

## 8. Open

1. **Rounds 2–4** to complete the S01/fold-3 chain (~12 h each on this host), and subjects S02–S04.
2. The 5 breaches are characterised but not explained to the ulp. If it matters, the device log
   records every loss as raw fp32 bits, so they can be decoded and compared offline without re-running.
3. `exp6_acc_epo_revised` finds that **~100 epochs reaches within ~0.4 pp of 200** for float ZO in
   simulation — worth a device round at 100 epochs, which would halve these 385 G cycles.
