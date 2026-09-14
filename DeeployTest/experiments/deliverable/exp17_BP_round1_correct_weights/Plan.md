# exp17 — BP round-1 on-device fine-tuning, on the CORRECT (lab-machine) pretrained weights

**Date opened:** 2026-09-14 · **Branch:** `feat/GAP9_w_NE16` (TrainDeeploy + Onnx4Deeploy)
**Supersedes:** `exp4_BP_round1` (same recipe, wrong base checkpoint)

---

## 1. Why this experiment exists

Every on-device experiment so far (`exp1`, `exp4`, and the QZO chain) started from
`SilentWear/artifacts/…/leave_one_session_out_fold_3.pt` — a checkpoint we reproduced **locally on the
Mac**. That reproduction has since been shown **not to match the paper**. The user re-ran the
SilentWear reproduction on a lab machine with a GPU, it **did** match the paper, and those artifacts
are now at `SilentWear/SilentWear/artifacts_reference/`.

The two base checkpoints are not small perturbations of one another:

| | our Mac run (`artifacts/`) | lab machine (`artifacts_reference/`) |
|---|---|---|
| epochs actually run | 31 | **48** |
| best val accuracy | 0.8972 | **0.9139** |
| `num_batches_tracked` (all 5 BN layers) | *N* | ***N* + 765** |
| tensors differing | — | **37 / 37** |

765 = 17 extra epochs × 45 batches/epoch (1440 train windows ÷ batch 32) — i.e. the lab run simply
trained 17 epochs longer before early stopping fired. Every downstream on-device accuracy number is
therefore quoted against a base model that is **worse-converged than the paper's**, which confounds
any statement about the on-device method itself.

**Goal:** re-run the round-1 BP on-device simulation, unchanged in every respect except the base
checkpoint, so that the on-device accuracy can be compared against the paper-matching reference.

## 2. What is held fixed (the "correct on-device BP setting")

Recipe **S2 — full-model, frozen-BN**, exactly as `BP_FLOW.md §A.0` and `exp4_BP_round1/FINDING.md`:

| knob | value |
|---|---|
| scope | full model (conv + BN γ/β + fc), **no BN folding** |
| BN | **frozen pretrained stats**, train ≡ inference (`--bn-frozen-stats`, device `BN_FROZEN_STATS=ON`) |
| optimizer | SGD, no momentum, no weight decay |
| lr | 3e-4, static |
| n_accum | 4, **SUM** grads (effective batch 1) |
| epochs | 40 |
| FT data | 54 windows = 30 %, 6/class, seed 42, `--stratified` |
| device steps | **540** = 40 × 54 ÷ 4 |
| MaxPool | argmax-mask + transpose-dedup (`--maxpool-argmax-mask`) |
| memory | `--l1 128000 --l2 1500000` (GAP9's 1.5 MB L2) |
| cores | 8 |
| cell | S01 / session 3 / **vocalized** / fold 3 / batch 1 → eval batch 2 |

**The one and only change:** `--pretrained-weights` now points at `artifacts_reference/`.

Fixture dirs are suffixed `_ref` so exp4's artifacts are not overwritten and the two can be diffed.

## 3. Phases

| # | phase | container | cost | gate |
|---|---|---|---|---|
| 0 | Record checkpoint provenance + diff vs. old | host | s | — |
| 1 | **Zero-shot b1 eval** on the new ckpt (export infer + device eval) | both | ~min | **must match the lab reference's zero-shot b1**; if it does not, the weight-loading path is wrong and phases 2-6 are wasted |
| 2 | Export TRAIN fixture b1 (`…_train_fullfrozen_b1_fold3_ref`) | `agitated_hugle` | ~min | optimizer sibling dir auto-created |
| 3 | Device TRAIN, 540 steps, `DUMP_WEIGHTS=ON` | `traindeeploy` | **hours** | final `[WDUMP s=539 …]` present |
| 4 | Extract device weights → carry ckpt | `traindeeploy` | s | 32 tensors matched |
| 5 | Export INFER fixture for batch 2 with carry weights | `agitated_hugle` | ~min | — |
| 6 | Device EVAL on batch 2's 180 windows → balanced accuracy | `traindeeploy` | ~min | bit-exactness vs ORT |

Phase 1 is deliberately first: it is cheap and it is the only phase that can fail for a *setup* reason
rather than a *scientific* reason.

## 4. Acceptance criteria

1. Zero-shot b1 (phase 1) reproduces the lab-machine reference's zero-shot b1 for S01/fold-3.
2. Phase 6 produces a batch-2 balanced accuracy with **0 bit-exactness errors** vs the ORT reference
   (device inference == its own reference inference).
3. The result is reported **alongside** exp4's 87.22 %, with the delta attributed to the base
   checkpoint alone (everything else is byte-identical in configuration).

## 5. Known non-failures (carried over from exp4)

- Per-step loss "errors" during training (~416/2160 in exp4, onset ≈ step 133) are the inherent
  device-vs-ORT **MaxPool argmax tie-flips** from fp reduction-order differences. Accuracy, not
  per-step loss, is the acceptance criterion (`BP_FLOW.md §B.5`).
- `BENCH train_cycles` is a `uint32` that **overflows** for a 540-step round. Do not quote it; the
  faithful figure is ~130–140 M cycles/step × 540 ≈ 75.8 G cycles ≈ 3.4 min on GAP9 @ 370 MHz.

## 6. Gotchas to respect

- **Kill orphan GVSoC by PID before every device run** — `killall`/`pkill` by name do not reach
  `gvsoc_launcher`, and orphans starve the new sim: `pgrep -f gvsoc_launcher | xargs -r kill -9`.
- The train fixture dir name **must contain `_train`**, or no `_optimizer` sibling is generated and
  tiling fails at `SoftmaxCrossEntropyLoss`.
- Device stdout flushes only at `main()` return — an in-progress log looks empty.
