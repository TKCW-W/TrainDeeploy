# exp18 — ZO (MeZO, forward-only) round-1 on-device fine-tuning, on the CORRECT pretrained weights

**Date opened:** 2026-09-15 · **Branch:** `feat/GAP9_w_NE16` (TrainDeeploy + Onnx4Deeploy)
**Supersedes:** [`exp5_ZO_round1`](../exp5_ZO_round1/FINDING.md) — same recipe, superseded base checkpoint
**Sibling:** [`exp17`](../exp17_BP_round1_correct_weights/Findings.md) (BP) · [`exp19`](../exp19_QZO_round1_correct_weights/Plan.md) (QZO)

---

## 1. Why

`exp5` established that forward-only zeroth-order (MeZO) fine-tuning runs on device fully bit-exact
(0/21600) and matched/beat BP on accuracy. But it started from `SilentWear/artifacts/` — the Mac
reproduction that does **not** match the paper. The authoritative base weights are now
`SilentWear/SilentWear/artifacts_reference/` (lab-machine GPU run, matches the paper; all 37 tensors
differ, 47 vs 30 epochs, best val acc 0.9139 vs 0.8972 — see
[`exp17 Findings §2`](../exp17_BP_round1_correct_weights/Findings.md)).

**Goal:** re-run the ZO round-1 simulation with the recipe unchanged in every respect except the base
checkpoint, and report batch-2 accuracy + bit-exactness.

## 2. What is held fixed (recipe, from exp5's own device log)

Read back from `exp5_ZO_round1/logs/round1_gvsoc_zo.log.gz`, not from prose:

```
N_TRAIN_STEPS=2700  N_ACCUM_STEPS=4  DATA_INPUTS=2
ZO_EPS=0.010000  ZO_LR=0.000003000  ZO_Q=1  ZO_SEED=42
BN_FROZEN_STATS=ON  DUMP_WEIGHTS=ON
Generation: --cores=8 --defaultMemLevel=L2 --l1=128000 --l2=2000000
            --memAllocStrategy=MiniMalloc --searchStrategy=random-max
```

| knob | value |
|---|---|
| method | MeZO, forward-only, 2 forwards/sample (±ε) |
| perturbation | Rademacher, ε 0.01, q 1 dir/step, seed 42 |
| scope | all 22 trainable params (10 conv + 10 BN γ/β + 2 fc) |
| BN | frozen pretrained stats (`--bn-frozen-stats`, `BN_FROZEN_STATS=ON`) |
| lr | 3e-6 static |
| epochs / steps | 200 epochs → **2700 update steps**, n_accum 4 |
| FT data | 54 windows (30 %, 6/class, seed 42, `--stratified`) |
| cell | S01 / session 3 / vocalized / fold 3 · train batch 1 → eval batch 2 |

**The one and only change:** `--pretrained-weights` → `artifacts_reference/`.
Fixtures suffixed `_ref` so exp5's artifacts are preserved and the two can be diffed.

## 3. Phases

| # | phase | container | cost | gate |
|---|---|---|---|---|
| 1 | Export ZO fixture (`-mode zo-train`) | `agitated_hugle` | min | `n_steps=2700 q=1 lr=3e-06 eps=0.01 seed=42`, 54 stratified |
| 1b | **Verify the 54 FT windows are byte-identical to exp17's BP draw** | host | s | must be `True` — otherwise ZO-vs-BP is not a controlled comparison |
| 2 | Pack into the two-dir layout (`_train` / `_update`) | `traindeeploy` | s | train 24 inputs, update 22 in/22 out |
| 3 | Device ZO round, 2700 steps, `DUMP_WEIGHTS=ON` | `traindeeploy` | **~18 h** | `Errors: 0 out of 21600` expected (ZO is forward-only) |
| 4 | Extract `[WDUMP s=2699]` → carry checkpoint | `traindeeploy` | s | 22 tensors; BN stats must stay frozen |
| 5 | Export INFER fixture for batch 2 with carry weights | `agitated_hugle` | min | — |
| 6 | Device EVAL on batch 2 (180 windows) | `traindeeploy` | ~35 min | bit-exactness vs ORT |

## 4. Acceptance criteria

1. **Bit-exactness:** `Errors: 0 out of 21600` on the training round. exp5 achieved a perfect 0 —
   ZO is forward-only, so there is no MaxPool *backward* and none of the argmax tie-flips that give BP
   its drift. Any nonzero count here is a genuine regression and must be explained, not waved through.
2. Batch-2 evaluation: all 180 windows bit-exact vs ORT (`sim_errors=0`).
3. Report batch-2 balanced accuracy against the reference table (no-FT 76.67, paper-FT 88.89) and
   against exp17 (BP, 87.78) on the **same 54 windows**.

## 5. Runtime note — this is the long one

exp5's faithful counter gives **385.3 G cycles** for the round (vs BP's ~75.8 G), i.e. ~5×, driven by
5× the epochs. On this host exp17's BP round took 3 h 40 min for ~75.8 G cycles (≈0.345 G cyc/min under
x86_64 emulation on Apple Silicon), so exp18 extrapolates to **≈18 h**.

Run it **concurrently with exp19** using a private build directory:
`PYTEST_XDIST_WORKER=zo` → `TEST_SIRACUSA/build_zo` (`testUtils/deeployRunner.py:219`). The generation
dir is already keyed on the fixture name, so `build_master` was the only shared state.

## 6. Gotchas

- **Launch detached**: `docker exec -d` + `nohup` + `python3 -u`, log redirected to the bind mount. A
  foreground `docker exec` streams stdout to a host process that host memory pressure can kill, which
  orphans the run and loses the `[WDUMP]` block (exp17 §5.3).
- **The fixture is TWO dirs.** Pass `--optimizer-dir` explicitly. If the `-t` name lacks `_train`, the
  auto-derive resolves to the same dir and the runner loads the perturbed-forward graph as the update
  graph → generation fails.
- `pgrep -f "[g]vsoc_launcher"` — bracket it, or it kills its own shell.
