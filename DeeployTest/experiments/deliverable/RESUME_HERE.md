# ⏸ PAUSED 2026-09-15 — READ THIS FIRST TO RESUME (exp18 / exp19)

Everything below is **frozen, not cancelled**. The Mac was closed and taken away; macOS sleep
suspends the Docker VM, so the container processes stop getting CPU and resume on wake. Nothing was
killed deliberately.

## State at pause

| | status |
|---|---|
| **exp19 QZO** | ✅ **COMPLETE** through phase 5. Result below. Only `Findings.md` is left to write. |
| **exp18 ZO** | ⏳ device round **890 / 2700**, running in `traindeeploy` as `PYTEST_XDIST_WORKER=zo`. ~8 h of awake time left at 3.6 upd/min. |

## First thing to do on resume

```bash
docker exec traindeeploy pgrep -cf 'zo_train_b1_fold3_ref'     # expect >=1
tail -1 exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1.log
grep -oE 'update [0-9]+/2700' exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1.log | tail -1
```
* **Advancing** → nothing to do; re-arm a monitor for `BENCH train_cycles` in that log.
* **Dead / Docker wedged** → the round has no checkpointing; restart from scratch with
  `bash exp18_ZO_round1_correct_weights/scripts/run_round1.sh 3` (≈12.5 h).

Then finish exp18 phases 4→6: `bash exp18_ZO_round1_correct_weights/scripts/run_round1.sh 4`, then `5`, then `6`.

## ⚠️ Before launching ANY eval next to a live device run

`exp9_QZO_round1/qzo_accuracy_eval_untiled.py` used to `rm -rf TEST_SIRACUSA` and
`pgrep -f gvsoc_launcher | xargs kill -9` — machine-wide. That **destroyed the first exp18 round at
update 1566/2700** (~7 h lost). Fixed in commit `93458a0`: both steps are now scoped to
`PYTEST_XDIST_WORKER`. Other harnesses in this tree still contain the unscoped pattern — **read a
tool's startup block before running it beside a long simulation.** See `exp18/Plan.md` §7.

## exp19 QZO — RESULT (complete)

Cell: S01 / session 3 / vocalized / fold 3 · train batch 1 → eval batch 2 · base weights
`artifacts_reference/`.

| | value |
|---|---|
| device-trained, **on-device** untiled eval, batch 2 | **85.00 %** |
| same weights, host-executor | 85.56 % |
| device zero-shot (fresh calibration) | 75.00 % |
| PyTorch fc-float reference (fresh calibration) | 87.78 % |
| **full-round bit-exactness** | **`Errors: 9949 / 21600`** (46 %) — ffast-math, expected |
| inference bit-exact fails | 116 / 180 — accepted, does not change predictions |
| train cycles | 20·2³² + 3 612 894 973 = **89.51 G** |
| opt cycles | 877.9 M |

Reference for the same cell (`artifacts_reference` ft_summary.csv): zero-shot b2 **76.67**,
paper-recipe FT b2 **88.89**. exp17 (BP, same 54 windows) reached **87.78**.

**exp19's headline improvement over exp12:** exp12 exported its reference with the default
`--n-epochs 1`, so only **104 of 21 600** forwards were ever compared. exp19 exports with
`--n-epochs 200` (`n_batches=10800`) → the error count covers the **whole round**.

**Calibration was regenerated** (required — thresholds are calibrated *from the pretrained network*).
1 of 12 sites unchanged, and it is exactly the weight-independent one:
`blocks.0.conv.input_quant` 2854.0 → 2854.0 (0.0 %), while e.g. `blocks.2.conv.output_quant`
11.3644 → 18.9531 (**+66.8 %**).

## Controls already verified (do not re-litigate)

* 54 FT windows byte-identical across exp17 (BP) / exp18 (ZO) / exp19 (QZO).
* 180 batch-2 eval windows byte-identical (max abs diff 0.0) between QZO's cached `evX1` and exp17's
  eval fixture.
  → exp17/18/19 differ **only** in optimization method.

## Open work

1. exp18 phases 4–6, then `exp18/Findings.md`.
2. `exp19/Findings.md` (all numbers are in `results/` and `logs/`; nothing left to run).
3. Still outstanding from before: re-point `exp21_BP_faithful` at `artifacts_reference/`, and run
   `exp22` (frozen-BN with everything else the paper's) for the BN ablation.
