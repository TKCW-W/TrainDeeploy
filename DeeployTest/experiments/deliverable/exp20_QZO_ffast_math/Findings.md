# exp20 — Findings: `-ffast-math` off costs **+3.84 %**

**Run:** 2026-09-16 · **Branch:** `feat/GAP9_w_NE16` · **Plan:** [`Plan.md`](./Plan.md)
**Verdict:** affordable. Strict fp32 is the right default for the QZO training graph.

---

## 1. Headline

| | ffast (default) | strict fp32 | delta | overhead |
|---|---:|---:|---:|---:|
| train cycles | 8,364,474 | 8,697,847 | +333,373 | **+3.99 %** |
| opt cycles | 330,760 | 331,516 | +756 | +0.23 % |
| **total** | **8,695,234** | **9,029,363** | **+334,129** | **+3.84 %** |

At 370 MHz that is **23.50 ms → 24.40 ms** per update step (one accumulation, i.e. 2 forwards +
1 optimizer pass). Scaled to a full round-1 QZO fine-tune — 540 update steps × 4 accumulations —
**4.2 min → 4.4 min** of device time. Both arms reported `Errors: 0 out of 2`.

Bit-exactness was never the question here (a *single* step is bit-exact under either build; the
divergence `exp13`/`exp14` chased only accumulates over thousands of steps). The question was the
price of the flags that keep it exact for a whole round, and the answer is under four percent.

## 2. Where the overhead lands

Per-kernel cycles, `|delta| >= 500`:

| kernel | ffast | strict | delta | % |
|---|---:|---:|---:|---:|
| `bnBatchNormalization` | 1,109,754 | 1,423,434 | **+313,680** | **+28.3 %** |
| `bn_1BatchNormalization` | 56,649 | 73,967 | +17,318 | +30.6 % |
| `bn_2BatchNormalization` | 12,732 | 17,761 | +5,029 | +39.5 % |
| `QCDQ_blocks_0_conv_input_quant_1Clip_Quant` | 73,492 | 76,487 | +2,995 | +4.1 % |
| `QCDQ_blocks_1_conv_input_quant_1Clip_Quant` | 73,931 | 76,408 | +2,477 | +3.4 % |
| `QCDQ_blocks_2_conv_input_quant_1Clip_Quant` | 37,859 | 39,230 | +1,371 | +3.6 % |
| `wrappedInnerForwardImpl_5Gemm` | 7,407 | 8,617 | +1,210 | +16.3 % |
| `bn_3BatchNormalization` | 6,660 | 7,861 | +1,201 | +18.0 % |
| `QCDQ_blocks_4_conv_input_quant_1Clip_Quant` | 10,414 | 11,060 | +646 | +6.2 % |
| `_MERGE_CONVRQ_PASS_4` | 60,528 | 61,139 | +611 | +1.0 % |
| `_MERGE_CONVRQ_PASS_2` | 224,728 | 225,324 | +596 | +0.3 % |
| `poolMaxPool` | 1,543,552 | 1,523,064 | **−20,488** | −1.3 % |

Two things to read out of this table.

**The cost is concentrated, and exactly where the theory says.** The five `BatchNorm` kernels
account for **+338,228** cycles — more than the whole net delta. `bnBatchNormalization` alone is
94 % of it. That is the one kernel whose inner loop is `(x − mean) * inv_std * gamma + beta`, i.e.
two back-to-back multiply-adds per element; forbidding `fmadd.s` turns each into a separate `fmul.s`
+ `fadd.s`, and the arithmetic is memory-light enough that the extra instruction is not hidden.
The +28 % is the FMA contraction ratio showing up almost undiluted.

**`poolMaxPool` moved, and it should not have.** It is a pure int8 comparison kernel with no
floating-point in it at all; `-fno-fast-math` cannot change its semantics. It nevertheless got
1.3 % *faster*. Of the 96 profiled kernels, **zero** had identical cycle counts between the two
runs. So there is a floor of a few tenths of a percent of pure code-layout noise — different
inlining, different I-cache alignment — riding on every number in this table. The BatchNorm
figures are far above that floor and are real; a sub-1 % entry like `_MERGE_CONVRQ_PASS_2` is
not distinguishable from it.

Net of the BatchNorm block and the Gemm, the rest of the graph is ~flat, which is the expected
result: `DEEPLOY_STRICT_FP32_FILES` only touched `BatchNorm.c` and `Gemm.c`, and the generated
`TrainingNetwork.c` / `OptimizerNetwork.c` are mostly orchestration.

## 3. Is it affordable?

Yes, and the framing matters. Three reference points from earlier experiments:

* QZO round 1 on device (`exp19`) is **89.51 G cycles**. +3.84 % is ~3.4 G cycles — under 10 s of
  device time on a run measured in minutes, and a rounding error next to the 385 G cycles ZO
  (`exp18`) spends.
* The alternative to paying it is `9,949 / 21,600` mismatched training losses (`exp19` again,
  built with default ffast-math), with divergence onset at mini-batch 5127. Strict fp32 buys
  `0 / 21,600`.
* It is not a *deployment* cost in the usual sense: inference is unaffected unless you build it
  with the same flags. This is the price of being able to *validate* on-device training against a
  host reference.

There is also headroom left on the table if it ever stops being affordable: the fix is
file-granular, so `DEEPLOY_STRICT_FP32_FILES=BatchNorm.c` alone would recover the Gemm's 1,210
cycles — and `exp14`'s fusion counts (13 in `BatchNorm.c`, 154 in `Gemm.c`) say `Gemm.c` is the
one carrying the risk, so that trade is the wrong way round. Keep both.

## 4. Reproduction

Both arms run in `deeploy_arm_mounted`, on exp19's existing fixture. **Kill stale simulators
first** — orphaned `gvsoc_launcher` processes starve new runs:

```bash
docker exec deeploy_arm_mounted bash -lc \
  'pgrep -f "[g]vsoc_launcher" | xargs -r kill -9'
```

The bracket in `[g]vsoc_launcher` is required: without it `pgrep -f` matches its own command line
and the pipeline kills its shell (exit 137, empty log).

**Arm A — default (`-ffast-math` on):**

```bash
docker exec -e PYTEST_XDIST_WORKER=ffast deeploy_arm_mounted bash -lc '
cd /app/ETH/TrainDeeploy/DeeployTest &&
python testRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo19_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo19_update \
  --cores 8 --l1 128000 --l2 2000000 --defaultMemLevel L2 \
  --memAllocStrategy MiniMalloc --searchStrategy random-max \
  --profileTiling --n-steps 1 --n-accum 1 \
  -D BN_FROZEN_STATS=ON' 2>&1 | tee ffast_default/logs/run.log
```

**Arm B — strict fp32:** identical, with `PYTEST_XDIST_WORKER=strict` and

```bash
  -D BN_FROZEN_STATS=ON DEEPLOY_STRICT_FP32=ON 'DEEPLOY_STRICT_FP32_FILES=BatchNorm.c;Gemm.c'
```

Confirm the flags actually reached the compiler — the build prints them:

```
-- [QW strict-fp32] BatchNorm.c: -fno-fast-math -ffp-contract=off
-- [QW strict-fp32] Gemm.c: -fno-fast-math -ffp-contract=off
-- [QW strict-fp32] TrainingNetwork.c / OptimizerNetwork.c: -fno-fast-math -ffp-contract=off
```

(`strict_fp32/logs/run.log.gz`, lines 54-63.) Without this check a silently-ignored flag looks exactly like
"strict fp32 is free".

**Two things that will cost you a run if you skip them:**

1. **Separate build directories.** `PYTEST_XDIST_WORKER` selects `TEST_SIRACUSA/build_<worker>`
   (`testUtils/deeployRunner.py:219`). Sharing one directory lets arm B reuse arm A's objects,
   and the whole experiment measures nothing.
2. **Never `rm -rf TEST_SIRACUSA` while another run is live**, and never kill gvsoc by a bare
   pattern. Scope both to the worker. A machine-wide cleanup killed a live 7-hour ZO run earlier
   in this project.

**Artifacts:**

| file | contents |
|---|---|
| `ffast_default/logs/run.log.gz` | arm A, full build + simulation |
| `strict_fp32/logs/run.log.gz` | arm B, ditto (flags at lines 54–63) |
| `results/summary.txt` | the two tables above, as generated |
| `results/profile.json` | all 96 kernels, both arms, machine-readable |

## 5. Related

* `exp13` / `exp14` — why the flags exist: clang FMA + reassociation in `BatchNorm.c` / `Gemm.c`
  under `-ffast-math`, micro-probed fusion counts, and the full round at `0 / 21600`.
* `exp19` — QZO round 1 on `artifacts_reference`, built *without* the flags: 85.00 %,
  9,949 / 21,600 breaches.
* `cmake/pulp/pulp.cmake:12` — where the global `-ffast-math` is set.
