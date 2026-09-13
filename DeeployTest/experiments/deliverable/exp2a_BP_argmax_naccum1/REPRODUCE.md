# exp2a — BP argmax-mask, single step **at n_accum 1** (matched to exp3 ZO)

Re-run of exp2 (BP, MaxPool argmax-mask ON) with `--n-steps 1 --n-accum 1` so the per-step cycle count is
directly comparable to exp3 ZO (which also used n_accum 1). Same fixture as exp2, only `--n-accum` changed.

## Result — PASS
- **`BENCH train_cycles = 32,708,926`** (opt_cycles 56,277) for **1 fwd+bwd** — PASSED, 0 errors
  (`N_TRAIN_STEPS=1 N_ACCUM_STEPS=1`).
- Memory unchanged (static allocation is n_accum-independent): L2 peak = 1,511,308 B (see `results/`).

## Command (traindeeploy)
```bash
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/deliverable_bp_train_argmaxmask \
  --n-steps 1 --n-accum 1 --cores 8 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON --plotMemAlloc
```

## Matched single-step comparison — BP vs ZO at n_accum 1
| | work per step | train_cycles | @370 MHz |
|---|---|--:|--:|
| **exp2a BP argmax-mask** | 1 fwd+bwd | **32,708,926** | 88.4 µs |
| **exp3 ZO** | 2 forwards (±ε) | **35,643,810** | 96.3 µs |

**ZO is ~9% more per step** (+2,934,884 cyc): it spends 2 forwards to estimate the gradient where BP does
1 forward + 1 backward, and here a backward costs *less* than a forward, so 2 forwards slightly exceed one
fwd+bwd. Per-forward: ZO = 17.82 M; BP fwd+bwd = 32.71 M ≈ 1.84 × a ZO forward.
