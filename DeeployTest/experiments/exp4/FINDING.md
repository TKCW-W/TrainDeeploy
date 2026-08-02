# exp4 — BP (backprop) GAP9 Deployment: on-device FT fits GAP9 memory

**Date:** 2026-08-03
**Branch:** TrainDeeploy `feat/BNFRozen_OptionB`
**Question:** does the on-device backprop fine-tuning graph (SpeechNet, full-model + frozen-BN, with the
exp3 MaxPool argmax-mask + transpose-dedup) fit **GAP9**'s memory for deployment?

## GAP9 memory spec (verified)
- **L1 TCDM = 128 KiB = 131,072 B**, shared among the 9 cluster RISC-V cores.
- **L2 = 1.5 MB** (interleaved). L1 is quoted in **KiB** (binary), so L2's "1.5 MB" is most likely
  **1,572,864 B (1.5 MiB)**; the pessimistic decimal reading is 1,500,000 B.
- (Also external L3 / off-chip; not the binding constraint here.)
Sources: GAP9 overview + on-device-training papers (links below).

## Result — IT FITS (even on the pessimistic decimal L2)
Built the exp3 argmax-mask + dedup graph (`speechnet_train_argmask_b1`) on the tiled runner with a **strict
`--l2 1,500,000`** and `--l1 128000` (both ≤ GAP9's actual budgets), same `--memAllocStrategy MiniMalloc
--searchStrategy random-max`:
- ✅ **Compiles + runs on GVSoC, loss bit-exact (Errors 0/16).**
- ✅ **L2 buffer peak = 1,482,636 B → ~17,364 B headroom** under 1,500,000 (more under 1,572,864).
- ✅ **L1 fit** at `--l1 128000`, which is *below* GAP9's actual 131,072 B → fits with a bit more room than
  tested.

So a **feasible tiling within GAP9's L2 (1.5 MB) exists** — the workload is GAP9-deployable memory-wise.
(The CP solver tightens when constrained: the same graph packed to 1,511,308 B under a loose 2 MB budget,
but repacks to 1,482,636 B when limited to 1.5 MB.)

## Why exp3 was the enabler
| build | L2 peak | fits GAP9 1.5 MB? |
|---|--:|:--:|
| recompute baseline (pre-exp3) | 1,793,800 | ❌ |
| argmax-mask, no transpose-dedup | 1,825,356 | ❌ |
| **argmax-mask + dedup (exp3)** | **1,482,636** (@1.5 MB budget) | ✅ (~17 KB spare) |

The recompute baseline and the un-deduped argmax-mask both exceed 1.5 MB. The exp3 argmax-mask +
`MergeSiblingTransposesPass` is what brings it under GAP9's L2.

## Caveats
1. **Headroom is thin (~17 KB @ decimal, ~90 KB @ binary L2).** Little room for graph/model growth. Margin
   levers (deferred, from exp3): dedup the forward `Conv_input_*_transposed` buffers (the current peak
   driver, blocks 2–3), and a real uint8 mask.
2. This is an **allocation-fit test on the tiled-Siracusa runner** (L1/L2 budgets set to GAP9's) — it proves
   a valid tiling exists within GAP9's memory. A true GAP9 build would use the GAP9 platform binding; the
   memory tiling/allocation is the constraint answered here.
3. Static allocation → footprint is independent of `--n-steps`; the 4-step run is representative.
4. Drift caveat carries over from exp3: bit-exactness here is at 16 micro-batches (below the ~step-133
   drift onset); the device-vs-ORT MaxPool argmax-drift is inherent and unaffected.

## Artifacts
`logs/gap9_l2_1p5M.log`, `results/memory_alloc_gap9_1p5M.html`. Depends on exp3 (dedup commit `c6aa5bc`).

## Sources
- GAP9 overview: https://www.emergentmind.com/topics/gap9-microcontroller
- On-Device Training on GAP (memory hierarchy): https://arxiv.org/html/2407.03644v1
- GAP9Shield module (GAP9 L1/L2): https://arxiv.org/html/2407.13706v1
