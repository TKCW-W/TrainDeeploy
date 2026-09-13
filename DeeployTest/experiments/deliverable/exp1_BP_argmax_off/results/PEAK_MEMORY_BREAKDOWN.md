# L2 peak‑memory breakdown — per live tensor (exp1 BP, argmax‑off, single step)

Peak = the busiest step of the schedule, **step 24**, with **1,552.2 KB** live in L2 (arena reserves 1,695 KB).
Source: `memory_alloc_deeployStates.html`, all tensors whose lifetime spans step 24. Roles use the corrected
classifier (a tensor is `backward-grad` only when the `grad` marker is on its own output, `__N_grad`).

## Where we are in the forward pass at the peak
The tensor created exactly at step 24 is **Block‑2's MaxPool** (`blocks_2_3_MaxPool` input‑transpose + pooled
output); Block‑2's ReLU finished at `x[22..24]`. So the peak is **mid‑forward, Block 2 of 5, at the pooling stage**:

```
Block 0: Conv→BN→ReLU→MaxPool   ✓ done, activations STASHED for its backward
Block 1: Conv→BN→ReLU→MaxPool   ✓ done, STASHED
Block 2: Conv→BN→ReLU ✓ … ▶ MaxPool   ← executing now (step 24)
Block 3: not started (only its pre‑transposed conv weight resident)
Block 4: not started (only its pre‑transposed conv weight resident)
```
It is the peak because the two **largest** blocks (0, 1) still hold all their forward activations (needed by
their backward) and we have just added Block 2's — the cumulative early‑block stash is maximal here. Blocks 3–4
add only 14+28 KB (no pooling, small maps), so nothing later exceeds this moment.

---

## Per‑tensor list at peak

### Block 0 — 958.2 KB  (the stash of the first, largest block)
| KB | role | tensor (short) | what it is |
|--:|---|---|---|
| 306.69 | fwd‑transpose | `blocks_0_0_Conv__0 …_transpose_in_var` | **Conv output**, transposed → feeds BatchNorm (fwd) / BatchNormGrad (bwd) |
| 306.69 | fwd‑activation | `blocks_0_1_BatchNormalization__0_tensor` | **BatchNorm output** activation |
| 306.69 | fwd‑transpose | `blocks_0_3_MaxPool_… Relu__0_tensor_transposed` | **ReLU output**, transposed → feeds MaxPool  ← *this is the "ReLU transposed"* |
| 38.06 | fwd‑transpose | `blocks_0_3_MaxPool__0 …_ConvGradW_transpose_in_var` | **MaxPool output** (= Block‑1 Conv input), transposed, stashed for Block‑1 ConvGrad‑W |
| 0.07 | — | 4 tiny buffers (BN running_mean/var, tokens) | frozen BN stats |
| **958.2** | | **= 306.69×3 + 38.06 + 0.07** | ✓ |

### Block 1 — 250.4 KB
| KB | role | tensor (short) | what it is |
|--:|---|---|---|
| 77.00 | fwd‑transpose | `blocks_1_0_Conv__0 …_transpose_in_var` | Conv output transposed → BatchNorm |
| 77.00 | fwd‑activation | `blocks_1_1_BatchNormalization__0_tensor` | BatchNorm output |
| 77.00 | fwd‑transpose | `blocks_1_3_MaxPool_… Relu__0_tensor_transposed` | ReLU output transposed → MaxPool |
| 19.25 | fwd‑transpose | `blocks_1_3_MaxPool__0 …_ConvGradW_transpose_in_var` | MaxPool output transposed, stashed for Block‑2 ConvGrad‑W |
| 0.13 | — | 4 tiny BN stats | |
| **250.4** | | **= 77×3 + 19.25 + 0.13** | ✓ |

### Block 2 — 80.6 KB  (currently computing)
| KB | role | tensor (short) | what it is |
|--:|---|---|---|
| 20.12 | fwd‑transpose | `blocks_2_0_Conv__0 …_transpose_in_var` | Conv output transposed → BatchNorm |
| 20.12 | fwd‑activation | `blocks_2_1_BatchNormalization__0_tensor` | BatchNorm output |
| 20.12 | fwd‑activation | `blocks_2_2_Relu__0_tensor` | **ReLU output** (native — just produced at `x[22..24]`) |
| 20.12 | fwd‑transpose | `blocks_2_3_MaxPool_… Relu__0_tensor_transposed` | ReLU output transposed → MaxPool (just created, `x[24..]`) |
| 0.13 | — | 4 tiny BN stats | |
| **80.6** | | **= 20.12×4 + 0.13** | ✓ |

### Block 3 — 14.3 KB  (not computed yet)
| KB | role | tensor (short) | what it is |
|--:|---|---|---|
| 14.00 | fwd‑transpose | `blocks_3_0_Conv_Conv_input_14_transposed` | Block‑3 **conv weight**, pre‑transposed (persistent) |
| 0.26 | — | 4 tiny BN stats | |
| **14.3** | | **= 14.00 + 0.26** | ✓ |

### Block 4 — 28.3 KB  (not computed yet)
| KB | role | tensor (short) | what it is |
|--:|---|---|---|
| 28.00 | fwd‑transpose | `blocks_4_0_Conv_Conv_input_18_transposed` | Block‑4 **conv weight**, pre‑transposed (persistent) |
| 0.26 | — | 4 tiny BN stats | |
| **28.3** | | **= 28.00 + 0.26** | ✓ |

### Input / weights / other — 220.5 KB
| KB | tensor | what it is |
|--:|---|---|
| 38.28 | `input_0` | EMG input window `[1,1,14,700]` |
| 60.5 (sum) | `input_2..input_23` | 22 trainable **weights** (conv w/b, BN γ/β, fc w/b) |
| 60.5 (sum) | `input_24..input_46` | 22 **gradient‑accumulation** buffers (SUM over n_accum) |
| 60.5 (sum) | `output_5..output_21` | 22 **gradient‑output** buffers (weight‑shaped) |
| ~4.8 | 67 sub‑0.5 KB buffers | biases, BN γ/β, frozen running_mean/var, saved stats, def/token placeholders |
| **220.5** | | **≈ 38.3 + 3×60.5 + 4.8** | ✓ |

Biggest single entries in this band: `input_0`=38.3, `input_18`/`input_40`/`output_17`=28 each,
`input_14`/`input_36`/`output_13`=14 each, `input_6/10/28/32`+`output_9/5`=8 each.

### Grand total
```
Block 0   958.2
Block 1   250.4
Block 2    80.6
Block 3    14.3
Block 4    28.3
In/weights 220.5
--------------- +
TOTAL    1552.2 KB   ✓  (matches the peak-live figure)
```

---

## Q: Is the ReLU‑transposed missing from the breakdown plot?
**No — it is present, just attributed to MaxPool.** For Block 0 it is the **306.69 KB "MaxPool" segment** in
fig 8: the tensor `blocks_0_3_MaxPool_… Relu__0_tensor_transposed` is the **ReLU output transposed to feed the
MaxPool**. Deeploy names/attributes an *input* transpose to the *consuming* node (MaxPool), so in the fig‑8
"MaxPool 345 KB" bar it is `307 (ReLU‑out transposed) + 38 (MaxPool‑out transposed)`. The breakdown is complete
and sums to 958 KB.

Note also: Block 0 has **no native ReLU or native Conv activation** at the peak — those were consumed and freed
early (steps ~8–12). What survives are the **transposed copies + the BN output**, stashed for the backward.
(Block 2, which is computing *now*, is the only block showing both a native ReLU output **and** its transpose.)

If you want the plot to say `ReLU→Pool` instead of `MaxPool` for that 307 KB segment, I can relabel it.

## Q: Does the weight transpose also need L1?
**Yes.** Each conv transposes its weight into an **L1** buffer for the kernel (the conv computes on L1‑resident
data), in addition to any L2 copy. From the L1 map (`input_*_transposed`):

| conv | L1 weight‑transpose buffer | KB in L1 |
|---|---|--:|
| Block 0 | `input_2_transposed` | 0.12 |
| Block 1 | `input_6_transposed` | 8.0 |
| Block 2 | `input_10_transposed` | 8.0 |
| Block 3 | `input_14_transposed` | 14.0 |
| Block 4 | `input_18_transposed` | 28.0 |
| (Block 0 input) | `input_0_transposed` | 38.3 |

These are **transient** — each is live only while its conv runs (they don't co‑exist), so they don't raise the
L2 peak, but they do consume L1 during that conv. Block‑4's weight transpose alone reserves **28 KB of the
128 KB L1**, and the Block‑0 input transpose reserves **38 KB** — i.e. the transpose is *not free* on L1, and on
the tighter L1 budget these are a meaningful fraction. (L1 is the binding resource at 99.8 %.)
