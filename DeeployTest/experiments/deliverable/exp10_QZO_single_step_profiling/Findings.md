# exp10 — QZO single-step latency profiling — Findings

Date: **2026-09-03** · GVSoC/Siracusa, 8 cores · fixture = exp9 (pooled@99.99 baked, lr 1e-5)
Raw: `logs/profiletiling.log` (614 profiled node sections, run PASSED), `results/breakdown.json`,
plots `results/{per_class_comparison,total_comparison}.png` · Parser: `analyze_profile.py`
Float-ZO counterpart: `../exp6_ZO_single_step_latency/logs/profiletiling.log`

## Question

exp6 (float ZO) vs exp7 (QZO) single-step logs show the quantized step costs **10.8×** more
cycles (386.8M vs 35.7M per loss pair). Where does the time go — compute or DMA, and which
operators?

## Method

`deeployMezoRunner_tiled_siracusa.py --profileTiling` on the exp9 QZO fixture (1 update step,
n_accum 4 → normalized to ONE loss pair = 2 forwards to match exp6's n_accum 1). Per tile the
profiler reports Pre-Kernel / Kernel / Post-Kernel cycles: Kernel = compute, Pre+Post = tiling +
DMA. Sanity: profiled QZO total 376.4M/pair vs unprofiled exp7 386.8M — within 3%, attribution
trustworthy.

## Result — per loss pair (2 forwards + update share)

| class | float ZO (exp6) | QZO (exp10) |
|---|---|---|
| **Quant/Dequant (QCDQ boundaries)** | — | **368.6M (97.9%)** |
| Conv | 29.4M (82.2%) | **3.1M (0.8%)** |
| MaxPool | 1.8M | 1.8M |
| BatchNorm | 1.4M | 1.4M |
| Transpose | 2.2M | 0.9M |
| ReLU | 0.5M | 0.5M |
| Perturb / Loss / FC / GAP / other | 0.5M | 0.1M |
| **total** | **35.8M** | **376.4M** |
| compute vs DMA/tiling | 95.8% / 4.2% | 99.7% / 0.3% |

Top offending nodes (QZO, per pair):

| node | cycles | tensor | ≈cyc/element |
|---|---|---|---|
| QCDQ_blocks_0_conv_output_dequantMul_Dequant | 218.2M | (8,14,700)=78,400 ×2 fwd | **~1,390** |
| QCDQ_blocks_1_conv_output_dequantMul_Dequant | 51.8M | (16,14,87) ×2 | ~1,330 |
| QCDQ_blocks_1_conv_input_quant_Clip_Quant | 29.2M | | ~1,400 |
| QCDQ_blocks_0_conv_input_quant_Clip_Quant | 28.7M | (1,14,700)=9,800 ×2 | ~1,460 |
| blocks_2..4 Quant/Dequant | 39.8M | | similar |

## Conclusions

1. **The user-observed 10.8× slowdown is 97.9% attributable to the Quant/Dequant boundary ops**
   (the QCDQ Clip-Quant / dequant-Mul nodes at each conv's int8↔fp32 edge). It is neither DMA
   (0.3% overhead) nor the quantized compute.
2. **The int8 convolutions are 9.5× FASTER than the float convolutions** (3.1M vs 29.4M/pair) —
   quantization delivers exactly the speedup it promises; the datapath design is sound.
3. **~1,400 cycles/element for an elementwise mul+round+clip is pathological** (a sane 8-core
   PULP implementation runs ~1–4 cyc/element). These nodes are hitting a naive/scalar fallback
   kernel (likely single-core, per-element float call overhead), i.e. a **kernel-mapping /
   implementation problem, not a structural cost of quantization**. The BN-unfolded design does
   force int8→fp32→int8 at every block, but at proper elementwise speed those boundaries would
   cost ~0.5M cycles total, not 368M.
4. **Upside if fixed**: QZO pair without the QCDQ pathology ≈ 7.8M cycles vs float's 35.8M —
   the quantized ZO step would be **~4.6× faster** than float ZO, flipping the current 10.8×
   penalty into the speedup the QZO story is supposed to deliver. Fix = give Quant/Dequant real
   parallel PULP kernels (or fold them into the adjacent RequantShift/Conv), a Deeploy
   binding/kernel task.
5. Practical implication for exp9: the full 2700-step device round at current speed costs ~11×
   the float round (multi-day GVSoC). Options: fix the kernels first (correct long-term move),
   or run the long round as-is for the accuracy datapoint.

## THE FIX (added same day) — shipped parallel templates + fp32 casts: QZO now 4.3× FASTER than float

The shipped Deeploy reference **already delivers** 8-core parallel Quant/Dequant templates
(`ETH/Deeploy/Deeploy/Targets/PULPOpen/Templates/{Quant,Dequant}Template.py`, per-core chunking
via `pi_core_id()`), but its own `Bindings.py` — mirrored by our vendored copy — wires the
**Generic single-core** templates instead (`BEGIN_SINGLE_CORE`, 7 cores idle). Additionally both
templates interpolate the scale as a bare double literal (`* 0.04484...` without `f`), promoting
every element to double-precision **soft-float** on the fp32-only cluster FPU.

Fix in the vendored `TrainDeeploy/Deeploy` (shipped repo untouched):
1. Copied the shipped parallel templates into `Targets/PULPOpen/Templates/`.
2. Switched `BasicQuantBindings`/`BasicDequantBindings` to them (originals kept commented).
3. Added explicit `(float32_t)` casts on scale/zero_point in the (vendored) templates.

Staged measurement (profiled single step, 4 pairs; bit-exactness re-validated at every stage —
all runs PASS against the export reference, stage 2 `Errors: 0/8`):

| stage | step (4 pairs) | per pair | vs float ZO (35.8M) | attribution |
|---|---|---|---|---|
| 0 · Generic single-core + double | 1505.5M | 376.4M | 10.5× slower | baseline |
| 1 · shipped 8-core templates | 261.8M | 65.4M | 1.8× slower | ÷5.75 from parallelization |
| 2 · + fp32 casts | **33.5M** | **8.37M** | **4.3× FASTER** | ÷7.8 from killing soft-double |

Fixed per-class profile (per pair, `results/breakdown_stage2.json`; classifier corrected
2026-09-04 — the QZO perturb nodes `rqsp_*`/`pert_*` were previously absorbed into Conv/Other):
Conv 2.54M (30.2%), MaxPool 1.82M (21.6%), BatchNorm 1.41M (16.8%), Transpose 0.90M (10.8%),
**Perturb 0.68M (8.1%)**, **Quant/Dequant 0.54M (6.5%** — down 680×**)**, ReLU 0.49M. The step
is conv-dominated as a healthy int8 pipeline should be — and with the reclassification the true
int8-conv advantage is **29.4M → 2.5M = 11.6× faster convs** than float. Exact-config
comparison (both n_steps=1 n_accum=1, exp10a): QZO 8.62M vs float 35.82M = **4.15×**.

Consequence: the exp9 full round-1 (2700 steps) drops from ~4.2T to ~22.6G cycles — from
multi-day to ~1 h of GVSoC, and 17× cheaper than the float-ZO round (385G).

## Reproduction

```bash
docker exec traindeeploy bash -lc 'cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
  python3 deeployMezoRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo9_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo9_update \
  --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --profileTiling \
  > experiments/deliverable/exp10_QZO_single_step_profiling/logs/profiletiling.log 2>&1'
# then: python3 analyze_profile.py   (in agitated_hugle for the plots)
```
