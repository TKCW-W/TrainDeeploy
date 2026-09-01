# exp8 meeting follow-up — (1) L+/L− log ordering, (2) LSB stall vs calibration hypothesis

Two points raised in the 2026-08-28 meeting, each answered below with conceptual + technical evidence.

---

## Point 1 — Why the log prints all L+ then all L− (is the +/− pass order correct?)

**Verdict: the harness is correct and consistent. The "all + then all −" ordering is a *reporting* artifact, not the execution order.**

### Execution order (per mini-batch, interleaved) — `deeploymezotest.c`
Inside the accumulation loop, each mini-batch runs a proper antithetic pair, `+ε` immediately followed by `−ε`:
```c
/* ② +eps pass -> L+ */
perturbation_sign = 1u;                                  // +eps
RunTrainingNetwork();
memcpy(&stored_loss_plus[mb],  &lp_bits, 4);             // store into the PLUS array
/* ③ -eps pass -> L- : same seed_base, flip sign only */
perturbation_sign = 0u;                                  // -eps
RunTrainingNetwork();
memcpy(&stored_loss_minus[mb], &lm_bits, 4);             // store into the MINUS array
/* ④ acc += (L+ - L-) */
```
So on device the sequence is `L+₀, L−₀, L+₁, L−₁, …` — the correct MeZO antithetic pair per window.

### Why the log looks "all + then all −"
The two results go into **two separate arrays** (`stored_loss_plus[]`, `stored_loss_minus[]`, lines 131-132), and the post-training **comparison** checks them as two independent output tensors (lines 478/490):
```c
{ .computed = stored_loss_plus,  .tag = "loss+", ... }   // printed first, i = 0..N-1
{ .computed = stored_loss_minus, .tag = "loss-", ... }   // printed second, i = 0..N-1
```
The printer (line 203) loops each array in full, so you see `[loss+ 0..7]` then `[loss- 0..7]`. This is the **output-verification layout**, decoupled from the interleaved compute. No inconsistency, no fix needed. (If a strictly chronological log is ever wanted, it's a one-line reorder of the two comparison structs into an interleaved print — cosmetic only.)

---

## Point 2 — "The LSB stall should not happen; the issue might be calibration"

Supervisor's suggestion: pretrain → PTQ → **freeze activation scales** → fine-tune in Brevitas, and *measure how many weights move (%) per update*. We carried this out as a controlled PyTorch/Brevitas experiment.

### Experiment design (`brevitas_stall_study/brevitas_stall_experiment.py`)
Build the real Brevitas `QuantSpeechNet`, load the pretrained fold-3 checkpoint, PTQ-calibrate on **real SilentWear** windows, freeze scales, then run MeZO (lr=1e-5, ε=0.01, n_accum=4, seed=42, 100 update steps) in **two regimes that differ in *only one thing*** — everything else (frozen s_w, the z direction each step, the windows, the g_proj formula) is identical:

- **Regime A — master weights** (= standard QAT / what Brevitas fine-tuning does): keep a latent fp32 weight, `W_latent += coeff·z` (unrounded), re-quantize to int8 each step with the frozen scale.
- **Regime B — direct int8** (= our current device update): `w_int8 += round(coeff·z / s_w)`, sub-LSB discarded each step.

Metric = **% of int8 conv-weight elements that change per update step** (his exact ask). All four validation checkpoints passed (B reproduces the device stall for the first steps; A eventually moves; step-0 int8 round-trips; s_w data-free).

### Results

**(a) The weight scale is DATA-FREE — calibration cannot touch it.**
`s_w[c] = max(|W[c]|)/127` computed with vs without activation calibration:
```
max |s_w(no-calib) − s_w(calib)|  =  0.0        (exactly identical, all 5 conv layers + fc)
```
Activation calibration sets *activation* scales; it has **zero** effect on the *weight* grid. So calibration cannot be the cause of a *weight* stall. (Measured s_w ≈ 0.0010–0.0030 per channel — the grid our earlier analysis used, now confirmed.)

**(a2) Could a per-channel OUTLIER be inflating `s_w`? — checked per channel, worst case.**
`s_w = max|W[c]|/127` is outlier-sensitive, so one large weight in a channel would coarsen that channel's grid.
Measured across all **104 output channels** (ratio = `max|w| / p99|w|`; the quantization scale cancels):

| block | elem/channel | median `s_w` | max/p99 (median ch) | max/p99 (**worst ch**) | channels with ratio > 2 |
|---|---|---|---|---|---|
| 0 | 4 | 0.00339 | 1.01 | 1.01 | **0** |
| 1 | 128 | 0.00130 | 1.15 | 1.34 | **0** |
| 2 | 128 | 0.00177 | 1.14 | 1.51 | **0** |
| 3 | 112 | 0.00150 | 1.10 | 1.47 | **0** |
| 4 | 224 | 0.00134 | 1.14 | **1.80** | **0** |

The worst channel in the whole network is inflated only **1.80×**, and **no** channel exceeds 2×. The decisive
test — would outlier-clipping un-stall *any* channel? (step-0 `coeff = 9.4e-5`):
```
current s_w : median = 0.065 LSB, max = 0.095 LSB  ->  channels reaching 0.5 LSB:  0 / 104
p99-clipped : median = 0.077 LSB, max = 0.108 LSB  ->  channels reaching 0.5 LSB:  0 / 104
need 7.7x finer s_w to un-stall the median channel; outlier-clipping offers 1.18x
```
Even the **most favorable channel** with **aggressive p99 clipping** reaches only 0.108 LSB — **4.6× short** of the
0.5 threshold. These weights are near-Gaussian per channel (typical of a small trained CNN; pathological weight
outliers are a transformer phenomenon), so abs-max is already near-optimal and there is nothing to reclaim.

**So the calibration/scale hypothesis fails in all three of its possible forms:**

| form | test | result |
|---|---|---|
| activation calibration inflates the weight grid | `s_w` with vs without calibration | **Δ = 0.0 exactly** (data-free) |
| a per-channel outlier inflates `s_w` | per-channel `max/p99`, worst case | **1.80× worst; 0/104 channels > 2×** |
| a better weight-scale method would fix it | best channel + p99 clipping vs 0.5 LSB | **0.108 vs 0.5 — 4.6× short** |

**(b) The per-step stall is REAL and dominant for direct-int8; master weights move every step.** (lr=1e-5, 100 steps)

| | Regime A (master) | Regime B (direct int8 = device) |
|---|---|---|
| steps with **0%** weights moved | **0 / 100** | **98 / 100** |
| first step any weight moves | step 0 | step 15 |
| cumulative % moved @100 steps | 71.7% | 61.1% |
| how it moves | smooth every step (fp32 latent accumulates) | only on 2 gradient **spikes** |
| fp32 latent drift @100 | grows steadily to 4.76 LSB | n/a (no latent) |

> **Methodology correction (2026-09-01).** The first version of this experiment pre-quantized the weights with
> a local `fake_quant()` and injected the result, intending to bypass Brevitas' scale re-tracking. That was
> **not** a no-op: Brevitas re-derived a slightly finer scale from the injected tensor and re-rounded, shifting
> some elements by 1 LSB (`brevitas_evidence/idem.py` reproduces it). The script now does it properly — it
> pins `weight_quant.tensor_quant.scaling_impl` to a constant (`ConstScale`) and hands Brevitas the **latent**
> weight, so **Brevitas itself performs the quantization** with a provably frozen scale. A new freeze gate
> asserts this: `max scale drift = 0.0`, `max |brevitas_quant_weight − frozen-scale reference| = 0.0`.
> Numbers above are from the corrected run (Regime B cumulative moved 55.1% → 61.1%); **every conclusion is
> unchanged** — A moves on 100/100 steps, B stalls on 98/100.

**(c) Direct-int8's movement is NOISE, not descent.** Regime B's entire movement comes from **2 of 100 steps**:
```
step 15:  g_proj = 67  → 44.2% of weights flip at once
step 42:  g_proj = 82  → 77.5% of weights flip at once
(median |g_proj| = 8;  LSB-cross for the finest channel needs |g_proj| > 49)
```
On the 98 "normal" steps (|g_proj|≈8, `coeff/s_w ≈ 0.06 LSB`) it moves **nothing** — exactly the exp8 device result. The only movement is 2 rare noisy directions with a freakishly steep `|L+−L−|` that cross the LSB threshold for *most channels simultaneously*. That is the worst kind of update: stall punctuated by giant noise jumps, not gradient descent. Master weights, using the *same* g_proj, instead integrate the small consistent signal on every step.

**(d) The stall is about update-magnitude-vs-grid (lr sweep), independent of calibration.**

| lr | A (master) cum% | B (direct) cum% |
|---|---|---|
| **1e-5** | 58.7% | 44.2% (2 spikes only) |
| 1e-4 | 95.1% | 95.2% |
| 1e-3 | 99.1% | 99.3% |

At lr≥1e-4 the per-step update itself exceeds ½ LSB, so even direct-int8 moves smoothly — but ZO at those learning rates diverges (empirically lr≥1e-3 → NaN). The stall is the collision of *ZO needs a tiny lr* × *coarse weight grid*; nothing here depends on calibration.

### Verdict on the supervisor's statement

| claim | verdict | evidence |
|---|---|---|
| "issue might be **calibration**" | **Refuted** | `s_w` is data-free (Δ=0.0); activation calibration cannot change the weight grid or the stall. Also: our calibration is already correct (loss ≈ real, not the random-calib 9.07 regime). |
| "the LSB stall **should not happen**" | **Right — *for master-weight training*** | In standard QAT / Brevitas fine-tuning (Regime A) weights move on every step and never stall. His suggested Brevitas flow *is* the master-weight scheme — so it demonstrates the **fix**, not a calibration bug. |

### Precise statement of the Brevitas / master-weight claim (please use this wording)

Brevitas **performs no weight update at all** — it never writes `self.weight`; it is a forward-time quantizer
that reads the weight and returns a transient value (`nn/mixin/parameter.py:49`). What is true is narrower:

> **Brevitas' parameter interface is fp32-only**, so any training loop whose trainable state is Brevitas' own
> parameters necessarily accumulates in fp32 — master-weight training, without the author choosing it.
> Evidence: the module holds **zero int8 tensors** (params all `float32`); `torch.optim.SGD(m.parameters())`
> captures `conv.weight` itself; and assigning an int8 tensor raises
> `RuntimeError: data set to a tensor that requires gradients must be floating point`.
> The fp32 accumulation happens in **`torch/optim/sgd.py:366` `param.add_(grad, alpha=-lr)`** — outside Brevitas.

**Therefore the stall is decided by the update implementation, not by Brevitas.** In our experiment the two
regimes differ by exactly two lines (`:302` fp32 accumulate vs `:307` round-to-grid) on the *same* model with
the *same* frozen scales. On device we store only int8 (a choice made in `build_int8_forward`), so there is no
fp32 to accumulate into. "Master weights" is therefore **not a bug fix but the other storage design** — paying
~60 KB of L2 for what fake-quant gets for free.

*(Avoid saying "Brevitas is intrinsically master-weight" — he can correctly reply that he never implemented
master weights. Full runnable evidence + Brevitas source extracts: `brevitas_evidence/`.)*
| run the Brevitas experiment, measure % moving | **Exactly the right diagnostic** | It isolates master-vs-direct as *the* deciding variable and points straight at master weights. |

**Bottom line:** the stall is caused by our **direct-int8 update discarding the sub-LSB gradient**, not by calibration. The remedy is **master weights** (latent fp32 that accumulates the sub-LSB updates and re-quantizes with the *frozen* scale) — which is precisely what the supervisor's Brevitas fine-tuning does. His intuition ("weights should move") is correct *because* he was implicitly assuming the master-weight regime; his calibration guess is not the mechanism.

### Honest correction to our earlier framing
Our exp8 note called the stall a "hard floor." That was **too strong**: rare large-gradient (noise) steps *do* cross the LSB and flip weights (2/100 here), so over a very long run the direct-int8 weights are not *permanently* frozen. The accurate statement is: **direct-int8 stalls on ~98% of steps and moves only via occasional noise spikes — it does not perform proper gradient descent on the conv weights; master weights do.** The exp8 device result (0/14880 after 2 steps) is the *typical* per-step behavior, not a 2-step artifact.

### Artifacts
`brevitas_stall_study/`: `brevitas_stall_experiment.py` (reproducible; runs in the `agitated_hugle` container),
`results.json` (per-step arrays, lr sweep, s_w, validation flags), `cumulative_pct_moved.png` (A vs B staircase),
`run.log.gz`. Source experiment dir: `Onnx4Deeploy/QZO_exp/exp_brevitas_stall/`.

### Recommended next step
Build the **master-weight** variant (fp32 shadow of the int8 conv weights, ~52 KB L2, `round(W_master/s_w)` into the int8 buffer each update — a `Quant` op in the update graph) and re-run exp8-style: expect the conv weights to move smoothly like Regime A. Then the QZO accuracy round can use it. This is the concrete follow-up the experiment justifies.
