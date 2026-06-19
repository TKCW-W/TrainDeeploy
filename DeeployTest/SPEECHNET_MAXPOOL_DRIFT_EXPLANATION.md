# Why MaxPool on-device training drifts from the reference — detailed mechanism

After the four correctness fixes, on-device MaxPool fine-tuning is **bit-exact (~1e-6) for the
first tens of steps, then the per-step loss drifts** from the ORT reference into a bounded band
(e.g. 90-step run: 52/90 over TOL=0.001, max ~0.06; 2-sample run: 9/30, drift onset ~step 21).
The model still learns correctly. This note explains *why*, with measurements.

## The drift is trajectory divergence, not a per-step error

The on-device path (8-core PULP fp32) and the reference (ORT fp32) are two different fp32
implementations of the same SGD. Floating-point addition is non-associative, so they sum in
different orders and disagree each step by ~1e-6 (reduction-order rounding) — unavoidable on
parallel fp32 hardware. The two weight trajectories then slowly separate; the loss-landscape
curvature (fine-tuning near a minimum) bounds the separation, so it **plateaus, not blows up**.

That the per-step *computation* is correct is proven by the bit-exact opening segment.

## The amplifier is MaxPool's discrete argmax — directly demonstrated

Running two identical fp32 trainings differing only by a ~1e-6 per-step perturbation, and counting
pooling-window argmax disagreements vs the loss difference:

```
step | loss_diff | argmax flips
  5  | 0.000008  |   0     ← bit-exact, no flips
  9  | 0.000010  |   1     ← FIRST flip
 10  | 0.000143  |   8     ← loss diff jumps
 27  | 0.007336  | 104
>>> first argmax flip at step 9 ; first loss-diff>1e-3 at step 12
```

The loss divergence begins right after the first argmax flips and grows with the flip count.

## Why argmax flips happen (the heart of it)

A flip needs the window's **top two values within ~1e-6** of each other. But measuring the actual
gap across ~284k windows: **~0%** are that close at any frozen instant (median gap 0.20). So flips
are **not** static coincidences.

They are **dynamic crossings**. Each SGD step moves the weights by `lr·grad ≈ 1e-3` — a thousand
times larger than the fp noise — so window values, and the top1–top2 gap, evolve. When the
runner-up overtakes the leader, the gap passes through zero. Measured rate:

```
~1.0–1.5% of positive-max windows change their argmax per epoch (≈2,500–3,700 of ~255,000)
```

At each such crossing the window is transiently a near-tie, and the ~1e-6 implementation
difference decides which side of zero each trajectory lands on → they pick different argmax →
the gradient routes differently. Why so many crossings here: the MaxPool input is a smoothed
convolution of a continuous, band-limited EMG signal, so within an 8-sample (16 ms) window
several positions have comparable values; *which* is the single max is a soft choice a small
weight nudge can swap.

Early on the trajectories are ~1e-6 apart so they cross at nearly the same step and rarely
disagree; each disagreement nudges the weights apart, so later crossings disagree more often — a
self-reinforcing growth that saturates against the loss curvature.

**Corroboration (2-sample run):** with only 2 samples cycled 15×, the weights overfit faster, so
features evolve faster, so the drift onset is *earlier* (step ~21 vs ~36 for 18 samples) — exactly
as the mechanism predicts (onset is governed by how fast features move).

## Why precision can't fix it

- It is **not** a reduction-precision problem: if it were, the loss would differ from step 1 and
  grow smoothly. Instead it is bit-exact then a sudden onset — the reductions already match to 1e-6.
- Higher precision only **delays** the first flip (a window must drift closer to a crossing before
  even smaller noise flips it); it cannot remove flips, because training itself manufactures the
  crossings. A precision sweep confirmed precision delays the onset strongly for AvgPool but only
  ~3–4× as much for MaxPool (its discreteness).
- fp64 accumulation is also **non-viable** on this fp32 core: `double` is soft-float — a test run
  spent 42 min without completing one step (~10–50× slowdown). (That change was reverted.)
- The only way to zero flips is **bit-identical forward arithmetic** between device and reference
  (same kernels / reduction order) so every crossing breaks the same way — a validation-methodology
  alignment, not a numerical-accuracy change.

## Why it does not matter in practice

- The per-step gradient is correct; both trajectories are valid SGD paths differing only in
  **arbitrary tie-breaking at windows where two maxima are essentially equal** (the "right" choice
  is genuinely ambiguous). Neither is wrong; the model trains correctly (loss decreases).
- The drift is bounded (~3% of the loss).
- Real on-device fine-tuning is short (a few steps to adapt) → squarely in the bit-exact regime.
- The "errors" are failures to bit-match one particular fp32 reference trajectory over many steps,
  not training errors.

**One line:** fine-tuning continuously slides ~1% of pooling windows per epoch through an argmax
crossing, and at each crossing the unavoidable ~1e-6 difference between two fp32 implementations
sends the gradient to a different position — a discrete, training-driven event that precision can
delay but not eliminate, while leaving the actual training correct.
