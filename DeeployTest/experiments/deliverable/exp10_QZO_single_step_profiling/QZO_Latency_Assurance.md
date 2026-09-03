# Assuring the QZO latency — how the quantized ZO step went from 10.5× slower to 4.3× faster than float ZO

Date: 2026-09-03 · Companion to `Findings.md` (measurement detail) · All numbers = one ZO loss
pair (2 antithetic forwards + update share), GVSoC/Siracusa 8-core cluster, exp9 fixture.

| stage | Quant/Dequant impl | step per pair | vs float ZO (35.8M cyc) |
|---|---|---|---|
| 1 (as found) | Generic template: single-core + double math | 376.4M | **10.5× slower** |
| 2 (shipped, wired by us) | shipped PULP template: 8-core, still double math | 65.4M | 1.8× slower |
| 3 (our fix) | + fp32 casts on the interpolated constants | **8.37M** | **4.3× faster** |

Bit-exactness was re-validated at every stage (each run PASSES the export-reference loss check;
stage 3: `Errors: 0 out of 8`).

---

## 1 · Why was QZO slower than float ZO? The kernels actually used, and what made them slow

Profiling (`--profileTiling`, `analyze_profile.py`) attributed **97.9%** of the original QZO step
(368.6M of 376.4M cycles) to the ten `Quant` / `Dequant` nodes that sit at each conv's int8↔fp32
boundary — ~**1,400 cycles per element** for what is a multiply-round-clamp. Neither DMA (0.3%)
nor the int8 convolutions (3.1M — already 9.5× *faster* than the float convs) were at fault.

**The kernel that ran** is the *Generic* target's template, wired in by the PULPOpen bindings:

`Deeploy/Targets/Generic/Templates/QuantTemplate.py` (vendored TrainDeeploy/Deeploy — original):

```c
// Quantization (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

    for (uint32_t i=0; i<${size}; i++) {
        // quantization formula
        float32_t input_val = ${data_in}[i];
        float32_t scaled_val = input_val * ${scale};  // Multiply instead of divide
        float32_t shifted_val = scaled_val + ${zero_point};

        // Round to nearest integer
        int32_t quantized = (int32_t)(shifted_val + 0.5f * (shifted_val >= 0 ? 1 : -1));

        // Clamp the value
        if (quantized < ${min_val}) quantized = ${min_val};
        if (quantized > ${max_val}) quantized = ${max_val};

        // Assign directly with explicit cast
        ${data_out}[i] = (${data_out_type.referencedType.typeName})quantized;
    }
END_SINGLE_CORE
```

wired by `Deeploy/Targets/PULPOpen/Bindings.py` (original lines, now kept commented):

```python
from Deeploy.Targets.Generic.Templates import AddTemplate, ConcatTemplate, DequantTemplate, ...
    GatherTemplate, QuantTemplate, ...

BasicQuantBindings = [
    NodeBinding(QuantChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), QuantTemplate.referenceTemplate,
                ForkTransformer),
]
```

Two independent defects make this loop pathologically slow:

**(a) Single-core execution.** `BEGIN_SINGLE_CORE` is
(`TargetLibraries/PULPOpen/inc/DeeployPULPMath.h:18`):

```c
#define BEGIN_SINGLE_CORE if (pi_core_id() == 0) {
```

The binding's `ForkTransformer` forks the code onto all 8 cluster cores — and then this guard
makes 7 of them idle while core 0 walks the whole tensor. Cost: 8×.

**(b) Double-precision soft-float math.** The Mako placeholder `${scale}` renders the Python
float as a bare C literal. In the generated `TrainingNetwork.c` (from the profiled build):

```c
float32_t scaled_val = input_val * 0.04484933426769446; // Multiply instead of divide
```

`0.04484933426769446` has no `f` suffix, so it is a **double** constant; C's promotion rules
then evaluate `float × double` in double precision. The Siracusa cluster FPU is **fp32-only**,
so every such multiply/add compiles to a soft-float library call (`__muldf3`/`__adddf3`,
~100–400 cycles each) instead of a 1-cycle hardware `fmul.s`. Cost: another ~40× on top of the
core count, giving the observed ~1,400 cycles/element. The largest tensor
(block-0 conv output, 8×14×700 = 78,400 elements) alone cost 218M cycles per pair.

---

## 2 · What we took from the shipped reference, how — and why it is better but still 1.8× slower

The shipped Deeploy repo **already delivers implementation-ready parallel templates** that were
never wired into any binding (its own `Bindings.py` imports the Generic ones, same as above):

`ETH/Deeploy/Deeploy/Targets/PULPOpen/Templates/QuantTemplate.py` (shipped — original; Dequant
is analogous):

```c
// Quantization (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});

for (uint32_t i=${nodeName}_chunk_start; i<${nodeName}_chunk_stop; i++) {
    // quantization formula
    float32_t input_val = ${data_in}[i];
    float32_t scaled_val = input_val * ${scale};  // Multiply instead of divide
    ...
```

Instead of the single-core guard, every forked core computes its own `[chunk_start, chunk_stop)`
range from `pi_core_id()` and processes 1/8 of the tensor — a standard PULP parallel elementwise
pattern, drop-in compatible with the existing `ForkTransformer` binding and the tiling flow.

**How we integrated it** (vendored `TrainDeeploy/Deeploy` only; the shipped repo stays
untouched): copied the two shipped template files into
`Deeploy/Targets/PULPOpen/Templates/{Quant,Dequant}Template.py` and switched the bindings —
`Deeploy/Targets/PULPOpen/Bindings.py` (our change, originals kept commented):

```python
from Deeploy.Targets.PULPOpen.Templates import QuantTemplate as PULPQuantTemplate  # -- QW
from Deeploy.Targets.PULPOpen.Templates import DequantTemplate as PULPDequantTemplate  # -- QW

BasicQuantBindings = [
    NodeBinding(QuantChecker([PointerClass(float32_t)], [PointerClass(int8_t)]),
                PULPQuantTemplate.referenceTemplate, ForkTransformer),  # -- QW parallel
    # QW: original single-core Generic mapping (kept for reference):
    # NodeBinding(QuantChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), QuantTemplate.referenceTemplate,
    #             ForkTransformer),
]
```

**Result: 376.4M → 65.4M per pair (÷5.75)** — close to the ideal 8× minus chunking remainder and
the parts of the step that were already parallel.

**Why still 1.8× slower than float ZO:** the shipped template carries the *same* `${scale}` bare
double literal as the Generic one (defect (b) is inherited). Each core now handles ~1/8 of the
elements, but every element still pays the double-precision soft-float calls. The remaining
QCDQ cost, ~57M cycles/pair (~175 cyc/element effective), is almost exactly the stage-1 excess
over the float step.

---

## 3 · What we did on top, and what it unblocks

Our addition — in the **vendored** copies of the PULP templates
(`TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/{Quant,Dequant}Template.py`), each line marked
`-- QW`:

```c
float32_t scaled_val = input_val * ((float32_t)${scale});  // Multiply instead of divide -- QW: fp32 cast (bare literal is double -> soft-float on fp32-only FPU)
float32_t shifted_val = scaled_val + ((float32_t)${zero_point});  // -- QW fp32 cast
```

and in Dequant:

```c
float32_t shifted_val = (float32_t)quantized - ((float32_t)${zero_point});  // -- QW fp32 cast
float32_t dequantized = shifted_val * ((float32_t)${scale});  // -- QW fp32 cast (bare literal is double -> soft-float)
```

**What the cast unblocks:** the constant becomes a compile-time fp32 value, so the whole
expression stays in single precision — and single-precision mul/add **is what the cluster FPU
implements in hardware** (1-cycle `fmul.s`/`fadd.s`). The soft-float library calls disappear
entirely; the loop becomes load → fmul → fadd → round/clamp → store, ~3–4 cycles/element/core.
Measured: **65.4M → 8.37M per pair (÷7.8)**; the Quant/Dequant class collapsed from 368.6M to
**0.54M** (680×) and the step is now conv-dominated (37.7%), as a healthy int8 pipeline should be.

**Why this is correct, not just fast:** the ONNX `Quant`/`Dequant` scale is an fp32 graph
attribute, and the export's host reference computes these ops in fp32 (onnxruntime float32
kernels) — so fp32 device math matches the operator's declared precision *and* the reference
more faithfully than the accidental double math did. The double precision was an artifact of C
literal-promotion rules, not a design decision. Validation: the stage-3 run re-passed the full
export-reference loss comparison (`Errors: 0 out of 8`, test PASSED). Caveat, stated: it is a
deviation from the shipped template text (a latent upstream bug worth reporting), minimal by
construction (casts only), and gated behind the bit-exactness check.

**Net effect:** the QZO step costs 8.37M cycles vs float ZO's 35.8M — **4.3× faster** — turning
the quantized-training efficiency story right-side up, and shrinking the full 2700-step
on-device round from ~4.2T to ~22.6G cycles (multi-day → ~1 h of GVSoC; 17× cheaper than the
float-ZO round's 385G).

---

### Artifact index

| file | content |
|---|---|
| `logs/profiletiling.log.gz` | stage-1 profile (Generic single-core) |
| `logs/profiletiling_parallel.log.gz` | stage-2 profile (shipped 8-core) |
| `logs/profiletiling_parallel_fp32.log.gz` | stage-3 profile (+ fp32 casts) |
| `results/breakdown.json`, `results/breakdown_stage2.json` | per-class cycle attribution |
| `results/per_class_comparison.png`, `results/total_comparison.png` | plots |
| `analyze_profile.py` | log parser / plot generator |
| commit `21b2910` (TrainDeeploy) | the template + binding changes |
