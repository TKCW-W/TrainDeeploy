# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (exp16c_SDK_port phase 2): rewrite a plain `1xK` / `Kx1` Conv into the NE16 form, automatically.

Until now the NE16 form of a 1xK convolution -- the per-tap stacked bit-serial weight, the
`ne16_taps` / `weight_offset` / `ne16_weight_preencoded` attributes, and (for a runtime weight) the
`NE16WeightEncode` node -- was authored by hand in the experiment's fixture builder. That is fine
for a single-layer fixture and useless for the real QZO graph, which contains none of it. This pass
does the rewrite, so an ordinary exported network compiles to NE16 with no fixture surgery.

Two ordering constraints decide where it must run, and both were learned the hard way:

1.  **After engine coloring, before the NHWC lowering pass.** `PULPNCHWtoNHWCPass` permutes every
    rank-4 conv input and derives the spatial rank from the *weight's* rank; both are wrong for a
    bit-serial weight. `_NCHWtoNHWC_fun` already skips a weight marked `ne16_weight_preencoded`, but
    only if the mark is there *by then* -- so this pass has to have run first. `NE16Deployer`
    therefore inserts it at index 1 of the lowering pipeline, immediately after the first
    `EngineColoringPass`, rather than appending it with the other NE16 passes (which run after all
    coloring and after NHWC).

2.  **The node must still be a `Conv`, not a `RequantizedConv`.** streamin forces 32-bit output
    (`assert(!(streamin && quantization_bits != 32))`, gvsoc fsm.cpp:50), so the RequantShift has to
    stay a separate cluster node. `PULPConvRequantMergePass` is guarded against merging an
    NE16-coloured 1xK conv -- see `Targets/PULPOpen/TopologyOptimizationPasses/Passes.py`.

Signedness of the input activation is decided by walking the producer chain (§`_inputIsNonNegative`).
NE16 reads activations as UNSIGNED and GAP9's NE16 has no signed-input bit (CONFIG0[28] exists only
in the `ne16v2` model, which no GAP9 chip instantiates -- see ETH/WorkLog/GAP9_SDK_NE16_Corner_Cases.md
section 8). A conv whose input can be negative is therefore LEFT TO THE CLUSTER by this pass rather
than silently miscomputed; see the note in `_ne16_prepare_1xk_fun`.

-- QW
"""

from functools import partial
from typing import Optional

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic
from Deeploy.Targets.NE16.TopologyOptimizationPasses.Passes import _weightEncode

NE16_WEIGHT_OFFSET = -128  # fixed, never values.min() -- see the exp16c Findings
NE16_CIN_SUBTILE = 16

# Ops that cannot make a non-negative tensor negative. Walking back through them looking for a
# ReLU is what tells blocks 1-4 (MaxPool <- Relu) apart from block 0 (straight from the input).
_SIGN_TRANSPARENT = {"Quant", "Dequant", "MaxPool", "AveragePool", "GlobalAveragePool", "Reshape",
                     "Transpose", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Pad"}
_NON_NEGATIVE_OPS = {"Relu", "ReluN"}


def _inputIsNonNegative(tensor: gs.Tensor, maxDepth: int = 8) -> bool:
    """True when the producer chain guarantees values >= 0, so NE16's unsigned read is already right.

    Every `Quant` in the QZO graph declares `signed = 1`, so the *type* says nothing -- the
    distinction is structural. A `Clip` counts only when its lower bound is a constant >= 0.
    """
    cur = tensor
    for _ in range(maxDepth):
        if not cur.inputs:
            return False  # a graph input: no guarantee
        producer = cur.inputs[0]
        if producer.op in _NON_NEGATIVE_OPS:
            return True
        if producer.op == "Clip" and len(producer.inputs) > 1 and isinstance(producer.inputs[1], gs.Constant):
            return bool(np.all(np.asarray(producer.inputs[1].values) >= 0))
        if producer.op not in _SIGN_TRANSPARENT or not producer.inputs:
            return False
        cur = producer.inputs[0]
    return False


def _tapSlice(values: np.ndarray, j: int, tapAxisIsH: bool) -> np.ndarray:
    """Tap j of a (cout, cin, H, W) kernel as its own (cout, cin, 1, 1) pointwise kernel."""
    return values[:, :, j:j + 1, :] if tapAxisIsH else values[:, :, :, j:j + 1]


def _ne16_prepare_1xk_fun(graph: gs.Graph, match: Match, name: str, ne16EngineName: str) -> gs.Graph:
    node = list(match.nodes_map.values())[0]

    if node.attrs.get("engine", None) != ne16EngineName:
        return graph
    if "ne16_taps" in node.attrs:
        return graph  # already prepared -- the pass is idempotent

    ks = node.attrs.get("kernel_shape", None)
    if ks is None or len(ks) != 2:
        return graph
    kh, kw = int(ks[0]), int(ks[1])
    if not ((kh == 1 and kw > 1) or (kw == 1 and kh > 1)):
        return graph
    if int(node.attrs.get("group", 1)) != 1:
        return graph
    if list(node.attrs.get("dilations", [1, 1])) != [1, 1]:
        return graph
    if list(node.attrs.get("strides", [1, 1])) != [1, 1]:
        return graph

    # QW: PADDING IS NOT MODELLED, and getting this wrong is SILENT. NE16's 1x1 mode cannot pad at
    #     all (its guard in gvsoc fsm.cpp:53 is commented out, so a padded 1x1 job produces wrong
    #     results rather than trapping), and the template emits `.padding = 0` with
    #     `input_addr_offset = 0`. Every experiment so far pre-padded the activation in the FIXTURE
    #     DATA; a real graph carries the padding on the node instead. Refuse those and let the
    #     cluster have them.
    #
    #     This is the single biggest remaining gap: SpeechNet blocks 0/1/2 are padded
    #     ([0,2,0,2], [0,8,0,8], [0,4,0,4]) and therefore stay on the cluster, leaving only the two
    #     7x1 blocks on NE16. The GAP9 SDK's own 1-D kernel solves it with pointer arithmetic plus
    #     border-subtile computation (`NE16_ComputeBorders`, CNN_BasicKernels_NE16.c) rather than
    #     with NE16 padding -- that is the shape of the fix, and it belongs in
    #     NE161xKConstraint.serializeTilingSolution. See the exp16c_SDK_port Findings. -- QW
    if any(int(p) != 0 for p in node.attrs.get("pads", [0, 0, 0, 0])):
        node.attrs.pop("engine", None)
        return graph

    # QW: NE16 reads activations as unsigned and GAP9's NE16 has no signed-input bit. Rather than
    #     emit silently wrong results, hand the node back to the cluster. `ConvEngineDiscolorationPass`
    #     (already in the NE16 pipeline) re-decides the colour after this mutation.
    #     Lifting this needs an input-offset op (x -> x+128 as uint8) to pair with the existing
    #     NE16SignedInputBias correction; see the exp16c_SDK_port Findings.
    #     `ne16_unsigned_input=1` on the node is an explicit override for graphs where the producer
    #     chain cannot show what the author knows -- e.g. a single-layer fixture whose activation is
    #     a graph input. (GAP9's only `Relu` binding is fp32, `Bindings.py:381`, so an int8 Relu
    #     cannot simply be spliced in to express it.) In the real QZO graph no override is needed:
    #     the chain really is Quant <- MaxPool <- Relu for blocks 1-4, and Quant <- graph input for
    #     block 0, so the analysis separates them on its own.
    if int(node.attrs.get("ne16_unsigned_input", 0)) != 1 and not _inputIsNonNegative(node.inputs[0]):
        node.attrs.pop("engine", None)
        return graph

    taps = kh * kw
    tapAxisIsH = (kh != 1)
    weight = node.inputs[1]

    if isinstance(weight, gs.Constant):
        # Host-side: restack the K taps into (taps*cout, cinMajor, encBytes), exactly as a
        # standalone 1x1 conv per tap would be encoded. Rank 3 on purpose -- a rank-4 conv input is
        # permuted by the NHWC pass, which would destroy the bit-serial layout.
        values = np.asarray(weight.values)
        if values.ndim != 4:
            return graph
        enc = np.concatenate([
            _weightEncode((_tapSlice(values, j, tapAxisIsH).astype(np.int32) - NE16_WEIGHT_OFFSET).astype(np.uint8),
                          bits = 8) for j in range(taps)
        ],
                             axis = 0)
        weight.values = enc
        weight.name = f"{name}_{weight.name}"
    else:
        # Runtime weight (the QZO case: an RQSPerturbRademacher output, recomputed twice per ZO
        # step). Insert the device-side encoder between producer and conv.
        shape = [int(s) for s in weight.shape]
        if len(shape) != 4:
            return graph
        cout, cin = shape[0], shape[1]
        cinMajor = (cin + NE16_CIN_SUBTILE - 1) // NE16_CIN_SUBTILE
        encShape = [taps * cout, cinMajor, 8 * NE16_CIN_SUBTILE // 8]
        encTensor = gs.Variable(name = f"{node.name}_weight_enc", dtype = np.uint8, shape = encShape)
        encNode = gs.Node(op = "NE16WeightEncode",
                          name = f"{node.name}_wenc",
                          inputs = [weight],
                          outputs = [encTensor],
                          attrs = {
                              "ne16_taps": taps,
                              "ne16_bits": 8
                          })
        graph.nodes.append(encNode)
        node.inputs[1] = encTensor

    node.attrs["ne16_taps"] = taps
    node.attrs["weight_offset"] = NE16_WEIGHT_OFFSET
    node.attrs["ne16_weight_preencoded"] = 1

    # QW: the RequantShift is deliberately NOT merged into this conv (streamin forces int32 out), so
    #     it ends up OUTSIDE the NHWC region `PULPNCHWtoNHWCPass` wraps around the conv: the pass
    #     transposes the conv's int32 output back to NCHW, and the RequantShift consumes that.
    #     GAP9's `default_channels_first` is False, so without saying so explicitly
    #     `RequantShiftLayer.computeShapes` takes the channel count from the LAST axis and the
    #     per-channel mul/add fail to broadcast ("Could not broadcast rqs_mul_tensor from (32,) to
    #     [1, 5]"). NCHW is simply the truth for this node; DeeployTypes.py lets a node attribute
    #     override the default. -- QW
    for consumer in node.outputs[0].outputs:
        if consumer.op in ("RequantShift", "RequantizedConv"):
            consumer.attrs["channels_first"] = 1

    graph.cleanup().toposort()
    return graph


@contextagnostic
class NE16Prepare1xKPass(ReplaceSequentialPatternPass):
    """Rewrites NE16-coloured `1xK` / `Kx1` convolutions into the NE16 per-tap form."""

    def __init__(self, ne16EngineName: str = "NE16"):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(graph, partial(_ne16_prepare_1xk_fun, ne16EngineName = ne16EngineName),
                         "_NE16_PREPARE_1XK_PASS", NonBranchingMatcher(regex_op = True))
