# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the ZO_Conv prototype fixture -- QW (dual-pass-fusion, step 1).
#
# ZO_Conv is SpeechNet block 0's convolution lifted VERBATIM out of the float-ZO training
# graph, together with the two PerturbRademacher nodes that feed it:
#
#     blocks_0_0_weight --PerturbRademacher(idx=0)--> 0_blocks_0_0_weight --.
#     blocks_0_0_bias   --PerturbRademacher(idx=1)--> 1_blocks_0_0_bias ----+--> Conv --> out
#     input ---------------------------------------------------------------'
#
# Nothing is re-authored: node names, `idx`/`seed`/`eps` attributes and the Conv attributes
# are copied from the source graph, so Deeploy parses, binds and tiles these three nodes
# exactly as it does inside the full ZO network -- including the z-stream, which is keyed
# on the `idx` ATTRIBUTE rather than on schedule position.
#
# Single step, one window, n_accum = 1: the npz carries exactly one input window and no
# mb<N>_ mini-batch variants.
#
# Usage (from DeeployTest/):
#     python3 experiments/dual_pass_fusion/build_zo_conv_fixture.py

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zo_host_ref import conv2d_im2col_hwc, perturb_rademacher  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLOY_TEST = os.path.abspath(os.path.join(HERE, "..", ".."))

# artifacts_reference checkpoint (the paper-matching one). The non-_ref fixture carries the
# superseded artifacts/ weights -- see memory note "SilentWear reference checkpoints".
DEFAULT_SRC = os.path.join(DEEPLOY_TEST,
                           "Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3_ref")
DEFAULT_DST = os.path.join(DEEPLOY_TEST, "Tests/Kernels/FP32/ZO_Conv")

# The three source nodes, in schedule order.
W_PERTURB = "perturbrademacher_0_blocks_0_0_weight"
B_PERTURB = "perturbrademacher_1_blocks_0_0_bias"
CONV = "node_1_blocks_0_blocks_0_0_Conv"

# Graph inputs of the full ZO network, in declaration order -> their npz slot.
# arr_0000 is the window, arr_0001 the label (dropped: ZO_Conv has no loss), arr_0002/3 the
# block-0 conv weight/bias.
SRC_WINDOW = "arr_0000"
SRC_WEIGHT = "arr_0002"
SRC_BIAS = "arr_0003"


def main() -> None:
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default = DEFAULT_SRC, help = "source ZO training fixture dir")
    ap.add_argument("--dst", default = DEFAULT_DST, help = "output fixture dir")
    ap.add_argument("--window", type = int, default = 0,
                    help = "which mini-batch window to bake in (0 = the base arr_* entry)")
    args = ap.parse_args()

    src_model = onnx.load(os.path.join(args.src, "network.onnx"))
    src_graph = src_model.graph
    src_npz = np.load(os.path.join(args.src, "inputs.npz"))

    nodes_by_name = {n.name: n for n in src_graph.node}
    for name in (W_PERTURB, B_PERTURB, CONV):
        assert name in nodes_by_name, f"{name} not in {args.src}/network.onnx"
    nodes = [nodes_by_name[n] for n in (W_PERTURB, B_PERTURB, CONV)]

    # --- graph interface ------------------------------------------------------------
    # Reuse the source ValueInfoProtos verbatim so dtype/shape are whatever the ZO graph
    # declares, not something we invent.
    src_inputs = {vi.name: vi for vi in src_graph.input}
    graph_inputs = [src_inputs[n] for n in ("input", "blocks_0_0_weight", "blocks_0_0_bias")]

    conv_out_name = nodes_by_name[CONV].output[0]
    x_shape = [d.dim_value for d in src_inputs["input"].type.tensor_type.shape.dim]
    w_shape = [d.dim_value for d in src_inputs["blocks_0_0_weight"].type.tensor_type.shape.dim]
    conv_attrs = {a.name: onnx.helper.get_attribute_value(a) for a in nodes_by_name[CONV].attribute}
    pads, strides = conv_attrs["pads"], conv_attrs["strides"]
    h_out = (x_shape[2] + pads[0] + pads[2] - w_shape[2]) // strides[0] + 1
    w_out = (x_shape[3] + pads[1] + pads[3] - w_shape[3]) // strides[1] + 1
    out_shape = [x_shape[0], w_shape[0], h_out, w_out]
    graph_output = helper.make_tensor_value_info(conv_out_name, onnx.TensorProto.FLOAT, out_shape)

    # The perturb outputs must keep their ValueInfo: ZO puts the conv weight on a graph EDGE
    # rather than in an initializer, and Deeploy's NCHW->NHWC lowering reads
    # `node.inputs[1].shape` to count spatial dims
    # (Deeploy/.../LoweringOptimizationPasses.py:267). Drop the value_info and that pass
    # crashes on a shapeless Variable.
    src_value_info = {vi.name: vi for vi in src_graph.value_info}
    value_info = [src_value_info[n.output[0]] for n in nodes[:2]]

    graph = helper.make_graph(nodes, "ZO_Conv", graph_inputs, [graph_output],
                              initializer = [], value_info = value_info)
    model = helper.make_model(graph, opset_imports = list(src_model.opset_import))
    model.ir_version = src_model.ir_version
    model.producer_name = "build_zo_conv_fixture.py"

    # --- tensors --------------------------------------------------------------------
    win_key = SRC_WINDOW if args.window == 0 else f"mb{args.window}_{SRC_WINDOW}"
    x = src_npz[win_key].astype(np.float32)
    w = src_npz[SRC_WEIGHT].astype(np.float32)
    b = src_npz[SRC_BIAS].astype(np.float32)
    assert list(x.shape) == x_shape and list(w.shape) == w_shape

    # --- host reference -------------------------------------------------------------
    # Defaults of ZORuntime.c: perturbation_sign = 1, i.e. the +eps pass of the ZO step.
    pert = {n.name: {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            for n in nodes[:2]}
    w_pert = perturb_rademacher(w, seed = pert[W_PERTURB]["seed"], node_idx = pert[W_PERTURB]["idx"],
                                eps = pert[W_PERTURB]["eps"], direction = 1)
    b_pert = perturb_rademacher(b, seed = pert[B_PERTURB]["seed"], node_idx = pert[B_PERTURB]["idx"],
                                eps = pert[B_PERTURB]["eps"], direction = 1)
    y = conv2d_im2col_hwc(x, w_pert, b_pert, pads, strides)
    assert list(y.shape) == out_shape

    # --- write ----------------------------------------------------------------------
    # generateNetwork.py consumes inputs.npz/outputs.npz POSITIONALLY (`inputs.files`
    # order zipped onto the graph's inputs), so the write order below is load-bearing;
    # the key names are documentation only.
    os.makedirs(args.dst, exist_ok = True)
    onnx.save(model, os.path.join(args.dst, "network.onnx"))
    np.savez(os.path.join(args.dst, "inputs.npz"),
             **{"input": x, "blocks_0_0_weight": w, "blocks_0_0_bias": b})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{conv_out_name: y})

    print(f"[ZO_Conv] wrote {args.dst}")
    print(f"  nodes       : {[f'{n.op_type}:{n.name}' for n in nodes]}")
    print(f"  value_info  : {[vi.name for vi in value_info]}")
    print(f"  input       : {x.shape} from {os.path.basename(args.src)}/{win_key}")
    print(f"  weight/bias : {w.shape} / {b.shape}  (perturbed on-device, idx 0 / 1, seed 42, eps 0.01)")
    print(f"  output      : {conv_out_name} {y.shape}  range [{y.min():.6f}, {y.max():.6f}]")
    print(f"  |w_pert-w|  : {np.abs(w_pert - w).max():.6f} (expect eps)")


if __name__ == "__main__":
    main()
