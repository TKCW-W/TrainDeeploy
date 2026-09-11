#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp16c -- build the full on-device QZO weight path for ANY SpeechNet conv block.

For block N this emits two fixtures, fed identical real data:

  b{N}_ref        the original Conv + RequantShift, plain int8 weight      -> cluster reference
  b{N}_ne16       RQSPerturbRademacher -> NE16WeightEncode -> NE16 Conv    -> the NE16 run

The NE16 fixture is the real thing: the device receives the **unperturbed** int8 weight and does
the perturbation AND the bit-serial encoding itself. Nothing about the weight is precomputed on the
host except the golden.

Why this file exists separately from `../exp16a_PW_single_layer/build_fixtures.py`: that one is
hardwired to block 1 and to `1xK` kernels, and it also still carries exp16a's abandoned
Slice/Add exploration. exp16c needs every block, including the `Kx1` ones, so generalising in place
would have destabilised exp16a's and exp16b's reproducibility. Both import Deeploy's own
`_weightEncode`, so the two can never disagree about the encoding.

SpeechNet convs (from the training graph):

  block  kernel  pads         Cin->Cout   activation
    0    1x4     0,2,0,2      1->8        SIGNED  -- needs BLOCKER 3, not handled here
    1    1x16    0,8,0,8      8->16       >= 0
    2    1x8     0,4,0,4      16->16      >= 0
    3    7x1     0,0,0,0      16->32      >= 0     taps along H
    4    7x1     0,0,0,0      32->32      >= 0     taps along H

Run in agitated_hugle:
  docker exec agitated_hugle bash -lc 'cd /app && \
    PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy python3 \
    TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16c_PW_single_layer/build_fixtures.py --block 2'
"""

import argparse
import ast
import os

import numpy as np
import onnx
import onnx.shape_inference
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper

from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

TRAIN = "TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_qzo12_train"
WEIGHT_OFFSET = -128  # fixed -- see exp16a Plan.md, blocker 1


def _load_deeploy_weight_encode(path = "TrainDeeploy/Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py"):
    """Lift Deeploy's own encoder out by AST -- agitated_hugle has no mako, so `import Deeploy` fails.

    Reimplementing it would risk the fixture and the compiler disagreeing about the bit layout.
    """
    fn = next(n for n in ast.parse(open(path).read()).body
              if isinstance(n, ast.FunctionDef) and n.name == "_weightEncode")
    import numpy.typing as npt
    ns = {"np": np, "npt": npt}
    mod = ast.Module(body = [fn], type_ignores = [])
    ast.fix_missing_locations(mod)
    exec(compile(mod, path, "exec"), ns)
    return ns["_weightEncode"]


_weightEncode = _load_deeploy_weight_encode()


def _rqs_node(name, inp, mul_name, add_name, out, attr_protos, domain = ""):
    n = helper.make_node("RequantShift", [inp, mul_name, add_name], [out], name = name, domain = domain)
    n.attribute.extend(attr_protos)  # verbatim: `div` MUST stay TENSOR-typed
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default = TRAIN)
    ap.add_argument("--out", default = "TrainDeeploy/DeeployTest/Tests/Models/NE16")
    ap.add_argument("--block", type = int, default = 2, help = "SpeechNet block index 0..4")
    ap.add_argument("--suffix", default = "")
    ap.add_argument("--neg-pmul", action = "store_true", default = False, dest = "neg_pmul",
                    help = "negate the perturbation multiplier -> the L- pass (RandomNoiseQuant.c:31)")
    args = ap.parse_args()

    N_ = args.block
    CONV = "/wrappedInnerForwardImpl/Conv" if N_ == 0 else f"/wrappedInnerForwardImpl_{N_}/Conv"
    ACT = f"/blocks_{N_}_conv_input_quant_1/Clip_output_0"
    WRAW = f"blocks.{N_}.conv.weight_int8"
    WPERT = f"{WRAW}_pert"

    src = os.path.join(args.train, "network.onnx")
    model = onnx.load(src)
    g = gs.import_onnx(model)
    conv = next(n for n in g.nodes if n.name == CONV)
    rqs = next(n for n in g.nodes if n.op == "RequantShift" and n.inputs[0] is conv.outputs[0])
    rqs_proto = next(n for n in model.graph.node if n.name == rqs.name)
    rqs_protos, rqs_domain = list(rqs_proto.attribute), rqs_proto.domain

    pert = next(n for n in g.nodes if n.op == "RQSPerturbRademacher" and n.outputs[0].name == WPERT)
    pert_proto = next(n for n in model.graph.node if n.name == pert.name)
    pert_protos, pert_domain = list(pert_proto.attribute), pert_proto.domain

    pads = [int(v) for v in conv.attrs["pads"]]        # [h_beg, w_beg, h_end, w_end]
    kh, kw = (int(v) for v in conv.attrs["kernel_shape"])
    assert 1 in (kh, kw), f"block {N_} kernel {kh}x{kw} is not 1xK or Kx1"
    K = kh * kw
    tapAxisH = (kh != 1)                               # True for the 7x1 blocks
    print(f"[block] {N_}: {CONV}  kernel {kh}x{kw} (K={K}, taps along {'H' if tapAxisH else 'W'})  pads={pads}")

    # ---- real data straight out of the training graph -----------------------------------------
    npz = np.load(os.path.join(args.train, "inputs.npz"))
    base = [npz[k] for k in sorted(k for k in npz.files if k.startswith("arr_"))]
    feed = {i.name: np.asarray(v) for i, v in zip(model.graph.input, base)}
    act, wpert = (np.asarray(t) for t in run_onnx_graph(src, feed, output_names = [ACT, WPERT]))
    w_raw = np.asarray(feed[WRAW]).astype(np.int8)
    w_pmul = np.asarray(pert.inputs[1].values).astype(np.int32)

    if args.neg_pmul:
        # L-: negating the Rademacher sign == negating M (RandomNoiseQuant.c:31)
        pg = helper.make_graph(
            [helper.make_node("RQSPerturbRademacher", ["w", "pmul"], ["wp"], name = "p", domain = pert_domain)],
            "perturb_only",
            [helper.make_tensor_value_info("w", onnx.TensorProto.INT8, list(w_raw.shape))],
            [helper.make_tensor_value_info("wp", onnx.TensorProto.INT8, list(w_raw.shape))],
            initializer = [numpy_helper.from_array((-w_pmul).astype(np.int32), "pmul")])
        pg.node[0].attribute.extend(pert_protos)
        pm = helper.make_model(pg, opset_imports = list(model.opset_import))
        pm.ir_version = model.ir_version
        onnx.save(pm, "/tmp/_exp16c_pneg.onnx")
        wneg = np.asarray(run_onnx_graph("/tmp/_exp16c_pneg.onnx", {"w": w_raw}, output_names = ["wp"])[0])
        print(f"[pert-] L-: {int((wneg != wpert).sum())} of {wpert.size} weight elements differ from L+")
        wpert = wneg

    mul = np.asarray(rqs.inputs[1].values).astype(np.int32)
    add_t = rqs.inputs[2]
    add = np.asarray(add_t.values).astype(np.int32) if isinstance(add_t, gs.Constant) else \
        np.asarray(run_onnx_graph(src, feed, output_names = [add_t.name])[0]).astype(np.int32)

    Nb, Cin, H, W = act.shape
    Cout = wpert.shape[0]
    Hout = H + pads[0] + pads[2] - kh + 1
    Wout = W + pads[1] + pads[3] - kw + 1
    print(f"[act  ] {act.shape} range [{act.min()},{act.max()}]   [wpert] {wpert.shape} range [{wpert.min()},{wpert.max()}]")
    print(f"[shape] in [{Nb},{Cin},{H},{W}] * [{Cout},{Cin},{kh},{kw}] -> out [{Nb},{Cout},{Hout},{Wout}]")
    # QW (exp16c phase 4 / BLOCKER 3): NE16 reads activations as UNSIGNED and CONFIG0 bit 26
    #     (PR #183's `input_signed`) is undecoded by the hardware. Block 0's activation is genuinely
    #     signed ([-66,127]), so it is fed as x+128 (uint8) and the induced per-output-channel term
    #     `128 * sum_w` removed afterwards. Because `add` lands AFTER the multiply in
    #     (acc*mul + add) >> log2(div), the correction to apply is `add - 128*sum_w*mul`, computed
    #     ON DEVICE by NE16SignedInputBias -- sum_w is not a host constant when the weight is a
    #     per-step RQSPerturbRademacher output. -- QW
    signedAct = bool(act.min() < 0)
    if signedAct:
        print(f"[signd] block {N_} activation is SIGNED (min {act.min()}) -> +128 input, "
              f"NE16SignedInputBias correction")
    # Every SpeechNet conv pads only along its TAP axis; the decomposition expresses padding as a
    # pre-pad of the fixture data (NE16's 1x1 mode cannot pad, and its guard in fsm.cpp:53 is
    # commented out -> silent wrong results).
    assert (pads[0] == pads[2] == 0) if not tapAxisH else (pads[1] == pads[3] == 0), \
        f"padding on the non-tap axis is not modelled; pads={pads}"

    opset = list(model.opset_import)

    # ---- 1. cluster reference ------------------------------------------------------------------
    rdir = os.path.join(args.out, f"b{N_}_ref{args.suffix}")
    os.makedirs(rdir, exist_ok = True)
    ref = helper.make_graph(
        [helper.make_node("Conv", ["input", "weight"], ["conv_out"], name = f"b{N_}_conv",
                          kernel_shape = [kh, kw], pads = pads, strides = [1, 1],
                          dilations = [1, 1], group = 1),
         _rqs_node(f"b{N_}_rqs", "conv_out", "rqs_mul", "rqs_add", "output", rqs_protos, rqs_domain)],
        f"b{N_}_ref{args.suffix}",
        [helper.make_tensor_value_info("input", onnx.TensorProto.INT8, [Nb, Cin, H, W]),
         helper.make_tensor_value_info("weight", onnx.TensorProto.INT8, list(wpert.shape))],
        [helper.make_tensor_value_info("output", onnx.TensorProto.INT8, [Nb, Cout, Hout, Wout])],
        initializer = [numpy_helper.from_array(mul, "rqs_mul"), numpy_helper.from_array(add, "rqs_add")])
    mr = helper.make_model(ref, opset_imports = opset)
    mr.ir_version = model.ir_version
    onnx.save(onnx.shape_inference.infer_shapes(mr, strict_mode = False), os.path.join(rdir, "network.onnx"))
    feed_ref = {"input": act.astype(np.int8), "weight": wpert.astype(np.int8)}
    ref_out = np.asarray(run_onnx_graph(os.path.join(rdir, "network.onnx"), feed_ref, output_names = ["output"])[0])
    np.savez(os.path.join(rdir, "inputs.npz"), **feed_ref)
    np.savez(os.path.join(rdir, "outputs.npz"), output = ref_out.astype(np.int8))
    print(f"[ref  ] output {ref_out.shape} range [{ref_out.min()},{ref_out.max()}] -> {rdir}")

    # ---- 2. the NE16 fixture: perturb -> encode -> conv, NHWC-native ----------------------------
    kdir = os.path.join(args.out, f"b{N_}_ne16{args.suffix}")
    os.makedirs(kdir, exist_ok = True)

    # Weight layout: per-tap NE16 encoding stacked (taps*cout, cinMajor, encBytes). RANK 3 on
    # purpose -- a rank-4 conv input is permuted by PULPNCHWtoNHWCPass, which would destroy the
    # bit-serial encoding. Computed here only as the shape/golden; the DEVICE produces the bytes.
    taps = [wpert[:, :, j:j + 1, :] if tapAxisH else wpert[:, :, :, j:j + 1] for j in range(K)]
    enc_taps = np.concatenate([_weightEncode((t.astype(np.int32) - WEIGHT_OFFSET).astype(np.uint8), bits = 8)
                               for t in taps], axis = 0)

    conv_n = helper.make_node("Conv", ["input", "weight_enc"], ["conv_out"], name = f"b{N_}_conv1xk",
                              kernel_shape = [kh, kw], pads = [0, 0, 0, 0], strides = [1, 1],
                              dilations = [1, 1], group = 1)
    for k, v in (("ne16_weight_preencoded", 1), ("weight_offset", WEIGHT_OFFSET),
                 ("ne16_taps", K), ("channels_first", 0)):
        conv_n.attribute.append(helper.make_attribute(k, v))
    rqsAddName = "rqs_add_corr" if signedAct else "rqs_add"
    rqs_n = _rqs_node(f"b{N_}_rqs", "conv_out", "rqs_mul", rqsAddName, "output", rqs_protos, rqs_domain)
    rqs_n.attribute.append(helper.make_attribute("channels_first", 0))

    pert_n = helper.make_node("RQSPerturbRademacher", ["weight", "w_pmul"], ["weight_pert"],
                              name = f"b{N_}_wpert", domain = pert_domain)
    pert_n.attribute.extend(pert_protos)   # verbatim -> idx/seed match, so the RNG stream matches
    enc_n = helper.make_node("NE16WeightEncode", ["weight_pert"], ["weight_enc"], name = f"b{N_}_wenc")
    enc_n.attribute.append(helper.make_attribute("ne16_taps", K))
    enc_n.attribute.append(helper.make_attribute("ne16_bits", 8))
    extraNodes, extraVinfo = [], []
    if signedAct:
        bias_n = helper.make_node("NE16SignedInputBias", ["weight_pert", "rqs_mul", "rqs_add"],
                                  ["rqs_add_corr"], name = f"b{N_}_sbias")
        bias_n.attribute.append(helper.make_attribute("ne16_input_offset", 128))
        extraNodes.append(bias_n)
        extraVinfo.append(helper.make_tensor_value_info("rqs_add_corr", onnx.TensorProto.INT32,
                                                        list(add.shape)))

    # The device input is pre-padded along the tap axis; its extent there is exactly out + (K-1).
    actp = np.pad(act, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
                  mode = "constant", constant_values = 0).astype(np.int8)
    Hp, Wp = actp.shape[2], actp.shape[3]
    assert (Hp if tapAxisH else Wp) == (Hout if tapAxisH else Wout) + K - 1, \
        f"tap-axis extent {(Hp if tapAxisH else Wp)} != out+{K-1}"
    k_in = actp.transpose(0, 2, 3, 1).copy()           # NHWC-native: no layout Transposes at all
    if signedAct:
        k_in = (k_in.astype(np.int32) + 128).astype(np.uint8)   # x_u = x + 128, read as uint8
    k_out = ref_out.transpose(0, 2, 3, 1).copy()

    kg = helper.make_graph(
        [pert_n] + extraNodes + [enc_n, conv_n, rqs_n],
        f"b{N_}_ne16{args.suffix}",
        [helper.make_tensor_value_info("input",
                                       onnx.TensorProto.UINT8 if signedAct else onnx.TensorProto.INT8,
                                       [Nb, Hp, Wp, Cin]),
         helper.make_tensor_value_info("weight", onnx.TensorProto.INT8, list(w_raw.shape))],
        [helper.make_tensor_value_info("output", onnx.TensorProto.INT8, [Nb, Hout, Wout, Cout])],
        initializer = [numpy_helper.from_array(mul, "rqs_mul"), numpy_helper.from_array(add, "rqs_add"),
                       numpy_helper.from_array((-w_pmul if args.neg_pmul else w_pmul).astype(np.int32), "w_pmul")],
        value_info = [helper.make_tensor_value_info("weight_pert", onnx.TensorProto.INT8, list(wpert.shape)),
                      helper.make_tensor_value_info("weight_enc", onnx.TensorProto.UINT8, list(enc_taps.shape)),
                      helper.make_tensor_value_info("conv_out", onnx.TensorProto.INT32,
                                                    [Nb, Hout, Wout, Cout])] + extraVinfo)
    mk = helper.make_model(kg, opset_imports = opset)
    mk.ir_version = model.ir_version
    onnx.save(mk, os.path.join(kdir, "network.onnx"))   # no shape inference: shapes are explicit
    np.savez(os.path.join(kdir, "inputs.npz"), input = k_in, weight = w_raw.astype(np.int8))
    np.savez(os.path.join(kdir, "outputs.npz"), output = k_out.astype(np.int8))
    np.savez(os.path.join(kdir, "weight_enc_golden.npz"), weight_enc = enc_taps)
    print(f"[ne16 ] {K} dispatches, weight_enc {enc_taps.shape} ({enc_taps.nbytes} B) encoded ON DEVICE, "
          f"input {list(k_in.shape)} NHWC-native -> {kdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
