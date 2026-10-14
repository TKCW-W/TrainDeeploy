# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp8): `PerturbRademacherFixed` -- the fp32 ZO perturbation with the sign fixed at codegen
time instead of read from the runtime `perturbation_sign`, for the dual-pass (fused) training graph, which computes
both ε-passes in one execution:

    mode 0  out [2*size] = [w + eps*z ; w - eps*z]   (stacked, e.g. BN gamma/beta on 2*C channels)
    mode 1  out [size]   =  w + eps*z
    mode 2  out [size]   =  w - eps*z

Same chunking, seed expression (`idx` attribute as node id), eps and kernel (`ApplyRademacherPerturbation`) as
FloatPerturbRademacherTemplate, so every half is bit-identical to what the original node writes in the matching
pass. Untiled (tensors of tens to hundreds of elements); tile_seed_offset stays 0 as for a single-tile original.
"""
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatPerturbRademacherFixedTemplate(NodeTemplate):
    """pad_cols > 0 (modes 1 / 2, 2-D weight [rows, cols]): the output is [rows, pad_cols] with the perturbed weight in
    columns [col_offset, col_offset + cols) and zeros elsewhere -- the dual-pass FC takes the whole stacked [1, 2C] vector
    with W+ padded to the left / W- to the right half (Deeploy's tiled Slice cannot cut the vector, exp8). The weight is
    perturbed compactly into an L1 scratch first (same chunks / seed as the unpadded node), then scattered."""

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        operatorRepresentation['tile_seed_offset'] = 0
        operatorRepresentation.setdefault('ctxtBuffer', 'NULL')  # Mako strict mode: defined for unpadded nodes too
        return ctxt, operatorRepresentation, []

    @staticmethod
    def computeTransientBuffersSize(ctxt: NetworkContext,
                                    operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, int]]:
        size = 4 * int(operatorRepresentation['size']) if operatorRepresentation.get('pad_cols', 0) else 4
        return [(operatorRepresentation['nodeName'] + "_scratch", size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # hoisted for every node (4 B when unpadded): the tiler looks up whatever computeTransientBuffersSize names
        name, size = _FloatPerturbRademacherFixedTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


referenceTemplate = _FloatPerturbRademacherFixedTemplate("""
// PerturbRademacherFixed (Name: ${nodeName}, Op: ${nodeOp}, mode ${mode}) -- QW exp8
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});
uint32_t ${nodeName}_local_size = ${nodeName}_chunk_stop - ${nodeName}_chunk_start;
uint32_t ${nodeName}_seed = ((${seed} + perturb_seed_base) + NUM_CORES * ${node_id} + ${nodeName}_core_id) ^ (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps = perturb_eps_use_override ? perturb_eps_override : ${eps}f;
% if pad_cols:
float32_t *${nodeName}_dst = (float32_t *) ${ctxtBuffer};
% else:
float32_t *${nodeName}_dst = (float32_t *) ${data_out};
% endif
% if mode in (0, 1):
ApplyRademacherPerturbation((const float32_t *) &${data_in}[${nodeName}_chunk_start],
                            &${nodeName}_dst[${nodeName}_chunk_start],
                            ${nodeName}_seed, 1u, ${nodeName}_local_size, ${nodeName}_eps);
% endif
% if mode == 2:
ApplyRademacherPerturbation((const float32_t *) &${data_in}[${nodeName}_chunk_start],
                            &${nodeName}_dst[${nodeName}_chunk_start],
                            ${nodeName}_seed, 0u, ${nodeName}_local_size, ${nodeName}_eps);
% endif
% if pad_cols:
pi_cl_team_barrier();
for (uint32_t r = ${nodeName}_core_id; r < ${rows}u; r += NUM_CORES) {
  float32_t *o = (float32_t *) ${data_out} + r * ${pad_cols}u;
  const float32_t *w = ${nodeName}_dst + r * ${cols}u;
  for (uint32_t j = 0; j < ${pad_cols}u; j++)
    o[j] = (j >= ${col_offset}u && j < ${col_offset + cols}u) ? w[j - ${col_offset}u] : 0.0f;
}
pi_cl_team_barrier();
% endif
% if mode == 0:
ApplyRademacherPerturbation((const float32_t *) &${data_in}[${nodeName}_chunk_start],
                            (float32_t *) &${data_out}[${size} + ${nodeName}_chunk_start],
                            ${nodeName}_seed, 0u, ${nodeName}_local_size, ${nodeName}_eps);
% endif
""")
