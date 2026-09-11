# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (exp16a / STEP 2b): tile constraint for the 1xK -> K-pointwise NE16 decomposition.

Two things make this different from NE16PWConv2DTileConstraint:

1. **Win != Wout.** A 1xK convolution shrinks W by K-1 (or, with the input pre-padded, keeps it).
   The pointwise constraint asserts `outputWidthVar == inputWidthVar`, which a 1xK node violates.

2. **The output tile must stay resident across all K dispatches.** Taps 1..K-1 use CONFIG0[14]
   streamin, which preloads the accumulator from `outfeat_addr` -- so every tap must write the
   SAME output buffer, and that buffer must not be evicted or re-DMA'd between dispatches.

The simplest constraint that satisfies both is to **not tile at all**: pin every dimension of
every tensor to its maximum, so the whole layer is one tile resident in L1. For SpeechNet block 1
that fits comfortably:

    input  [1,14,103,8] int8   =  11.5 KB   (K=16, pre-padded)
    weight (16,16,1,16) uint8  =   4.1 KB
    output [1,14,88,16] int32  =  78.8 KB
                       total   =  94.4 KB   <  110 KB (--l1 110000)

This is a deliberate, documented limitation of exp16a, not a general solution: a layer whose
int32 output exceeds L1 needs real tiling, and then the halo (K-1 extra input columns per tile)
and the streamin residency both have to be modelled. Recorded as future work in Findings.md. -- QW
"""

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class NE161xKConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for key in ('data_in', 'weight', 'data_out'):
            tilerModel.addTensorDimToModel(ctxt, parseDict[key])

        # Single-tile policy: every dimension pinned to its full extent. This simultaneously
        # sidesteps the Win != Wout mismatch and guarantees the streamin residency requirement
        # (one output buffer, never re-staged between the K dispatches).
        for key in ('data_in', 'weight', 'data_out'):
            buf = ctxt.lookup(parseDict[key])
            for dimIdx in range(len(buf.shape)):
                var = tilerModel.getTensorDimVar(tensorName = buf.name, dimIdx = dimIdx)
                tilerModel.addConstraint(var == var.Max())
        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        return tilerModel  # nothing to prefer: the geometry is already fully determined

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = ['data_in', 'weight', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inBuf: VariableBuffer = ctxt.lookup(operatorRepresentation['data_in'])
        wBuf: VariableBuffer = ctxt.lookup(operatorRepresentation['weight'])

        # One tile: each tensor is transferred whole, exactly once.
        from Deeploy.TilingExtension.TilingCodegen import HyperRectangle
        inputLoadSchedule = [{
            "data_in": HyperRectangle((0,) * len(inBuf.shape), tuple(inBuf.shape)),
            "weight": HyperRectangle((0,) * len(wBuf.shape), tuple(wBuf.shape)),
        }]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]

        # Nothing varies per tile, so there is nothing to replace per iteration.
        return VariableReplacementScheme({}, {}), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
