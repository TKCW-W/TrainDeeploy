/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* QW (exp16c / blocker 1b): device-side NE16 bit-serial weight encoder.
 *
 * NE16 reads weights bit-plane transposed. exp16a/exp16b let the HOST do that, which only works
 * while the weight is a gs.Constant. In QZO the conv weight is produced on device by
 * RQSPerturbRademacher, twice per ZO step, so the encoding has to happen on device too.
 *
 * This is a C port of `_weightEncode`
 * (Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py:24) for the case the 1xK -> K
 * pointwise decomposition needs: H*W == 1 per tap, bits == 8, dense (non-depthwise).
 *
 * The host function does two things; only the second survives here:
 *
 *   1. `weight_offset = values.min()` then `values -= weight_offset`.  DROPPED -- exp16a hardwires
 *      weight_offset = -128 (a data-dependent offset cannot work for a weight that changes every
 *      step), so the offset step degenerates to the unsigned reinterpretation w_u = w + 128,
 *      i.e. a single XOR with 0x80.
 *   2. The bit-plane transpose.  A fixed, data-INDEPENDENT permutation, reproduced below.
 *
 * Layouts
 * -------
 *   src: (cout, cin, H, W) int8, NCHW-contiguous, with H*W == taps. Tap j of output channel co,
 *        input channel ci is src[(co*cin + ci)*taps + j] -- which holds for BOTH a 1xK kernel
 *        (H=1, W=K) and a Kx1 one (H=K, W=1), since the tap is the last index either way.
 *   dst: (taps*cout, cinMajor, 16) uint8, matching
 *        `concat_j _weightEncode(w[:,:,:,j:j+1] + 128)` along axis 0 -- exactly what exp16a's
 *        build_fixtures.py produces on the host.
 *
 * For one (row, cinMajor) pair: gather the 16 input-channel lanes of that subtile (zero-padded
 * where cin is not a multiple of 16, matching numpy's `np.pad(..., constant_values=0)` which runs
 * AFTER the offset), then emit 16 bytes where output byte `b*2 + k` collects bit `b` of source
 * lanes 8k..8k+7, LSB-first (`bitorder="little"`).
 *
 * Plain C, no NE16 registers, no pulp-nnx dependency -- nothing here is an ISA change. -- QW
 */

#include "DeeployGAP9Math.h"

#define NE16_CIN_SUBTILE 16
#define NE16_WEIGHT_BITS 8

void NE16WeightEncode_i8_u8(const int8_t *__restrict__ src, uint8_t *__restrict__ dst,
                            const uint32_t cout, const uint32_t cin, const uint32_t taps,
                            const uint32_t row_start, const uint32_t row_count) {

  const uint32_t cinMajor = (cin + NE16_CIN_SUBTILE - 1) / NE16_CIN_SUBTILE;
  const uint32_t dstRowBytes = cinMajor * NE16_WEIGHT_BITS * (NE16_CIN_SUBTILE / 8);

  for (uint32_t row = row_start; row < row_start + row_count; row++) {
    // row == j*cout + co  (taps-major, matching the host's concatenate along axis 0)
    const uint32_t j = row / cout;
    const uint32_t co = row - j * cout;
    const int8_t *srcCo = src + (uint32_t)co * cin * taps + j;
    uint8_t *dstRow = dst + (uint32_t)row * dstRowBytes;

    for (uint32_t cm = 0; cm < cinMajor; cm++) {
      const uint32_t ciBase = cm * NE16_CIN_SUBTILE;

      for (uint32_t k = 0; k < NE16_CIN_SUBTILE / 8; k++) {
        // eight bit-plane accumulators for this half-subtile
        uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        uint32_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

        for (uint32_t t = 0; t < 8; t++) {
          const uint32_t ci = ciBase + 8 * k + t;
          // w_u = w + 128; lanes past cin are zero (NOT 128) -- the host pads after the offset
          const uint32_t v = (ci < cin) ? (uint32_t)((uint8_t)srcCo[ci * taps] ^ 0x80u) : 0u;
          acc0 |= ((v >> 0) & 1u) << t;
          acc1 |= ((v >> 1) & 1u) << t;
          acc2 |= ((v >> 2) & 1u) << t;
          acc3 |= ((v >> 3) & 1u) << t;
          acc4 |= ((v >> 4) & 1u) << t;
          acc5 |= ((v >> 5) & 1u) << t;
          acc6 |= ((v >> 6) & 1u) << t;
          acc7 |= ((v >> 7) & 1u) << t;
        }

        uint8_t *o = dstRow + cm * (NE16_WEIGHT_BITS * (NE16_CIN_SUBTILE / 8)) + k;
        o[0 * 2] = (uint8_t)acc0;
        o[1 * 2] = (uint8_t)acc1;
        o[2 * 2] = (uint8_t)acc2;
        o[3 * 2] = (uint8_t)acc3;
        o[4 * 2] = (uint8_t)acc4;
        o[5 * 2] = (uint8_t)acc5;
        o[6 * 2] = (uint8_t)acc6;
        o[7 * 2] = (uint8_t)acc7;
      }
    }
  }
}
