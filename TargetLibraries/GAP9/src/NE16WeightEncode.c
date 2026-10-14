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
    // row == tap*cout + co  (taps-major, matching the host's concatenate along axis 0)
    const uint32_t tap = row / cout;
    const uint32_t co = row - tap * cout;
    const int8_t *srcCo = src + (uint32_t)co * cin * taps + tap;
    uint8_t *dstRow = dst + (uint32_t)row * dstRowBytes;

    for (uint32_t cm = 0; cm < cinMajor; cm++) {
      const uint32_t ciBase = cm * NE16_CIN_SUBTILE;

      for (uint32_t half = 0; half < NE16_CIN_SUBTILE / 8; half++) {
        // eight bit planes of this half-subtile: plane<b> collects bit b of its 8 lanes
        uint32_t plane0 = 0, plane1 = 0, plane2 = 0, plane3 = 0;
        uint32_t plane4 = 0, plane5 = 0, plane6 = 0, plane7 = 0;

        for (uint32_t lane = 0; lane < 8; lane++) {
          const uint32_t ci = ciBase + 8 * half + lane;
          // w_u = w + 128; lanes past cin are zero (NOT 128) -- the host pads after the offset
          const uint32_t w_u = (ci < cin) ? (uint32_t)((uint8_t)srcCo[ci * taps] ^ 0x80u) : 0u;
          plane0 |= ((w_u >> 0) & 1u) << lane;
          plane1 |= ((w_u >> 1) & 1u) << lane;
          plane2 |= ((w_u >> 2) & 1u) << lane;
          plane3 |= ((w_u >> 3) & 1u) << lane;
          plane4 |= ((w_u >> 4) & 1u) << lane;
          plane5 |= ((w_u >> 5) & 1u) << lane;
          plane6 |= ((w_u >> 6) & 1u) << lane;
          plane7 |= ((w_u >> 7) & 1u) << lane;
        }

        uint8_t *dstHalf = dstRow + cm * (NE16_WEIGHT_BITS * (NE16_CIN_SUBTILE / 8)) + half;
        dstHalf[0 * 2] = (uint8_t)plane0;
        dstHalf[1 * 2] = (uint8_t)plane1;
        dstHalf[2 * 2] = (uint8_t)plane2;
        dstHalf[3 * 2] = (uint8_t)plane3;
        dstHalf[4 * 2] = (uint8_t)plane4;
        dstHalf[5 * 2] = (uint8_t)plane5;
        dstHalf[6 * 2] = (uint8_t)plane6;
        dstHalf[7 * 2] = (uint8_t)plane7;
      }
    }
  }
}

/* QW (exp16c phase 4 / BLOCKER 3): signed-activation bias correction.
 *
 * NE16 reads input activations as UNSIGNED. SpeechNet block 0's activation is genuinely signed
 * (range [-66, 127]), and CONFIG0 bit 26 -- what PR #183 calls `input_signed` -- is undecoded by
 * the hardware (ne16_regfile.cpp:200-231), so there is no register to flip.
 *
 * The standard fix: feed x_u = x + 128 and remove the induced term afterwards. Exactly,
 *
 *     sum_k w_k * x_k  =  sum_k w_k * (x_u,k - 128)  =  sum_k w_k * x_u,k  -  128 * sum_k w_k
 *
 * and `128 * sum_k w_k` is a per-OUTPUT-CHANNEL constant. It cannot be folded on the host, because
 * in QZO the weight is produced on device by RQSPerturbRademacher and changes every ZO step -- so
 * the correction is recomputed here, on device, from the same perturbed weight the encoder sees.
 *
 * Where it is applied matters. The RequantShift computes
 *
 *     out = (acc * mul + add) >> log2(div)
 *
 * so `add` is applied AFTER the multiply. Correcting `acc` by `-128*sum_w` is therefore equivalent
 * to correcting `add` by `-128 * sum_w * mul`:
 *
 *     out = ((acc_meas - 128*sum_w) * mul + add) >> s  =  (acc_meas * mul + [add - 128*sum_w*mul]) >> s
 *
 * Range: |128 * sum_w * mul| <= 128 * 127 * cin*H*W * max(mul). For block 0 that is
 * 128 * 508 * 454 ~= 3.0e7, comfortably inside int32 (checked against the real tensors before this
 * was written). A layer with a large `mul` and a wide receptive field could overflow; the parser
 * therefore records the shapes and this bound is documented rather than assumed.
 *
 * Parallelised over output channels, like the encoder. -- QW
 */
void NE16SignedInputBias_i32(const int8_t *__restrict__ weight, const int32_t *__restrict__ mul,
                             const int32_t *__restrict__ add, int32_t *__restrict__ out,
                             const uint32_t cout, const uint32_t cinTaps, const int32_t offset,
                             const uint32_t co_start, const uint32_t co_count) {
  (void)cout;
  for (uint32_t co = co_start; co < co_start + co_count; co++) {
    const int8_t *w = weight + (uint32_t)co * cinTaps;
    int32_t w_sum = 0;
    for (uint32_t i = 0; i < cinTaps; i++) {
      w_sum += (int32_t)w[i];
    }
    out[co] = add[co] - offset * w_sum * mul[co];
  }
}
