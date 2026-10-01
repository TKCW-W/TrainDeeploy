/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp6, task 2): the cluster-side preparation for the FUSED dual-pass conv on NE16.
 *
 * On NE16 the two eps-passes fuse by stacking the output channels: one 1x1 job with Cout' = 2*Cout,
 * rows [0, Cout) = W+ and rows [Cout, 2*Cout) = W-, over the SAME column buffer (block 0's input is
 * pass-independent). NE16 retires TP_OUT = 32 output channels per pass whether 8 or 16 are used, so
 * the second pass rides along in lanes that were idle. Two small cluster kernels feed that job:
 *
 *   PULP_NE16DualPerturbEncode : W (int8, NCHW [cout][K]) + wmul  ->  weight_enc (uint8, [2*cout][cinMajor][16])
 *       phase 1  every core perturbs ITS chunk of W for both signs (DualPerturb.h == exp5's phase 1)
 *                into a scratch [2*cout][K] int8 matrix (W+ rows, then W- rows); barrier;
 *       phase 2  every core bit-serial-encodes ITS share of the 2*cout rows (NE16WeightEncode_i8_u8,
 *                taps = 1, cin = K: the folded 1x1 weight of the im2col route).
 *
 *   PULP_NE16DualBias : W, wmul, add, bmul, mul  ->  add2 (int32, [2*cout])
 *       re-derives the same W+/W- (deterministic: same seeds, same chunking) and add+/add-, then
 *       applies the signed-input correction of exp16c per row:
 *           add2[c]        = add+[c] - off * mul[c] * sum_k W+[c][k]
 *           add2[cout + c] = add-[c] - off * mul[c] * sum_k W-[c][k]
 *       (NE16 reads the +off column buffer; the induced term is per output row.)
 *
 * Two nodes instead of one because Deeploy's tiler assumes single-output nodes (exp5 Findings §4);
 * the duplicated perturbation of 32 weights costs a few hundred cycles. Both run once per step,
 * untiled (tensors are tens of bytes). -- QW
 */

#include "DeeployGAP9Math.h"
#include "kernel/DualPerturb.h"
#include "pmsis.h"

void PULP_NE16DualPerturbEncode(const int8_t *pW, const int32_t *pWMul, uint8_t *pWEnc, int8_t *pScratch,
                                uint32_t seed_w_base, uint32_t tile_seed_xor, uint32_t log2D_w,
                                float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K) {
  const uint32_t core_id = pi_core_id();
  const uint32_t wsize = ch_out * K;
  int8_t *wp = pScratch;         /* rows [0, ch_out)        */
  int8_t *wm = pScratch + wsize; /* rows [ch_out, 2*ch_out) */

  uint32_t s, n;
  dc_chunk(wsize, num_cores, core_id, &s, &n);
  if (n)
    dc_perturb_i8(pW, wp, wm, pWMul, K, (seed_w_base + core_id) ^ tile_seed_xor, s, n, log2D_w, eps_scale);
  pi_cl_team_barrier();

  /* encode the stacked [2*ch_out][K] matrix as a 1x1 weight with K input channels */
  const uint32_t rows = 2u * ch_out;
  const uint32_t chunk = (rows + num_cores - 1u) / num_cores;
  uint32_t r0 = chunk * core_id;
  if (r0 > rows)
    r0 = rows;
  uint32_t r1 = r0 + chunk;
  if (r1 > rows)
    r1 = rows;
  if (r1 > r0)
    NE16WeightEncode_i8_u8(pScratch, pWEnc, rows, K, 1u, r0, r1 - r0);
  pi_cl_team_barrier();
}

void PULP_NE16DualBias(const int8_t *pW, const int32_t *pWMul, const int32_t *pAdd, const int32_t *pBMul,
                       const int32_t *pMul, int32_t *pAddOut, int8_t *pScratch, uint32_t seed_w_base,
                       uint32_t seed_b_base, uint32_t tile_seed_xor, uint32_t log2D_w, uint32_t log2D_b,
                       float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K, int32_t input_offset) {
  const uint32_t core_id = pi_core_id();
  const uint32_t wsize = ch_out * K;
  int8_t *wp = pScratch;
  int8_t *wm = pScratch + wsize;
  const uint32_t off_add = (2u * wsize + 3u) & ~3u;
  int32_t *addp = (int32_t *)(pScratch + off_add);
  int32_t *addm = addp + ch_out;

  uint32_t s, n;
  dc_chunk(wsize, num_cores, core_id, &s, &n);
  if (n)
    dc_perturb_i8(pW, wp, wm, pWMul, K, (seed_w_base + core_id) ^ tile_seed_xor, s, n, log2D_w, eps_scale);
  dc_chunk(ch_out, num_cores, core_id, &s, &n);
  if (n)
    dc_perturb_i32(pAdd, addp, addm, pBMul, 1u, (seed_b_base + core_id) ^ tile_seed_xor, s, n, log2D_b,
                   eps_scale);
  pi_cl_team_barrier();

  const uint32_t rows = 2u * ch_out;
  const uint32_t chunk = (rows + num_cores - 1u) / num_cores;
  uint32_t r0 = chunk * core_id;
  if (r0 > rows)
    r0 = rows;
  uint32_t r1 = r0 + chunk;
  if (r1 > rows)
    r1 = rows;
  for (uint32_t r = r0; r < r1; r++) {
    const uint32_t c = (r < ch_out) ? r : r - ch_out;
    const int8_t *w = pScratch + r * K; /* row r of the stacked [W+; W-] matrix */
    int32_t acc = 0;
    for (uint32_t k = 0; k < K; k++)
      acc += (int32_t)w[k];
    const int32_t a = (r < ch_out) ? addp[c] : addm[c];
    pAddOut[r] = a - input_offset * acc * pMul[c];
  }
  pi_cl_team_barrier();
}
