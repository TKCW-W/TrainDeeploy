/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp7, step 3): FUSED DUAL-PASS quantized convolution for SpeechNet blocks 1-4.
 *
 * Block 0's DualConv.c reads ONE input for both passes (the network input is the same for theta+eps*z and
 * theta-eps*z). From block 1 on the two passes have different activations, so this kernel takes x+ and x-:
 *
 *      A+ = requant( conv(x+, W+), add+ )        A- = requant( conv(x-, W-), add- )
 *
 *   phase 1  every core perturbs ITS chunk of the UNPERTURBED weight -- given in RAW CHW order
 *            [co][ci][ky][kx], i.e. NOT transposed by the layout pass (attr weight_chw) -- walking the chunk in
 *            CHW order so noise bit i lands on CHW element i exactly as in the reference
 *            RQSPerturbRademacher node, and writes W+ / W- at the HWC position the conv loop reads
 *            (dc_perturb_i8_chw). Requant-add: dc_perturb_i32 as in DualConv.c. Both signs from one stream.
 *            DualConv.c perturbs the HWC-transposed weight, which is only correct for ch_in = 1.
 *   phase 2  Conv4x2.c's structure (step 2a, bit-identical to pulp-nn): a flat range of output pixels per
 *            core, two pixels at a time; for every block of 4 output channels the 4x2 dot loop runs once
 *            for the + pass (x+, W+) and once for the - pass (x-, W-) -- the two passes share no operand, so
 *            they cannot share registers; what they share is the pixel loop, the padding test and the
 *            receptive-field addressing (identical positions in x+ and x-). Interior fields are read in
 *            place; fields that touch the padding are gathered (out of line, they are rare).
 *
 * Layouts (axis names as in pulp_nn_conv_i8_i8_i8: `x` = Deeploy's dim_im_in_y, the long / time axis):
 *   x+, x-  : HWC [dim_in_y][dim_in_x][ch_in], pixel stride ch_in (two separate tensors)
 *   weight  : CHW [ch_out][ch_in][dim_kernel_y][dim_kernel_x]   (unperturbed)
 *   output  : A+ at pOutP, A- at pOutM, both HWC with pixel stride `out_stride`
 *             (stacked [A+ | A-] output: pOutP = out, pOutM = out + ch_out, out_stride = 2*ch_out)
 * Preconditions: K = ch_in*kh*kw and kw*ch_in multiples of 4 (all SpeechNet blocks).
 *
 * Scratch (L1, DualConv2InTemplate.computeTransientBuffersSize -- keep in sync):
 *   [0, wsize) W+ (HWC)   [wsize, 2*wsize) W- (HWC)   wsize = ch_out*K
 *   off_addp (4-aligned) add+ [ch_out] i32   off_addm add- [ch_out] i32
 *   off_pat  four K-byte patches per core (x+ pixel A/B, x- pixel A/B)
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"
#include <math.h>

#include "kernel/DualPerturb.h"

typedef signed char d2_v4s __attribute__((vector_size(4)));

/* requant exactly as pulp_nn_bn_quant_i8 (32-bit variant), DualConv.c and Conv4x2.c */
#define D2_RQ(acc, mul, add)                                                                                   \
  ((int8_t)__builtin_pulp_clip_r((int32_t)((uint32_t)(acc) * (uint32_t)(mul) + (uint32_t)(add)) >> out_shift, 127))

/* Gather one receptive field into `patch` (rows contiguous, zeros in the padding). Out of line: only pixels
 * whose field touches the padding come here, and keeping it out of the pixel loop keeps that loop small. */
static void __attribute__((noinline)) d2_gather(const int8_t *pIn, int8_t *patch, int32_t iy0, int32_t ix0,
                                                uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in,
                                                uint16_t dim_kernel_x, uint16_t dim_kernel_y) {
  /* QW (exp7 step 5d): word path. In HWC the valid taps of one kernel row are ONE contiguous run of
   * (kx_hi - kx_lo)*ch_in bytes, so per row: zero-fill the padded words, copy the run word by word. The byte path
   * below cost ~2 x 128 byte copies per border pixel on block 1 and was the source of the phase-2 imbalance
   * (border pixels per core 22..30). Needs ch_in % 4 == 0 and a word-aligned input (as the in-place reads). */
  if (((ch_in & 3u) == 0u) && ((((uint32_t)pIn) & 3u) == 0u)) {
    const uint32_t cw = (uint32_t)ch_in >> 2; /* words per tap */
    int32_t kx_lo = -ix0, kx_hi = (int32_t)dim_in_x - ix0;
    if (kx_lo < 0)
      kx_lo = 0;
    if (kx_hi > (int32_t)dim_kernel_x)
      kx_hi = (int32_t)dim_kernel_x;
    if (kx_hi < kx_lo)
      kx_hi = kx_lo;
    int32_t *pw = (int32_t *)patch;
    for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
      const int32_t iy = iy0 + (int32_t)ky;
      if (iy < 0 || iy >= (int32_t)dim_in_y) {
        for (uint32_t k = 0; k < (uint32_t)dim_kernel_x * cw; k++)
          *pw++ = 0;
        continue;
      }
      for (uint32_t k = 0; k < (uint32_t)kx_lo * cw; k++)
        *pw++ = 0;
      const int32_t *src = (const int32_t *)(pIn + ((uint32_t)iy * dim_in_x + (uint32_t)(ix0 + kx_lo)) * ch_in);
      for (uint32_t k = 0; k < (uint32_t)(kx_hi - kx_lo) * cw; k++)
        *pw++ = *src++;
      for (uint32_t k = 0; k < (uint32_t)((int32_t)dim_kernel_x - kx_hi) * cw; k++)
        *pw++ = 0;
    }
    return;
  }
  int8_t *pp = patch;
  for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
    const int32_t iy = iy0 + (int32_t)ky;
    for (uint32_t kx = 0; kx < dim_kernel_x; kx++) {
      const int32_t ix = ix0 + (int32_t)kx;
      if (iy < 0 || iy >= (int32_t)dim_in_y || ix < 0 || ix >= (int32_t)dim_in_x) {
        for (uint32_t ci = 0; ci < ch_in; ci++)
          *pp++ = 0;
      } else {
        const int8_t *src = pIn + ((uint32_t)iy * dim_in_x + (uint32_t)ix) * ch_in;
        for (uint32_t ci = 0; ci < ch_in; ci++)
          *pp++ = src[ci];
      }
    }
  }
}

void PULP_DualConv2d_2in_i8_i8_i8_HWC(
    const int8_t *pInP, const int8_t *pInM, const int8_t *pWeight, const int32_t *pMul, const int32_t *pAdd,
    const int32_t *pWMul, const int32_t *pBMul, int8_t *pOutP, int8_t *pOutM, uint32_t out_stride,
    int8_t *pScratch, uint32_t seed_w_base, uint32_t seed_b_base, uint32_t tile_seed_xor, uint32_t log2D_w,
    uint32_t log2D_b, float32_t eps_scale, uint32_t num_cores, uint16_t out_shift, uint16_t dim_in_x,
    uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
    uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t pad_y_top, uint16_t pad_y_bottom, uint16_t pad_x_left,
    uint16_t pad_x_right, uint16_t stride_x, uint16_t stride_y) {
  const uint32_t core_id = pi_core_id();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_x * dim_kernel_y;  // block 1: 128
  const uint32_t row_len = (uint32_t)dim_kernel_x * ch_in;           // bytes of one kernel row (HWC)
  const uint32_t row_words = row_len >> 2;
  const uint32_t wsize = (uint32_t)ch_out * K;                        // block 1: 2048

  /* scratch layout -- must match DualConv2InTemplate.computeTransientBuffersSize */
  const uint32_t off_addp = (2u * wsize + 3u) & ~3u;
  const uint32_t off_addm = off_addp + 4u * ch_out;
  const uint32_t off_pat = off_addm + 4u * ch_out;
  int8_t *wp = pScratch;
  int8_t *wm = pScratch + wsize;
  int32_t *addp = (int32_t *)(pScratch + off_addp);
  int32_t *addm = (int32_t *)(pScratch + off_addm);
  int8_t *patch = pScratch + off_pat + core_id * 4u * K; /* xP A | xP B | xM A | xM B */

#ifdef DUALCONV2IN_PROBE /* -D DUALCONV2IN_PROBE=ON: core-0 cycles of each phase (incl. its closing barrier) */
  uint32_t probe_t0 = 0, probe_t1 = 0;
  if (core_id == 0) {
    pi_perf_conf(1 << PI_PERF_CYCLES);
    pi_perf_reset();
    pi_perf_start();
    probe_t0 = pi_perf_read(PI_PERF_CYCLES);
  }
#endif
  /* ---------------- phase 1: perturb my CHW chunk, both signs, write HWC ------------------------- */
  uint32_t s, n;
  dc_chunk(wsize, num_cores, core_id, &s, &n);
  if (n)
    dc_perturb_i8_chw(pWeight, wp, wm, pWMul, ch_in, dim_kernel_y, dim_kernel_x,
                      (seed_w_base + core_id) ^ tile_seed_xor, s, n, log2D_w, eps_scale);
  dc_chunk((uint32_t)ch_out, num_cores, core_id, &s, &n);
  if (n)
    dc_perturb_i32(pAdd, addp, addm, pBMul, 1u, (seed_b_base + core_id) ^ tile_seed_xor, s, n, log2D_b,
                   eps_scale);
  pi_cl_team_barrier();
#ifdef DUALCONV2IN_PROBE
  if (core_id == 0)
    probe_t1 = pi_perf_read(PI_PERF_CYCLES);
#endif

  /* ---------------- phase 2: flat pixel range (or channel range, 5c), two pixels at a time, both passes -- */
  const uint32_t in_place = ((ch_in & 3u) == 0u) && ((((uint32_t)pInP | (uint32_t)pInM) & 3u) == 0u);
  const uint32_t in_row = (uint32_t)dim_in_x * ch_in;
  const uint32_t total = (uint32_t)dim_out_y * dim_out_x;
  /* QW (exp7 step 5c): split the PIXELS (each core: a pixel range, all channels) or the OUTPUT CHANNELS (each core:
   * a range of 4-channel blocks, all pixels). Measured (block 3 / block 4 phase 2): pixels 37.6 k / 20.3 k, channels
   * 40.0 k / 15.9 k -- one 4-channel block per pixel pair pays the per-pair overhead 8x as often, so channels only win
   * when the pixel split leaves cores idle (block 4: 10 pixels, chunk 2 -> 5 of 8 cores) and the channel split keeps
   * more of them busy. A cost model counting only 4x2 steps chose channels for block 3 too, wrongly. Needs
   * ch_out % 4 == 0. -- QW */
  const uint32_t nblk = ch_out >> 2;
  const uint32_t px_chunk = (total + num_cores - 1u) / num_cores;
  const uint32_t px_busy = px_chunk ? (total + px_chunk - 1u) / px_chunk : 0u;
  const uint32_t blk_chunk = (nblk + num_cores - 1u) / num_cores;
  const uint32_t ch_busy = blk_chunk ? (nblk + blk_chunk - 1u) / blk_chunk : 0u;
#if defined(DUALCONV2IN_SPLIT) && DUALCONV2IN_SPLIT == 1 /* measurement override: 1 = pixels, 2 = channels */
  const uint32_t split_ch = 0u;
  (void)px_busy;
  (void)ch_busy;
#elif defined(DUALCONV2IN_SPLIT) && DUALCONV2IN_SPLIT == 2
  const uint32_t split_ch = ((ch_out & 3u) == 0u);
  (void)px_busy;
  (void)ch_busy;
#else
  const uint32_t split_ch = ((ch_out & 3u) == 0u) && (px_busy < num_cores) && (ch_busy > px_busy);
#endif
  uint32_t p0, p1, c0, c1;
  if (split_ch) {
    p0 = 0;
    p1 = total;
    c0 = 4u * blk_chunk * core_id;
    if (c0 > ch_out)
      c0 = ch_out;
    c1 = c0 + 4u * blk_chunk;
    if (c1 > ch_out)
      c1 = ch_out;
    if (c0 == c1)
      p1 = 0; /* no channels for this core */
  } else {
    c0 = 0;
    c1 = ch_out;
    p0 = px_chunk * core_id;
    if (p0 > total)
      p0 = total;
    p1 = p0 + px_chunk;
    if (p1 > total)
      p1 = total;
  }
  uint32_t oy = p0 / dim_out_x;
  uint32_t ox = p0 - oy * dim_out_x;

  for (uint32_t p = p0; p < p1; p += 2u) {
    const uint32_t two = (p + 1u < p1); /* the last pixel of an odd range runs alone (B duplicates A) */
    /* receptive-field addresses: identical positions in x+ and x-, so the padding test is shared */
    const int8_t *fP[2], *fM[2];
    uint32_t fs[2];
    for (uint32_t q = 0; q < 2u; q++) {
      if (q == 1u && !two) {
        fP[1] = fP[0];
        fM[1] = fM[0];
        fs[1] = fs[0];
        break;
      }
      const int32_t iy0 = (int32_t)(oy * stride_y) - (int32_t)pad_y_top;
      const int32_t ix0 = (int32_t)(ox * stride_x) - (int32_t)pad_x_left;
      if (in_place && iy0 >= 0 && iy0 + (int32_t)dim_kernel_y <= (int32_t)dim_in_y && ix0 >= 0 &&
          ix0 + (int32_t)dim_kernel_x <= (int32_t)dim_in_x) {
        const uint32_t off = ((uint32_t)iy0 * dim_in_x + (uint32_t)ix0) * ch_in;
        fP[q] = pInP + off;
        fM[q] = pInM + off;
        fs[q] = in_row;
      } else {
        int8_t *pa = patch + q * K;
        int8_t *pm = patch + (2u + q) * K;
        d2_gather(pInP, pa, iy0, ix0, dim_in_x, dim_in_y, ch_in, dim_kernel_x, dim_kernel_y);
        d2_gather(pInM, pm, iy0, ix0, dim_in_x, dim_in_y, ch_in, dim_kernel_x, dim_kernel_y);
        fP[q] = pa;
        fM[q] = pm;
        fs[q] = row_len;
      }
      if (++ox == dim_out_x) {
        ox = 0;
        oy++;
      }
    }

    /* ---- both passes with ONE copy of the 4x2 loop: pass 0 = (x+, W+, add+), pass 1 = (x-, W-, add-) ---- */
    for (uint32_t pass = 0; pass < 2u; pass++) {
      const int8_t *srcA = pass ? fM[0] : fP[0];
      const int8_t *srcB = pass ? fM[1] : fP[1];
      const int8_t *W = pass ? wm : wp;
      const int32_t *addv = pass ? addm : addp;
      int8_t *oA = (pass ? pOutM : pOutP) + p * out_stride;
      int8_t *oB = oA + out_stride;
      const uint32_t sA = fs[0], sB = fs[1];

      uint32_t c = c0;
      for (; c + 4u <= c1; c += 4u) {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        const int8_t *w = W + c * K;
        for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
          const d2_v4s *xa = (const d2_v4s *)(srcA + ky * sA);
          const d2_v4s *xb = (const d2_v4s *)(srcB + ky * sB);
          const d2_v4s *w0 = (const d2_v4s *)(w + ky * row_len);
          const d2_v4s *w1 = (const d2_v4s *)(w + K + ky * row_len);
          const d2_v4s *w2 = (const d2_v4s *)(w + 2u * K + ky * row_len);
          const d2_v4s *w3 = (const d2_v4s *)(w + 3u * K + ky * row_len);
          for (uint32_t k = 0; k < row_words; k++) {
            const d2_v4s va = *xa++;
            const d2_v4s vb = *xb++;
            const d2_v4s v0 = *w0++;
            const d2_v4s v1 = *w1++;
            const d2_v4s v2 = *w2++;
            const d2_v4s v3 = *w3++;
            a0 = __builtin_pulp_sdotsp4(va, v0, a0);
            b0 = __builtin_pulp_sdotsp4(vb, v0, b0);
            a1 = __builtin_pulp_sdotsp4(va, v1, a1);
            b1 = __builtin_pulp_sdotsp4(vb, v1, b1);
            a2 = __builtin_pulp_sdotsp4(va, v2, a2);
            b2 = __builtin_pulp_sdotsp4(vb, v2, b2);
            a3 = __builtin_pulp_sdotsp4(va, v3, a3);
            b3 = __builtin_pulp_sdotsp4(vb, v3, b3);
          }
        }
        oA[c] = D2_RQ(a0, pMul[c], addv[c]);
        oA[c + 1u] = D2_RQ(a1, pMul[c + 1u], addv[c + 1u]);
        oA[c + 2u] = D2_RQ(a2, pMul[c + 2u], addv[c + 2u]);
        oA[c + 3u] = D2_RQ(a3, pMul[c + 3u], addv[c + 3u]);
        if (two) {
          oB[c] = D2_RQ(b0, pMul[c], addv[c]);
          oB[c + 1u] = D2_RQ(b1, pMul[c + 1u], addv[c + 1u]);
          oB[c + 2u] = D2_RQ(b2, pMul[c + 2u], addv[c + 2u]);
          oB[c + 3u] = D2_RQ(b3, pMul[c + 3u], addv[c + 3u]);
        }
      }
      /* channel tail (ch_out % 4, pixel split only), 1 x 2 -- empty for every SpeechNet block */
      for (; c < c1; c++) {
        int32_t a = 0, b = 0;
        const int8_t *w = W + c * K;
        for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
          const d2_v4s *xa = (const d2_v4s *)(srcA + ky * sA);
          const d2_v4s *xb = (const d2_v4s *)(srcB + ky * sB);
          const d2_v4s *w0 = (const d2_v4s *)(w + ky * row_len);
          for (uint32_t k = 0; k < row_words; k++) {
            const d2_v4s v0 = *w0++;
            a = __builtin_pulp_sdotsp4(*xa++, v0, a);
            b = __builtin_pulp_sdotsp4(*xb++, v0, b);
          }
        }
        oA[c] = D2_RQ(a, pMul[c], addv[c]);
        if (two)
          oB[c] = D2_RQ(b, pMul[c], addv[c]);
      }
    }
  }
  pi_cl_team_barrier();
#ifdef DUALCONV2IN_PROBE
  if (core_id == 0) {
    const uint32_t probe_t2 = pi_perf_read(PI_PERF_CYCLES);
    printf("[DC2PROBE] phase1 %u phase2 %u (core-0 cycles, K=%u ch_out=%u)\n", (unsigned)(probe_t1 - probe_t0),
           (unsigned)(probe_t2 - probe_t1), (unsigned)K, (unsigned)ch_out);
  }
#endif
}
