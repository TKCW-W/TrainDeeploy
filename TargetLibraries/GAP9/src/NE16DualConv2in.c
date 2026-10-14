/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp7, step 7): BOTH ZO passes of a 1xK / Kx1 quantized conv (SpeechNet blocks 1-4) on NE16
 * in ONE node, with the GAP9 SDK's pipeline (Ker_MM_Conv1DSmallv2_NE16):
 *
 *   - the column buffer is built from the HWC input, WORD-WISE: in HWC the valid taps of one kernel row are ONE
 *     contiguous run of (kx_hi - kx_lo)*ch_in bytes, so per row: fill the padded words with the pad value 0x80808080,
 *     copy the run as words XOR 0x80808080 (= x + 128, NE16 reads unsigned). Column order (ky, kx, ci), so the weight
 *     is encoded in the same order (NE16DualPerturbEncode with ne16_hwc = 1). exp6's Im2ColU8.c built the column from
 *     NCHW byte by byte (block 1: 291 k cycles per pass on GAP9);
 *   - one NE16 job = 3 x w consecutive output pixels of one pass (w = job_w, from an L1 budget; a flat pixel range
 *     presented to NE16 as a 3 x w block, so its 3x3 output subtiles are full), a 1x1 conv over K = ch_in*kh*kw input
 *     channels; the remainder of a pass is at most one 3 x (r/3) and one 1 x (r%3) job. Pass + first;
 *   - DOUBLE BUFFERED (pipelined = 1): two column buffers; while NE16 computes job j from one, the 8 workers build the
 *     column of job j+1 into the other. Job j+1 is queued behind job j (NE16's 2-deep job queue), so NE16 starts it
 *     without waiting for the cores. Buffer reuse needs no job ids (GVSoC's running_id is broken): right after job j
 *     is dispatched, ne16_nnx_dispatch_wait() returns only when a queue slot is free, i.e. when job j-1 -- the last
 *     user of the buffer job j+1 will fill -- has finished. pipelined = 0 runs the same jobs strictly in sequence
 *     (build, dispatch, wait) to measure what the overlap is worth.
 *
 * Output: HWC [dim_out_y][dim_out_x][2*ch_out], A+ in channels [0, ch_out), A- in [ch_out, 2*ch_out) -- NE16 writes each
 * pass with an output pixel stride of 2*ch_out. Requant on NE16 (32-bit scale per channel, bias, shift), configured
 * exactly as Deeploy's pointwise template (conf0 is computed by the Deeploy template and passed in).
 * Inputs: x+ / x- HWC int8; pWEnc2 = [W+; W-] encoded, 2*ch_out rows; pAdd2 = [add+'; add-'] (signed-input corrected).
 * Runs on the cluster MASTER core (NE16 engine); pColBuf = 2 * 3 * job_w * K bytes of L1.
 * Built only on GAP9_w_NE16 (pulp-nnx present). -- QW
 */

#if __has_include("pulp_nnx_ne16.h")

#include "DeeployGAP9Math.h"
#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "pmsis.h"
#include "pulp_nnx_ne16.h"
#include "pulp_nnx_util.h"

#define N16D_PAD_WORD 0x80808080u

typedef struct {
  const int8_t *x;
  uint8_t *col;
  uint32_t p0, np; /* flat output-pixel range [p0, p0+np) of the tile, may span rows */
  uint32_t in_stride; /* bytes between input pixels: ch_in, or 2*ch_in for the stacked [x+ | x-] input (exp8) */
  uint16_t dim_in_y, dim_in_x, ch_in, dim_out_x, kh, kw, pad_top, pad_left, stride_y, stride_x;
} n16d_col_args_t;

/* worker body: the columns of a flat range of output pixels, split over the cores */
static void n16d_build_col(void *arg) {
  const n16d_col_args_t *a = (const n16d_col_args_t *)arg;
  const uint32_t core = pi_core_id(), cores = pi_cl_cluster_nb_cores();
  const uint32_t cw = (uint32_t)a->ch_in >> 2;          /* words per tap */
  const uint32_t K = (uint32_t)a->kh * a->kw * a->ch_in;
  const uint32_t chunk = (a->np + cores - 1u) / cores;
  uint32_t q0 = chunk * core, q1 = q0 + chunk;
  if (q0 > a->np)
    q0 = a->np;
  if (q1 > a->np)
    q1 = a->np;
  uint32_t oy = (a->p0 + q0) / a->dim_out_x;
  uint32_t ox = (a->p0 + q0) - oy * a->dim_out_x;
  for (uint32_t q = q0; q < q1; q++) {
    uint32_t *dst = (uint32_t *)(a->col + q * K);
    const int32_t iy0 = (int32_t)(oy * a->stride_y) - (int32_t)a->pad_top;
    const int32_t ix0 = (int32_t)(ox * a->stride_x) - (int32_t)a->pad_left;
    int32_t lo = -ix0, hi = (int32_t)a->dim_in_x - ix0;
    if (lo < 0)
      lo = 0;
    if (hi > (int32_t)a->kw)
      hi = (int32_t)a->kw;
    if (hi < lo)
      hi = lo;
    for (uint32_t ky = 0; ky < a->kh; ky++) {
      const int32_t iy = iy0 + (int32_t)ky;
      if (iy < 0 || iy >= (int32_t)a->dim_in_y) {
        for (uint32_t k = 0; k < (uint32_t)a->kw * cw; k++)
          *dst++ = N16D_PAD_WORD;
        continue;
      }
      for (uint32_t k = 0; k < (uint32_t)lo * cw; k++)
        *dst++ = N16D_PAD_WORD;
      if (a->in_stride == a->ch_in) { /* one contiguous run */
        const uint32_t *src =
            (const uint32_t *)(a->x + ((uint32_t)iy * a->dim_in_x + (uint32_t)(ix0 + lo)) * a->ch_in);
        for (uint32_t k = 0; k < (uint32_t)(hi - lo) * cw; k++)
          *dst++ = *src++ ^ N16D_PAD_WORD;
      } else { /* exp8: stacked input -- cw words per tap, taps in_stride apart */
        for (int32_t kx = lo; kx < hi; kx++) {
          const uint32_t *src =
              (const uint32_t *)(a->x + ((uint32_t)iy * a->dim_in_x + (uint32_t)(ix0 + kx)) * a->in_stride);
          for (uint32_t k = 0; k < cw; k++)
            *dst++ = src[k] ^ N16D_PAD_WORD;
        }
      }
      for (uint32_t k = 0; k < (uint32_t)((int32_t)a->kw - hi) * cw; k++)
        *dst++ = N16D_PAD_WORD;
    }
    if (++ox == a->dim_out_x) {
      ox = 0;
      oy++;
    }
  }
}

/* one NE16 job: a 1x1 conv over the columns of h x w consecutive output pixels (K input channels), presented to NE16
 * as an h x w block -- the field values are Deeploy's NE162DPWConvTemplate for an h x w tile, with the output pixel
 * stride 2*ch_out (A+ and A- interleaved per pixel). The output pixels of a tile are contiguous in HWC, so a flat pixel
 * range is a valid h x w block with row stride w * pixel stride. */
static void n16d_task(ne16_task_t *t, const uint8_t *col, const uint8_t *w, int8_t *out, const int32_t *mul,
                      const int32_t *add, uint32_t h, uint32_t wd, uint32_t K, uint32_t ch_out, uint32_t out_px_stride,
                      uint32_t conf0, int32_t weight_offset) {
  const uint32_t nKi = (K + 15u) / 16u, nKo = (ch_out + 31u) / 32u, nHo = (h + 2u) / 3u, nWo = (wd + 2u) / 3u;
  const uint32_t bKi = (K % 16u) ? (K % 16u) : 16u, bKo = (ch_out % 32u) ? (ch_out % 32u) : 32u;
  const uint32_t bHo = (h % 3u) ? (h % 3u) : 3u, bWo = (wd % 3u) ? (wd % 3u) : 3u;
  t->data.weights_addr = (uint32_t)w;
  t->data.infeat_addr = (uint32_t)col;
  t->data.outfeat_addr = (uint32_t)out;
  t->data.scale_addr = (uint32_t)mul;
  t->data.scale_shift_addr = (uint32_t)NULL;
  t->data.scale_bias_addr = (uint32_t)add;
  t->data.cfg.input_stride = (ne16_stride_t){.d0 = K, .d1 = wd * K, .d2 = 0};
  t->data.cfg.output_stride =
      (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = out_px_stride, .d2 = wd * out_px_stride};
  t->data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16 * nKi, .d2 = 0};
  t->data.cfg.subtile.number.KoKi = nnx_concat_half(nKo, nKi);
  t->data.cfg.subtile.number.HoWo = nnx_concat_half(nHo, nWo);
  t->data.cfg.subtile.remainder.KoKi = nnx_concat_half(bKo, bKi);
  t->data.cfg.subtile.remainder.HoWo = nnx_concat_half(bHo, bWo);
  t->data.cfg.subtile.remainder.HiWi = nnx_concat_half(bHo, bWo);
  t->data.cfg.padding = 0;
  t->data.cfg.weight_offset_factor = weight_offset;
  t->data.cfg.filter_mask = 0;
  t->data.cfg.conf0 = conf0;
  t->weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
  t->qw = 8;
  t->subtile_output_channel = 32;
  t->kernel_shape = 1;
  t->depthwise = 0;
}

/* jobs of one pass over `total` output pixels: n_full jobs of 3 x wj, then the remainder r as 3 x (r/3) and 1 x (r%3)
 * (at most one each) -- a single 1 x r job would leave two of three subtile rows idle. */
static uint32_t n16d_jobs_per_pass(uint32_t total, uint32_t wj) {
  const uint32_t n_full = total / (3u * wj), r = total - n_full * 3u * wj;
  return n_full + ((r / 3u) ? 1u : 0u) + ((r % 3u) ? 1u : 0u);
}

static void n16d_job_geometry(uint32_t j, uint32_t total, uint32_t wj, uint32_t *p0, uint32_t *h, uint32_t *wd) {
  const uint32_t n_full = total / (3u * wj), r = total - n_full * 3u * wj;
  if (j < n_full) {
    *p0 = j * 3u * wj;
    *h = 3u;
    *wd = wj;
  } else if (j == n_full && (r / 3u)) {
    *p0 = n_full * 3u * wj;
    *h = 3u;
    *wd = r / 3u;
  } else {
    *p0 = total - (r % 3u);
    *h = 1u;
    *wd = r % 3u;
  }
}

void PULP_NE16DualConv2in(const int8_t *pInP, const int8_t *pInM, const uint8_t *pWEnc2, const int32_t *pMul,
                          const int32_t *pAdd2, int8_t *pOut, uint8_t *pColBuf, uint32_t num_cores, uint32_t conf0,
                          uint16_t dim_in_y, uint16_t dim_in_x, uint16_t ch_in, uint16_t dim_out_y, uint16_t dim_out_x,
                          uint16_t ch_out, uint16_t dim_kernel_y, uint16_t dim_kernel_x, uint16_t pad_top,
                          uint16_t pad_bottom, uint16_t pad_left, uint16_t pad_right, uint16_t stride_y,
                          uint16_t stride_x, int32_t weight_offset, uint32_t pipelined, uint32_t job_w,
                          uint32_t in_stride) {
  (void)pad_bottom;
  (void)pad_right;
  const ne16_dev_t *dev = ne16_pulp_get_dev();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_y * dim_kernel_x;
  const uint32_t wsize = (uint32_t)ch_out * ((K + 15u) / 16u) * 16u; /* encoded bytes of one pass */
  const uint32_t ch2 = 2u * ch_out;
  /* job = 3 x job_w output pixels (NE16 computes 3x3 output subtiles); job_w is set by the template from an L1 budget
   * (two buffers of 3 * job_w * K bytes) */
  const uint32_t wj = job_w ? job_w : 1u;
  uint8_t *col[2] = {pColBuf, pColBuf + 3u * wj * K};
  const uint32_t total = (uint32_t)dim_out_y * dim_out_x;
  const uint32_t jpp = n16d_jobs_per_pass(total, wj); /* jobs per pass */
  const uint32_t njobs = 2u * jpp;
  uint32_t p0, h, wd;
  n16d_job_geometry(0, total, wj, &p0, &h, &wd);
  n16d_col_args_t a = {.x = pInP, .col = col[0], .p0 = p0, .np = h * wd, .in_stride = in_stride,
                       .dim_in_y = dim_in_y, .dim_in_x = dim_in_x,
                       .ch_in = ch_in, .dim_out_x = dim_out_x, .kh = dim_kernel_y, .kw = dim_kernel_x,
                       .pad_top = pad_top, .pad_left = pad_left, .stride_y = stride_y, .stride_x = stride_x};
  ne16_task_t task;

  /* columns of job 0 */
  pi_cl_team_fork(num_cores, n16d_build_col, &a);

  for (uint32_t j = 0; j < njobs; j++) {
    const uint32_t pass = j / jpp;
    n16d_job_geometry(j - pass * jpp, total, wj, &p0, &h, &wd);
    n16d_task(&task, col[j & 1u], pWEnc2 + pass * wsize, pOut + p0 * ch2 + pass * ch_out, pMul, pAdd2 + pass * ch_out,
              h, wd, K, ch_out, ch2, conf0, weight_offset);
    ne16_nnx_dispatch_wait(dev);
    ne16_nnx_dispatch(dev, &task);
    if (!pipelined)
      ne16_nnx_resolve_wait(dev, &task); /* sequential: job j done before its successor's columns are built */
    if (j + 1u < njobs) {
      if (pipelined)
        ne16_nnx_dispatch_wait(dev); /* a free slot => job j-1 done => col[(j+1)&1] is free; job j keeps running */
      const uint32_t pn = (j + 1u) / jpp;
      n16d_job_geometry(j + 1u - pn * jpp, total, wj, &p0, &h, &wd);
      a.x = pn ? pInM : pInP;
      a.p0 = p0;
      a.np = h * wd;
      a.col = col[(j + 1u) & 1u];
      pi_cl_team_fork(num_cores, n16d_build_col, &a);
    }
  }
  ne16_nnx_resolve_wait(dev, &task); /* GVSoC: waits until the queue is empty */
}

#endif /* pulp-nnx present */
