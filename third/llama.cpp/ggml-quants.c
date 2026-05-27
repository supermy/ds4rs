// Extracted from llama.cpp ggml/src/ggml-quants.c
// Contains quantization/dequantization code for IQ2_XXS, IQ2_XS, and Q2_K formats.
//
// Original source: https://github.com/ggerganov/llama.cpp
// License: MIT
//
// This file is self-contained: it includes FP16 conversion, helper functions,
// kgrid lookup tables, and all quantize/dequantize functions for the above types.
// It depends on ggml-common.h for block structures (block_q2_K, block_iq2_xxs,
// block_iq2_xs) and grid tables (iq2xxs_grid, iq2xs_grid, ksigns_iq2xs, kmask_iq2xs).

#define GGML_COMMON_IMPL_C
#include "ggml-quants.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stdlib.h> // for qsort, malloc, free
#include <stdio.h>  // for printf (used in quantize error messages)

// ============================================================================
// FP16 <-> FP32 conversion (from ggml-impl.h)
// ============================================================================

static inline float fp32_from_bits(uint32_t w) {
    union { uint32_t as_bits; float as_value; } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union { float as_value; uint32_t as_bits; } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_half h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_half ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

// ============================================================================
// Compatibility macros and constants
// ============================================================================

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define GGML_UNUSED(x) (void)(x)
#define UNUSED GGML_UNUSED

#define GGML_ASSERT(x) assert(x)
#define GGML_ABORT(msg) do { fprintf(stderr, "ABORT: %s\n", msg); abort(); } while(0)

// GGML type enum values (subset needed for IQ2 grid init)
enum {
    EXTRACTED_GGML_TYPE_Q2_K    = 10,
    EXTRACTED_GGML_TYPE_IQ2_XXS = 16,
    EXTRACTED_GGML_TYPE_IQ2_XS  = 17,
    EXTRACTED_GGML_TYPE_IQ1_S   = 19,
    EXTRACTED_GGML_TYPE_IQ2_S   = 22,
    EXTRACTED_GGML_TYPE_IQ1_M   = 29,
};

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ2_S 1e-8f

// NGRID_IQ1S needed by iq2_grid_size for IQ1_S/IQ1_M types
#ifndef NGRID_IQ1S
#define NGRID_IQ1S 2048
#endif

// ============================================================================
// Helper functions
// ============================================================================

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// make_qkx2_quants: used by quantize_row_q2_K_ref
static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; i++) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; i++) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// make_qkx3_quants: used by quantize_row_q2_K_impl
static float make_qkx3_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0]*x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        float w = weights ? weights[i] : x[i]*x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                float w = weights ? weights[i] : x[i]*x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; i++) L[i] = Laux[i];
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// make_qp_quants: used by quantize_row_q2_K_impl
static float make_qp_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, const float * quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    if (max < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        float w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            float diff = x[i] - scale_is*l;
            float w = quant_weights[i];
            mse += w*diff*diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

// ============================================================================
// Q2_K dequantization
// ============================================================================

void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

// ============================================================================
// Q2_K quantization (reference, no importance matrix)
// ============================================================================

void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) max_scale = scale;
            float min = mins[j];
            if (min > max_min) max_min = min;
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

// ============================================================================
// Q2_K quantization (with importance matrix)
// ============================================================================

static void quantize_row_q2_K_impl(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int k, const float * GGML_RESTRICT quant_weights) {
    GGML_ASSERT(quant_weights);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    const bool requantize = true;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float mins[QK_K/16];
    float scales[QK_K/16];
    float sw[QK_K/16];
    float weight[16];
    uint8_t Ls[QK_K/16], Lm[QK_K/16];

    for (int i = 0; i < nb; i++) {
        memset(sw, 0, QK_K/16*sizeof(float));
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += x[j]*x[j];
        float sigma2 = sumx2/QK_K;
        for (int j = 0; j < QK_K/16; ++j) {
            const float * GGML_RESTRICT qw = quant_weights + QK_K * i + 16*j;
            for (int l = 0; l < 16; ++l) weight[l] = qw[l] * sqrtf(sigma2 + x[16*j + l]*x[16*j + l]);
            for (int l = 0; l < QK_K/16; ++l) sw[j] += weight[l];
            scales[j] = make_qkx3_quants(16, 3, x + 16*j, weight, L + 16*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float dm, mm;
        dm  = make_qp_quants(QK_K/16, 15, scales, Ls, sw);
        mm  = make_qp_quants(QK_K/16, 15, mins,   Lm, sw);

        y[i].d    = GGML_FP32_TO_FP16(dm);
        y[i].dmin = GGML_FP32_TO_FP16(mm);
        dm        = GGML_FP16_TO_FP32(y[i].d);
        mm        = GGML_FP16_TO_FP32(y[i].dmin);

        for (int j = 0; j < QK_K/16; ++j) {
            y[i].scales[j] = Ls[j] | (Lm[j] << 4);
        }

        if (requantize) {
            for (int j = 0; j < QK_K/16; ++j) {
                const float d = dm * (y[i].scales[j] & 0xF);
                if (!d) continue;
                const float m = mm * (y[i].scales[j] >> 4);
                for (int ii = 0; ii < 16; ++ii) {
                    int l = nearest_int((x[16*j + ii] + m)/d);
                    l = MAX(0, MIN(3, l));
                    L[16*j + ii] = l;
                }
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    // Compute row size: sizeof(block_q2_K) per QK_K elements
    const size_t row_size = (n_per_row / QK_K) * sizeof(block_q2_K);
    if (!quant_weights) {
        quantize_row_q2_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    else {
        char * qrow = (char *)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q2_K_impl(src, (block_q2_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

// ============================================================================
// IQ2_XXS dequantization
// ============================================================================

void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ============================================================================
// IQ2_XS dequantization
// ============================================================================

void dequantize_row_iq2_xs(const block_iq2_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ============================================================================
// IQ2 grid data and initialization (for quantization)
// ============================================================================

typedef struct {
    uint64_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq2_entry_t;

static iq2_entry_t iq2_data[4] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int iq2_data_index(int type) {
    GGML_ASSERT(type == EXTRACTED_GGML_TYPE_IQ2_XXS || type == EXTRACTED_GGML_TYPE_IQ2_XS ||
                type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M ||
                type == EXTRACTED_GGML_TYPE_IQ2_S);
    return type == EXTRACTED_GGML_TYPE_IQ2_XXS ? 0 :
           type == EXTRACTED_GGML_TYPE_IQ2_XS  ? 1 :
           type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M ? 2 : 3;
}

static inline int iq2_grid_size(int type) {
    GGML_ASSERT(type == EXTRACTED_GGML_TYPE_IQ2_XXS || type == EXTRACTED_GGML_TYPE_IQ2_XS ||
                type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M ||
                type == EXTRACTED_GGML_TYPE_IQ2_S);
    return type == EXTRACTED_GGML_TYPE_IQ2_XXS ? 256 :
           type == EXTRACTED_GGML_TYPE_IQ2_XS  ? 512 :
           type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M ? NGRID_IQ1S : 1024;
}

static int iq2_compare_func(const void * left, const void * right) {
    const int * l = (const int *)left;
    const int * r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq2xs_init_impl(int type) {
    const int gindex = iq2_data_index(type);
    const int grid_size = iq2_grid_size(type);
    if (iq2_data[gindex].grid) {
        return;
    }
    static const uint16_t kgrid_2bit_256[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };
    static const uint16_t kgrid_2bit_512[512] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
          260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
          352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
          640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
         1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
         1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
         2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
         2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
         4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
         4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
         5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
         5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
         8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
         8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
        10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
        16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
        16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
        16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
        17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
        18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
        20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
        21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
        22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
        24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
        32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
        33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
        33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
        35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
        37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
        40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
        42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
    };
    static const uint16_t kgrid_2bit_1024[1024] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   102,   105,   128,   130,   133,   136,   145,   148,   160,
          165,   170,   257,   260,   262,   265,   272,   274,   277,   280,   289,   292,   320,   322,   325,   328,
          337,   340,   342,   345,   352,   357,   360,   385,   388,   400,   402,   405,   417,   420,   512,   514,
          517,   520,   529,   532,   544,   554,   577,   580,   582,   585,   592,   597,   640,   645,   650,   660,
          674,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1062,  1065,  1088,  1090,  1093,
         1096,  1098,  1105,  1108,  1110,  1113,  1120,  1122,  1125,  1153,  1156,  1158,  1161,  1168,  1173,  1176,
         1185,  1188,  1280,  1282,  1285,  1288,  1290,  1297,  1300,  1302,  1305,  1312,  1317,  1320,  1345,  1348,
         1350,  1353,  1360,  1362,  1365,  1368,  1377,  1380,  1408,  1410,  1413,  1416,  1425,  1428,  1440,  1537,
         1540,  1542,  1545,  1552,  1557,  1600,  1605,  1608,  1617,  1620,  1632,  1665,  1668,  1680,  2048,  2050,
         2053,  2056,  2065,  2068,  2070,  2073,  2080,  2085,  2090,  2113,  2116,  2118,  2121,  2128,  2130,  2133,
         2136,  2145,  2148,  2176,  2181,  2196,  2218,  2305,  2308,  2320,  2322,  2325,  2328,  2337,  2368,  2373,
         2376,  2385,  2388,  2400,  2433,  2448,  2560,  2577,  2580,  2594,  2600,  2602,  2640,  2713,  4097,  4100,
         4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4134,  4160,  4162,  4165,  4168,  4177,  4180,  4182,
         4185,  4192,  4194,  4197,  4200,  4225,  4228,  4230,  4240,  4245,  4248,  4257,  4260,  4352,  4354,  4357,
         4360,  4362,  4369,  4372,  4374,  4377,  4384,  4386,  4389,  4392,  4417,  4420,  4422,  4425,  4432,  4434,
         4437,  4440,  4449,  4452,  4480,  4482,  4485,  4488,  4497,  4500,  4609,  4612,  4617,  4624,  4629,  4641,
         4644,  4672,  4677,  4689,  4692,  4737,  4740,  4752,  5120,  5122,  5125,  5128,  5137,  5140,  5142,  5145,
         5152,  5157,  5160,  5185,  5188,  5190,  5193,  5200,  5202,  5205,  5208,  5217,  5220,  5248,  5250,  5253,
         5256,  5265,  5268,  5280,  5377,  5380,  5382,  5385,  5392,  5394,  5397,  5400,  5409,  5412,  5440,  5442,
         5445,  5448,  5457,  5460,  5472,  5505,  5508,  5520,  5632,  5637,  5640,  5649,  5652,  5664,  5697,  5700,
         5712,  5760,  5802,  6145,  6148,  6150,  6153,  6160,  6165,  6168,  6177,  6208,  6210,  6213,  6216,  6225,
         6228,  6240,  6273,  6276,  6400,  6402,  6405,  6408,  6417,  6420,  6432,  6465,  6468,  6480,  6505,  6562,
         6660,  6672,  6720,  6742,  8192,  8194,  8197,  8200,  8209,  8212,  8214,  8217,  8224,  8229,  8234,  8257,
         8260,  8272,  8274,  8277,  8292,  8320,  8330,  8340,  8362,  8449,  8452,  8464,  8466,  8469,  8481,  8512,
         8514,  8517,  8529,  8532,  8544,  8577,  8580,  8592,  8704,  8714,  8738,  8744,  8746,  8772,  8784,  8840,
         8842,  8872,  9217,  9220,  9222,  9225,  9232,  9237,  9240,  9249,  9252,  9280,  9282,  9285,  9288,  9297,
         9300,  9312,  9345,  9348,  9360,  9472,  9477,  9480,  9489,  9492,  9504,  9537,  9540,  9552,  9574,  9600,
         9729,  9732,  9744,  9792,  9817, 10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497, 10500,
        10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16410,
        16417, 16420, 16422, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470, 16473, 16480, 16482, 16485, 16513,
        16516, 16528, 16533, 16536, 16545, 16548, 16640, 16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674,
        16677, 16705, 16708, 16710, 16713, 16720, 16722, 16725, 16728, 16737, 16740, 16768, 16770, 16773, 16776, 16785,
        16788, 16800, 16897, 16900, 16912, 16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992, 17025,
        17028, 17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430, 17433, 17440, 17442, 17445, 17448, 17473, 17476,
        17478, 17481, 17488, 17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544, 17553, 17556, 17568, 17665,
        17668, 17670, 17673, 17680, 17682, 17685, 17688, 17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760,
        17770, 17793, 17796, 17808, 17920, 17922, 17925, 17928, 17937, 17940, 17952, 17985, 17988, 18000, 18048, 18085,
        18433, 18436, 18441, 18448, 18450, 18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516, 18528,
        18564, 18576, 18688, 18690, 18693, 18696, 18705, 18708, 18720, 18753, 18756, 18768, 18816, 18838, 18945, 18948,
        18960, 19008, 20480, 20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517, 20520, 20545, 20548,
        20550, 20553, 20560, 20562, 20565, 20568, 20577, 20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740,
        20742, 20745, 20752, 20754, 20757, 20760, 20769, 20772, 20800, 20802, 20805, 20808, 20817, 20820, 20832, 20865,
        20868, 20880, 20992, 20997, 21000, 21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508, 21510,
        21513, 21520, 21522, 21525, 21528, 21537, 21540, 21568, 21570, 21573, 21576, 21585, 21588, 21600, 21633, 21636,
        21648, 21760, 21762, 21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017, 22020, 22032, 22054,
        22080, 22528, 22530, 22533, 22536, 22545, 22548, 22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800,
        22848, 23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592, 24594, 24597, 24600, 24609, 24612, 24640, 24645,
        24648, 24657, 24660, 24672, 24708, 24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900, 24912,
        24960, 24985, 25092, 25104, 25152, 25174, 25249, 25600, 25605, 25608, 25617, 25620, 25632, 25665, 25668, 25680,
        25728, 25857, 25860, 25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640, 26725, 26776, 26880,
        26922, 27202, 27297, 32768, 32770, 32773, 32776, 32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850,
        32853, 32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033, 33040, 33042, 33045, 33048, 33057, 33060,
        33088, 33090, 33093, 33096, 33105, 33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300, 33345,
        33348, 33360, 33793, 33796, 33798, 33801, 33808, 33810, 33813, 33816, 33825, 33856, 33858, 33861, 33864, 33873,
        33876, 33888, 33921, 33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113, 34116, 34128, 34176,
        34186, 34305, 34308, 34320, 34345, 34368, 34816, 34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076,
        35136, 35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880, 36882, 36885, 36888, 36900, 36928,
        36930, 36933, 36936, 36945, 36948, 36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188, 37200,
        37210, 37377, 37380, 37392, 37440, 37542, 37888, 37890, 37893, 37896, 37905, 37908, 37920, 37953, 37956, 37968,
        38016, 38038, 38145, 38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916, 38928, 38950, 38976,
        39081, 39168, 39241, 39250, 39568, 40960, 40965, 40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130,
        41280, 41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640, 41985, 41988, 42000, 42048, 42121,
        42148, 42240, 42265, 42577, 43018, 43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656, 43690,
    };

    const int kmap_size = 43692;
    const int nwant = type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M ? 3 : type == EXTRACTED_GGML_TYPE_IQ2_S ? 1 : 2;
    const uint16_t * kgrid = type == EXTRACTED_GGML_TYPE_IQ2_XXS ? kgrid_2bit_256 :
                             type == EXTRACTED_GGML_TYPE_IQ2_XS  ? kgrid_2bit_512 :
                             type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M ? NULL : kgrid_2bit_1024;
    // Note: IQ1_S/IQ1_M use kgrid_1bit_2048 which is not included in this extraction.
    // If you need IQ1_S/IQ1_M quantization, you must provide the 1-bit grid separately.
    if (type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M) {
        GGML_ABORT("IQ1_S/IQ1_M grid not included in this extraction");
    }

    uint64_t * kgrid_q2xs;
    int      * kmap_q2xs;
    uint16_t * kneighbors_q2xs;

    uint64_t * the_grid = (uint64_t *)malloc(grid_size*sizeof(uint64_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        for (int i = 0; i < 8; ++i) {
            int l = (kgrid[k] >> 2*i) & 0x3;
            pos[i] = 2*l + 1;
        }
    }
    kgrid_q2xs = the_grid;
    iq2_data[gindex].grid = the_grid;
    kmap_q2xs = (int *)malloc(kmap_size*sizeof(int));
    iq2_data[gindex].map = kmap_q2xs;
    for (int i = 0; i < kmap_size; ++i) kmap_q2xs[i] = -1;
    uint64_t aux64;
    uint8_t * aux8 = (uint8_t *)&aux64;
    for (int i = 0; i < grid_size; ++i) {
        aux64 = kgrid_q2xs[i];
        uint16_t index = 0;
        for (int k=0; k<8; ++k) {
            uint16_t q = (aux8[k] - 1)/2;
            index |= (q << 2*k);
        }
        kmap_q2xs[index] = i;
    }
    int8_t pos[8];
    int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
    int num_neighbors = 0, num_not_in_map = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q2xs[i] >= 0) continue;
        ++num_not_in_map;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2*k) & 0x3;
            pos[k] = 2*l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
        int n = 0; int d2 = dist2[0];
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2*j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    kneighbors_q2xs = (uint16_t *)malloc((num_neighbors + num_not_in_map)*sizeof(uint16_t));
    iq2_data[gindex].neighbours = kneighbors_q2xs;
    int counter = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q2xs[i] >= 0) continue;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2*k) & 0x3;
            pos[k] = 2*l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
        kmap_q2xs[i] = -(counter + 1);
        int d2 = dist2[0];
        uint16_t * start = &kneighbors_q2xs[counter++];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2*j];
                ++nhave;
            }
            kneighbors_q2xs[counter++] = dist2[2*j+1];
            ++n;
        }
        *start = n;
    }
    free(dist2);
}

void iq2xs_free_impl(int type) {
    GGML_ASSERT(type == EXTRACTED_GGML_TYPE_IQ2_XXS || type == EXTRACTED_GGML_TYPE_IQ2_XS ||
                type == EXTRACTED_GGML_TYPE_IQ1_S || type == EXTRACTED_GGML_TYPE_IQ1_M ||
                type == EXTRACTED_GGML_TYPE_IQ2_S);
    const int gindex = iq2_data_index(type);
    if (iq2_data[gindex].grid) {
        free(iq2_data[gindex].grid);       iq2_data[gindex].grid = NULL;
        free(iq2_data[gindex].map);        iq2_data[gindex].map  = NULL;
        free(iq2_data[gindex].neighbours); iq2_data[gindex].neighbours = NULL;
    }
}

// ============================================================================
// IQ2 quantization helper: find best neighbor
// ============================================================================

static int iq2_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint64_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

// ============================================================================
// IQ2_XXS quantization (with importance matrix)
// ============================================================================

static void quantize_row_iq2_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq2_data_index(EXTRACTED_GGML_TYPE_IQ2_XXS);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call iq2xs_init_impl()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call iq2xs_init_impl()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call iq2xs_init_impl()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_xxs * y = vy;

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            const float * qw = quant_weights + QK_K*ibl + 32*ib;
            for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
            float eff_max = scale*kMaxQ;
            if (eff_max <= 0) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/eff_max;
                float this_scale = 1/id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            if (scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + grid_index);
                    for (int i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
                q2[2*ib+1] |= (block_signs[k] << 7*k);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            q2[2*ib+1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}

// ============================================================================
// IQ2_XS quantization (with importance matrix)
// ============================================================================

static void quantize_row_iq2_xs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq2_data_index(EXTRACTED_GGML_TYPE_IQ2_XS);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kmap_q2xs       && "forgot to call iq2xs_init_impl()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call iq2xs_init_impl()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call iq2xs_init_impl()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_xs * y = vy;

    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2*(QK_K/16)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);
        memset(y[ibl].scales, 0, QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            const float * qw = quant_weights + QK_K*ibl + 16*ib;
            for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2*ib+k] = grid_index | (block_signs[k] << 9);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);

    }
}

size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xxs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq2_xxs);
    }
    return nrow * nblock * sizeof(block_iq2_xxs);
}

size_t quantize_iq2_xs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq2_xs);
    }
    return nrow * nblock * sizeof(block_iq2_xs);
}
