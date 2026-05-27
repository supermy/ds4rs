// Extracted from llama.cpp ggml/src/ggml-quants.h
// Contains function declarations for IQ2_XXS, IQ2_XS, and Q2_K quantization/dequantization.
//
// Original source: https://github.com/ggerganov/llama.cpp
// License: MIT

#pragma once

#ifndef GGML_COMMON_DECL_C
#define GGML_COMMON_DECL_C
#endif
#include "ggml-common.h"

#include <stdint.h>
#include <stddef.h>

// GGML_RESTRICT: from ggml.h
#ifndef GGML_RESTRICT
#ifdef __cplusplus
#    if defined(__GNUC__)
#        define GGML_RESTRICT __restrict__
#    elif defined(__clang__)
#        define GGML_RESTRICT __restrict
#    elif defined(_MSC_VER)
#        define GGML_RESTRICT __restrict
#    else
#        define GGML_RESTRICT
#    endif
#else
#    if defined (_MSC_VER) && (__STDC_VERSION__ < 201112L)
#        define GGML_RESTRICT __restrict
#    else
#        define GGML_RESTRICT restrict
#    endif
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Q2_K
// ============================================================================

// Reference quantization (no importance matrix)
void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);

// Dequantization
void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization with importance matrix
size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// ============================================================================
// IQ2_XXS
// ============================================================================

// Dequantization
void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization with importance matrix
size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// ============================================================================
// IQ2_XS
// ============================================================================

// Dequantization
void dequantize_row_iq2_xs(const block_iq2_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization with importance matrix
size_t quantize_iq2_xs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// ============================================================================
// IQ2 grid initialization / cleanup
// ============================================================================

// Initialize IQ2 grid data for the given type.
// type must be one of: GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_S
void iq2xs_init_impl(int type);
void iq2xs_free_impl(int type);

#ifdef __cplusplus
}
#endif
