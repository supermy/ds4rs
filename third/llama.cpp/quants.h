/* quants.h — 精简桩头文件，仅提供 quants_generic.c 编译所需的函数声明。
 * 从 llama.cpp ggml/src/ggml-cpu/quants.h 抽取。
 */
#pragma once

#include "ggml-common.h"
#include "ggml-quants.h"

#ifdef __cplusplus
extern "C" {
#endif

// quantize_row 函数声明（由 ggml-quants.h 提供）
// vec_dot 函数声明

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
