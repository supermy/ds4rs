/* ggml-cpu-impl.h — 精简桩头文件，仅提供 quants_generic.c / quants_x86.c 编译所需的定义。
 * 从 llama.cpp ggml/src/ggml-cpu/ggml-cpu-impl.h 抽取。
 */
#pragma once

#include "ggml-common.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGML_RESTRICT 已在 ggml-common.h 中定义

#ifndef GGML_UNUSED
#define GGML_UNUSED(x) (void)(x)
#endif

#ifndef GGML_ASSERT
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "GGML_ASSERT: %s\n", #x); abort(); } } while (0)
#endif

// x86 SIMD 头文件
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// SIMD 相关宏 — 仅 x86 AVX-512 (Ryzen 5 7600)
#if defined(_MSC_VER)
#define m512bh(p) p
#define m512i(p) p
#else
#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)
#endif

#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#ifdef __cplusplus
}
#endif
