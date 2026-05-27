/* ggml-impl.h — 精简桩头文件，提供 ggml 内部实现所需的宏和内联函数。
 * 从 llama.cpp ggml/src/ggml-impl.h 抽取。
 */
#pragma once

#include "ggml-common.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GGML_ASSERT
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "GGML_ASSERT: %s\n", #x); abort(); } } while (0)
#endif

#ifndef GGML_UNREACHABLE
#define GGML_UNREACHABLE() do { fprintf(stderr, "GGML_UNREACHABLE\n"); abort(); } while (0)
#endif

#ifndef GGML_UNUSED
#define GGML_UNUSED(x) (void)(x)
#endif

#ifdef __cplusplus
}
#endif
