/* simd-mappings.h — 精简桩头文件，仅提供 quants_generic.c / quants_x86.c 编译所需的定义。
 * 从 llama.cpp ggml/src/ggml-cpu/simd-mappings.h 抽取。
 */
#pragma once

// 此文件在 llama.cpp 中定义 SIMD 内联函数映射
// ds4rs 的 quants 代码不需要额外的 SIMD 映射
