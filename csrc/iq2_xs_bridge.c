/**
 * IQ2_XS 桥接层 — 提供 FP16 转换实现 + Rust 可调用的 C 接口
 *
 * iq2_xs.h 中声明了 extern ggml_fp16_to_fp32 / ggml_fp32_to_fp16，
 * 此文件提供实现，并暴露 Rust FFI 可用的包装函数。
 */
#include "iq2_xs.h"

// ============================================================================
// FP16 ↔ FP32 转换实现（软件模拟，兼容 IEEE 754 半精度）
// ============================================================================

float ggml_fp16_to_fp32(uint16_t h) {
    // IEEE 754 半精度 → 单精度
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    float result;
    if (exp == 0) {
        if (mant == 0) {
            // 零
            result = 0.0f;
        } else {
            // 次正规数: 手动反规格化
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp++; }
            mant &= 0x3ff;
            uint32_t f32_bits = (sign << 31) | ((127 - 15 - exp + 1) << 23) | (mant << 13);
            memcpy(&result, &f32_bits, sizeof(float));
        }
    } else if (exp == 31) {
        // 无穷大 / NaN
        uint32_t f32_bits = (sign << 31) | (0xff << 23) | (mant << 13);
        memcpy(&result, &f32_bits, sizeof(float));
    } else {
        // 正规数
        uint32_t f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        memcpy(&result, &f32_bits, sizeof(float));
    }
    return result;
}

uint16_t ggml_fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(uint32_t));

    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;

    if (exp <= 0) {
        // 次正规数 / 零
        if (exp < -10) return (uint16_t)(sign << 15);
        mant |= 0x800000;
        uint16_t h_mant = (uint16_t)(mant >> (14 - exp));
        return (uint16_t)(sign << 15) | h_mant;
    } else if (exp >= 31) {
        // 溢出 → 无穷大
        return (uint16_t)(sign << 15) | 0x7c00;
    } else {
        // 正规数（四舍五入）
        uint16_t h_mant = (uint16_t)((mant + 0x1000) >> 13);
        if (h_mant & 0x400) { h_mant = 0; exp++; } // 进位
        if (exp >= 31) return (uint16_t)(sign << 15) | 0x7c00;
        return (uint16_t)(sign << 15) | ((uint16_t)exp << 10) | (h_mant & 0x3ff);
    }
}

// ============================================================================
// Rust FFI 接口
// ============================================================================

// 初始化 IQ2_XS 运行时数据（kmap, kneighbors）
void iq2xs_init_ffi(void) {
    iq2xs_init();
}

// 释放 IQ2_XS 运行时数据
void iq2xs_free_ffi(void) {
    iq2xs_free();
}

// 量化: float32 → IQ2_XS
// src:     [nrow * n_per_row] float32 输入
// dst:     [nrow * (n_per_row / 256) * 74] 字节输出（block_iq2_xs 数组）
// nrow:    行数
// n_per_row: 每行元素数（必须是 256 的倍数）
// 返回:    写入 dst 的字节数
size_t quantize_iq2_xs_ffi(
    const float * src, void * dst, int64_t nrow, int64_t n_per_row)
{
    // 构造全 1 权重矩阵（无重要性加权）
    float * weights = (float *)malloc(nrow * n_per_row * sizeof(float));
    for (int64_t i = 0; i < nrow * n_per_row; i++) weights[i] = 1.0f;

    size_t result = quantize_iq2_xs(src, dst, nrow, n_per_row, weights);

    free(weights);
    return result;
}

// 反量化: IQ2_XS → float32
// blocks: block_iq2_xs 数组
// dst:    [n] float32 输出
// n:      元素数（必须是 256 的倍数）
void dequantize_iq2_xs_ffi(const void * blocks, float * dst, int64_t n) {
    dequantize_row_iq2_xs((const block_iq2_xs *)blocks, dst, n);
}

// 获取 block_iq2_xs 的大小（74 字节）
size_t iq2_xs_block_size_ffi(void) {
    return sizeof(block_iq2_xs);
}

// 获取 QK_K 常量（256）
int iq2_xs_qk_k_ffi(void) {
    return QK_K;
}
