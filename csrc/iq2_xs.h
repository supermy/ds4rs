/**
 * IQ2_XS 量化与反量化 — 从 llama.cpp 抽取的独立实现
 *
 * 来源文件:
 *   ggml/src/ggml-common.h          — block_iq2_xs, block_q8_K 结构体, iq2xs_grid, ksigns_iq2xs, kmask_iq2xs
 *   ggml/src/ggml-quants.c           — quantize_row_iq2_xs_impl, quantize_iq2_xs, iq2xs_init_impl, iq2_find_best_neighbour, kgrid_2bit_512
 *   ggml/src/ggml-cpu/quants.c       — ggml_vec_dot_iq2_xs_q8_K_generic
 *   ggml/src/ggml-cpu/arch/x86/quants.c — ggml_vec_dot_iq2_xs_q8_K (AVX2/AVX 优化)
 *
 * 数据结构:
 *   block_iq2_xs: { d: fp16, qs[32]: u16, scales[8]: u8 } = 74 字节 / 256 元素 = 2.3125 bpw
 *   block_q8_K:   { d: float, qs[256]: i8, bsums[16]: i16 } = 292 字节 (中间量化/点积用)
 *
 * 量化流程 (quantize_row_iq2_xs_impl):
 *   1. 对每个 16 元素子块，计算符号（保证偶数负数）+ 绝对值 xval
 *   2. 尝试 19 个 scale 候选值，找最优 scale（sumqx²/sumq2 最大化）
 *   3. 用 L 值计算 + kmap 查找 grid 索引
 *   4. off-grid 修正 + 负 scale 处理
 *   5. d = max_scale / 31
 *   6. scale_4bit = nearest_int(0.5 * (id * scale - 1))
 *
 * 反量化流程 (dequantize_row_iq2_xs):
 *   1. d = FP16_TO_FP32(block.d)
 *   2. db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25
 *   3. db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25
 *   4. grid = iq2xs_grid[qs & 511]
 *   5. signs = ksigns_iq2xs[qs >> 9]
 *   6. y = db * grid * (signs & mask ? -1 : 1)
 */
#ifndef IQ2_XS_H
#define IQ2_XS_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 常量
// ============================================================================
#define QK_K           256     // super-block 大小（每组 256 个元素）
#define GROUP_MAX_EPS  1e-15f  // 全零检测阈值（来源: ggml-quants.c）
#define IQ2_XS_KMAXQ  3       // L 值范围 [0, kMaxQ-1] = [0, 2]
// kmap 大小: L 值为 0/1/2 时，8 个 2-bit 组合最大值为 0xAAAA=43690，加 2 余量
#define IQ2_XS_KMAP_SIZE 43692

// ============================================================================
// 数据结构
// ============================================================================
// FP16 简化表示（实际平台需提供 ggml_fp16_to_fp32 / ggml_fp32_to_fp16 实现）
typedef uint16_t ggml_half;

// IQ2_XS 量化块 — 来源: ggml-common.h
// 每个 super-block 编码 256 个元素，占用 74 字节 = 2.3125 bpw
typedef struct {
    ggml_half d;               // 全局缩放因子 (FP16)
    uint16_t qs[QK_K/8];      // 32 个 uint16_t: 低 9-bit = grid 索引, 高 7-bit = 符号索引
    uint8_t  scales[QK_K/32]; // 8 个 uint8_t: 高低 4-bit 各编码一个子块缩放
} block_iq2_xs;
// 总大小: 2 + 32*2 + 8 = 74 字节

// Q8_K 中间量化块 — 来源: ggml-common.h
// 用于量化中间表示和点积计算
typedef struct {
    float    d;                // delta（缩放因子）
    int8_t   qs[QK_K];        // 量化值
    int16_t  bsums[QK_K/16];  // 每 16 个量化值的和（用于矩阵乘法分块求和优化）
} block_q8_K;

// ============================================================================
// 查找表: kmask_iq2xs[8] — 符号位掩码
// 来源: ggml-common.h
// ============================================================================
static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

// ============================================================================
// 查找表: ksigns_iq2xs[128] — 符号反转查找表
// 来源: ggml-common.h
// 只包含偶数位 1 的值（128 个），用于 7-bit 符号编码
// 原理: 7-bit 编码 8 个符号位，但只有偶数个负数的组合有效（128 种）
// ============================================================================
static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// ============================================================================
// 查找表: iq2xs_grid[512] — 物理网格（反量化用）
// 来源: ggml-common.h
// 每个 uint64_t 编码 8 个字节，每字节取值: 0x08=8, 0x19=25, 0x2b=43
// 对应 L=0,1,2 的量化值 2*L+1={1,3,5} 乘以 8
// 反量化时直接作为 uint8_t 数组读取
// ============================================================================
static const uint64_t iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

// ============================================================================
// 查找表: kgrid_2bit_512[512] — 量化用紧凑网格索引
// 来源: ggml-quants.c (iq2xs_init_impl 内部)
// 每个 uint16_t 的每 2-bit 编码一个量化级别 l (0-2)
// 运行时展开为逻辑网格值 pos[i] = 2*l + 1 = {1, 3, 5}
// ============================================================================
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

// ============================================================================
// 辅助函数
// ============================================================================

// nearest_int: 快速浮点转整数（来源: ggml-quants.c）
// 利用 IEEE 754 浮点数特性，通过加 12582912.0 (2^23 + 2^22) 实现快速舍入
// 输入范围: |fval| <= 4194303 (2^22 - 1)
static inline int nearest_int(float fval) {
    //assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline float fmaxf3(float a, float b) { return a > b ? a : b; }
static inline float fminf3(float a, float b) { return a < b ? a : b; }
static inline int   imax(int a, int b) { return a > b ? a : b; }
static inline int   imin(int a, int b) { return a < b ? a : b; }

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// FP16 ↔ FP32 转换（需由使用方提供实现）
extern float ggml_fp16_to_fp32(uint16_t h);
extern uint16_t ggml_fp32_to_fp16(float f);

// ============================================================================
// IQ2_XS 初始化数据（运行时构建 kmap 和 kneighbors）
// 来源: ggml-quants.c (iq2xs_init_impl)
// ============================================================================
typedef struct {
    uint64_t * grid;       // 512 项逻辑网格（值域 {1, 3, 5}）
    int      * map;        // kmap: L 值编码 → grid 索引（off-grid 为负值）
    uint16_t * neighbours; // kneighbors: off-grid 点的最近邻列表
} iq2xs_data_t;

static iq2xs_data_t g_iq2xs_data = {NULL, NULL, NULL};

// iq2_find_best_neighbour: 在邻居列表中找最佳 grid 索引
// 来源: ggml-quants.c
// 对 off-grid 点，遍历其最近邻 grid 点，找加权距离最小的
static int iq2_find_best_neighbour(
    const uint16_t * neighbours,
    const uint64_t * grid,
    const float    * xval,
    const float    * weight,
    float            scale,
    int8_t         * L) {

    int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int grid_index = -1;

    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }

    // 从最佳 grid 点反推 L 值
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;

    return grid_index;
}

// iq2_compare_func: qsort 比较函数（稳定排序，先按距离再按索引）
// 来源: ggml-quants.c
static int iq2_compare_func(const void * a, const void * b) {
    const int * l = (const int *)a;
    const int * r = (const int *)b;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

// iq2xs_init: 初始化 IQ2_XS 量化所需的运行时数据
// 来源: ggml-quants.c (iq2xs_init_impl, 简化为仅 IQ2_XS)
// 构建: 逻辑网格 → kmap（L编码→grid索引）→ kneighbors（off-grid最近邻）
static void iq2xs_init(void) {
    if (g_iq2xs_data.grid) return;

    const int grid_size = 512;
    const int kmap_size = IQ2_XS_KMAP_SIZE;
    const int nwant = 2; // IQ2_XS: 每个距离层保留 2 个邻居

    // 步骤 1: 从 kgrid_2bit_512 展开逻辑网格（值域 {1, 3, 5}）
    uint64_t * the_grid = (uint64_t *)malloc(grid_size * sizeof(uint64_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        for (int i = 0; i < 8; ++i) {
            int l = (kgrid_2bit_512[k] >> 2 * i) & 0x3;
            pos[i] = 2 * l + 1;
        }
    }
    g_iq2xs_data.grid = the_grid;

    // 步骤 2: 构建 kmap（从逻辑网格值反推 16-bit 索引）
    int * kmap = (int *)malloc(kmap_size * sizeof(int));
    for (int i = 0; i < kmap_size; ++i) kmap[i] = -1;

    uint64_t aux64;
    uint8_t * aux8 = (uint8_t *)&aux64;
    for (int i = 0; i < grid_size; ++i) {
        aux64 = the_grid[i];
        uint16_t index = 0;
        for (int k = 0; k < 8; ++k) {
            uint16_t q = (aux8[k] - 1) / 2;
            index |= (q << 2 * k);
        }
        kmap[index] = i;
    }

    // 步骤 3: 为 off-grid 点构建 kneighbors
    int8_t pos[8];
    int * dist2 = (int *)malloc(2 * grid_size * sizeof(int));
    int num_neighbors = 0, num_not_in_map = 0;

    // 第一遍: 计算总邻居数
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap[i] >= 0) continue;
        ++num_not_in_map;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2 * k) & 0x3;
            pos[k] = 2 * l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(the_grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        int n = 0, d2 = dist2[0], nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2 * j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }

    // 第二遍: 填充 kneighbors 数组
    uint16_t * kneighbors = (uint16_t *)malloc((num_neighbors + num_not_in_map) * sizeof(uint16_t));
    int counter = 0;

    for (int i = 0; i < kmap_size; ++i) {
        if (kmap[i] >= 0) continue;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2 * k) & 0x3;
            pos[k] = 2 * l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(the_grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        kmap[i] = -(counter + 1);
        int d2 = dist2[0];
        uint16_t * start = &kneighbors[counter++];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2 * j];
                ++nhave;
            }
            kneighbors[counter++] = dist2[2 * j + 1];
            ++n;
        }
        *start = n;
    }
    free(dist2);

    g_iq2xs_data.map = kmap;
    g_iq2xs_data.neighbours = kneighbors;
}

// iq2xs_free: 释放运行时数据
static void iq2xs_free(void) {
    if (g_iq2xs_data.grid) {
        free(g_iq2xs_data.grid);       g_iq2xs_data.grid = NULL;
        free(g_iq2xs_data.map);        g_iq2xs_data.map = NULL;
        free(g_iq2xs_data.neighbours); g_iq2xs_data.neighbours = NULL;
    }
}

// ============================================================================
// 反量化: dequantize_row_iq2_xs
// 来源: ggml-quants.c
// 将 block_iq2_xs 解码为 float 数组
// ============================================================================
static void dequantize_row_iq2_xs(const block_iq2_xs * x, float * y, int64_t k) {
    // k 必须是 QK_K 的倍数
    const int64_t nb = k / QK_K;
    float db[2];

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            // 每个 scales[ib32] 的高低 4-bit 各编码一个子块缩放
            // db[0] = d * (0.5 + low4bit) * 0.25, db[1] = d * (0.5 + high4bit) * 0.25
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;

            for (int l = 0; l < 4; ++l) {
                // qs 低 9-bit = grid 索引 (0-511), 高 7-bit = 符号索引
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4 * ib32 + l] & 511));
                const uint8_t   signs = ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];

                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ============================================================================
// 量化: quantize_row_iq2_xs_impl
// 来源: ggml-quants.c
// GGUF 官方量化算法
//
// quant_weights: 重要性权重矩阵（不能为 NULL，无权重时传入全 1 数组）
// ============================================================================
static void quantize_row_iq2_xs_impl(
    const float * x, void * vy, int64_t n, const float * quant_weights)
{
    iq2xs_init(); // 确保初始化

    const uint64_t * kgrid_q2xs      = g_iq2xs_data.grid;
    const int      * kmap_q2xs       = g_iq2xs_data.map;
    const uint16_t * kneighbors_q2xs = g_iq2xs_data.neighbours;

    const int kMaxQ = IQ2_XS_KMAXQ;
    const int64_t nbl = n / QK_K;
    block_iq2_xs * y = (block_iq2_xs *)vy;

    float scales[QK_K / 16];    // 16 个子块缩放
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2 * (QK_K / 16)]; // 32 个 qs 值

    for (int ibl = 0; ibl < nbl; ++ibl) {
        y[ibl].d = ggml_fp32_to_fp16(0.f);
        memset(q2, 0, QK_K / 4);
        memset(y[ibl].scales, 0, QK_K / 32);

        float max_scale = 0;
        const float * xbl = x + QK_K * ibl;

        // 计算方差（用于重要性权重）
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = sumx2 / QK_K;

        // 对每个 16 元素子块
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            const float * xb = xbl + 16 * ib;
            const float * qw = quant_weights + QK_K * ibl + 16 * ib;

            // 重要性权重: weight[i] = qw[i] * sqrt(sigma2 + xb[i]^2)
            for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);

            // 步骤 1: 符号提取 — 将每 8 个元素分为 2 组，确保每组负数个数为偶数
            for (int k = 0; k < 2; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        ++nflip;
                        s |= (1 << i);
                    }
                }
                // 确保负数个数为偶数（7-bit 符号编码只能表示偶数个负数）
                if (nflip % 2) {
                    int imin = 0;
                    float min_val = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min_val) { min_val = ax; imin = i; }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }

            // 步骤 2: 寻找最佳 scale 和量化索引
            float max_val = xval[0];
            for (int i = 1; i < 16; ++i) max_val = MAX(max_val, xval[i]);
            memset(L, 0, 16);
            if (max_val < GROUP_MAX_EPS) { scales[ib] = 0; continue; }

            float best = 0;
            float scale = max_val / (2 * kMaxQ - 1);
            is_on_grid[0] = is_on_grid[1] = true;

            // 尝试 19 个 scale 候选值
            for (int is = -9; is <= 9; ++is) {
                float id = (2 * kMaxQ - 1 + is * 0.1f) / max_val;
                float this_scale = 1.0f / id;

                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        Laux[8 * k + i] = MAX(0, MIN(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8 * k + i] << 2 * i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(
                            neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                    }
                }

                // 计算加权最优 scale
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k < 2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }

            // 步骤 3: 对不在 grid 上的点进行修正
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1.0f / scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        l = MAX(0, MIN(kMaxQ - 1, l));
                        u |= (l << 2 * i);
                        L[8 * k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(
                            neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            // 步骤 4: 处理负 scale（翻转所有符号）
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }

            // 步骤 5: 编码 grid 索引 + 符号到 q2
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8 * k + i] << 2 * i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    // 不应到达此处（off-grid 已在步骤 3 处理）
                    fprintf(stderr, "IQ2_XS: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) fprintf(stderr, " %d", L[8 * k + i]);
                    fprintf(stderr, "\n");
                    grid_index = 0; // 降级处理
                }
                q2[2 * ib + k] = grid_index | (block_signs[k] << 9);
            }

            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        // 步骤 6: 编码全局 scale d 和子块 scale
        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K / 4);
            continue;
        }

        float d = max_scale / 31.0f;
        y[ibl].d = ggml_fp32_to_fp16(d);
        float id = 1.0f / d;

        for (int ib = 0; ib < QK_K / 16; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = MAX(0, MIN(15, l));
            if (ib % 2 == 0) y[ibl].scales[ib / 2] = l;
            else y[ibl].scales[ib / 2] |= (l << 4);
        }

        memcpy(y[ibl].qs, q2, QK_K / 4);
    }
}

// 量化入口函数
// 来源: ggml-quants.c (quantize_iq2_xs)
static size_t quantize_iq2_xs(
    const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights)
{
    int64_t nblock = n_per_row / QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq2_xs);
    }
    return nrow * nblock * sizeof(block_iq2_xs);
}

// ============================================================================
// 点积: ggml_vec_dot_iq2_xs_q8_K_generic
// 来源: ggml-cpu/quants.c
// IQ2_XS 与 Q8_K 的通用点积（纯 C 实现）
// 结果乘以 0.125 (1/8)，因为 grid 值域 {8, 25, 43} 对应 {1,3,5}*8
// ============================================================================
static void ggml_vec_dot_iq2_xs_q8_K_generic(
    int n, float * s,
    const void * vx, const void * vy)
{
    const block_iq2_xs * x = (const block_iq2_xs *)vx;
    const block_q8_K   * y = (const block_q8_K   *)vy;
    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        const uint16_t * q2 = x[i].qs;
        const uint8_t  * sc = x[i].scales;
        const int8_t   * q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            // 每个 scales[ib32] 编码两个子块缩放: ls1=低4bit, ls2=高4bit
            const uint16_t ls1 = 2 * (sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2 * (sc[ib32] >>  4) + 1;
            int32_t sumi = 0;

            // 前 2 个 qs 值（16 个元素）
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;

            // 后 2 个 qs 值（16 个元素）
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

// ============================================================================
// 点积: ggml_vec_dot_iq2_xs_q8_K (x86 AVX2/AVX 优化)
// 来源: ggml-cpu/arch/x86/quants.c
// 结果乘以 0.125 (1/8)
// ============================================================================
#if defined(__AVX2__) || defined(__AVX__)

#include <immintrin.h>

// 辅助: 8 个 float 的水平求和
static inline float hsum_float_8(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    lo = _mm_add_ps(lo, hi);     // 0+4, 1+5, 2+6, 3+7
    hi = _mm_movehl_ps(hi, lo);  // 2+6, 3+7, -, -
    lo = _mm_add_ps(lo, hi);     // 0+2+4+6, 1+3+5+7, -, -
    hi = _mm_shuffle_ps(lo, lo, 0x01); // 1+3+5+7, -, -, -
    lo = _mm_add_ss(lo, hi);     // 0+1+2+3+4+5+6+7
    return _mm_cvtss_f32(lo);
}

// 辅助: scale shuffle 掩码（将 4-bit scale 展开到 16-bit）
static inline __m128i get_scale_shuffle(int ib32) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm_loadu_si128((const __m128i *)(k_shuffle + 16 * ib32));
}

// MM256_SET_M128I 辅助（兼容旧版 GCC）
#ifndef MM256_SET_M128I
#define MM256_SET_M128I(e1, e0) _mm256_insertf128_si256(_mm256_castsi128_si256(e0), (e1), 1)
#endif

#if defined(__AVX2__)

// ============================================================================
// AVX2 优化路径
// ============================================================================
static void ggml_vec_dot_iq2_xs_q8_K_avx2(
    int n, float * s,
    const void * vx, const void * vy)
{
    const block_iq2_xs * x = (const block_iq2_xs *)vx;
    const block_q8_K   * y = (const block_q8_K   *)vy;
    const int nb = n / QK_K;

    const __m256i mone = _mm256_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };

    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i *)bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i *)block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i *)block_sign_shuffle_mask_2);
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i *)k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        const uint16_t * q2 = x[i].qs;
        const int8_t   * q8 = y[i].qs;

        // 解码 scales: 8 个 uint8 → 16 个 4-bit scale → 16 个 int8
        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {
            // 加载 16 个 qs 值
            const __m256i q2_data = _mm256_loadu_si256((const __m256i *)q2); q2 += 16;

            // 提取 grid 索引（低 9-bit）
            aux_gindex = _mm256_and_si256(q2_data, m511);

            // 提取符号位（bit 9-12 和 bit 13-15）
            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);

            // 恢复完整符号位（奇偶性修正）
            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            // 加载 Q8_K 值（4 × 32 = 128 字节）
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            // 从 grid 查找量化值
            const __m256i q2_1 = _mm256_set_epi64x(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                     iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                     iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            const __m256i q2_3 = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                     iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            const __m256i q2_4 = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                     iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            // 应用符号到 Q8_K 值
            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = MM256_SET_M128I(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = MM256_SET_M128I(full_signs_h, full_signs_h);

            __m256i signs;
            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            // maddubs: uint8 × int8 → int16 点积
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i dot3  = _mm256_maddubs_epi16(q2_3, q8s_3);
            const __m256i dot4  = _mm256_maddubs_epi16(q2_4, q8s_4);

            // 乘以 scales
            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 3)));

            // madd: int16 × int16 → int32
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);
}

#elif defined(__AVX__)

// ============================================================================
// AVX 优化路径（无 AVX2）
// ============================================================================
static void ggml_vec_dot_iq2_xs_q8_K_avx(
    int n, float * s,
    const void * vx, const void * vy)
{
    const block_iq2_xs * x = (const block_iq2_xs *)vx;
    const block_q8_K   * y = (const block_q8_K   *)vy;
    const int nb = n / QK_K;

    const __m128i mone = _mm_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i bit_selector_mask_0 = _mm_loadu_si128((const __m128i *)bit_selector_mask_bytes);
    const __m128i bit_selector_mask_1 = _mm_loadu_si128((const __m128i *)bit_selector_mask_bytes + 1);
    const __m128i block_sign_shuffle_1_0 = _mm_loadu_si128((const __m128i *)block_sign_shuffle_mask_1);
    const __m128i block_sign_shuffle_1_1 = _mm_loadu_si128((const __m128i *)block_sign_shuffle_mask_1 + 1);
    const __m128i block_sign_shuffle_2_0 = _mm_loadu_si128((const __m128i *)block_sign_shuffle_mask_2);
    const __m128i block_sign_shuffle_2_1 = _mm_loadu_si128((const __m128i *)block_sign_shuffle_mask_2 + 1);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m128i bit_helper_0 = _mm_loadu_si128((const __m128i *)k_bit_helper);
    const __m128i bit_helper_1 = _mm_loadu_si128((const __m128i *)k_bit_helper + 1);
    const __m128i m511 = _mm_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        const uint16_t * q2 = x[i].qs;
        const int8_t   * q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {

            const __m128i q2_data_0 = _mm_loadu_si128((const __m128i *)q2);
            const __m128i q2_data_1 = _mm_loadu_si128((const __m128i *)q2 + 1);  q2 += 16;
            aux_gindex = MM256_SET_M128I(_mm_and_si128(q2_data_1, m511), _mm_and_si128(q2_data_0, m511));

            const __m128i partial_sign_bits_0 = _mm_srli_epi16(q2_data_0, 9);
            const __m128i partial_sign_bits_1 = _mm_srli_epi16(q2_data_1, 9);
            const __m128i partial_sign_bits_upper_0 = _mm_srli_epi16(q2_data_0, 13);
            const __m128i partial_sign_bits_upper_1 = _mm_srli_epi16(q2_data_1, 13);
            const __m128i partial_sign_bits_for_counting_0 = _mm_xor_si128(partial_sign_bits_0, partial_sign_bits_upper_0);
            const __m128i partial_sign_bits_for_counting_1 = _mm_xor_si128(partial_sign_bits_1, partial_sign_bits_upper_1);

            const __m128i odd_bits_0 = _mm_shuffle_epi8(bit_helper_0, partial_sign_bits_for_counting_0);
            const __m128i odd_bits_1 = _mm_shuffle_epi8(bit_helper_1, partial_sign_bits_for_counting_1);
            const __m128i full_sign_bits_0 = _mm_or_si128(partial_sign_bits_0, odd_bits_0);
            const __m128i full_sign_bits_1 = _mm_or_si128(partial_sign_bits_1, odd_bits_1);

            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i q2_1_0 = _mm_set_epi64x(iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]]);
            const __m128i q2_3_0 = _mm_set_epi64x(iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
            const __m128i q2_3_1 = _mm_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]]);
            const __m128i q2_4_0 = _mm_set_epi64x(iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
            const __m128i q2_4_1 = _mm_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]]);

            __m128i signs_0, signs_1;
            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(signs_1, mone));

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const __m128i dot3_0  = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
            const __m128i dot3_1  = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
            const __m128i dot4_0  = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
            const __m128i dot4_1  = _mm_maddubs_epi16(q2_4_1, q8s_4_1);

            __m128i sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0));
            const __m128i sc1_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc1_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1));
            const __m128i sc2_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc2_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2));
            const __m128i sc3_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc3_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3));
            const __m128i sc4_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc4_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, sc1_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, sc1_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, sc2_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, sc2_1));
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot3_0, sc3_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot3_1, sc3_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot4_0, sc4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot4_1, sc4_1));
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d),
            _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))),
            accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);
}

#endif // __AVX2__ / __AVX__

// 统一入口: 根据编译时 CPU 特性选择最优路径
static void ggml_vec_dot_iq2_xs_q8_K(
    int n, float * s,
    const void * vx, const void * vy)
{
#if defined(__AVX2__)
    ggml_vec_dot_iq2_xs_q8_K_avx2(n, s, vx, vy);
#elif defined(__AVX__)
    ggml_vec_dot_iq2_xs_q8_K_avx(n, s, vx, vy);
#else
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, vx, vy);
#endif
}

#endif // __AVX2__ || __AVX__

#ifdef __cplusplus
}
#endif

#endif // IQ2_XS_H
