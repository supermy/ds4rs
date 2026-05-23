/**
 * 验证 GGUF IQ2_XS 量化与反量化的正确性。
 *
 * 功能：
 *   1. 读取 GGUF 文件中的 IQ2_XS 张量
 *   2. 使用 llama.cpp 相同的反量化公式将 IQ2_XS 反量化为 FP32
 *   3. 输出反量化结果供 Python 对比
 *
 * 编译:
 *   gcc -O2 -o verify_gguf_iq2xs verify_gguf_iq2xs.c -lm
 *
 * 用法:
 *   ./verify_gguf_iq2xs <gguf_file> [tensor_name] [max_blocks]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ---- GGUF 常量 ---- */
#define GGUF_MAGIC     0x46554747  /* "GGUF" 小端 */
#define GGUF_VERSION   3
#define GGML_TYPE_IQ2_XS 28
#define QK_K           256
#define IQ2_XS_BLOCK_BYTES 74

/* ---- IQ2_XS block 结构 (与 ggml-common.h 一致) ---- */
typedef struct {
    uint16_t d;           /* ggml_half (FP16) */
    uint16_t qs[QK_K/8]; /* 32 个 uint16: 低 9-bit = grid 索引, 高 7-bit = 符号索引 */
    uint8_t  scales[QK_K/32]; /* 8 个 uint8: 高低 4-bit 各编码一个子块缩放 */
} block_iq2_xs;
/* static_assert: 2 + 64 + 8 = 74 bytes */

/* ---- 查找表 (来自 ggml-common.h) ---- */
static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

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

static const uint64_t iq2xs_grid[512] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,
    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL,
    0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,
    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL,
    0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL,
    0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL,
    0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL,
    0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL,
    0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL,
    0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL,
    0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL,
    0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL,
    0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL,
    0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL,
    0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL,
    0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL,
    0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL,
    0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL,
    0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL,
    0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL,
    0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL,
    0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,
    0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL,
    0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL,
    0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,
    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL,
    0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL,
    0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL,
    0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL,
    0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL,
    0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL,
    0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL,
    0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL,
    0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL,
    0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL,
    0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL,
    0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL,
    0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL,
    0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL,
    0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL,
    0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL,
    0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL,
    0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,
    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL,
    0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL,
    0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL,
    0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL,
    0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL,
    0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL,
    0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL,
    0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL,
    0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL,
    0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL,
    0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL,
    0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL,
    0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL,
    0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL,
    0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL,
    0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL,
    0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL,
    0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL,
    0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL,
    0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL,
    0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL,
    0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL,
    0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL,
    0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL,
    0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL,
    0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL,
    0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL,
    0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL,
    0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL,
    0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL,
    0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL,
    0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL,
    0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL,
    0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL,
    0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL,
    0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL,
    0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL,
    0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL,
    0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL,
    0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL,
    0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL,
    0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL,
    0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL,
    0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL,
    0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL,
    0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL,
    0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL,
    0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL,
    0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL,
    0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL,
    0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL,
    0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL,
    0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL,
    0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL,
    0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL,
    0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL,
    0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL,
    0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,
    0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL,
    0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL,
    0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL,
    0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL,
    0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL,
    0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL,
    0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL,
    0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL,
    0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL,
    0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL,
    0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL,
    0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL,
    0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL,
    0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL,
    0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL,
    0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL,
    0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL,
    0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL,
    0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL,
    0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL,
};

/* ---- FP16 转 FP32 (简易实现) ---- */
static float fp16_to_fp32(uint16_t h) {
    /* IEEE 754 半精度 → 单精度 */
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;

    float result;
    if (exponent == 0) {
        if (mantissa == 0) {
            /* 零 */
            result = 0.0f;
        } else {
            /* 非规格化数 */
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3ff;
            result = ldexpf((float)mantissa / 1024.0f, exponent - 15);
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            result = INFINITY;
        } else {
            result = NAN;
        }
    } else {
        result = ldexpf(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
    }

    return sign ? -result : result;
}

/* ---- GGUF 张量信息 ---- */
typedef struct {
    char name[256];
    int n_dims;
    int64_t dims[4];
    int32_t ggml_type;
    uint64_t offset;  /* 相对于数据区起始 */
} tensor_info_t;

/* ---- GGUF 读取器 ---- */
static int read_gguf_header(FILE *f, int64_t *n_tensors, int64_t *n_kv) {
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1) return -1;
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "无效的 GGUF 文件: magic=0x%08x\n", magic);
        return -1;
    }
    if (fread(&version, 4, 1, f) != 1) return -1;
    if (version != GGUF_VERSION) {
        fprintf(stderr, "不支持的 GGUF 版本: %u\n", version);
        return -1;
    }
    if (fread(n_tensors, 8, 1, f) != 1) return -1;
    if (fread(n_kv, 8, 1, f) != 1) return -1;
    return 0;
}

static int read_gguf_string(FILE *f, char *buf, int bufsize) {
    uint64_t len;
    if (fread(&len, 8, 1, f) != 1) return -1;
    if (len >= (uint64_t)bufsize) {
        fprintf(stderr, "字符串太长: %lu\n", (unsigned long)len);
        return -1;
    }
    if (fread(buf, 1, len, f) != len) return -1;
    buf[len] = '\0';
    return 0;
}

static int skip_kv_pairs(FILE *f, int64_t n_kv) {
    for (int64_t i = 0; i < n_kv; i++) {
        char key[256];
        if (read_gguf_string(f, key, sizeof(key)) != 0) return -1;
        int32_t vtype;
        if (fread(&vtype, 4, 1, f) != 1) return -1;

        switch (vtype) {
            case 0: /* UINT8 */   fseek(f, 1, SEEK_CUR); break;
            case 1: /* INT8 */    fseek(f, 1, SEEK_CUR); break;
            case 2: /* UINT16 */  fseek(f, 2, SEEK_CUR); break;
            case 3: /* INT16 */   fseek(f, 2, SEEK_CUR); break;
            case 4: /* UINT32 */  fseek(f, 4, SEEK_CUR); break;
            case 5: /* INT32 */   fseek(f, 4, SEEK_CUR); break;
            case 6: /* FLOAT32 */ fseek(f, 4, SEEK_CUR); break;
            case 7: /* BOOL */    fseek(f, 1, SEEK_CUR); break;
            case 8: { /* STRING */
                uint64_t slen;
                if (fread(&slen, 8, 1, f) != 1) return -1;
                fseek(f, slen, SEEK_CUR);
                break;
            }
            case 9: { /* ARRAY */
                int32_t arr_type;
                uint64_t arr_len;
                if (fread(&arr_type, 4, 1, f) != 1) return -1;
                if (fread(&arr_len, 8, 1, f) != 1) return -1;
                int elem_size = 0;
                switch (arr_type) {
                    case 0: elem_size = 1; break;
                    case 1: elem_size = 1; break;
                    case 4: elem_size = 4; break;
                    case 5: elem_size = 4; break;
                    default: elem_size = 4; break;
                }
                fseek(f, arr_len * elem_size, SEEK_CUR);
                break;
            }
            case 10: /* UINT64 */  fseek(f, 8, SEEK_CUR); break;
            case 11: /* INT64 */   fseek(f, 8, SEEK_CUR); break;
            case 12: /* FLOAT64 */ fseek(f, 8, SEEK_CUR); break;
            default:
                fprintf(stderr, "不支持的 KV 类型: %d\n", vtype);
                return -1;
        }
    }
    return 0;
}

/* ---- 反量化 IQ2_XS block 到 FP32 ---- */
static void dequantize_iq2_xs_block(const block_iq2_xs *block, float *output) {
    float d = fp16_to_fp32(block->d);

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        /* 每个 ib32 包含 32 个元素, 分为 4 组 (il=0..3), 每组 8 个元素 */
        /* scales[ib] 的高低 4-bit 各编码一个子块缩放 */
        uint8_t scale_byte = block->scales[ib];
        /* ls1 = 2*(scale_byte & 0xf) + 1, 用于 il=0,1 */
        /* ls2 = 2*(scale_byte >> 4) + 1, 用于 il=2,3 */
        /* 这与 llama.cpp ggml_vec_dot_iq2_xs_q8_K_generic 一致 */

        for (int il = 0; il < 4; ++il) {
            int qs_idx = 4 * ib + il;
            uint16_t qs_val = block->qs[qs_idx];

            int grid_idx = qs_val & 511;
            int sign_idx = qs_val >> 9;

            const uint8_t *grid = (const uint8_t *)(iq2xs_grid + grid_idx);
            uint8_t signs = ksigns_iq2xs[sign_idx];

            /* 计算 scale: 与 llama.cpp 一致
             * il=0,1 用 ls1 = 2*(scale_byte & 0xf) + 1
             * il=2,3 用 ls2 = 2*(scale_byte >> 4) + 1
             * 最终: y = d * ls * 0.125 * grid[j] * sign
             * 等价于: y = d * (0.5 + scale_4bit) * 0.25 * grid[j] * sign
             *   因为 ls * 0.125 = (2*scale_4bit + 1) * 0.125 = (0.5 + scale_4bit) * 0.25
             */
            int ls;
            if (il < 2) {
                ls = 2 * (scale_byte & 0xf) + 1;
            } else {
                ls = 2 * (scale_byte >> 4) + 1;
            }

            float scale_factor = d * ls * 0.125f;

            for (int j = 0; j < 8; ++j) {
                int k_global = ib * 32 + il * 8 + j;
                float sign = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                output[k_global] = scale_factor * grid[j] * sign;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <gguf_file> [tensor_name] [max_blocks]\n", argv[0]);
        return 1;
    }

    const char *filepath = argv[1];
    const char *target_name = (argc >= 3) ? argv[2] : NULL;
    int max_blocks = (argc >= 4) ? atoi(argv[3]) : 4;  /* 默认只验证前 4 个 block */

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "无法打开文件: %s\n", filepath);
        return 1;
    }

    /* 读取 GGUF 头 */
    int64_t n_tensors, n_kv;
    if (read_gguf_header(f, &n_tensors, &n_kv) != 0) {
        fclose(f);
        return 1;
    }
    printf("GGUF: n_tensors=%ld, n_kv=%ld\n", (long)n_tensors, (long)n_kv);

    /* 跳过 KV 对 */
    if (skip_kv_pairs(f, n_kv) != 0) {
        fclose(f);
        return 1;
    }

    /* 读取张量信息 */
    tensor_info_t *tensors = calloc(n_tensors, sizeof(tensor_info_t));
    for (int64_t i = 0; i < n_tensors; i++) {
        if (read_gguf_string(f, tensors[i].name, sizeof(tensors[i].name)) != 0) {
            fprintf(stderr, "读取张量名称失败\n");
            goto cleanup;
        }
        if (fread(&tensors[i].n_dims, 4, 1, f) != 1) goto cleanup;
        for (int d = 0; d < tensors[i].n_dims; d++) {
            if (fread(&tensors[i].dims[d], 8, 1, f) != 1) goto cleanup;
        }
        if (fread(&tensors[i].ggml_type, 4, 1, f) != 1) goto cleanup;
        if (fread(&tensors[i].offset, 8, 1, f) != 1) goto cleanup;
    }

    /* 计算数据区偏移 (32 字节对齐) */
    long meta_end = ftell(f);
    long data_offset = ((meta_end + 31) / 32) * 32;
    printf("数据区偏移: %ld (0x%lx)\n", data_offset, (unsigned long)data_offset);

    /* 查找目标张量 */
    int target_idx = -1;
    for (int64_t i = 0; i < n_tensors; i++) {
        if (tensors[i].ggml_type == GGML_TYPE_IQ2_XS) {
            if (target_name == NULL || strcmp(tensors[i].name, target_name) == 0) {
                target_idx = i;
                if (target_name != NULL) break;
            }
        }
    }

    if (target_idx < 0) {
        printf("未找到 IQ2_XS 张量");
        if (target_name) printf(" '%s'", target_name);
        printf("\n可用的 IQ2_XS 张量:\n");
        for (int64_t i = 0; i < n_tensors; i++) {
            if (tensors[i].ggml_type == GGML_TYPE_IQ2_XS) {
                printf("  [%ld] %s  dims=[", (long)i, tensors[i].name);
                for (int d = 0; d < tensors[i].n_dims; d++) {
                    printf("%s%ld", d ? "," : "", (long)tensors[i].dims[d]);
                }
                printf("]\n");
            }
        }
        goto cleanup;
    }

    tensor_info_t *t = &tensors[target_idx];
    int64_t n_elements = 1;
    for (int d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];
    int64_t n_blocks_total = (n_elements + QK_K - 1) / QK_K;
    int n_blocks = (max_blocks > 0 && max_blocks < n_blocks_total) ? max_blocks : (int)n_blocks_total;

    printf("\n目标张量: %s\n", t->name);
    printf("  类型: IQ2_XS (%d)\n", t->ggml_type);
    printf("  dims: [");
    for (int d = 0; d < t->n_dims; d++) printf("%s%ld", d ? "," : "", (long)t->dims[d]);
    printf("]\n");
    printf("  n_elements: %ld\n", (long)n_elements);
    printf("  n_blocks: %ld (验证前 %d 个)\n", (long)n_blocks_total, n_blocks);

    /* 读取 block 数据 */
    fseek(f, data_offset + t->offset, SEEK_SET);
    block_iq2_xs *blocks = malloc(n_blocks * sizeof(block_iq2_xs));
    if (!blocks) {
        fprintf(stderr, "内存分配失败\n");
        goto cleanup;
    }

    if (fread(blocks, IQ2_XS_BLOCK_BYTES, n_blocks, f) != (size_t)n_blocks) {
        fprintf(stderr, "读取 block 数据失败\n");
        free(blocks);
        goto cleanup;
    }

    /* 反量化并输出 */
    printf("\n=== 反量化结果 (前 %d 个 block) ===\n", n_blocks);

    float *output = malloc(n_blocks * QK_K * sizeof(float));
    if (!output) {
        fprintf(stderr, "内存分配失败\n");
        free(blocks);
        goto cleanup;
    }

    for (int b = 0; b < n_blocks; b++) {
        dequantize_iq2_xs_block(&blocks[b], output + b * QK_K);
    }

    /* 输出统计信息 */
    float global_max = 0, global_min = 0, global_sum = 0;
    int total = n_blocks * QK_K;
    for (int i = 0; i < total; i++) {
        if (output[i] > global_max) global_max = output[i];
        if (output[i] < global_min) global_min = output[i];
        global_sum += output[i];
    }
    printf("  值域: [%f, %f]\n", global_min, global_max);
    printf("  均值: %f\n", global_sum / total);

    /* 输出第一个 block 的前 32 个值供对比 */
    printf("\n--- Block 0 前 32 个值 ---\n");
    for (int i = 0; i < 32; i++) {
        printf("  [%3d] %f\n", i, output[i]);
    }

    /* 输出 block 0 的原始量化参数 */
    printf("\n--- Block 0 原始参数 ---\n");
    printf("  d (FP16 raw): 0x%04x -> FP32: %f\n", blocks[0].d, fp16_to_fp32(blocks[0].d));
    printf("  qs[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
           blocks[0].qs[0], blocks[0].qs[1], blocks[0].qs[2], blocks[0].qs[3]);
    printf("  scales[0..1]: 0x%02x 0x%02x\n", blocks[0].scales[0], blocks[0].scales[1]);

    /* 输出所有 block 的 d 值统计 */
    printf("\n--- 所有 block 的 d 值 ---\n");
    for (int b = 0; b < n_blocks; b++) {
        float d_val = fp16_to_fp32(blocks[b].d);
        printf("  block[%d].d = %f (0x%04x)\n", b, d_val, blocks[b].d);
    }

    /* 输出为 Python 可读的格式 */
    printf("\n=== PYTHON_DATA_START ===\n");
    printf("# Block 0 反量化结果 (前 256 个 FP32 值)\n");
    printf("import numpy as np\n");
    printf("w_c = np.array([\n");
    for (int i = 0; i < QK_K; i++) {
        printf("%.8e%s", output[i], (i < QK_K - 1) ? "," : "");
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("], dtype=np.float32)\n");
    printf("# Block 0 原始参数\n");
    printf("d_raw = 0x%04x  # FP16 -> FP32: %f\n", blocks[0].d, fp16_to_fp32(blocks[0].d));
    printf("qs_raw = [0x%04x", blocks[0].qs[0]);
    for (int i = 1; i < 32; i++) printf(", 0x%04x", blocks[0].qs[i]);
    printf("]\n");
    printf("scales_raw = [0x%02x", blocks[0].scales[0]);
    for (int i = 1; i < 8; i++) printf(", 0x%02x", blocks[0].scales[i]);
    printf("]\n");
    printf("=== PYTHON_DATA_END ===\n");

    free(output);
    free(blocks);

cleanup:
    free(tensors);
    fclose(f);
    return 0;
}
