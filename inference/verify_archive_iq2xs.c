/**
 * 验证 experts.iq2xs 归档文件的 IQ2_XS 量化数据正确性。
 *
 * 功能：
 *   1. 读取归档文件头和索引表
 *   2. 抽取指定专家的 IQ2_XS block 数据
 *   3. 使用 llama.cpp 相同的反量化公式反量化为 FP32
 *   4. 输出反量化结果供 Python 对比
 *
 * 编译:
 *   gcc -O2 -o verify_archive_iq2xs verify_archive_iq2xs.c -lm
 *
 * 用法:
 *   ./verify_archive_iq2xs <archive_file> [layer_id expert_id weight_type]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ---- 归档格式常量 ---- */
#define ARCHIVE_MAGIC    "IQ2X"
#define ARCHIVE_VERSION  1
#define HEADER_SIZE      64
#define INDEX_ENTRY_SIZE 32
#define QK_K             256
#define BLOCK_BYTES      74

/* ---- IQ2_XS block 结构 (与 ggml-common.h 一致) ---- */
typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8]; /* 32 */
    uint8_t  scales[QK_K/32]; /* 8 */
} block_iq2_xs;

/* ---- 查找表 (来自 ggml-common.h) ---- */
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

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

/* ---- FP16 转 FP32 ---- */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    float result;
    if (exponent == 0) {
        if (mantissa == 0) { result = 0.0f; }
        else {
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3ff;
            result = ldexpf((float)mantissa / 1024.0f, exponent - 15);
        }
    } else if (exponent == 31) {
        result = (mantissa == 0) ? INFINITY : NAN;
    } else {
        result = ldexpf(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
    }
    return sign ? -result : result;
}

/* ---- 反量化 IQ2_XS block ---- */
static void dequantize_iq2_xs_block(const block_iq2_xs *block, float *output) {
    float d = fp16_to_fp32(block->d);
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        uint8_t scale_byte = block->scales[ib];
        for (int il = 0; il < 4; ++il) {
            int qs_idx = 4 * ib + il;
            uint16_t qs_val = block->qs[qs_idx];
            int grid_idx = qs_val & 511;
            int sign_idx = qs_val >> 9;
            const uint8_t *grid = (const uint8_t *)(iq2xs_grid + grid_idx);
            uint8_t signs = ksigns_iq2xs[sign_idx];
            int ls = (il < 2) ? (2 * (scale_byte & 0xf) + 1) : (2 * (scale_byte >> 4) + 1);
            float scale_factor = d * ls * 0.125f;
            for (int j = 0; j < 8; ++j) {
                int k_global = ib * 32 + il * 8 + j;
                float sign = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                output[k_global] = scale_factor * grid[j] * sign;
            }
        }
    }
}

/* ---- 索引条目 ---- */
typedef struct {
    uint32_t layer_id;
    uint32_t expert_id;
    uint32_t weight_type;
    uint32_t n_blocks;
    uint32_t out_dim;
    uint32_t in_dim;
    uint64_t offset;
} index_entry_t;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <archive_file> [layer_id expert_id weight_type]\n", argv[0]);
        return 1;
    }

    const char *filepath = argv[1];
    int target_layer = (argc >= 3) ? atoi(argv[2]) : 0;
    int target_expert = (argc >= 4) ? atoi(argv[3]) : 0;
    int target_weight = (argc >= 5) ? atoi(argv[4]) : 0;

    FILE *f = fopen(filepath, "rb");
    if (!f) { fprintf(stderr, "无法打开文件: %s\n", filepath); return 1; }

    /* 读取文件头 */
    char header[HEADER_SIZE];
    if (fread(header, 1, HEADER_SIZE, f) != HEADER_SIZE) {
        fprintf(stderr, "读取文件头失败\n"); fclose(f); return 1;
    }

    if (memcmp(header, ARCHIVE_MAGIC, 4) != 0) {
        fprintf(stderr, "无效的归档文件: magic=%.4s\n", header); fclose(f); return 1;
    }

    uint32_t version, n_layers, n_experts, n_weights;
    uint64_t index_offset, index_size, data_offset;
    memcpy(&version, header + 4, 4);
    memcpy(&n_layers, header + 8, 4);
    memcpy(&n_experts, header + 12, 4);
    memcpy(&n_weights, header + 16, 4);
    memcpy(&index_offset, header + 20, 8);
    memcpy(&index_size, header + 28, 8);
    memcpy(&data_offset, header + 36, 8);

    printf("归档文件: %s\n", filepath);
    printf("  版本: %u, 层数: %u, 专家数: %u, 权重数: %u\n", version, n_layers, n_experts, n_weights);
    printf("  索引偏移: %lu, 索引大小: %lu, 数据偏移: %lu\n",
           (unsigned long)index_offset, (unsigned long)index_size, (unsigned long)data_offset);

    uint32_t n_entries = index_size / INDEX_ENTRY_SIZE;
    printf("  索引条目数: %u\n", n_entries);

    /* 读取索引表 */
    index_entry_t *entries = calloc(n_entries, sizeof(index_entry_t));
    fseek(f, index_offset, SEEK_SET);
    for (uint32_t i = 0; i < n_entries; i++) {
        uint8_t buf[INDEX_ENTRY_SIZE];
        if (fread(buf, 1, INDEX_ENTRY_SIZE, f) != INDEX_ENTRY_SIZE) {
            fprintf(stderr, "读取索引条目 %u 失败\n", i); goto cleanup;
        }
        memcpy(&entries[i].layer_id, buf, 4);
        memcpy(&entries[i].expert_id, buf + 4, 4);
        memcpy(&entries[i].weight_type, buf + 8, 4);
        memcpy(&entries[i].n_blocks, buf + 12, 4);
        memcpy(&entries[i].out_dim, buf + 16, 4);
        memcpy(&entries[i].in_dim, buf + 20, 4);
        memcpy(&entries[i].offset, buf + 24, 8);
    }

    /* 查找目标专家 */
    int target_idx = -1;
    for (uint32_t i = 0; i < n_entries; i++) {
        if (entries[i].layer_id == target_layer &&
            entries[i].expert_id == target_expert &&
            entries[i].weight_type == target_weight) {
            target_idx = i;
            break;
        }
    }

    if (target_idx < 0) {
        printf("\n未找到 layer=%d expert=%d weight=%d\n", target_layer, target_expert, target_weight);
        /* 显示前几个条目 */
        printf("前 5 个索引条目:\n");
        for (int i = 0; i < 5 && i < (int)n_entries; i++) {
            printf("  [%d] layer=%u expert=%u weight=%u n_blocks=%u out_dim=%u in_dim=%u offset=%lu\n",
                   i, entries[i].layer_id, entries[i].expert_id, entries[i].weight_type,
                   entries[i].n_blocks, entries[i].out_dim, entries[i].in_dim,
                   (unsigned long)entries[i].offset);
        }
        goto cleanup;
    }

    index_entry_t *e = &entries[target_idx];
    const char *weight_names[] = {"w1", "w2", "w3"};
    printf("\n目标: layer=%d expert=%d %s\n", target_layer, target_expert,
           weight_names[target_weight]);
    printf("  n_blocks=%u, out_dim=%u, in_dim=%u, offset=%lu\n",
           e->n_blocks, e->out_dim, e->in_dim, (unsigned long)e->offset);

    /* 验证 n_blocks 与维度是否一致 */
    uint32_t n_blocks_per_row = e->in_dim / QK_K;
    uint32_t expected_blocks = e->out_dim * n_blocks_per_row;
    if (e->n_blocks != expected_blocks) {
        printf("  警告: n_blocks=%u != expected=%u (out_dim=%u * in_dim/%u=%u)\n",
               e->n_blocks, expected_blocks, e->out_dim, QK_K, n_blocks_per_row);
    }

    /* 读取 block 数据 */
    int n_blocks = e->n_blocks;
    int max_blocks = 4;  /* 只验证前 4 个 block */
    if (max_blocks > n_blocks) max_blocks = n_blocks;

    fseek(f, e->offset, SEEK_SET);
    block_iq2_xs *blocks = malloc(max_blocks * sizeof(block_iq2_xs));
    if (fread(blocks, BLOCK_BYTES, max_blocks, f) != (size_t)max_blocks) {
        fprintf(stderr, "读取 block 数据失败\n"); free(blocks); goto cleanup;
    }

    /* 反量化 */
    float *output = malloc(max_blocks * QK_K * sizeof(float));
    for (int b = 0; b < max_blocks; b++) {
        dequantize_iq2_xs_block(&blocks[b], output + b * QK_K);
    }

    /* 统计 */
    float gmax = 0, gmin = 0, gsum = 0;
    int total = max_blocks * QK_K;
    for (int i = 0; i < total; i++) {
        if (output[i] > gmax) gmax = output[i];
        if (output[i] < gmin) gmin = output[i];
        gsum += output[i];
    }
    printf("\n反量化结果 (前 %d 个 block):\n", max_blocks);
    printf("  值域: [%f, %f]\n", gmin, gmax);
    printf("  均值: %f\n", gsum / total);

    /* Block 0 d 值 */
    printf("\n  Block 0: d=%f (0x%04x)\n", fp16_to_fp32(blocks[0].d), blocks[0].d);

    /* 输出 Python 对比数据 */
    printf("\n=== PYTHON_VERIFY ===\n");
    printf("import numpy as np\n");
    printf("w_c = np.array([\n");
    for (int i = 0; i < QK_K; i++) {
        printf("%.8e%s", output[i], (i < QK_K-1) ? "," : "");
        if ((i+1) % 8 == 0) printf("\n");
    }
    printf("], dtype=np.float32)\n");
    printf("d_raw = 0x%04x\n", blocks[0].d);
    printf("qs_raw = [0x%04x", blocks[0].qs[0]);
    for (int i = 1; i < 32; i++) printf(", 0x%04x", blocks[0].qs[i]);
    printf("]\n");
    printf("scales_raw = [0x%02x", blocks[0].scales[0]);
    for (int i = 1; i < 8; i++) printf(", 0x%02x", blocks[0].scales[i]);
    printf("]\n");
    printf("n_blocks = %d\n", n_blocks);
    printf("out_dim = %u\n", e->out_dim);
    printf("in_dim = %u\n", e->in_dim);
    printf("=== END_VERIFY ===\n");

    free(output);
    free(blocks);

cleanup:
    free(entries);
    fclose(f);
    return 0;
}
