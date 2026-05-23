"""优化版 IQ2_XS GEMM TileLang kernel

直接计算 y = x @ W^T，其中 W 从 IQ2_XS 反量化得到。
避免中间的 FP8 量化步骤，直接输出 BF16 结果。

反量化公式（与 csrc/iq2_xs.cu 第110-143行对齐）：
  y[j] = d * (0.5 + scale_4bit) * 0.25 * grid[j] * (signs & mask[j] ? -1 : 1)

其中:
  d = block.d (FP16 全局缩放)
  scale_4bit = (scales[ib] >> 4*(il/2)) & 0xf  (4-bit 子块缩放, 0-15)
  grid[j] = iq2xs_grid[qs[4*ib + il] & 511] 的第 j 个字节 (值为 8, 25, 43)
  signs = ksigns_iq2xs[qs[4*ib + il] >> 9]
  ib = 0..7 (32 元素 sub-block 索引)
  il = 0..3 (8 元素组索引)

优化策略：
  1. Shared Memory 缓存：缓存 x 和 W 的块，减少全局内存访问
  2. Tensor Core 加速：使用 T.gemm 利用 Tensor Core
  3. 优化的 Thread Block 配置：16x64 block，128 threads
  4. 流水线优化：2 阶段流水线隐藏内存延迟
  5. 融合反量化：在 kernel 内部完成 IQ2_XS → BF16 反量化
  6. 查找表缓存：iq2xs_grid 和 ksigns_iq2xs 加载到 shared memory
"""
import os
import torch
import tilelang
import tilelang.language as T
from typing import Tuple

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"

# IQ2_XS 超级块大小 — 每个 block 编码 256 个元素
QK_K = 256

# ============================================================================
# IQ2_XS 查找表数据
# 来源: csrc/iq2_xs.cuh 第四部分
# ============================================================================

# 符号位掩码 — 用于反量化时判断每个元素的符号
# 来源: ggml/src/ggml-common.h
KMASK_IQ2XS = [1, 2, 4, 8, 16, 32, 64, 128]

# 符号反转查找表 — 128 项, 用于 7-bit 符号编码
# 只包含偶数位 1 的值 (popcnt 为偶数), 用于将 7-bit 符号索引展开为 8-bit 符号字节
# 来源: csrc/iq2_xs.cuh 第140-149行
KSIGNS_IQ2XS = [
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
]

# 物理网格查找表 — 512 项, 每项 8 字节 (8 个 int8)
# 字节取值: 0x08=8, 0x19=25, 0x2b=43 (对应 L=0,1,2 的 2*L+1=1,3,5 乘以 8)
# 来源: csrc/iq2_xs.cuh 第154-283行
IQ2XS_GRID_U64 = [
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
]


def _make_grid_u8_tensor(device='cuda'):
    """将 iq2xs_grid 从 uint64[512] 展开为 uint8[512, 8] 张量。

    每个 uint64 项包含 8 个字节, 小端序排列。
    展开后方便 TileLang kernel 按 [grid_idx, byte_idx] 索引。
    """
    grid_u8 = torch.zeros(512, 8, dtype=torch.uint8, device='cpu')
    for i in range(512):
        val = IQ2XS_GRID_U64[i]
        for j in range(8):
            grid_u8[i, j] = (val >> (8 * j)) & 0xFF
    return grid_u8.to(device)


def _make_ksigns_tensor(device='cuda'):
    """将 ksigns_iq2xs 从 list 转为 uint8[128] 张量。"""
    return torch.tensor(KSIGNS_IQ2XS, dtype=torch.uint8, device=device)


@tilelang.jit(pass_configs=pass_configs)
def iq2xs_gemm_kernel_optimized(N: int, K: int):
    """优化的 IQ2_XS GEMM TileLang kernel。

    计算：C[M, N] = A[M, K] @ B_dequant[N, K]^T
    其中 B_dequant 从 IQ2_XS 格式反量化得到。

    反量化公式（来源: csrc/iq2_xs.cu 第110-143行）:
      y[j] = d * (0.5 + scale_4bit) * 0.25 * grid[j] * (signs & mask[j] ? -1 : 1)

    IQ2_XS 数据布局（来源: csrc/iq2_xs.cuh 第89-98行 block_iq2_xs）:
      d:      [N, n_blocks] float16 — 全局缩放因子
      qs:     [N, n_blocks, 32] uint16 — 低 9-bit = grid 索引, 高 7-bit = 符号索引
      scales: [N, n_blocks, 8] uint8 — 高低 4-bit 各编码一个子块缩放

    每个 block 编码 256 个元素 (QK_K=256):
      - 32 个 uint16_t qs: 每个 qs 解码为 8 个元素
        - ib=0..7 (32元素 sub-block), il=0..3 (8元素组)
        - qs[4*ib + il] 的低 9-bit = grid 索引, 高 7-bit = 符号索引
      - 8 个 uint8_t scales: scales[ib] 的高低 4-bit 各编码一个子块缩放
        - (scales[ib] >> 4*(il/2)) & 0xf = scale_4bit

    参数:
        N: 输出列数（B 矩阵的行数）
        K: 缩减维度（A 的列数 = B 的列数，必须是 256 的倍数）
    """
    M = T.symbolic("M")

    # block_K 必须是 QK_K(256) 的倍数，因为 IQ2_XS 反量化需要完整的 256 元素 block
    block_M = 16
    block_N = 64
    block_K = QK_K  # 256 — 每个 K 迭代处理一个完整的 IQ2_XS block
    threads = 128
    n_blocks = K // QK_K  # K 维度上的 IQ2_XS block 数量

    @T.prim_func
    def main(
        A: T.Tensor[(M, K), BF16],
        qs: T.Tensor[(N, n_blocks, 32), "uint16"],       # grid 索引 + 符号索引
        scales: T.Tensor[(N, n_blocks, 8), "uint8"],     # 4-bit 打包的子块缩放
        d: T.Tensor[(N, n_blocks), "float16"],            # 全局缩放因子
        grid: T.Tensor[(512, 8), "uint8"],                # iq2xs_grid 查找表
        ksigns: T.Tensor[(128), "uint8"],                 # ksigns_iq2xs 查找表
        C: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), BF16)
            B_shared = T.alloc_shared((block_N, block_K), BF16)
            C_shared = T.alloc_shared((block_M, block_N), BF16)

            # IQ2_XS 量化数据 shared memory
            B_qs_shared = T.alloc_shared((block_N, block_K // QK_K, 32), "uint16")
            B_scales_shared = T.alloc_shared((block_N, block_K // QK_K, 8), "uint8")
            B_d_shared = T.alloc_shared((block_N, block_K // QK_K), "float16")

            C_local = T.alloc_fragment((block_M, block_N), FP32)
            C_local_accum = T.alloc_fragment((block_M, block_N), FP32)

            # 查找表 shared memory — 所有线程共享，只需加载一次
            grid_shared = T.alloc_shared((512, 8), "uint8")
            ksigns_shared = T.alloc_shared((128), "uint8")

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            # 加载查找表到 shared memory
            # iq2xs_grid: 512*8 = 4096 字节
            for i, j in T.Parallel(512, 8):
                grid_shared[i, j] = grid[i, j]
            # ksigns_iq2xs: 128 字节
            for i in T.Parallel(128):
                ksigns_shared[i] = ksigns[i]

            K_iters = T.ceildiv(K, block_K)

            for k in T.Pipelined(K_iters, num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)

                # 加载 IQ2_XS 量化数据到 shared memory
                # block_K=QK_K=256，每个 K 迭代正好处理 1 个 IQ2_XS block
                n_blocks_per_iter = block_K // QK_K
                for i, nb, q in T.Parallel(block_N, n_blocks_per_iter, 32):
                    row_idx = bx * block_N + i
                    block_idx = k * n_blocks_per_iter + nb
                    B_qs_shared[i, nb, q] = qs[row_idx, block_idx, q]

                for i, nb, s in T.Parallel(block_N, n_blocks_per_iter, 8):
                    row_idx = bx * block_N + i
                    block_idx = k * n_blocks_per_iter + nb
                    B_scales_shared[i, nb, s] = scales[row_idx, block_idx, s]

                for i, nb in T.Parallel(block_N, n_blocks_per_iter):
                    row_idx = bx * block_N + i
                    block_idx = k * n_blocks_per_iter + nb
                    B_d_shared[i, nb] = d[row_idx, block_idx]

                # IQ2_XS 反量化：在 shared memory 中完成
                # 反量化公式（来源: csrc/iq2_xs.cu 第110-143行）:
                #   y[j] = d * (0.5 + scale_4bit) * 0.25 * grid[j] * sign
                #
                # 每个 IQ2_XS block 编码 256 个元素:
                #   ib=0..7 (32元素 sub-block), il=0..3 (8元素组)
                #   k_global = nb*256 + ib*32 + il*8 + j  (0..255)
                #   qs_idx = 4*ib + il  (0..31)
                #   scale_4bit = (scales[ib] >> 4*(il/2)) & 0xf
                #   grid_idx = qs[qs_idx] & 511
                #   sign_idx = qs[qs_idx] >> 9
                #   grid_val = grid[grid_idx][j]  (8, 25, 或 43)
                #   signs = ksigns[sign_idx]
                #   sign = (signs & kmask[j]) ? -1 : 1
                for i, j in T.Parallel(block_N, block_K):
                    row_idx = bx * block_N + i
                    # j 是 block_K 内的偏移 (0..block_K-1)
                    # 需要映射到 IQ2_XS block 内的位置
                    nb_local = j // QK_K       # IQ2_XS block 索引 (在 block_K 内)
                    j_in_block = j % QK_K      # block 内偏移 (0..255)

                    ib = j_in_block // 32      # sub-block 索引 (0..7)
                    j_in_sub = j_in_block % 32 # sub-block 内偏移 (0..31)
                    il = j_in_sub // 8         # 8元素组索引 (0..3)
                    j_in_group = j_in_sub % 8  # 组内偏移 (0..7)

                    # qs 索引: qs[4*ib + il]
                    qs_idx = 4 * ib + il
                    qs_val = B_qs_shared[i, nb_local, qs_idx]

                    # grid 索引: qs_val & 511 (低 9-bit)
                    grid_idx = qs_val & 511
                    # 符号索引: qs_val >> 9 (高 7-bit)
                    sign_idx = qs_val >> 9

                    # 从 grid 查找表获取量化值 (8, 25, 或 43)
                    grid_val = T.Cast(FP32, grid_shared[grid_idx, j_in_group])

                    # 从 ksigns 查找表获取符号字节
                    signs_val = ksigns_shared[sign_idx]

                    # 符号判断: (signs_val & kmask[j_in_group]) ? -1 : 1
                    # kmask = [1, 2, 4, 8, 16, 32, 64, 128]
                    kmask_val = 1 << j_in_group
                    sign_bit = signs_val & kmask_val
                    # sign_bit != 0 时 sign = -1, 否则 sign = 1
                    # 用条件表达式: sign = 1 - 2 * (sign_bit != 0)
                    sign_val = T.Cast(FP32, 1 - 2 * T.Cast("int32", sign_bit != 0))

                    # 解码 4-bit 子块缩放
                    # scale_4bit = (scales[ib] >> 4*(il/2)) & 0xf
                    scale_byte = B_scales_shared[i, nb_local, ib]
                    shift_amount = 4 * (il // 2)
                    scale_4bit = T.Cast(FP32, (scale_byte >> shift_amount) & 0xf)

                    # 全局缩放因子
                    d_val = T.Cast(FP32, B_d_shared[i, nb_local])

                    # 反量化: y = d * (0.5 + scale_4bit) * 0.25 * grid_val * sign
                    w_val = d_val * (0.5 + scale_4bit) * 0.25 * grid_val * sign_val
                    B_shared[i, j] = T.Cast(BF16, w_val)

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]

                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


_kernel_cache = {}

# 查找表 GPU 缓存：避免每次调用 iq2xs_gemm_optimized 重复创建张量
_grid_tensor_cache = {}
_ksigns_tensor_cache = {}


def _get_cached_grid_tensor(device):
    """获取缓存的 grid 查找表张量（按设备缓存，GPU 常驻）。"""
    if device not in _grid_tensor_cache:
        _grid_tensor_cache[device] = _make_grid_u8_tensor(device)
    return _grid_tensor_cache[device]


def _get_cached_ksigns_tensor(device):
    """获取缓存的 ksigns 查找表张量（按设备缓存，GPU 常驻）。"""
    if device not in _ksigns_tensor_cache:
        _ksigns_tensor_cache[device] = _make_ksigns_tensor(device)
    return _ksigns_tensor_cache[device]


def get_iq2xs_gemm_kernel(N: int, K: int):
    """获取或编译 IQ2_XS GEMM kernel（带缓存）。"""
    key = (N, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = iq2xs_gemm_kernel_optimized(N, K)
    return _kernel_cache[key]


def iq2xs_gemm_optimized(
    x: torch.Tensor,
    qs: torch.Tensor,
    scales: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """优化的 IQ2_XS GEMM：y = x @ W^T。

    参数:
        x: [M, K] BF16 输入矩阵
        qs: [N, n_blocks, 32] uint16 — grid 索引 + 符号索引
            每个 uint16: 低 9-bit = grid 索引 (0-511), 高 7-bit = 符号索引
        scales: [N, n_blocks, 8] uint8 — 4-bit 打包的子块缩放
            每个 uint8: 高 4-bit 和低 4-bit 各编码一个子块缩放 (0-15)
        d: [N, n_blocks] FP16 — 全局缩放因子

    返回:
        y: [M, N] BF16 输出矩阵
    """
    assert x.is_cuda and qs.is_cuda and scales.is_cuda and d.is_cuda
    assert x.is_contiguous()

    M = x.size(0)
    K = x.size(1)
    N = qs.size(0)
    assert K % QK_K == 0, f"K={K} 必须是 {QK_K} 的倍数"

    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    # 准备查找表张量（使用缓存，避免每次调用重复创建）
    grid_tensor = _get_cached_grid_tensor(x.device)
    ksigns_tensor = _get_cached_ksigns_tensor(x.device)

    kernel = get_iq2xs_gemm_kernel(N, K)
    kernel(x, qs, scales, d, grid_tensor, ksigns_tensor, y)

    return y


def benchmark_iq2xs_gemm():
    """性能测试：对比优化版与原始实现。"""
    import time

    print("=" * 70)
    print("优化版 IQ2_XS GEMM 性能测试")
    print("=" * 70)

    M, N, K = 1, 2048, 7168
    n_blocks = K // QK_K

    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    # IQ2_XS 格式数据
    qs = torch.randint(0, 32768, (N, n_blocks, 32), dtype=torch.int16, device='cuda').to(torch.uint16)
    scales = torch.randint(0, 256, (N, n_blocks, 8), dtype=torch.uint8, device='cuda')
    d = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda')

    print(f"\n输入形状:")
    print(f"  x: [{M}, {K}]")
    print(f"  W: [{N}, {K}] (IQ2_XS)")
    print(f"  qs: [{N}, {n_blocks}, 32] uint16")
    print(f"  scales: [{N}, {n_blocks}, 8] uint8")
    print(f"  d: [{N}, {n_blocks}] float16")
    print(f"  输出: [{M}, {N}]")

    print("\n[预热] 编译 kernel...")
    y = iq2xs_gemm_optimized(x, qs, scales, d)
    torch.cuda.synchronize()
    print("[预热] 完成")

    n_iter = 100
    start = time.perf_counter()
    for _ in range(n_iter):
        y = iq2xs_gemm_optimized(x, qs, scales, d)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iter * 1000

    print(f"\n[结果]")
    print(f"  IQ2_XS GEMM 时间: {elapsed:.3f} ms")
    print(f"  目标 (FP8 GEMM):  0.036 ms")
    print(f"  性能比: {elapsed / 0.036:.2f}x")

    return elapsed


def compare_with_fp8_gemm():
    """对比 IQ2_XS GEMM 与 FP8 GEMM 性能。"""
    import time
    from kernel import fp8_gemm

    print("\n" + "=" * 70)
    print("IQ2_XS GEMM vs FP8 GEMM 性能对比")
    print("=" * 70)

    M, N, K = 1, 2048, 7168
    n_blocks = K // QK_K

    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    x_fp8 = x_bf16.to(torch.float8_e4m3fn)
    x_scale = torch.ones(M, K // 128, dtype=torch.float32, device='cuda')

    # IQ2_XS 格式数据
    qs = torch.randint(0, 32768, (N, n_blocks, 32), dtype=torch.int16, device='cuda').to(torch.uint16)
    scales = torch.randint(0, 256, (N, n_blocks, 8), dtype=torch.uint8, device='cuda')
    d = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda')

    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    w_scale = torch.ones(N // 128, K // 128, dtype=torch.float32, device='cuda')

    print(f"\n矩阵形状: x=[{M}, {K}], W=[{N}, {K}]")

    y_iq2xs = iq2xs_gemm_optimized(x_bf16, qs, scales, d)

    torch.set_default_dtype(torch.bfloat16)
    y_fp8 = fp8_gemm(x_fp8, x_scale, w_fp8, w_scale)
    torch.cuda.synchronize()

    n_iter = 100

    start = time.perf_counter()
    for _ in range(n_iter):
        y_iq2xs = iq2xs_gemm_optimized(x_bf16, qs, scales, d)
    torch.cuda.synchronize()
    iq2xs_time = (time.perf_counter() - start) / n_iter * 1000

    start = time.perf_counter()
    for _ in range(n_iter):
        y_fp8 = fp8_gemm(x_fp8, x_scale, w_fp8, w_scale)
    torch.cuda.synchronize()
    fp8_time = (time.perf_counter() - start) / n_iter * 1000

    print(f"\n[性能对比]")
    print(f"  IQ2_XS GEMM: {iq2xs_time:.3f} ms")
    print(f"  FP8 GEMM:    {fp8_time:.3f} ms")
    print(f"  性能比:      {iq2xs_time / fp8_time:.2f}x")

    return iq2xs_time, fp8_time


def test_correctness():
    """正确性测试：对比 TileLang kernel 与 PyTorch 参考实现。

    参考实现使用与 C CUDA 代码相同的反量化公式:
      y[j] = d * (0.5 + scale_4bit) * 0.25 * grid[j] * (signs & mask[j] ? -1 : 1)
    """
    print("\n" + "=" * 70)
    print("IQ2_XS GEMM 正确性测试")
    print("=" * 70)

    M, N, K = 2, 128, 512
    n_blocks = K // QK_K

    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    # 生成 IQ2_XS 格式数据
    # qs: 低 9-bit = grid 索引 (0-511), 高 7-bit = 符号索引 (0-127)
    qs = torch.zeros(N, n_blocks, 32, dtype=torch.uint16, device='cuda')
    for i in range(N):
        for j in range(n_blocks):
            for q in range(32):
                grid_idx = torch.randint(0, 512, (1,)).item()
                sign_idx = torch.randint(0, 128, (1,)).item()
                qs[i, j, q] = grid_idx | (sign_idx << 9)

    # scales: 高低 4-bit 各编码一个子块缩放 (0-15)
    scales = torch.zeros(N, n_blocks, 8, dtype=torch.uint8, device='cuda')
    for i in range(N):
        for j in range(n_blocks):
            for s in range(8):
                lo = torch.randint(0, 16, (1,)).item()
                hi = torch.randint(0, 16, (1,)).item()
                scales[i, j, s] = lo | (hi << 4)

    # d: 全局缩放因子
    d = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda')

    # TileLang kernel 计算
    y_tilelang = iq2xs_gemm_optimized(x, qs, scales, d)

    # PyTorch 参考实现 — 使用与 C CUDA 相同的反量化公式
    # 来源: csrc/iq2_xs.cu 第110-143行
    grid_u8 = _make_grid_u8_tensor('cuda')
    ksigns_tensor = _make_ksigns_tensor('cuda')

    w_ref = torch.zeros(N, K, dtype=torch.float32, device='cuda')
    for i in range(N):
        for j in range(n_blocks):
            d_val = d[i, j].item()
            for ib in range(8):       # 32元素 sub-block 索引
                for il in range(4):   # 8元素组索引
                    qs_idx = 4 * ib + il
                    qs_val = qs[i, j, qs_idx].item()
                    grid_idx = qs_val & 511
                    sign_idx = qs_val >> 9

                    # 解码 4-bit 子块缩放
                    scale_byte = scales[i, j, ib].item()
                    shift_amount = 4 * (il // 2)
                    scale_4bit = (scale_byte >> shift_amount) & 0xf

                    # 从 grid 查找表获取量化值
                    # 从 ksigns 查找表获取符号字节
                    signs_val = ksigns_tensor[sign_idx].item()

                    # 反量化: y = d * (0.5 + scale_4bit) * 0.25 * grid_val * sign
                    scale_factor = d_val * (0.5 + scale_4bit) * 0.25
                    for jj in range(8):
                        k_global = j * QK_K + ib * 32 + il * 8 + jj
                        grid_val = grid_u8[grid_idx, jj].item()
                        kmask_val = 1 << jj
                        sign = -1.0 if (signs_val & kmask_val) else 1.0
                        w_ref[i, k_global] = scale_factor * grid_val * sign

    y_ref = x.float() @ w_ref.T
    y_ref = y_ref.to(torch.bfloat16)

    error = (y_tilelang.float() - y_ref.float()).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    # 相对误差（避免除零）
    y_ref_abs = y_ref.float().abs()
    y_ref_max = y_ref_abs.max().item()
    relative_error = max_error / max(y_ref_max, 1.0)

    print(f"\n矩阵形状: x=[{M}, {K}], W=[{N}, {K}]")
    print(f"\n[误差分析]")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  平均误差: {mean_error:.6f}")
    print(f"  相对误差: {relative_error:.6f}")

    # BF16 精度约为 1/128 ≈ 0.78%，允许 2% 相对误差
    if relative_error < 0.02:
        print("\n[结果] ✓ 正确性测试通过（BF16 精度范围内）")
        return True
    else:
        print("\n[结果] ✗ 正确性测试失败")
        return False


if __name__ == "__main__":
    test_correctness()
    benchmark_iq2xs_gemm()
    compare_with_fp8_gemm()
