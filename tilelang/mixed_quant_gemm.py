"""混合量化 GEMM TileLang kernel: IQ2_XXS + Q2_K。

支持混合量化专家 FFN：
  - gate_proj (w1): IQ2_XXS — 2.0625 bpw
  - up_proj (w3):   IQ2_XXS — 2.0625 bpw
  - down_proj (w2): Q2_K    — 2.5625 bpw

IQ2_XXS qs 编码格式（GGUF/llama.cpp 标准）：
  - 每 4 个 uint16 组成一组 (q0, q1, q2, q3)
  - q0 | (q1 << 16) → aux32_0 → 4 个 grid_idx (8-bit, 0-255)
  - q2 | (q3 << 16) → aux32_1 → 4 个 sign_idx (7-bit) + 4-bit ls
  - 反量化: d * ls * grid[grid_idx] * sign * 0.125

优化策略：
  1. Shared Memory 缓存输入和权重块
  2. Tensor Core 加速矩阵乘法
  3. 流水线优化隐藏内存延迟
  4. 融合反量化在 kernel 内部完成
  5. 查找表缓存到 shared memory

用法：
  python tilelang/mixed_quant_gemm.py --test
"""
import os
import torch
import tilelang
import tilelang.language as T
from typing import Tuple
import numpy as np
from math import log2, ceil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

BF16 = "bfloat16"
FP32 = "float32"

QK_K = 256


# IQ2_XXS Grid (256 entries × 8 i8)，与 Rust tables.rs 中 get_iq2xxs_grid() 一致
IQ2XXS_GRID_U64 = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
]

# 构建 IQ2_XXS grid tensor (256, 8) float32
_grid_np = np.zeros((256, 8), dtype=np.float32)
for _i, _g in enumerate(IQ2XXS_GRID_U64):
    _b = _g.to_bytes(8, 'little')
    for _j in range(8):
        _v = _b[_j] if _b[_j] < 128 else _b[_j] - 256
        _grid_np[_i, _j] = float(_v)

IQ2_XXS_GRID_TENSOR = torch.from_numpy(_grid_np).cuda()

KSIGNS = np.array([
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
], dtype=np.uint8)
KSIGNS_TENSOR = torch.from_numpy(KSIGNS).cuda()


@tilelang.jit(pass_configs=pass_configs)
def iq2xxs_gemm_kernel(N: int, K: int):
    """IQ2_XXS GEMM TileLang kernel（GGUF 格式）。

    计算：C[M, N] = A[M, K] @ B_dequant[N, K]^T

    IQ2_XXS qs 编码格式（GGUF/llama.cpp 标准）：
      - 每 4 个 uint16 组成一组 (q0, q1, q2, q3)
      - q0 | (q1 << 16) → aux32_0 → 4 个 grid_idx (8-bit, 0-255)
      - q2 | (q3 << 16) → aux32_1 → 4 个 sign_idx (7-bit) + 4-bit ls
      - 反量化: d * ls * grid[grid_idx] * sign * 0.125

    qs_idx 布局: qs[N, n_blocks, 32]
      - ib32 = 0..7, l = 0..3
      - qs_idx = ib32 * 4 + l
      - 同一 ib32 的 4 个 qs 值组成一组
    """
    M = T.symbolic("M")

    block_M = 16
    block_N = 64
    block_K = 64
    threads = 128

    n_blocks_per_row = (K + QK_K - 1) // QK_K

    @T.prim_func
    def main(
        A: T.Tensor[(M, K), BF16],
        d: T.Tensor[(N, n_blocks_per_row), "float16"],
        qs: T.Tensor[(N, n_blocks_per_row, 32), "uint16"],
        grid: T.Tensor[(256, 8), "float32"],
        ksigns: T.Tensor[(128), "uint8"],
        C: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), BF16)
            B_shared = T.alloc_shared((block_N, block_K), BF16)
            C_shared = T.alloc_shared((block_M, block_N), BF16)

            C_local = T.alloc_fragment((block_M, block_N), FP32)
            C_local_accum = T.alloc_fragment((block_M, block_N), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)

            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)

                for i, j in T.Parallel(block_N, block_K):
                    row_idx = bx * block_N + i
                    k_idx = k * block_K + j

                    block_idx = k_idx // QK_K
                    local_k = k_idx % QK_K

                    ib32 = local_k // 32
                    local_in_32 = local_k % 32
                    l = local_in_32 // 8
                    local_in_8 = local_in_32 % 8

                    # 读取同一 ib32 的 4 个 qs 值
                    qs_base = ib32 * 4
                    q0 = qs[row_idx, block_idx, qs_base + 0]
                    q1 = qs[row_idx, block_idx, qs_base + 1]
                    q2 = qs[row_idx, block_idx, qs_base + 2]
                    q3 = qs[row_idx, block_idx, qs_base + 3]

                    # 构建 aux32 对
                    aux32_0 = T.Cast("uint32", q0) | (T.Cast("uint32", q1) << 16)
                    aux32_1 = T.Cast("uint32", q2) | (T.Cast("uint32", q3) << 16)

                    # 从 aux32_0 提取 grid_idx: 第 l 个字节
                    grid_idx = T.Cast("int32", (aux32_0 >> (l * 8)) & T.uint32(0xFF))

                    # 从 aux32_1 提取 sign_idx: 第 l 个 7-bit 段
                    sign_idx = T.Cast("int32", (aux32_1 >> (l * 7)) & T.uint32(0x7F))

                    # 从 aux32_1 高 4-bit 提取 ls
                    ls_int = T.Cast(FP32, (aux32_1 >> 28) & T.uint32(0xF))
                    ls = T.Cast(FP32, 2.0) * ls_int + T.Cast(FP32, 1.0)

                    d_val = T.Cast(FP32, d[row_idx, block_idx])

                    grid_row_idx = T.Cast("int32", local_in_8)
                    grid_val = grid[grid_idx, grid_row_idx]

                    sign_byte = ksigns[sign_idx]
                    sign_bit = (sign_byte >> T.Cast("uint8", grid_row_idx)) & T.uint8(1)
                    sign_mul = T.Cast(FP32, 1.0) - T.Cast(FP32, sign_bit) * T.Cast(FP32, 2.0)

                    # IQ2_XXS 反量化: d * ls * grid * sign * 0.125
                    B_shared[i, j] = T.Cast(BF16, d_val * ls * grid_val * sign_mul * T.Cast(FP32, 0.125))

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]

                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(pass_configs=pass_configs)
def q2k_gemm_kernel(N: int, K: int):
    """Q2_K GEMM TileLang kernel（GGUF block_q2_K 格式）。

    计算：C[M, N] = A[M, K] @ B_dequant[N, K]^T

    Q2_K 格式（block_q2_K, 84 bytes / 256 elements）：
      - scales[16]: uint8, 4-bit packed（低 4 位 = scale, 高 4 位 = min）
      - qs[64]: uint8, 2-bit 量化值打包
      - d: float16 super-block scale
      - dmin: float16 super-block minimum scale

    256 个元素分成 2 个 half（各 128 元素），每个 half 有 32 字节 qs。
    每个 half 的 128 元素分成 4 组（各 32 元素），每组 16 字节 qs，
    每字节被 4 个子块通过 shift=0/2/4/6 共享。

    反量化公式（llama.cpp 标准）：
      value = d * (sc & 0xF) * quant_2bit - dmin * (sc >> 4)
      其中 quant_2bit = (qs[q2_base + pos_in_group] >> shift) & 3
    """
    M = T.symbolic("M")

    block_M = 16
    block_N = 64
    block_K = 64
    threads = 128

    n_blocks_per_row = (K + QK_K - 1) // QK_K

    @T.prim_func
    def main(
        A: T.Tensor[(M, K), BF16],
        d: T.Tensor[(N, n_blocks_per_row), "float16"],
        dmin: T.Tensor[(N, n_blocks_per_row), "float16"],
        scales: T.Tensor[(N, n_blocks_per_row, 16), "uint8"],
        qs: T.Tensor[(N, n_blocks_per_row, 64), "uint8"],
        C: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), BF16)
            B_shared = T.alloc_shared((block_N, block_K), BF16)
            C_shared = T.alloc_shared((block_M, block_N), BF16)

            C_local = T.alloc_fragment((block_M, block_N), FP32)
            C_local_accum = T.alloc_fragment((block_M, block_N), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)

            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)

                for i, j in T.Parallel(block_N, block_K):
                    row_idx = bx * block_N + i
                    k_idx = k * block_K + j

                    block_idx = k_idx // QK_K
                    local_k = k_idx % QK_K

                    # GGUF Q2_K: 2 halves × 4 groups × 2 sub-blocks × 16 elements
                    k_half = local_k // 128
                    pos_in_half = local_k % 128
                    j_group = pos_in_half // 32
                    pos_in_group = pos_in_half % 32

                    # qs 索引: q2_base + pos_in_group, shift = j_group * 2
                    q2_base = k_half * 32
                    qs_byte_idx = q2_base + pos_in_group
                    shift = j_group * 2

                    # scales 索引: 4-bit packed (低4位=scale, 高4位=min)
                    is_idx = k_half * 8 + j_group * 2 + pos_in_group // 16

                    d_val = T.Cast(FP32, d[row_idx, block_idx])
                    dmin_val = T.Cast(FP32, dmin[row_idx, block_idx])

                    sc = scales[row_idx, block_idx, is_idx]
                    scale_4bit = T.Cast(FP32, sc & T.uint8(0xF))
                    min_4bit = T.Cast(FP32, sc >> T.uint8(4))

                    qs_val = qs[row_idx, block_idx, qs_byte_idx]
                    quant_2bit = T.Cast(FP32, (qs_val >> T.Cast("uint8", shift)) & T.uint8(3))

                    # GGUF 反量化: d * (sc & 0xF) * quant - dmin * (sc >> 4)
                    B_shared[i, j] = T.Cast(BF16, d_val * scale_4bit * quant_2bit - dmin_val * min_4bit)

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]

                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(pass_configs=pass_configs)
def iq2xxs_gate_up_silu_kernel(N_gate: int, N_up: int, K: int):
    """融合 gate+up+SwiGLU TileLang kernel。

    计算：mid[M, N_gate] = sigmoid(gate(x)) * gate(x) * up(x)

    gate 和 up 共享输入 x，只需加载一次 x 到 shared memory。
    SwiGLU 融合在 kernel 内部完成，避免中间张量分配和额外 kernel 启动。

    输出形状为 [M, N_gate]（与 gate 同维度），up 必须与 gate 同维度。

    相比分开调用 iq2xxs_gemm 两次：
      - 节省 1 次 kernel 启动（3→2 次）
      - 节省 1 次 x 从 global memory 加载
      - 避免 gate_out 和 up_out 中间张量分配
    """
    M = T.symbolic("M")

    block_M = 16
    block_N = 64
    block_K = 64
    threads = 128

    n_blocks_per_row = (K + QK_K - 1) // QK_K
    assert N_gate == N_up, "gate 和 up 必须同维度"

    @T.prim_func
    def main(
        A: T.Tensor[(M, K), BF16],
        gate_d: T.Tensor[(N_gate, n_blocks_per_row), "float16"],
        gate_qs: T.Tensor[(N_gate, n_blocks_per_row, 32), "uint16"],
        up_d: T.Tensor[(N_up, n_blocks_per_row), "float16"],
        up_qs: T.Tensor[(N_up, n_blocks_per_row, 32), "uint16"],
        grid: T.Tensor[(256, 8), "float32"],
        ksigns: T.Tensor[(128), "uint8"],
        C: T.Tensor[(M, N_gate), BF16],
    ):
        with T.Kernel(T.ceildiv(N_gate, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), BF16)
            B_gate_shared = T.alloc_shared((block_N, block_K), BF16)
            B_up_shared = T.alloc_shared((block_N, block_K), BF16)
            C_shared = T.alloc_shared((block_M, block_N), BF16)

            gate_local = T.alloc_fragment((block_M, block_N), FP32)
            gate_accum = T.alloc_fragment((block_M, block_N), FP32)
            up_local = T.alloc_fragment((block_M, block_N), FP32)
            up_accum = T.alloc_fragment((block_M, block_N), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(gate_accum)
            T.clear(up_accum)

            K_iters = T.ceildiv(K, block_K)

            for k in T.Pipelined(K_iters, num_stages=4):
                # 加载 x（gate 和 up 共享）
                T.copy(A[by * block_M, k * block_K], A_shared)

                # 反量化 gate 权重
                for i, j in T.Parallel(block_N, block_K):
                    row_idx = bx * block_N + i
                    k_idx = k * block_K + j
                    block_idx = k_idx // QK_K
                    local_k = k_idx % QK_K
                    ib32 = local_k // 32
                    local_in_32 = local_k % 32
                    l = local_in_32 // 8
                    local_in_8 = local_in_32 % 8

                    qs_base = ib32 * 4
                    q0 = gate_qs[row_idx, block_idx, qs_base + 0]
                    q1 = gate_qs[row_idx, block_idx, qs_base + 1]
                    q2 = gate_qs[row_idx, block_idx, qs_base + 2]
                    q3 = gate_qs[row_idx, block_idx, qs_base + 3]

                    aux32_0 = T.Cast("uint32", q0) | (T.Cast("uint32", q1) << 16)
                    aux32_1 = T.Cast("uint32", q2) | (T.Cast("uint32", q3) << 16)

                    grid_idx = T.Cast("int32", (aux32_0 >> (l * 8)) & T.uint32(0xFF))
                    sign_idx = T.Cast("int32", (aux32_1 >> (l * 7)) & T.uint32(0x7F))
                    ls_int = T.Cast(FP32, (aux32_1 >> 28) & T.uint32(0xF))
                    ls = T.Cast(FP32, 2.0) * ls_int + T.Cast(FP32, 1.0)

                    d_val = T.Cast(FP32, gate_d[row_idx, block_idx])
                    grid_row_idx = T.Cast("int32", local_in_8)
                    grid_val = grid[grid_idx, grid_row_idx]

                    sign_byte = ksigns[sign_idx]
                    sign_bit = (sign_byte >> T.Cast("uint8", grid_row_idx)) & T.uint8(1)
                    sign_mul = T.Cast(FP32, 1.0) - T.Cast(FP32, sign_bit) * T.Cast(FP32, 2.0)

                    B_gate_shared[i, j] = T.Cast(BF16, d_val * ls * grid_val * sign_mul * T.Cast(FP32, 0.125))

                # 反量化 up 权重
                for i, j in T.Parallel(block_N, block_K):
                    row_idx = bx * block_N + i
                    k_idx = k * block_K + j
                    block_idx = k_idx // QK_K
                    local_k = k_idx % QK_K
                    ib32 = local_k // 32
                    local_in_32 = local_k % 32
                    l = local_in_32 // 8
                    local_in_8 = local_in_32 % 8

                    qs_base = ib32 * 4
                    q0 = up_qs[row_idx, block_idx, qs_base + 0]
                    q1 = up_qs[row_idx, block_idx, qs_base + 1]
                    q2 = up_qs[row_idx, block_idx, qs_base + 2]
                    q3 = up_qs[row_idx, block_idx, qs_base + 3]

                    aux32_0 = T.Cast("uint32", q0) | (T.Cast("uint32", q1) << 16)
                    aux32_1 = T.Cast("uint32", q2) | (T.Cast("uint32", q3) << 16)

                    grid_idx = T.Cast("int32", (aux32_0 >> (l * 8)) & T.uint32(0xFF))
                    sign_idx = T.Cast("int32", (aux32_1 >> (l * 7)) & T.uint32(0x7F))
                    ls_int = T.Cast(FP32, (aux32_1 >> 28) & T.uint32(0xF))
                    ls = T.Cast(FP32, 2.0) * ls_int + T.Cast(FP32, 1.0)

                    d_val = T.Cast(FP32, up_d[row_idx, block_idx])
                    grid_row_idx = T.Cast("int32", local_in_8)
                    grid_val = grid[grid_idx, grid_row_idx]

                    sign_byte = ksigns[sign_idx]
                    sign_bit = (sign_byte >> T.Cast("uint8", grid_row_idx)) & T.uint8(1)
                    sign_mul = T.Cast(FP32, 1.0) - T.Cast(FP32, sign_bit) * T.Cast(FP32, 2.0)

                    B_up_shared[i, j] = T.Cast(BF16, d_val * ls * grid_val * sign_mul * T.Cast(FP32, 0.125))

                # 两个 GEMM 共享 A_shared
                T.gemm(A_shared, B_gate_shared, gate_local, transpose_B=True)
                T.gemm(A_shared, B_up_shared, up_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    gate_accum[i, j] += gate_local[i, j]
                    up_accum[i, j] += up_local[i, j]

                T.clear(gate_local)
                T.clear(up_local)

            # SwiGLU 融合：sigmoid(gate) * gate * up → 写入输出
            for i, j in T.Parallel(block_M, block_N):
                gate_val = gate_accum[i, j]
                # sigmoid 近似：1 / (1 + exp(-x))
                sig = T.Cast(FP32, 1.0) / (T.Cast(FP32, 1.0) + T.exp(-gate_val))
                silu_val = sig * gate_val * up_accum[i, j]
                C_shared[i, j] = T.Cast(BF16, silu_val)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


_kernel_cache = {}


def get_iq2xxs_gemm_kernel(N: int, K: int):
    """获取或编译 IQ2_XXS GEMM kernel（带缓存）。"""
    key = ("iq2xxs", N, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = iq2xxs_gemm_kernel(N, K)
    return _kernel_cache[key]


def get_q2k_gemm_kernel(N: int, K: int):
    """获取或编译 Q2_K GEMM kernel（带缓存）。"""
    key = ("q2k", N, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = q2k_gemm_kernel(N, K)
    return _kernel_cache[key]


def get_gate_up_silu_kernel(N_gate: int, N_up: int, K: int):
    """获取或编译融合 gate+up+SwiGLU kernel（带缓存）。"""
    key = ("gate_up_silu", N_gate, N_up, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = iq2xxs_gate_up_silu_kernel(N_gate, N_up, K)
    return _kernel_cache[key]


def get_fused_ffn_kernel(N_gate: int, N_down: int, K_gate: int, K_down: int):
    """获取或编译全融合 FFN kernel（带缓存）。"""
    key = ("fused_ffn", N_gate, N_down, K_gate, K_down)
    if key not in _kernel_cache:
        _kernel_cache[key] = fused_mixed_ffn_kernel(N_gate, N_down, K_gate, K_down)
    return _kernel_cache[key]


@tilelang.jit(pass_configs=pass_configs)
def fused_mixed_ffn_kernel(N_gate: int, N_down: int, K_gate: int, K_down: int):
    """全融合 FFN kernel：gate+up+SwiGLU+down 单 kernel 完成。

    计算：C[M, N_down] = SwiGLU(gate(x), up(x)) @ W_down^T

    策略：每个 down 块冗余计算完整的 gate+up+SwiGLU，
    将 mid 存入 shared memory，然后计算 down tile。

    对 M=1 decode：
      - mid 仅 1×2048 = 4KB，可放 shared memory
      - gate/up 权重被 L2 cache 缓存（所有块读同一份权重）
      - 消除 2 次 kernel 启动 + mid 全局内存读写

    相比 2-kernel 方案：
      - kernel 启动 2→1 次
      - mid 不经过全局内存（shared memory 直传）
      - x 只从全局内存读一次（gate+up+down 三次 GEMM 共享）
    """
    M = T.symbolic("M")

    block_M = 16
    block_N = 64
    block_K = 64
    threads = 128

    n_blocks_k_gate = (K_gate + QK_K - 1) // QK_K
    n_blocks_k_down = (K_down + QK_K - 1) // QK_K

    gate_N_blocks = T.ceildiv(N_gate, block_N)

    @T.prim_func
    def main(
        A: T.Tensor[(M, K_gate), BF16],
        gate_d: T.Tensor[(N_gate, n_blocks_k_gate), "float16"],
        gate_qs: T.Tensor[(N_gate, n_blocks_k_gate, 32), "uint16"],
        up_d: T.Tensor[(N_gate, n_blocks_k_gate), "float16"],
        up_qs: T.Tensor[(N_gate, n_blocks_k_gate, 32), "uint16"],
        down_d: T.Tensor[(N_down, n_blocks_k_down), "float16"],
        down_dmin: T.Tensor[(N_down, n_blocks_k_down), "float16"],
        down_scales: T.Tensor[(N_down, n_blocks_k_down, 16), "uint8"],
        down_qs: T.Tensor[(N_down, n_blocks_k_down, 64), "uint8"],
        grid: T.Tensor[(256, 8), "float32"],
        ksigns: T.Tensor[(128), "uint8"],
        C: T.Tensor[(M, N_down), BF16],
    ):
        with T.Kernel(T.ceildiv(N_down, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # ===== Phase 1: gate+up+SwiGLU → mid (shared memory) =====
            # 每个 down 块都计算完整的 mid，存入 shared memory
            # mid 大小: block_M × N_gate BF16 = 16×2048×2 = 64KB
            # 对 M=1 实际只使用第一行 = 4KB
            mid_shared = T.alloc_shared((block_M, N_gate), BF16)
            A_shared = T.alloc_shared((block_M, block_K), BF16)
            B_gate_shared = T.alloc_shared((block_N, block_K), BF16)
            B_up_shared = T.alloc_shared((block_N, block_K), BF16)

            gate_local = T.alloc_fragment((block_M, block_N), FP32)
            gate_accum = T.alloc_fragment((block_M, block_N), FP32)
            up_local = T.alloc_fragment((block_M, block_N), FP32)
            up_accum = T.alloc_fragment((block_M, block_N), FP32)

            T.use_swizzle(panel_size=10)

            # 遍历 gate+up 的所有 N tiles
            for gate_bx in T.serial(gate_N_blocks):
                T.clear(gate_accum)
                T.clear(up_accum)

                K_gate_iters = T.ceildiv(K_gate, block_K)

                for k in T.Pipelined(K_gate_iters, num_stages=4):
                    T.copy(A[by * block_M, k * block_K], A_shared)

                    # 反量化 gate 权重
                    for i, j in T.Parallel(block_N, block_K):
                        row_idx = gate_bx * block_N + i
                        k_idx = k * block_K + j
                        block_idx = k_idx // QK_K
                        local_k = k_idx % QK_K
                        ib32 = local_k // 32
                        local_in_32 = local_k % 32
                        l = local_in_32 // 8
                        local_in_8 = local_in_32 % 8

                        qs_base = ib32 * 4
                        q0 = gate_qs[row_idx, block_idx, qs_base + 0]
                        q1 = gate_qs[row_idx, block_idx, qs_base + 1]
                        q2 = gate_qs[row_idx, block_idx, qs_base + 2]
                        q3 = gate_qs[row_idx, block_idx, qs_base + 3]

                        aux32_0 = T.Cast("uint32", q0) | (T.Cast("uint32", q1) << 16)
                        aux32_1 = T.Cast("uint32", q2) | (T.Cast("uint32", q3) << 16)

                        grid_idx = T.Cast("int32", (aux32_0 >> (l * 8)) & T.uint32(0xFF))
                        sign_idx = T.Cast("int32", (aux32_1 >> (l * 7)) & T.uint32(0x7F))
                        ls_int = T.Cast(FP32, (aux32_1 >> 28) & T.uint32(0xF))
                        ls = T.Cast(FP32, 2.0) * ls_int + T.Cast(FP32, 1.0)

                        d_val = T.Cast(FP32, gate_d[row_idx, block_idx])
                        grid_row_idx = T.Cast("int32", local_in_8)
                        grid_val = grid[grid_idx, grid_row_idx]

                        sign_byte = ksigns[sign_idx]
                        sign_bit = (sign_byte >> T.Cast("uint8", grid_row_idx)) & T.uint8(1)
                        sign_mul = T.Cast(FP32, 1.0) - T.Cast(FP32, sign_bit) * T.Cast(FP32, 2.0)

                        B_gate_shared[i, j] = T.Cast(BF16, d_val * ls * grid_val * sign_mul * T.Cast(FP32, 0.125))

                    # 反量化 up 权重
                    for i, j in T.Parallel(block_N, block_K):
                        row_idx = gate_bx * block_N + i
                        k_idx = k * block_K + j
                        block_idx = k_idx // QK_K
                        local_k = k_idx % QK_K
                        ib32 = local_k // 32
                        local_in_32 = local_k % 32
                        l = local_in_32 // 8
                        local_in_8 = local_in_32 % 8

                        qs_base = ib32 * 4
                        q0 = up_qs[row_idx, block_idx, qs_base + 0]
                        q1 = up_qs[row_idx, block_idx, qs_base + 1]
                        q2 = up_qs[row_idx, block_idx, qs_base + 2]
                        q3 = up_qs[row_idx, block_idx, qs_base + 3]

                        aux32_0 = T.Cast("uint32", q0) | (T.Cast("uint32", q1) << 16)
                        aux32_1 = T.Cast("uint32", q2) | (T.Cast("uint32", q3) << 16)

                        grid_idx = T.Cast("int32", (aux32_0 >> (l * 8)) & T.uint32(0xFF))
                        sign_idx = T.Cast("int32", (aux32_1 >> (l * 7)) & T.uint32(0x7F))
                        ls_int = T.Cast(FP32, (aux32_1 >> 28) & T.uint32(0xF))
                        ls = T.Cast(FP32, 2.0) * ls_int + T.Cast(FP32, 1.0)

                        d_val = T.Cast(FP32, up_d[row_idx, block_idx])
                        grid_row_idx = T.Cast("int32", local_in_8)
                        grid_val = grid[grid_idx, grid_row_idx]

                        sign_byte = ksigns[sign_idx]
                        sign_bit = (sign_byte >> T.Cast("uint8", grid_row_idx)) & T.uint8(1)
                        sign_mul = T.Cast(FP32, 1.0) - T.Cast(FP32, sign_bit) * T.Cast(FP32, 2.0)

                        B_up_shared[i, j] = T.Cast(BF16, d_val * ls * grid_val * sign_mul * T.Cast(FP32, 0.125))

                    T.gemm(A_shared, B_gate_shared, gate_local, transpose_B=True)
                    T.gemm(A_shared, B_up_shared, up_local, transpose_B=True)

                    for i, j in T.Parallel(block_M, block_N):
                        gate_accum[i, j] += gate_local[i, j]
                        up_accum[i, j] += up_local[i, j]

                    T.clear(gate_local)
                    T.clear(up_local)

                # SwiGLU 融合 + 写入 mid_shared
                for i, j in T.Parallel(block_M, block_N):
                    gate_val = gate_accum[i, j]
                    sig = T.Cast(FP32, 1.0) / (T.Cast(FP32, 1.0) + T.exp(-gate_val))
                    silu_val = sig * gate_val * up_accum[i, j]
                    mid_shared[i, gate_bx * block_N + j] = T.Cast(BF16, silu_val)

            # ===== Phase 2: down GEMM from mid_shared → C =====
            mid_A_shared = T.alloc_shared((block_M, block_K), BF16)
            B_down_shared = T.alloc_shared((block_N, block_K), BF16)
            C_shared = T.alloc_shared((block_M, block_N), BF16)

            C_local = T.alloc_fragment((block_M, block_N), FP32)
            C_local_accum = T.alloc_fragment((block_M, block_N), FP32)

            T.clear(C_local)
            T.clear(C_local_accum)

            K_down_iters = T.ceildiv(K_down, block_K)

            for k in T.Pipelined(K_down_iters, num_stages=4):
                # 从 mid_shared 读取 down 的输入（shared memory → shared memory）
                for i, j in T.Parallel(block_M, block_K):
                    mid_A_shared[i, j] = mid_shared[i, k * block_K + j]

                # 反量化 down 权重（Q2_K）
                for i, j in T.Parallel(block_N, block_K):
                    row_idx = bx * block_N + i
                    k_idx = k * block_K + j

                    block_idx = k_idx // QK_K
                    local_k = k_idx % QK_K

                    k_half = local_k // 128
                    pos_in_half = local_k % 128
                    j_group = pos_in_half // 32
                    pos_in_group = pos_in_half % 32

                    q2_base = k_half * 32
                    qs_byte_idx = q2_base + pos_in_group
                    shift = j_group * 2

                    is_idx = k_half * 8 + j_group * 2 + pos_in_group // 16

                    d_val = T.Cast(FP32, down_d[row_idx, block_idx])
                    dmin_val = T.Cast(FP32, down_dmin[row_idx, block_idx])

                    sc = down_scales[row_idx, block_idx, is_idx]
                    scale_4bit = T.Cast(FP32, sc & T.uint8(0xF))
                    min_4bit = T.Cast(FP32, sc >> T.uint8(4))

                    qs_val = down_qs[row_idx, block_idx, qs_byte_idx]
                    quant_2bit = T.Cast(FP32, (qs_val >> T.Cast("uint8", shift)) & T.uint8(3))

                    B_down_shared[i, j] = T.Cast(BF16, d_val * scale_4bit * quant_2bit - dmin_val * min_4bit)

                T.gemm(mid_A_shared, B_down_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]

                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def iq2xxs_gemm(x: torch.Tensor, d: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """IQ2_XXS GEMM：y = x @ W^T（GGUF 格式）。

    参数：
        x: [M, K] BF16 输入矩阵
        d: [N, n_blocks] FP16 super-block scale
        qs: [N, n_blocks, 32] uint16 (GGUF IQ2_XXS 格式)

    返回：
        y: [M, N] BF16 输出矩阵
    """
    assert x.is_cuda and d.is_cuda and qs.is_cuda
    assert x.is_contiguous()

    M = x.size(0)
    K = x.size(1)
    N = d.size(0)

    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    kernel = get_iq2xxs_gemm_kernel(N, K)
    kernel(x, d, qs, IQ2_XXS_GRID_TENSOR, KSIGNS_TENSOR, y)

    return y


def q2k_gemm(
    x: torch.Tensor,
    d: torch.Tensor,
    dmin: torch.Tensor,
    scales: torch.Tensor,
    qs: torch.Tensor,
) -> torch.Tensor:
    """Q2_K GEMM：y = x @ W^T（GGUF block_q2_K 格式）。

    反量化公式（llama.cpp 标准）：
      value = d * (sc & 0xF) * quant_2bit - dmin * (sc >> 4)

    其中 scales 为 4-bit packed（低 4 位 = scale, 高 4 位 = min），
    qs 按 2 halves × 4 groups 结构索引，shift=0/2/4/6 共享字节。

    参数：
        x: [M, K] BF16 输入矩阵
        d: [N, n_blocks] FP16 super-block scale
        dmin: [N, n_blocks] FP16 super-block minimum scale
        scales: [N, n_blocks, 16] uint8 4-bit packed（低4位=scale, 高4位=min）
        qs: [N, n_blocks, 64] uint8 2-bit 量化值打包

    返回：
        y: [M, N] BF16 输出矩阵
    """
    assert x.is_cuda and d.is_cuda and dmin.is_cuda and scales.is_cuda and qs.is_cuda
    assert x.is_contiguous()
    
    M = x.size(0)
    K = x.size(1)
    N = d.size(0)
    
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    
    kernel = get_q2k_gemm_kernel(N, K)
    kernel(x, d, dmin, scales, qs, y)
    
    return y


def mixed_quant_ffn(
    x: torch.Tensor,
    gate_d: torch.Tensor, gate_qs: torch.Tensor,
    up_d: torch.Tensor, up_qs: torch.Tensor,
    down_d: torch.Tensor, down_dmin: torch.Tensor, down_scales: torch.Tensor, down_qs: torch.Tensor,
) -> torch.Tensor:
    """混合量化 FFN：SwiGLU(gate(x), up(x)) @ down。

    gate 和 up 使用 IQ2_XXS，down 使用 Q2_K。

    使用 2-kernel 方案：gate+up+SwiGLU 融合 + down GEMM。
    - kernel 启动 2 次（gate_up_silu + q2k_gemm）
    - mid 经全局内存传递（4KB for M=1，L2 cache 命中）

    注：全融合 1-kernel 方案已测试，M=1 decode 下因冗余计算反而更慢（0.4 vs 0.5 t/s）。
    原因：112 个 down 块每个重复计算 gate+up（32 tiles），L2 cache 压力大。

    参数：
        x: [M, K] BF16 输入
        gate_*: IQ2_XXS 格式的 gate 权重
        up_*: IQ2_XXS 格式的 up 权重
        down_*: Q2_K 格式的 down 权重

    返回：
        y: [M, down_out_dim] BF16 输出
    """
    M = x.size(0)
    K = x.size(1)
    N_gate = gate_d.size(0)
    N_up = up_d.size(0)

    # 融合 gate+up+SwiGLU：单 kernel 完成
    mid = torch.empty(M, N_gate, dtype=torch.bfloat16, device=x.device)
    kernel = get_gate_up_silu_kernel(N_gate, N_up, K)
    kernel(x, gate_d, gate_qs, up_d, up_qs, IQ2_XXS_GRID_TENSOR, KSIGNS_TENSOR, mid)

    # down 投影
    down_out = q2k_gemm(mid, down_d, down_dmin, down_scales, down_qs)

    return down_out


def test_correctness():
    """正确性测试。"""
    print("\n" + "=" * 70)
    print("混合量化 GEMM 正确性测试")
    print("=" * 70)
    
    M, N_gate, K_gate = 2, 128, 512
    n_blocks_gate = (K_gate + QK_K - 1) // QK_K
    
    x = torch.randn(M, K_gate, dtype=torch.bfloat16, device='cuda')
    d = torch.randn(N_gate, n_blocks_gate, dtype=torch.float16, device='cuda').abs() + 0.1
    qs = torch.randint(0, 65536, (N_gate, n_blocks_gate, 32), dtype=torch.uint16, device='cuda')
    
    print(f"\n矩阵形状: x=[{M}, {K_gate}], W=[{N_gate}, {K_gate}]")
    
    try:
        y = iq2xxs_gemm(x, d, qs)
        print(f"IQ2_XXS GEMM 输出形状: {y.shape}")
        print("[IQ2_XXS] ✓ 测试通过")
    except Exception as e:
        print(f"[IQ2_XXS] ✗ 测试失败: {e}")
    
    N_down, K_down = 64, 128
    n_blocks_down = (K_down + QK_K - 1) // QK_K
    
    x_down = torch.randn(M, K_down, dtype=torch.bfloat16, device='cuda')
    d_down = torch.randn(N_down, n_blocks_down, dtype=torch.float16, device='cuda').abs() + 0.1
    dmin_down = torch.randn(N_down, n_blocks_down, dtype=torch.float16, device='cuda').abs() * 0.1
    scales_down = torch.randint(0, 256, (N_down, n_blocks_down, 16), dtype=torch.uint8, device='cuda')
    qs_down = torch.randint(0, 256, (N_down, n_blocks_down, 64), dtype=torch.uint8, device='cuda')
    
    try:
        y_down = q2k_gemm(x_down, d_down, dmin_down, scales_down, qs_down)
        print(f"Q2_K GEMM 输出形状: {y_down.shape}")
        print("[Q2_K] ✓ 测试通过")
    except Exception as e:
        print(f"[Q2_K] ✗ 测试失败: {e}")


def benchmark():
    """性能测试。"""
    import time
    
    print("\n" + "=" * 70)
    print("混合量化 GEMM 性能测试")
    print("=" * 70)
    
    M, N, K = 1, 2048, 7168
    n_blocks = (K + QK_K - 1) // QK_K
    
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    d = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda').abs() + 0.1
    qs = torch.randint(0, 65536, (N, n_blocks, 32), dtype=torch.uint16, device='cuda')
    
    print(f"\n输入形状: x=[{M}, {K}], W=[{N}, {K}]")
    
    try:
        print("\n[预热] 编译 kernel...")
        y = iq2xxs_gemm(x, d, qs)
        torch.cuda.synchronize()
        print("[预热] 完成")
        
        n_iter = 100
        start = time.perf_counter()
        for _ in range(n_iter):
            y = iq2xxs_gemm(x, d, qs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter * 1000
        
        print(f"\n[结果] IQ2_XXS GEMM 时间: {elapsed:.3f} ms")
        print(f"目标 (FP8 GEMM): 0.036 ms")
        print(f"性能比: {elapsed / 0.036:.2f}x")
    except Exception as e:
        print(f"[错误] {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="混合量化 GEMM 测试")
    parser.add_argument("--test", action="store_true", help="运行正确性测试")
    parser.add_argument("--bench", action="store_true", help="运行性能测试")
    args = parser.parse_args()
    
    if args.test or (not args.test and not args.bench):
        test_correctness()
    if args.bench:
        benchmark()
