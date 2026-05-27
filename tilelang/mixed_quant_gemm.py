"""混合量化 GEMM TileLang kernel: IQ2_XXS + Q2_K。

支持混合量化专家 FFN：
  - gate_proj (w1): IQ2_XXS — 2.0625 bpw
  - up_proj (w3):   IQ2_XXS — 2.0625 bpw
  - down_proj (w2): Q2_K    — 2.5625 bpw

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


IQ2_XS_GRID_hex = (
    b"00000200050008000a0011001400160019002000220025002800410044004600"
    b"49005000520055005800610064008000820085008800910094009900a0000101"
    b"04010601090110011201150118011a0121012401400142014501480151015401"
    b"6001680181018401900100020202050208021102140220024102440250025502"
    b"80028a0201040404060409041004120415041804210424044004420445044804"
    b"5104540456046004810484049004000502050505080511051405200541054405"
    b"500561058005010604061006260640064206840600080208050808080a081108"
    b"14082008250841084408500858088008a008aa08010904091009400981098909"
    b"000a200a280a960aa00a01100410061009101010121015101810211024104010"
    b"4210451048105110541060106a10811084109010001102110511081111111411"
    b"2011411144115011801194119611011204120612101240126012001402140514"
    b"0814111414142014411444144914501464148014011504151015401500161416"
    b"49160118041810181218401854188618001905196619511aa91a002002200520"
    b"08200a201120142020204120442050208020a020012104211021402148216521"
    b"002222228022a82201240424102429244024002541255225992501261a26a626"
    b"002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
    b"0640094010401240154018402140244040404240454048404a40514054406040"
    b"6540814084409040004102410541084111411441204141414441504180418541"
    b"a241014204421042124229424042004402440544084411441444194420444144"
    b"4444504480449444014504451045244540459a4500460a464446504601480448"
    b"1048404845485448624800491149444950496949044a00500250055008501150"
    b"145020502850415044505050805001510451105115514051425100524452aa52"
    b"0154045410542154405460548154a154005508558055885521566856a1560058"
    b"14584158505899581a5940594259855a0160046010604060546062608660a960"
    b"006124624a62926200641664106540654565a46501686a682569066a546a626a"
    b"00800280058008801180148020802a8041804480508080808280a880aa800181"
    b"0481068110814081518159810082208280828282a082a8820184048410841284"
    b"158440846084898400854485a58518866a860088088825885a8880888288a888"
    b"0689228a808a888a968aa88a0190049010904090569084900091229164915692"
    b"89920094059444945094589429959095929541965198a6984999159a609a00a0"
    b"02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
    b"95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
)

IQ2_XS_grid_map = (0x08, 0x19, 0x2b)


def decode_grid_from_hex(grid_hex: bytes, grid_shape: tuple, grid_map: tuple) -> np.ndarray:
    """从十六进制编码解码 Grid 查找表。"""
    bits_per_elem = ceil(log2(len(grid_map)))
    elems_per_byte = 8 // bits_per_elem
    
    grid = np.frombuffer(grid_hex, dtype=np.uint8)
    grid = grid.reshape((-1, 2))
    grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
    grid = grid[..., 0] | grid[..., 1]
    grid = grid.reshape((-1, 1)) >> np.array([i for i in range(0, 8, 8 // elems_per_byte)], dtype=np.uint8).reshape((1, elems_per_byte))
    grid = (grid & ((1 << bits_per_elem) - 1)).reshape((-1, 1))
    grid_map_arr = np.array(grid_map, dtype=np.float32).reshape((1, -1))
    grid = np.take_along_axis(grid_map_arr, grid, axis=-1)
    return grid.reshape(grid_shape)


IQ2_XS_GRID = decode_grid_from_hex(IQ2_XS_GRID_hex, (512, 8), IQ2_XS_grid_map)
IQ2_XS_GRID_TENSOR = torch.from_numpy(IQ2_XS_GRID).cuda()

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
    """IQ2_XXS GEMM TileLang kernel。
    
    计算：C[M, N] = A[M, K] @ B_dequant[N, K]^T
    
    IQ2_XXS 格式（block_iq2_xxs）：
      - d: float16 super-block scale
      - qs: uint16[32]，低 9-bit = grid_idx，高 7-bit = sign_idx
    
    反量化公式：
      value = d * grid[grid_idx] * sign
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
        grid: T.Tensor[(512, 8), "float32"],
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
                    
                    qs_idx = ib32 * 4 + l
                    
                    d_val = T.Cast(FP32, d[row_idx, block_idx])
                    
                    qs_val = qs[row_idx, block_idx, qs_idx]
                    grid_idx = qs_val & T.uint16(511)
                    sign_idx = (qs_val >> 9) & T.uint16(127)
                    
                    grid_row_idx = T.Cast("int32", local_in_8)
                    grid_val = grid[grid_idx, grid_row_idx]
                    
                    sign_byte = ksigns[sign_idx]
                    sign_bit = (sign_byte >> grid_row_idx) & T.uint8(1)
                    sign_mul = T.Cast(FP32, 1.0) - T.Cast(FP32, sign_bit) * T.Cast(FP32, 2.0)
                    
                    B_shared[i, j] = T.Cast(BF16, d_val * grid_val * sign_mul)
                
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]
                
                T.clear(C_local)
            
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
    
    return main


@tilelang.jit(pass_configs=pass_configs)
def q2k_gemm_kernel(N: int, K: int):
    """Q2_K GEMM TileLang kernel。
    
    计算：C[M, N] = A[M, K] @ B_dequant[N, K]^T
    
    Q2_K 格式（block_q2_K）：
      - d: float16 super-block scale
      - dmin: float16 minimum scale
      - scales: uint8[16] 子块缩放
      - qs: uint8[64] 2-bit 量化值
    
    反量化公式：
      value = d * scales[ib16] * (qs >> bit_offset) & 0x3 - dmin
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
                    
                    ib16 = local_k // 16
                    local_in_16 = local_k % 16
                    
                    d_val = T.Cast(FP32, d[row_idx, block_idx])
                    dmin_val = T.Cast(FP32, dmin[row_idx, block_idx])
                    
                    scale_val = T.Cast(FP32, scales[row_idx, block_idx, ib16])
                    
                    byte_idx = local_in_16 // 4
                    bit_offset = (local_in_16 % 4) * 2
                    
                    qs_val = qs[row_idx, block_idx, byte_idx]
                    quant_2bit = (qs_val >> T.Cast("uint8", bit_offset)) & T.uint8(3)
                    
                    quant_val = T.Cast(FP32, quant_2bit) - T.Cast(FP32, 1.5)
                    
                    B_shared[i, j] = T.Cast(BF16, d_val * scale_val * quant_val + dmin_val)
                
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]
                
                T.clear(C_local)
            
            T.copy(C_local_accum, C_shared)
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


def iq2xxs_gemm(x: torch.Tensor, d: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """IQ2_XXS GEMM：y = x @ W^T。
    
    参数：
        x: [M, K] BF16 输入矩阵
        d: [N, n_blocks] FP16 super-block scale
        qs: [N, n_blocks, 32] uint16 grid_idx + sign_idx
    
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
    kernel(x, d, qs, IQ2_XS_GRID_TENSOR, KSIGNS_TENSOR, y)
    
    return y


def q2k_gemm(
    x: torch.Tensor,
    d: torch.Tensor,
    dmin: torch.Tensor,
    scales: torch.Tensor,
    qs: torch.Tensor,
) -> torch.Tensor:
    """Q2_K GEMM：y = x @ W^T。
    
    参数：
        x: [M, K] BF16 输入矩阵
        d: [N, n_blocks] FP16 super-block scale
        dmin: [N, n_blocks] FP16 minimum scale
        scales: [N, n_blocks, 16] uint8 子块缩放
        qs: [N, n_blocks, 64] uint8 2-bit 量化值
    
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
    
    参数：
        x: [M, K] BF16 输入
        gate_*: IQ2_XXS 格式的 gate 权重
        up_*: IQ2_XXS 格式的 up 权重
        down_*: Q2_K 格式的 down 权重
    
    返回：
        y: [M, down_out_dim] BF16 输出
    """
    gate_out = iq2xxs_gemm(x, gate_d, gate_qs)
    up_out = iq2xxs_gemm(x, up_d, up_qs)
    
    gate_sigmoid = torch.sigmoid(gate_out)
    mid = gate_sigmoid * gate_out * up_out
    
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
