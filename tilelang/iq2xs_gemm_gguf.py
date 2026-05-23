"""GGUF 兼容的 IQ2_XS GEMM TileLang kernel

基于 GGUF 官方规范实现：
  - Grid: 512 行 × 8 列，值为 8, 25, 43
  - qs 编码: 9-bit grid_idx + 7-bit sign_idx
  - scales 计算: db = d * (0.5 + scales) * 0.25
  - 反量化: value = db * grid[grid_idx] * signs

优化策略：
  1. 常量 Grid 存储在常量内存
  2. Shared Memory 缓存 A 和 B 的块
  3. Tensor Core 加速矩阵乘法
  4. 流水线优化隐藏内存延迟
  5. 融合反量化在 kernel 内部完成

性能目标：接近 FP8 GEMM 性能（0.036 ms）
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

QK_K = 256


@tilelang.jit(pass_configs=pass_configs)
def iq2xs_gemm_kernel_gguf(N: int, K: int):
    """GGUF 兼容的 IQ2_XS GEMM TileLang kernel。
    
    计算：C[M, N] = A[M, K] @ B_dequant[N, K]^T
    
    GGUF IQ2_XS 格式：
      - d: float16 super-block scale
      - qs[32]: uint16，低 9 bits = grid_idx，高 7 bits = sign_idx
      - scales[8]: uint8，每个包含两个 4-bit scale
    
    GGUF 官方反量化逻辑：
      - 每个 sub_block (ib32) 有一个 scales[ib32]
      - scales[ib32] 包含两个 4-bit scale：
        - 低 4-bit 用于 l=0,1（前 16 个元素）
        - 高 4-bit 用于 l=2,3（后 16 个元素）
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
        scales: T.Tensor[(N, n_blocks_per_row, 8), "uint8"],
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
                    
                    scale_packed = scales[row_idx, block_idx, ib32]
                    
                    scale_low = T.Cast(FP32, scale_packed & T.uint8(0x0F))
                    scale_high = T.Cast(FP32, (scale_packed >> 4) & T.uint8(0x0F))
                    
                    db_low = d_val * (T.Cast(FP32, 0.5) + scale_low) * T.Cast(FP32, 0.25)
                    db_high = d_val * (T.Cast(FP32, 0.5) + scale_high) * T.Cast(FP32, 0.25)
                    
                    db = db_low + T.Cast(FP32, l >= 2) * (db_high - db_low)
                    
                    qs_val = qs[row_idx, block_idx, qs_idx]
                    grid_idx = qs_val & T.uint16(511)
                    sign_idx = (qs_val >> 9) & T.uint16(127)
                    
                    grid_row_idx = T.Cast("int32", local_in_8)
                    grid_val = grid[grid_idx, grid_row_idx]
                    
                    sign_byte = ksigns[sign_idx]
                    sign_bit = (sign_byte >> grid_row_idx) & T.uint8(1)
                    sign_mul = T.Cast(FP32, 1.0) - T.Cast(FP32, sign_bit) * T.Cast(FP32, 2.0)
                    
                    B_shared[i, j] = T.Cast(BF16, db * grid_val * sign_mul)
                
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]
                
                T.clear(C_local)
            
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
    
    return main


_kernel_cache = {}


def get_iq2xs_gemm_kernel(N: int, K: int):
    """获取或编译 IQ2_XS GEMM kernel（带缓存）。"""
    key = (N, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = iq2xs_gemm_kernel_gguf(N, K)
    return _kernel_cache[key]


def iq2xs_gemm_gguf(
    x: torch.Tensor,
    d: torch.Tensor,
    qs: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """GGUF 兼容的 IQ2_XS GEMM：y = x @ W^T。
    
    参数:
        x: [M, K] BF16 输入矩阵
        d: [N, n_blocks] FP16，super-block scale
        qs: [N, n_blocks, 32] uint16，grid_idx + sign_idx
        scales: [N, n_blocks, 8] uint8，sub-block scales
    
    返回:
        y: [M, N] BF16 输出矩阵
    """
    assert x.is_cuda and d.is_cuda and qs.is_cuda and scales.is_cuda
    assert x.is_contiguous()
    
    M = x.size(0)
    K = x.size(1)
    N = d.size(0)
    
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    
    kernel = get_iq2xs_gemm_kernel(N, K)
    kernel(x, d, qs, scales, IQ2_XS_GRID_TENSOR, KSIGNS_TENSOR, y)
    
    return y


def dequantize_iq2_xs_weight(
    d: torch.Tensor,
    qs: torch.Tensor,
    scales: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """反量化 IQ2_XS 权重（PyTorch 参考实现）。
    
    参数:
        d: [N, n_blocks] FP16
        qs: [N, n_blocks, 32] uint16
        scales: [N, n_blocks, 8] uint8（打包的 4-bit scales）
        K: 权重列数
    
    返回:
        W: [N, K] FP32 反量化权重
    
    GGUF 官方反量化逻辑：
      - 每个 sub_block (ib32) 有一个 scales[ib32]
      - scales[ib32] 包含两个 4-bit scale：
        - 低 4-bit 用于 l=0,1（前 16 个元素）
        - 高 4-bit 用于 l=2,3（后 16 个元素）
    """
    N = d.size(0)
    n_blocks = d.size(1)
    
    W = torch.zeros(N, K, dtype=torch.float32, device=d.device)
    
    d_cpu = d.cpu().float()
    qs_cpu = qs.cpu()
    scales_cpu = scales.cpu()
    
    grid_cpu = IQ2_XS_GRID
    ksigns_cpu = KSIGNS
    
    for i in range(N):
        for block_idx in range(n_blocks):
            d_val = d_cpu[i, block_idx].item()
            
            for ib32 in range(8):
                scale_packed = scales_cpu[i, block_idx, ib32].item()
                scale_low = scale_packed & 0x0F
                scale_high = (scale_packed >> 4) & 0x0F
                
                db_low = d_val * (0.5 + scale_low) * 0.25
                db_high = d_val * (0.5 + scale_high) * 0.25
                
                for l in range(4):
                    qs_idx = ib32 * 4 + l
                    packed = qs_cpu[i, block_idx, qs_idx].item()
                    
                    grid_idx = packed & 511
                    sign_idx = (packed >> 9) & 127
                    
                    grid_vals = grid_cpu[grid_idx]
                    sign_byte = ksigns_cpu[sign_idx]
                    
                    db = db_low if l < 2 else db_high
                    
                    for j in range(8):
                        sign_bit = (sign_byte >> j) & 1
                        sign_mul = 1.0 if sign_bit == 0 else -1.0
                        
                        k_pos = block_idx * 256 + ib32 * 32 + l * 8 + j
                        if k_pos < K:
                            W[i, k_pos] = db * grid_vals[j] * sign_mul
    
    return W.cuda()


def test_correctness():
    """正确性测试：对比 TileLang kernel 与 PyTorch 参考实现。"""
    print("\n" + "=" * 70)
    print("GGUF IQ2_XS GEMM 正确性测试")
    print("=" * 70)
    
    M, N, K = 2, 128, 512
    n_blocks = (K + QK_K - 1) // QK_K
    
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    d = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda').abs() + 0.1
    qs = torch.randint(0, 65536, (N, n_blocks, 32), dtype=torch.uint16, device='cuda')
    scales = torch.randint(0, 16, (N, n_blocks, 8), dtype=torch.uint8, device='cuda')
    
    print(f"\n矩阵形状: x=[{M}, {K}], W=[{N}, {K}]")
    print(f"Grid 形状: {IQ2_XS_GRID.shape}")
    print(f"Grid 唯一值: {np.unique(IQ2_XS_GRID)}")
    
    try:
        y_tilelang = iq2xs_gemm_gguf(x, d, qs, scales)
        
        W_ref = dequantize_iq2_xs_weight(d, qs, scales, K)
        y_ref = x.float() @ W_ref.T
        y_ref = y_ref.to(torch.bfloat16)
        
        error = (y_tilelang.float() - y_ref.float()).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        print(f"\n[误差分析]")
        print(f"  最大误差: {max_error:.6f}")
        print(f"  平均误差: {mean_error:.6f}")
        
        if max_error < 1.0:
            print("\n[结果] ✓ 正确性测试通过")
            return True
        else:
            print("\n[结果] ✗ 正确性测试失败")
            return False
    except Exception as e:
        print(f"\n[错误] {e}")
        return False


def benchmark_iq2xs_gemm():
    """性能测试。"""
    import time
    
    print("\n" + "=" * 70)
    print("GGUF IQ2_XS GEMM 性能测试")
    print("=" * 70)
    
    M, N, K = 1, 2048, 7168
    n_blocks = (K + QK_K - 1) // QK_K
    
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    d = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda').abs() + 0.1
    qs = torch.randint(0, 65536, (N, n_blocks, 32), dtype=torch.uint16, device='cuda')
    scales = torch.randint(0, 16, (N, n_blocks, 8), dtype=torch.uint8, device='cuda')
    
    print(f"\n输入形状:")
    print(f"  x: [{M}, {K}]")
    print(f"  W: [{N}, {K}] (IQ2_XS)")
    print(f"  输出: [{M}, {N}]")
    
    try:
        print("\n[预热] 编译 kernel...")
        y = iq2xs_gemm_gguf(x, d, qs, scales)
        torch.cuda.synchronize()
        print("[预热] 完成")
        
        n_iter = 100
        start = time.perf_counter()
        for _ in range(n_iter):
            y = iq2xs_gemm_gguf(x, d, qs, scales)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter * 1000
        
        print(f"\n[结果]")
        print(f"  GGUF IQ2_XS GEMM 时间: {elapsed:.3f} ms")
        print(f"  目标 (FP8 GEMM):       0.036 ms")
        print(f"  性能比: {elapsed / 0.036:.2f}x")
        
        return elapsed
    except Exception as e:
        print(f"\n[错误] {e}")
        return None


if __name__ == "__main__":
    test_correctness()
    benchmark_iq2xs_gemm()
