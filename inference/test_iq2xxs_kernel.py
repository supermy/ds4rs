"""IQ2_XXS 和 Q2_K 内核正确性测试。

从 GGUF 文件加载真实专家权重，用纯 Python 实现 llama.cpp 的
ggml_vec_dot_iq2_xxs_q8_K_generic 和 ggml_vec_dot_q2_K_q8_K 参考算法，
与 Rust AVX-512/AVX2 内核的结果逐行对比。

用法：
  docker exec ds4rs-dev python /workspace/inference/test_iq2xxs_kernel.py
"""
import sys
import os
import struct
import numpy as np

# GGUF 读取
from gguf import GGUFReader

# Rust 扩展
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ds4rs import (
    Iq2XxsWeight, Q2KWeight,
    is_avx512_supported, init_tables, is_tables_initialized,
)

GGUF_PATH = os.environ.get("GGUF_PATH", "/workspace/gguf/experts_iq2xxs_q2k.gguf")

# ============================================================================
# IQ2_XXS 查找表 — 来自 llama.cpp ggml-common.h iq2xxs_grid
# 256 entries × 8 int8, 值域 {8, 25, 43}
# 与 Rust tables.rs get_iq2xxs_grid() 一致
# ============================================================================

_IQ2XXS_GRID_U64 = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0819080808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
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

# 解码 u64 打包的 grid 为 (256, 8) int8 数组
IQ2XXS_GRID = np.zeros((256, 8), dtype=np.int8)
for _i, _g in enumerate(_IQ2XXS_GRID_U64):
    _bytes = _g.to_bytes(8, 'little')
    for _j, _b in enumerate(_bytes):
        IQ2XXS_GRID[_i, _j] = np.int8(_b if _b < 128 else _b - 256)

# ksigns 表 — 与 iq2xs_gemm_tilelang.KSIGNS_IQ2XS 一致
# 128 项 uint8, 用于 7-bit 符号索引展开
KSIGNS_IQ2XS = np.array([
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
], dtype=np.uint8)

# 符号位掩码
KMASK = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]

QK_K = 256  # 每个 block 编码 256 个元素


# ============================================================================
# Q8 量化 — 匹配 llama.cpp quantize_f32_to_q8_block
# ============================================================================

def quantize_f32_to_q8_block(x_block):
    """将 256 个 f32 值量化为 Q8 格式。

    匹配 Rust kernel.rs 中 quantize_f32_to_q8_block 的逻辑：
      amax = max(|x[i]|)
      scale = 127.0 / amax  (if amax > 1e-6, else 0)
      q8[i] = round(x[i] * scale), clamped to [-128, 127]
      返回 inv_scale = amax / 127.0

    返回 (q8: int8 array[256], inv_scale: float)
    """
    amax = np.max(np.abs(x_block))
    if amax > 1e-6:
        scale = 127.0 / amax
        inv_scale = amax / 127.0
    else:
        scale = 0.0
        inv_scale = 0.0

    q8 = np.round(x_block * scale).clip(-128, 127).astype(np.int8)
    return q8, inv_scale


def quantize_f32_to_q8(x):
    """将整个 f32 向量量化为 Q8 格式（按 256 元素分块）。

    返回 (q8: int8 array, inv_scales: float32 array)
    """
    n = len(x)
    assert n % QK_K == 0, f"Length {n} must be multiple of {QK_K}"
    n_blocks = n // QK_K

    q8 = np.zeros(n, dtype=np.int8)
    inv_scales = np.zeros(n_blocks, dtype=np.float32)

    for blk in range(n_blocks):
        block = x[blk * QK_K:(blk + 1) * QK_K]
        q8_blk, inv_scales[blk] = quantize_f32_to_q8_block(block)
        q8[blk * QK_K:(blk + 1) * QK_K] = q8_blk

    return q8, inv_scales


# ============================================================================
# IQ2_XXS 参考实现 — 匹配 llama.cpp ggml_vec_dot_iq2_xxs_q8_K_generic
# ============================================================================

def iq2xxs_vec_dot_q8_ref(d, qs, q8, q8_inv_scales, n_blocks):
    """纯 Python 参考实现 IQ2_XXS vec_dot。

    匹配 Rust avx512.rs iq2xxs_vec_dot_q8 的逻辑：

    block_iq2_xxs 布局（66 bytes / 256 elements）：
      - d: fp16 super-block scale
      - qs[32]: uint16，编码 grid index + sign index + scale

    每 32 元素一个子块（ib32），读 4 个 uint16（8 bytes = 2 uint32）：
      - aux8[0..3] = 4 个 grid index（8-bit, 0-255），索引到 iq2xxs_grid（256 entries）
      - aux32[1] 包含 sign indices（7-bit × 4 = 28 bits）+ scale（4 bits）
      - ls = 2*(aux32[1] >> 28) + 1
      - sign = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]

    vec_dot 公式：result = 0.125 * d * Σ(ls * Σ(grid[j] * q8[j] * sign[j]))

    参数:
        d: float32 array [n_blocks]
        qs: uint16 array [n_blocks * 32]
        q8: int8 array [n_blocks * 256]
        q8_inv_scales: float32 array [n_blocks]
        n_blocks: int

    返回:
        float — 点积结果
    """
    sumf = 0.0

    for blk in range(n_blocks):
        d_val = d[blk] * q8_inv_scales[blk]
        qs_blk = qs[blk * 32:(blk + 1) * 32]
        q8_blk = q8[blk * 256:(blk + 1) * 256]

        bsum = 0
        q8_offset = 0

        for ib32 in range(8):
            # 每 32 元素一个子块，读 4 个 uint16
            q2_base = ib32 * 4
            qs_slice = qs_blk[q2_base:q2_base + 4]

            # 将 4 个 uint16 组合为 2 个 uint32（小端序）
            aux32_0 = int(qs_slice[0]) | (int(qs_slice[1]) << 16)
            aux32_1 = int(qs_slice[2]) | (int(qs_slice[3]) << 16)

            # aux8[0..3] = aux32_0 的 4 个字节 = 4 个 grid index
            aux8 = [
                aux32_0 & 0xFF,
                (aux32_0 >> 8) & 0xFF,
                (aux32_0 >> 16) & 0xFF,
                (aux32_0 >> 24) & 0xFF,
            ]

            # scale: 4 bits from top of aux32_1
            ls = 2 * ((aux32_1 >> 28) & 0xF) + 1

            sumi = 0
            for l in range(4):
                # grid lookup
                grid_idx = aux8[l]
                grid_row = IQ2XXS_GRID[grid_idx]  # 8 个 int8

                # sign lookup
                sign_idx = (aux32_1 >> (7 * l)) & 127
                sign_byte = KSIGNS_IQ2XS[sign_idx]

                for j in range(8):
                    grid_val = int(grid_row[j])
                    q8_val = int(q8_blk[q8_offset])
                    # sign: 如果 sign_byte 的第 j 位为 1，则负号
                    sign_val = -1 if (sign_byte & KMASK[j]) else 1
                    sumi += grid_val * q8_val * sign_val
                    q8_offset += 1

            bsum += sumi * ls

        sumf += d_val * bsum

    return 0.125 * sumf


# ============================================================================
# Q2_K 参考实现 — 匹配 llama.cpp ggml_vec_dot_q2_K_q8_K
# ============================================================================

def q2k_vec_dot_q8_ref(d, dmin, scales, qs, q8, q8_inv_scales, n_blocks):
    """纯 Python 参考实现 Q2_K vec_dot。

    匹配 Rust avx512.rs q2k_vec_dot_q8 的逻辑：

    block_q2_K 布局（84 bytes / 256 elements）：
      - scales[16]: 4-bit packed，低 4 位 = scale，高 4 位 = min
      - qs[64]: 2-bit 量化值打包（每字节 4 个 2-bit 值，0-3 无偏移）
      - d: fp16 super-block scale
      - dmin: fp16 super-block minimum scale

    vec_dot 公式：result = q8_d * (q2_d * isum - q2_dmin * summs)
      isum  = Σ (scales[j] & 0xF) * dot(q2_sub_block, q8_sub_block)
      summs = Σ bsums[j] * (scales[j] >> 4)

    参数:
        d: float32 array [n_blocks]
        dmin: float32 array [n_blocks]
        scales: uint8 array [n_blocks * 16]
        qs: uint8 array [n_blocks * 64]
        q8: int8 array [n_blocks * 256]
        q8_inv_scales: float32 array [n_blocks]
        n_blocks: int

    返回:
        float — 点积结果
    """
    sumf = 0.0

    for blk in range(n_blocks):
        d_val = d[blk]
        dmin_val = dmin[blk]
        sc = scales[blk * 16:(blk + 1) * 16]
        qs_blk = qs[blk * 64:(blk + 1) * 64]
        q8_blk = q8[blk * 256:(blk + 1) * 256]

        # min 贡献：summs = Σ bsums[j] * (sc[j] >> 4)
        summs = 0
        for j in range(16):
            bsum = 0
            for l in range(16):
                bsum += int(q8_blk[j * 16 + l])
            summs += bsum * (int(sc[j]) >> 4)

        # scale 贡献：isum = Σ (sc[is] & 0xF) * dot(q2_sub, q8_sub)
        isum = 0
        is_idx = 0
        q8_offset = 0

        for k in range(2):
            q2_base = k * 32
            shift = 0

            for _j in range(4):
                # 第一个子块
                sc_val = int(sc[is_idx]) & 0xF
                is_idx += 1
                isuml = 0
                for l in range(16):
                    q2_val = (int(qs_blk[q2_base + l]) >> shift) & 3
                    isuml += q2_val * int(q8_blk[q8_offset + l])
                isum += sc_val * isuml
                q8_offset += 16

                # 第二个子块
                sc_val = int(sc[is_idx]) & 0xF
                is_idx += 1
                isuml = 0
                for l in range(16):
                    q2_val = (int(qs_blk[q2_base + 16 + l]) >> shift) & 3
                    isuml += q2_val * int(q8_blk[q8_offset + l])
                isum += sc_val * isuml
                q8_offset += 16

                shift += 2

        # result = q8_d * (q2_d * isum - q2_dmin * summs)
        sumf += q8_inv_scales[blk] * (d_val * isum - dmin_val * summs)

    return sumf


# ============================================================================
# GGUF 解析
# ============================================================================

def parse_iq2xxs_block(data_bytes, n_blocks):
    """解析 IQ2_XXS 原始字节为 (d, qs) 数组。

    每个 block 66 bytes: d(fp16, 2B) + qs(u16×32, 64B)
    """
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    block_size = 66
    assert len(raw) == n_blocks * block_size, \
        f"Expected {n_blocks * block_size} bytes, got {len(raw)}"

    blocks = raw.reshape(n_blocks, block_size)

    # d: 前 2 字节 → fp16 → f32
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).ravel()

    # qs: 后 64 字节 → u16×32
    qs = blocks[:, 2:66].copy().view(np.uint16).reshape(n_blocks, 32).ravel().copy()

    return d, qs


def parse_q2k_block(data_bytes, n_blocks):
    """解析 Q2_K 原始字节为 (d, dmin, scales, qs) 数组。

    llama.cpp block_q2_K 布局（84 bytes）：
      scales: uint8[16] (4-bit packed scales+mins) — 前 16 字节
      qs:     uint8[64] (2-bit packed quants)      — 接下来 64 字节
      d:      fp16 (2 bytes)                        — 偏移 80
      dmin:   fp16 (2 bytes)                        — 偏移 82
    """
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    block_size = 84
    assert len(raw) == n_blocks * block_size, \
        f"Expected {n_blocks * block_size} bytes, got {len(raw)}"

    blocks = raw.reshape(n_blocks, block_size)

    # scales: bytes 0-15 → u8×16
    scales = blocks[:, 0:16].reshape(n_blocks, 16).ravel().copy()

    # qs: bytes 16-79 → u8×64
    qs = blocks[:, 16:80].reshape(n_blocks, 64).ravel().copy()

    # d: bytes 80-81 → fp16 → f32
    d = blocks[:, 80:82].copy().view(np.float16).astype(np.float32).ravel()

    # dmin: bytes 82-83 → fp16 → f32
    dmin = blocks[:, 82:84].copy().view(np.float16).astype(np.float32).ravel()

    return d, dmin, scales, qs


def load_tensor_from_gguf(reader, tensor_name):
    """从 GGUF 加载指定张量。"""
    for t in reader.tensors:
        if t.name == tensor_name:
            data = t.data.ravel().tobytes()
            ne0, ne1 = [int(x) for x in t.shape]
            n_blocks = ne0 * ne1 // 256
            type_name = t.tensor_type.name
            return data, ne0, ne1, n_blocks, type_name
    raise ValueError(f"Tensor {tensor_name} not found")


# ============================================================================
# 测试主逻辑
# ============================================================================

def init_iq2_tables():
    """初始化 IQ2 查找表（Rust 内核需要）。"""
    if not is_tables_initialized():
        from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS
        grid_u64 = list(IQ2XS_GRID_U64)
        ksigns = list(KSIGNS_IQ2XS)
        init_tables(grid_u64, ksigns)
        print(f"[Init] IQ2 tables initialized, AVX-512: {is_avx512_supported()}")


def test_iq2xxs_kernel():
    """测试 IQ2_XXS 内核：Python 参考 vs Rust 内核。"""
    print("\n" + "=" * 70)
    print("IQ2_XXS 内核正确性测试")
    print("=" * 70)

    reader = GGUFReader(GGUF_PATH)

    # 加载 layers.0.experts.0.w1 (IQ2_XXS)
    tensor_name = "layers.0.experts.0.w1"
    data, ne0, ne1, n_blocks, type_name = load_tensor_from_gguf(reader, tensor_name)
    print(f"\nTensor: {tensor_name}")
    print(f"  Type: {type_name}, Shape: ({ne0}, {ne1}), Blocks: {n_blocks}")

    assert type_name == 'IQ2_XXS', f"Expected IQ2_XXS, got {type_name}"

    d, qs = parse_iq2xxs_block(data, n_blocks)
    print(f"  d range: [{d.min():.6f}, {d.max():.6f}]")
    print(f"  qs range: [{qs.min()}, {qs.max()}]")

    out_dim = ne0
    in_dim = ne1
    n_blocks_per_row = in_dim // QK_K

    # 构造随机输入
    np.random.seed(42)
    x = np.random.randn(in_dim).astype(np.float32)

    # ---- Python 参考 ----
    # 量化 x 为 Q8
    q8, q8_inv_scales = quantize_f32_to_q8(x)

    ref_results = np.zeros(out_dim, dtype=np.float32)
    for row in range(out_dim):
        row_offset = row * n_blocks_per_row
        d_row = d[row_offset:row_offset + n_blocks_per_row]
        qs_row = qs[row_offset * 32:(row_offset + n_blocks_per_row) * 32]
        ref_results[row] = iq2xxs_vec_dot_q8_ref(
            d_row, qs_row, q8, q8_inv_scales, n_blocks_per_row
        )

    # ---- Rust 内核 ----
    rust_w = Iq2XxsWeight.from_numpy(d, qs, (out_dim, in_dim))
    rust_results = rust_w.matvec(x)

    # ---- 诊断输出 ----
    print(f"\n  前 5 行逐行对比:")
    print(f"  {'Row':>4s}  {'Python_ref':>14s}  {'Rust_kernel':>14s}  {'Diff':>14s}  {'RelDiff%':>10s}")
    for row in range(min(5, out_dim)):
        diff = abs(ref_results[row] - rust_results[row])
        rel_diff = diff / max(abs(ref_results[row]), 1e-8) * 100
        print(f"  {row:4d}  {ref_results[row]:14.6f}  {rust_results[row]:14.6f}  {diff:14.6f}  {rel_diff:10.4f}%")

    # 前 3 个 block 的详细诊断
    print(f"\n  前 3 个 block 详细诊断 (row 0):")
    row_offset = 0
    d_row = d[row_offset:row_offset + n_blocks_per_row]
    qs_row = qs[row_offset * 32:(row_offset + n_blocks_per_row) * 32]

    for blk in range(min(3, n_blocks_per_row)):
        d_val = d_row[blk] * q8_inv_scales[blk]
        qs_blk = qs_row[blk * 32:(blk + 1) * 32]
        q8_blk = q8[blk * 256:(blk + 1) * 256]

        blk_sum = 0
        for ib32 in range(8):
            q2_base = ib32 * 4
            qs_slice = qs_blk[q2_base:q2_base + 4]
            aux32_0 = int(qs_slice[0]) | (int(qs_slice[1]) << 16)
            aux32_1 = int(qs_slice[2]) | (int(qs_slice[3]) << 16)
            aux8 = [aux32_0 & 0xFF, (aux32_0 >> 8) & 0xFF,
                    (aux32_0 >> 16) & 0xFF, (aux32_0 >> 24) & 0xFF]
            ls = 2 * ((aux32_1 >> 28) & 0xF) + 1

            sub_sum = 0
            for l in range(4):
                grid_idx = aux8[l]
                sign_idx = (aux32_1 >> (7 * l)) & 127
                sign_byte = KSIGNS_IQ2XS[sign_idx]
                for j in range(8):
                    sign_val = -1 if (sign_byte & KMASK[j]) else 1
                    sub_sum += int(IQ2XXS_GRID[grid_idx, j]) * int(q8_blk[ib32 * 32 + l * 8 + j]) * sign_val
            blk_sum += sub_sum * ls

        blk_result = 0.125 * d_val * blk_sum
        print(f"    blk={blk}: d={d_row[blk]:.6f}, inv_scale={q8_inv_scales[blk]:.6f}, "
              f"ls0={2*((int(qs_blk[2])|(int(qs_blk[3])<<16))>>28)&0xF+1}, "
              f"bsum={blk_sum}, blk_result={blk_result:.6f}")

    # ---- 统计 ----
    diffs = np.abs(ref_results - rust_results)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)
    ref_abs = np.abs(ref_results)
    rel_diffs = diffs / np.maximum(ref_abs, 1e-8)
    max_rel_diff = np.max(rel_diffs) * 100
    mean_rel_diff = np.mean(rel_diffs) * 100

    print(f"\n  整体统计 ({out_dim} 行):")
    print(f"    Max  abs diff: {max_diff:.8f}")
    print(f"    Mean abs diff: {mean_diff:.8f}")
    print(f"    Median abs diff: {median_diff:.8f}")
    print(f"    Max  rel diff: {max_rel_diff:.4f}%")
    print(f"    Mean rel diff: {mean_rel_diff:.4f}%")
    print(f"    Ref range: [{ref_results.min():.6f}, {ref_results.max():.6f}]")
    print(f"    Rust range: [{rust_results.min():.6f}, {rust_results.max():.6f}]")

    # 判定
    threshold = 1e-3  # 量化误差容忍
    if max_diff < threshold:
        print(f"  [PASS] IQ2_XXS 内核测试通过 (max_diff={max_diff:.8f} < {threshold})")
    else:
        print(f"  [WARN] IQ2_XXS 内核差异较大 (max_diff={max_diff:.8f} >= {threshold})")
        # 找出差异最大的行
        worst_row = np.argmax(diffs)
        print(f"    Worst row: {worst_row}, ref={ref_results[worst_row]:.6f}, "
              f"rust={rust_results[worst_row]:.6f}, diff={diffs[worst_row]:.8f}")

    return ref_results, rust_results


def test_q2k_kernel():
    """测试 Q2_K 内核：Python 参考 vs Rust 内核。"""
    print("\n" + "=" * 70)
    print("Q2_K 内核正确性测试")
    print("=" * 70)

    reader = GGUFReader(GGUF_PATH)

    # 加载 layers.0.experts.0.w2 (Q2_K)
    tensor_name = "layers.0.experts.0.w2"
    data, ne0, ne1, n_blocks, type_name = load_tensor_from_gguf(reader, tensor_name)
    print(f"\nTensor: {tensor_name}")
    print(f"  Type: {type_name}, Shape: ({ne0}, {ne1}), Blocks: {n_blocks}")

    assert type_name == 'Q2_K', f"Expected Q2_K, got {type_name}"

    d, dmin, scales, qs = parse_q2k_block(data, n_blocks)
    print(f"  d range: [{d.min():.6f}, {d.max():.6f}]")
    print(f"  dmin range: [{dmin.min():.6f}, {dmin.max():.6f}]")
    print(f"  scales range: [{scales.min()}, {scales.max()}]")
    print(f"  qs range: [{qs.min()}, {qs.max()}]")

    out_dim = ne0
    in_dim = ne1
    n_blocks_per_row = in_dim // QK_K

    # 构造随机输入
    np.random.seed(42)
    x = np.random.randn(in_dim).astype(np.float32)

    # ---- Python 参考 ----
    q8, q8_inv_scales = quantize_f32_to_q8(x)

    ref_results = np.zeros(out_dim, dtype=np.float32)
    for row in range(out_dim):
        row_offset = row * n_blocks_per_row
        d_row = d[row_offset:row_offset + n_blocks_per_row]
        dmin_row = dmin[row_offset:row_offset + n_blocks_per_row]
        scales_row = scales[row_offset * 16:(row_offset + n_blocks_per_row) * 16]
        qs_row = qs[row_offset * 64:(row_offset + n_blocks_per_row) * 64]
        ref_results[row] = q2k_vec_dot_q8_ref(
            d_row, dmin_row, scales_row, qs_row, q8, q8_inv_scales, n_blocks_per_row
        )

    # ---- Rust 内核 ----
    rust_w = Q2KWeight.from_numpy(d, dmin, scales, qs, (out_dim, in_dim))
    rust_results = rust_w.matvec(x)

    # ---- 诊断输出 ----
    print(f"\n  前 5 行逐行对比:")
    print(f"  {'Row':>4s}  {'Python_ref':>14s}  {'Rust_kernel':>14s}  {'Diff':>14s}  {'RelDiff%':>10s}")
    for row in range(min(5, out_dim)):
        diff = abs(ref_results[row] - rust_results[row])
        rel_diff = diff / max(abs(ref_results[row]), 1e-8) * 100
        print(f"  {row:4d}  {ref_results[row]:14.6f}  {rust_results[row]:14.6f}  {diff:14.6f}  {rel_diff:10.4f}%")

    # 前 3 个 block 的详细诊断
    print(f"\n  前 3 个 block 详细诊断 (row 0):")
    row_offset = 0
    d_row = d[row_offset:row_offset + n_blocks_per_row]
    dmin_row = dmin[row_offset:row_offset + n_blocks_per_row]
    scales_row = scales[row_offset * 16:(row_offset + n_blocks_per_row) * 16]
    qs_row = qs[row_offset * 64:(row_offset + n_blocks_per_row) * 64]

    for blk in range(min(3, n_blocks_per_row)):
        d_val = d_row[blk]
        dmin_val = dmin_row[blk]
        sc = scales_row[blk * 16:(blk + 1) * 16]
        qs_blk = qs_row[blk * 64:(blk + 1) * 64]
        q8_blk = q8[blk * 256:(blk + 1) * 256]

        # min 贡献
        summs = 0
        for j in range(16):
            bsum = sum(int(q8_blk[j * 16 + l]) for l in range(16))
            summs += bsum * (int(sc[j]) >> 4)

        # scale 贡献
        isum = 0
        is_idx = 0
        q8_offset = 0
        for k in range(2):
            q2_base = k * 32
            shift = 0
            for _j in range(4):
                sc_val = int(sc[is_idx]) & 0xF
                is_idx += 1
                isuml = sum(
                    ((int(qs_blk[q2_base + l]) >> shift) & 3) * int(q8_blk[q8_offset + l])
                    for l in range(16)
                )
                isum += sc_val * isuml
                q8_offset += 16

                sc_val = int(sc[is_idx]) & 0xF
                is_idx += 1
                isuml = sum(
                    ((int(qs_blk[q2_base + 16 + l]) >> shift) & 3) * int(q8_blk[q8_offset + l])
                    for l in range(16)
                )
                isum += sc_val * isuml
                q8_offset += 16
                shift += 2

        blk_result = q8_inv_scales[blk] * (d_val * isum - dmin_val * summs)
        print(f"    blk={blk}: d={d_val:.6f}, dmin={dmin_val:.6f}, "
              f"isum={isum}, summs={summs}, blk_result={blk_result:.6f}")

    # ---- 统计 ----
    diffs = np.abs(ref_results - rust_results)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)
    ref_abs = np.abs(ref_results)
    rel_diffs = diffs / np.maximum(ref_abs, 1e-8)
    max_rel_diff = np.max(rel_diffs) * 100
    mean_rel_diff = np.mean(rel_diffs) * 100

    print(f"\n  整体统计 ({out_dim} 行):")
    print(f"    Max  abs diff: {max_diff:.8f}")
    print(f"    Mean abs diff: {mean_diff:.8f}")
    print(f"    Median abs diff: {median_diff:.8f}")
    print(f"    Max  rel diff: {max_rel_diff:.4f}%")
    print(f"    Mean rel diff: {mean_rel_diff:.4f}%")
    print(f"    Ref range: [{ref_results.min():.6f}, {ref_results.max():.6f}]")
    print(f"    Rust range: [{rust_results.min():.6f}, {rust_results.max():.6f}]")

    # 判定
    threshold = 1e-3
    if max_diff < threshold:
        print(f"  [PASS] Q2_K 内核测试通过 (max_diff={max_diff:.8f} < {threshold})")
    else:
        print(f"  [WARN] Q2_K 内核差异较大 (max_diff={max_diff:.8f} >= {threshold})")
        worst_row = np.argmax(diffs)
        print(f"    Worst row: {worst_row}, ref={ref_results[worst_row]:.6f}, "
              f"rust={rust_results[worst_row]:.6f}, diff={diffs[worst_row]:.8f}")

    return ref_results, rust_results


def main():
    print(f"GGUF: {GGUF_PATH}")
    print(f"AVX-512: {is_avx512_supported()}")

    init_iq2_tables()

    test_iq2xxs_kernel()
    test_q2k_kernel()

    print("\n" + "=" * 70)
    print("全部测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
