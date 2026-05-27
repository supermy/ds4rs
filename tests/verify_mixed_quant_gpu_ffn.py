"""混合量化 GPU FFN 正确性验证。

从 experts_iq2xxs_q2k.gguf 读取专家权重，构造随机输入，
分别用 CPU FFN (Rust AVX-512) 和 GPU FFN (TileLang + PyTorch) 计算，
对比两者输出。

GGUF IQ2_XXS 格式 (llama.cpp packed):
  每个 sub-block (32 元素) 用 4 个 uint16 编码:
    qs[0],qs[1] → aux32_0: 4×8-bit grid indices
    qs[2],qs[3] → aux32_1: 4×7-bit sign indices + 4-bit scale
  反量化: value = d * ls * grid[grid_idx][j] * sign * 0.125
  其中 ls = 2 * ((aux32_1 >> 28) & 0xF) + 1

TileLang IQ2_XXS kernel 期望 (per-group):
  每个 uint16: 低 9-bit = grid_idx, 高 7-bit = sign_idx
  反量化: value = d * grid[grid_idx][j] * sign

转换策略:
  1. 解包 GGUF IQ2_XXS → 提取 grid_idx, sign_idx, scale
  2. 转为 IQ2_XS 格式 (per-group qs + separate scales)
  3. 使用 iq2xs_gemm_optimized kernel + 自定义 IQ2_XXS grid

Q2_K 格式:
  GGUF: scales[16] 是 4-bit packed (low=scale, high=min)
  反量化: value = d * scale_4bit * quant_2bit - dmin * min_4bit
  TileLang kernel 使用不同公式，因此 Q2_K 使用反量化 + torch.matmul

用法:
  docker exec ds4rs-dev python /workspace/tests/verify_mixed_quant_gpu_ffn.py
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tilelang'))

# ============================================================================
# GGUF 读取
# ============================================================================
from gguf import GGUFReader

GGUF_PATH = os.environ.get("GGUF_PATH", "/workspace/gguf/experts_iq2xxs_q2k.gguf")
QK_K = 256

# ============================================================================
# IQ2_XXS 查找表 — 与 Rust tables.rs 中 get_iq2xxs_grid() 一致
# ============================================================================
# IQ2_XXS Grid: 与 Rust tables.rs 中 get_iq2xxs_grid() 的硬编码一致
_IQ2XXS_GRID_U64 = [
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

# 解码 IQ2_XXS grid: 256 entries × 8 int8
IQ2XXS_GRID = np.zeros((256, 8), dtype=np.int8)
for _i, _g in enumerate(_IQ2XXS_GRID_U64):
    _bytes = _g.to_bytes(8, 'little')
    for _j, _b in enumerate(_bytes):
        IQ2XXS_GRID[_i, _j] = np.int8(_b if _b < 128 else _b - 256)

# ksigns 表
KSIGNS_IQ2XXS = np.array([
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
], dtype=np.uint8)


# ============================================================================
# GGUF 解析
# ============================================================================
def parse_iq2xxs_block(data_bytes, n_blocks):
    """解析 IQ2_XXS: d(fp16,2B) + qs(u16×32,64B) = 66B/block"""
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    blocks = raw.reshape(n_blocks, 66)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).ravel()
    qs = blocks[:, 2:66].copy().view(np.uint16).reshape(n_blocks, 32).copy()
    return d, qs


def parse_q2k_block(data_bytes, n_blocks):
    """解析 Q2_K: scales(16B) + qs(64B) + d(fp16,2B) + dmin(fp16,2B) = 84B/block"""
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    blocks = raw.reshape(n_blocks, 84)
    scales = blocks[:, 0:16].reshape(n_blocks, 16).copy()
    qs = blocks[:, 16:80].reshape(n_blocks, 64).copy()
    d = blocks[:, 80:82].copy().view(np.float16).astype(np.float32).ravel()
    dmin = blocks[:, 82:84].copy().view(np.float16).astype(np.float32).ravel()
    return d, dmin, scales, qs


def load_expert_from_gguf(reader, layer_id, expert_id):
    """从 GGUF 加载单个专家的权重。"""
    weights = {}
    for wt_name, gguf_key in [('w1', 'gate'), ('w3', 'up'), ('w2', 'down')]:
        tensor_name = f"layers.{layer_id}.experts.{expert_id}.{wt_name}"
        found = None
        for t in reader.tensors:
            if t.name == tensor_name:
                found = t
                break
        if found is None:
            raise ValueError(f"Tensor {tensor_name} not found")

        data = found.data.ravel().tobytes()
        ne0, ne1 = [int(x) for x in found.shape]
        n_blocks = ne0 * ne1 // 256
        type_name = found.tensor_type.name

        # GGUF shape [ne0, ne1] 中 ne0=in_features, ne1=out_features
        # Rust 代码期望 shape = (out_features, in_features) = (ne1, ne0)
        logical_shape = (ne1, ne0)

        if type_name == 'IQ2_XXS':
            d, qs = parse_iq2xxs_block(data, n_blocks)
            weights[wt_name] = ('iq2xxs', d, qs, logical_shape)
        elif type_name == 'Q2_K':
            d, dmin, scales, qs = parse_q2k_block(data, n_blocks)
            weights[wt_name] = ('q2k', d, dmin, scales, qs, logical_shape)
        else:
            raise ValueError(f"Unsupported type: {type_name}")

    return weights


# ============================================================================
# IQ2_XXS → IQ2_XS 格式转换
# ============================================================================
def convert_iq2xxs_to_iq2xs(d_flat, qs_flat, ne0, ne1):
    """将 GGUF IQ2_XXS packed 格式转为 IQ2_XS per-group 格式。

    GGUF IQ2_XXS: 每个 sub-block (32 元素) 用 4 个 uint16:
      qs[0],qs[1] → aux32_0: 4×8-bit grid indices
      qs[2],qs[3] → aux32_1: 4×7-bit sign indices + 4-bit scale

    IQ2_XS per-group: 每个 uint16 = 9-bit grid_idx | 7-bit sign_idx
      + scales[8] per block (4-bit packed)

    反量化公式一致:
      IQ2_XXS: value = d * ls * grid[grid_idx][j] * sign * 0.125
      IQ2_XS:  value = d * (0.5 + scale_4bit) * 0.25 * grid[grid_idx][j] * sign
      其中 ls = 2*k+1, scale_4bit = k → (0.5+k)*0.25 = (2k+1)*0.125 = ls*0.125 ✓

    参数:
        d_flat: [total_blocks] float32
        qs_flat: [total_blocks, 32] uint16 (packed)
        ne0, ne1: GGUF shape

    返回:
        new_d: [N, n_blocks_per_row] float16
        new_qs: [N, n_blocks_per_row, 32] uint16 (per-group)
        new_scales: [N, n_blocks_per_row, 8] uint8 (4-bit packed)
    """
    N = ne1  # out_features
    K = ne0  # in_features
    n_blocks_per_row = K // QK_K
    total_blocks = N * n_blocks_per_row

    # Reshape flat arrays to [N, n_blocks_per_row, ...]
    d_2d = d_flat.reshape(N, n_blocks_per_row)
    qs_3d = qs_flat.reshape(N, n_blocks_per_row, 32)

    new_qs = np.zeros((N, n_blocks_per_row, 32), dtype=np.uint16)
    new_scales = np.zeros((N, n_blocks_per_row, 8), dtype=np.uint8)

    for row in range(N):
        for blk in range(n_blocks_per_row):
            for ib32 in range(8):  # 8 sub-blocks per block
                q2_base = ib32 * 4
                q0 = int(qs_3d[row, blk, q2_base + 0])
                q1 = int(qs_3d[row, blk, q2_base + 1])
                q2 = int(qs_3d[row, blk, q2_base + 2])
                q3 = int(qs_3d[row, blk, q2_base + 3])

                aux32_0 = q0 | (q1 << 16)
                aux32_1 = q2 | (q3 << 16)

                # 4 grid indices (8-bit each)
                grid_indices = [
                    aux32_0 & 0xFF,
                    (aux32_0 >> 8) & 0xFF,
                    (aux32_0 >> 16) & 0xFF,
                    (aux32_0 >> 24) & 0xFF,
                ]

                # 4 sign indices (7-bit each) + scale (4-bit)
                sign_indices = [
                    aux32_1 & 0x7F,
                    (aux32_1 >> 7) & 0x7F,
                    (aux32_1 >> 14) & 0x7F,
                    (aux32_1 >> 21) & 0x7F,
                ]
                scale_k = (aux32_1 >> 28) & 0xF  # 0-15

                # Pack per-group qs: 9-bit grid_idx | 7-bit sign_idx
                for l in range(4):
                    new_qs[row, blk, q2_base + l] = grid_indices[l] | (sign_indices[l] << 9)

                # Pack scales: IQ2_XS uses 4-bit packed scales
                # scale_4bit = k, same as IQ2_XXS's scale_k
                # scales[ib] packs two scale values: low 4-bit for il=0,1; high 4-bit for il=2,3
                # But in IQ2_XS: scale_4bit = (scales[ib] >> 4*(il/2)) & 0xf
                # So scales[ib] low nibble = k for il=0,1; high nibble = k for il=2,3
                # Since all 4 groups in one sub-block share the same scale_k:
                new_scales[row, blk, ib32] = scale_k | (scale_k << 4)

    new_d = d_2d.astype(np.float16)

    return new_d, new_qs, new_scales


# ============================================================================
# Q2_K 反量化 (CPU → BF16 GPU tensor)
# ============================================================================
def dequantize_q2k_to_bf16(d_flat, dmin_flat, scales_flat, qs_flat, ne0, ne1):
    """将 Q2_K 量化权重反量化为 BF16 矩阵。

    反量化公式 (per element):
      value = d * scale_4bit * quant_2bit - dmin * min_4bit

    参数:
        d_flat: [total_blocks] float32
        dmin_flat: [total_blocks] float32
        scales_flat: [total_blocks*16] uint8 (4-bit packed)
        qs_flat: [total_blocks*64] uint8 (2-bit packed)
        ne0, ne1: GGUF shape (ne0=in_features=K, ne1=out_features=N)

    返回:
        weight_bf16: [ne1, ne0] BF16 numpy array (逻辑矩阵形状)
    """
    N = ne1  # out_features
    K = ne0  # in_features
    n_blocks_per_row = K // QK_K

    d_2d = d_flat.reshape(N, n_blocks_per_row)
    dmin_2d = dmin_flat.reshape(N, n_blocks_per_row)
    scales_3d = scales_flat.reshape(N, n_blocks_per_row, 16)
    qs_3d = qs_flat.reshape(N, n_blocks_per_row, 64)

    weight = np.zeros((N, K), dtype=np.float32)

    for row in range(N):
        for blk in range(n_blocks_per_row):
            d_val = d_2d[row, blk]
            dmin_val = dmin_2d[row, blk]
            sc = scales_3d[row, blk]  # [16]
            qs_blk = qs_3d[row, blk]  # [64]

            for k_half in range(2):  # 2 halves of 128 elements
                q2_base = k_half * 32
                shift = 0
                is_idx = k_half * 8  # starting scale index

                for _j in range(4):
                    # First sub-block
                    scale_4bit = int(sc[is_idx]) & 0xF
                    min_4bit = int(sc[is_idx]) >> 4
                    is_idx += 1

                    for l in range(16):
                        global_k = blk * QK_K + k_half * 128 + _j * 32 + l
                        quant_2bit = (int(qs_blk[q2_base + l]) >> shift) & 3
                        weight[row, global_k] = d_val * scale_4bit * quant_2bit - dmin_val * min_4bit

                    # Second sub-block
                    scale_4bit = int(sc[is_idx]) & 0xF
                    min_4bit = int(sc[is_idx]) >> 4
                    is_idx += 1

                    for l in range(16):
                        global_k = blk * QK_K + k_half * 128 + _j * 32 + 16 + l
                        quant_2bit = (int(qs_blk[q2_base + 16 + l]) >> shift) & 3
                        weight[row, global_k] = d_val * scale_4bit * quant_2bit - dmin_val * min_4bit

                    shift += 2

    return weight.astype(np.float16)  # FP16 intermediate, will convert to BF16 on GPU


# ============================================================================
# IQ2_XXS 反量化 (CPU → BF16)
# ============================================================================
def dequantize_iq2xxs_to_bf16(d_flat, qs_flat, ne0, ne1):
    """将 IQ2_XXS 量化权重反量化为 BF16 矩阵。

    反量化公式:
      value = d * ls * grid[grid_idx][j] * sign * 0.125
      ls = 2 * ((aux32_1 >> 28) & 0xF) + 1
    """
    N = ne1
    K = ne0
    n_blocks_per_row = K // QK_K

    d_2d = d_flat.reshape(N, n_blocks_per_row)
    qs_3d = qs_flat.reshape(N, n_blocks_per_row, 32)

    weight = np.zeros((N, K), dtype=np.float32)

    for row in range(N):
        for blk in range(n_blocks_per_row):
            d_val = d_2d[row, blk]
            for ib32 in range(8):
                q2_base = ib32 * 4
                q0 = int(qs_3d[row, blk, q2_base + 0])
                q1 = int(qs_3d[row, blk, q2_base + 1])
                q2 = int(qs_3d[row, blk, q2_base + 2])
                q3 = int(qs_3d[row, blk, q2_base + 3])

                aux32_0 = q0 | (q1 << 16)
                aux32_1 = q2 | (q3 << 16)

                grid_indices = [
                    aux32_0 & 0xFF,
                    (aux32_0 >> 8) & 0xFF,
                    (aux32_0 >> 16) & 0xFF,
                    (aux32_0 >> 24) & 0xFF,
                ]
                sign_indices = [
                    aux32_1 & 0x7F,
                    (aux32_1 >> 7) & 0x7F,
                    (aux32_1 >> 14) & 0x7F,
                    (aux32_1 >> 21) & 0x7F,
                ]
                ls = 2 * ((aux32_1 >> 28) & 0xF) + 1

                for l in range(4):
                    grid_idx = grid_indices[l]
                    sign_idx = sign_indices[l]
                    sign_byte = KSIGNS_IQ2XXS[sign_idx]

                    for j in range(8):
                        global_k = blk * QK_K + ib32 * 32 + l * 8 + j
                        grid_val = int(IQ2XXS_GRID[grid_idx, j])
                        sign_val = -1 if (sign_byte & (1 << j)) else 1
                        weight[row, global_k] = d_val * ls * grid_val * sign_val * 0.125

    return weight.astype(np.float16)


# ============================================================================
# 主验证逻辑
# ============================================================================
def main():
    print("=" * 70)
    print("混合量化 GPU FFN 正确性验证")
    print("IQ2_XXS gate/up + Q2_K down")
    print("=" * 70)

    # 1. 读取 GGUF
    print("\n[1] 读取 GGUF 文件...")
    reader = GGUFReader(GGUF_PATH)
    layer_id, expert_id = 0, 0
    weights = load_expert_from_gguf(reader, layer_id, expert_id)

    wt = weights['w1']
    gate_N, gate_K = wt[-1]  # logical_shape = (out_features, in_features)
    print(f"  gate (w1): IQ2_XXS, logical shape=({gate_N}, {gate_K})")

    wt = weights['w3']
    up_N, up_K = wt[-1]
    print(f"  up   (w3): IQ2_XXS, logical shape=({up_N}, {up_K})")

    wt = weights['w2']
    down_N, down_K = wt[-1]
    print(f"  down (w2): Q2_K, logical shape=({down_N}, {down_K})")

    # 2. 构造随机输入
    K_gate = gate_K  # in_features for gate/up
    N_gate = gate_N  # out_features for gate/up
    K_down = down_K  # in_features for down
    N_down = down_N  # out_features for down

    np.random.seed(42)
    x_np = np.random.randn(K_gate).astype(np.float32)
    print(f"\n[2] 输入: x shape=({K_gate},), range=[{x_np.min():.4f}, {x_np.max():.4f}]")

    # 3. CPU FFN (Rust)
    print("\n[3] CPU FFN (Rust AVX-512)...")
    from ds4rs import (
        Iq2XxsWeight, Q2KWeight, mixed_ffn_pair_iq2xxs_q2k,
        is_avx512_supported, init_tables, is_tables_initialized,
    )

    if not is_tables_initialized():
        from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS
        init_tables(list(IQ2XS_GRID_U64), list(KSIGNS_IQ2XS))
        print(f"  AVX-512: {is_avx512_supported()}")
        print(f"  Tables initialized: {is_tables_initialized()}")

    wt = weights['w1']
    gate_w = Iq2XxsWeight.from_numpy(wt[1].copy(), wt[2].reshape(-1).copy(), wt[-1])
    wt = weights['w3']
    up_w = Iq2XxsWeight.from_numpy(wt[1].copy(), wt[2].reshape(-1).copy(), wt[-1])
    wt = weights['w2']
    down_w = Q2KWeight.from_numpy(wt[1].copy(), wt[2].copy(), wt[3].reshape(-1).copy(), wt[4].reshape(-1).copy(), wt[-1])

    cpu_out = mixed_ffn_pair_iq2xxs_q2k(x_np, gate_w, up_w, down_w, 1.0, 0.0)
    cpu_out = np.array(cpu_out, dtype=np.float32)
    print(f"  CPU output: shape={cpu_out.shape}, range=[{cpu_out.min():.4f}, {cpu_out.max():.4f}]")
    print(f"  CPU mean={cpu_out.mean():.6f}, std={cpu_out.std():.6f}")

    # 4. GPU FFN — 方案 A: 反量化 + torch.matmul (参考实现)
    print("\n[4] GPU FFN — 方案 A: 反量化 + torch.matmul...")
    device = 'cuda'

    # 反量化 gate 权重
    # dequantize 函数接收 GGUF shape (ne0, ne1)，其中 ne0=in_features, ne1=out_features
    wt = weights['w1']
    gate_ne0, gate_ne1 = gate_K, gate_N  # GGUF convention: ne0=K, ne1=N
    gate_weight_bf16 = dequantize_iq2xxs_to_bf16(wt[1], wt[2], gate_ne0, gate_ne1)
    gate_weight_gpu = torch.from_numpy(gate_weight_bf16).to(device, dtype=torch.bfloat16)

    # 反量化 up 权重
    wt = weights['w3']
    up_ne0, up_ne1 = up_K, up_N
    up_weight_bf16 = dequantize_iq2xxs_to_bf16(wt[1], wt[2], up_ne0, up_ne1)
    up_weight_gpu = torch.from_numpy(up_weight_bf16).to(device, dtype=torch.bfloat16)

    # 反量化 down 权重
    wt = weights['w2']
    down_ne0, down_ne1 = down_K, down_N
    down_weight_bf16 = dequantize_q2k_to_bf16(wt[1], wt[2], wt[3], wt[4], down_ne0, down_ne1)
    down_weight_gpu = torch.from_numpy(down_weight_bf16).to(device, dtype=torch.bfloat16)

    # 输入
    x_gpu = torch.from_numpy(x_np).to(device, dtype=torch.bfloat16).unsqueeze(0)  # [1, K]

    # gate + up + SwiGLU
    gate_out = x_gpu @ gate_weight_gpu.T  # [1, N_gate]
    up_out = x_gpu @ up_weight_gpu.T      # [1, N_gate]
    gate_sigmoid = torch.sigmoid(gate_out)
    mid = gate_sigmoid * gate_out * up_out  # SwiGLU

    # down
    gpu_out_a = mid @ down_weight_gpu.T  # [1, N_down]
    gpu_out_a = gpu_out_a.squeeze(0).float().cpu().numpy()

    print(f"  GPU-A output: shape={gpu_out_a.shape}, range=[{gpu_out_a.min():.4f}, {gpu_out_a.max():.4f}]")

    # 对比
    diff_a = np.abs(cpu_out - gpu_out_a)
    max_diff_a = np.max(diff_a)
    mean_diff_a = np.mean(diff_a)
    print(f"  CPU vs GPU-A: max_diff={max_diff_a:.6f}, mean_diff={mean_diff_a:.6f}")

    # 5. GPU FFN — 方案 B: IQ2_XS TileLang kernel (gate/up) + 反量化 down
    print("\n[5] GPU FFN — 方案 B: IQ2_XS TileLang kernel (gate/up) + 反量化 down...")
    try:
        from iq2xs_gemm_tilelang import iq2xs_gemm_optimized, get_iq2xs_gemm_kernel

        # 转换 IQ2_XXS → IQ2_XS 格式
        wt = weights['w1']
        gate_d, gate_qs, gate_scales = convert_iq2xxs_to_iq2xs(wt[1], wt[2], gate_ne0, gate_ne1)
        wt = weights['w3']
        up_d, up_qs, up_scales = convert_iq2xxs_to_iq2xs(wt[1], wt[2], up_ne0, up_ne1)

        # 创建自定义 IQ2_XXS grid (512, 8) uint8
        # 前 256 项 = IQ2_XXS grid, 后 256 项 = 填充 (复制前 256 项)
        custom_grid_u8 = np.zeros((512, 8), dtype=np.uint8)
        for _i, _g in enumerate(_IQ2XXS_GRID_U64):
            for _j in range(8):
                custom_grid_u8[_i, _j] = (_g >> (8 * _j)) & 0xFF
        # 填充后 256 项
        custom_grid_u8[256:, :] = custom_grid_u8[:256, :]
        custom_grid_tensor = torch.from_numpy(custom_grid_u8).to(device)

        # ksigns 与 IQ2_XXS 相同
        custom_ksigns = torch.from_numpy(KSIGNS_IQ2XXS.copy()).to(device)

        # 转为 GPU tensor
        gate_d_gpu = torch.from_numpy(gate_d.copy()).to(device)
        gate_qs_gpu = torch.from_numpy(gate_qs.copy()).to(device)
        gate_scales_gpu = torch.from_numpy(gate_scales.copy()).to(device)

        up_d_gpu = torch.from_numpy(up_d.copy()).to(device)
        up_qs_gpu = torch.from_numpy(up_qs.copy()).to(device)
        up_scales_gpu = torch.from_numpy(up_scales.copy()).to(device)

        # 使用 IQ2_XS kernel + 自定义 grid
        N_g = gate_ne1  # out_features
        K_g = gate_ne0  # in_features

        gate_kernel = get_iq2xs_gemm_kernel(N_g, K_g)
        gate_out_b = torch.empty(1, N_g, dtype=torch.bfloat16, device=device)
        gate_kernel(x_gpu, gate_qs_gpu, gate_scales_gpu, gate_d_gpu,
                    custom_grid_tensor, custom_ksigns, gate_out_b)

        up_kernel = get_iq2xs_gemm_kernel(N_g, K_g)
        up_out_b = torch.empty(1, N_g, dtype=torch.bfloat16, device=device)
        up_kernel(x_gpu, up_qs_gpu, up_scales_gpu, up_d_gpu,
                  custom_grid_tensor, custom_ksigns, up_out_b)

        # SwiGLU
        gate_sigmoid_b = torch.sigmoid(gate_out_b)
        mid_b = gate_sigmoid_b * gate_out_b * up_out_b

        # down (反量化 + torch.matmul)
        gpu_out_b = mid_b @ down_weight_gpu.T
        gpu_out_b = gpu_out_b.squeeze(0).float().cpu().numpy()

        print(f"  GPU-B output: shape={gpu_out_b.shape}, range=[{gpu_out_b.min():.4f}, {gpu_out_b.max():.4f}]")

        diff_b = np.abs(cpu_out - gpu_out_b)
        max_diff_b = np.max(diff_b)
        mean_diff_b = np.mean(diff_b)
        print(f"  CPU vs GPU-B: max_diff={max_diff_b:.6f}, mean_diff={mean_diff_b:.6f}")

    except Exception as e:
        print(f"  GPU-B 失败: {e}")
        import traceback
        traceback.print_exc()
        gpu_out_b = None
        max_diff_b = float('inf')

    # 6. GPU FFN — 方案 C: TileLang mixed_quant_gemm (直接传 GGUF 数据)
    print("\n[6] GPU FFN — 方案 C: TileLang mixed_quant_gemm (直接传 GGUF 数据)...")
    try:
        from mixed_quant_gemm import mixed_quant_ffn

        wt = weights['w1']
        gate_d_raw = torch.from_numpy(wt[1].reshape(gate_ne1, gate_ne0 // QK_K).astype(np.float16).copy()).to(device)
        gate_qs_raw = torch.from_numpy(wt[2].reshape(gate_ne1, gate_ne0 // QK_K, 32).copy()).to(device)

        wt = weights['w3']
        up_d_raw = torch.from_numpy(wt[1].reshape(up_ne1, up_ne0 // QK_K).astype(np.float16).copy()).to(device)
        up_qs_raw = torch.from_numpy(wt[2].reshape(up_ne1, up_ne0 // QK_K, 32).copy()).to(device)

        wt = weights['w2']
        down_d_raw = torch.from_numpy(wt[1].reshape(down_ne1, down_ne0 // QK_K).astype(np.float16).copy()).to(device)
        down_dmin_raw = torch.from_numpy(wt[2].reshape(down_ne1, down_ne0 // QK_K).astype(np.float16).copy()).to(device)
        down_scales_raw = torch.from_numpy(wt[3].reshape(down_ne1, down_ne0 // QK_K, 16).copy()).to(device)
        down_qs_raw = torch.from_numpy(wt[4].reshape(down_ne1, down_ne0 // QK_K, 64).copy()).to(device)

        gpu_out_c = mixed_quant_ffn(
            x_gpu,
            gate_d_raw, gate_qs_raw,
            up_d_raw, up_qs_raw,
            down_d_raw, down_dmin_raw, down_scales_raw, down_qs_raw,
        )
        gpu_out_c = gpu_out_c.squeeze(0).float().cpu().numpy()

        print(f"  GPU-C output: shape={gpu_out_c.shape}, range=[{gpu_out_c.min():.4f}, {gpu_out_c.max():.4f}]")

        diff_c = np.abs(cpu_out - gpu_out_c)
        max_diff_c = np.max(diff_c)
        mean_diff_c = np.mean(diff_c)
        print(f"  CPU vs GPU-C: max_diff={max_diff_c:.6f}, mean_diff={mean_diff_c:.6f}")

    except Exception as e:
        print(f"  GPU-C 失败: {e}")
        import traceback
        traceback.print_exc()
        gpu_out_c = None
        max_diff_c = float('inf')

    # 7. 总结
    print("\n" + "=" * 70)
    print("验证结果总结")
    print("=" * 70)

    print(f"\n方案 A (反量化 + torch.matmul):")
    print(f"  max_diff = {max_diff_a:.6f}")
    print(f"  {'PASS' if max_diff_a < 1.0 else 'FAIL'} (阈值: 1.0)")

    if gpu_out_b is not None:
        print(f"\n方案 B (IQ2_XS TileLang + 自定义 grid):")
        print(f"  max_diff = {max_diff_b:.6f}")
        print(f"  {'PASS' if max_diff_b < 1.0 else 'FAIL'} (阈值: 1.0)")

    if gpu_out_c is not None:
        print(f"\n方案 C (mixed_quant_gemm 直接传 GGUF 数据):")
        print(f"  max_diff = {max_diff_c:.6f}")
        print(f"  {'PASS' if max_diff_c < 1.0 else 'FAIL'} (阈值: 1.0)")
        print(f"  注意: GGUF IQ2_XXS/Q2_K 格式与 TileLang kernel 格式不兼容，")
        print(f"  结果偏差大是预期的")

    # 详细统计
    print(f"\n详细统计 (方案 A):")
    rel_diff_a = diff_a / np.maximum(np.abs(cpu_out), 1e-6)
    print(f"  max_abs_diff: {max_diff_a:.6f}")
    print(f"  mean_abs_diff: {mean_diff_a:.6f}")
    print(f"  median_abs_diff: {np.median(diff_a):.6f}")
    print(f"  max_rel_diff: {np.max(rel_diff_a)*100:.4f}%")
    print(f"  mean_rel_diff: {np.mean(rel_diff_a)*100:.4f}%")
    print(f"  p90_abs_diff: {np.percentile(diff_a, 90):.6f}")
    print(f"  p99_abs_diff: {np.percentile(diff_a, 99):.6f}")

    # 前 10 个输出对比
    print(f"\n前 10 个输出对比 (方案 A):")
    print(f"  {'Idx':>4s}  {'CPU':>12s}  {'GPU':>12s}  {'Diff':>12s}  {'RelDiff%':>10s}")
    for i in range(min(10, len(cpu_out))):
        d = abs(cpu_out[i] - gpu_out_a[i])
        rd = d / max(abs(cpu_out[i]), 1e-6) * 100
        print(f"  {i:4d}  {cpu_out[i]:12.6f}  {gpu_out_a[i]:12.6f}  {d:12.6f}  {rd:10.4f}%")

    return max_diff_a < 1.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
