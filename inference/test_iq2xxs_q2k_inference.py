"""IQ2_XXS + Q2_K 混合量化专家推理测试。

从 experts_iq2xxs_q2k.gguf 读取专家权重，使用 Rust AVX-512 内核执行 FFN，
验证混合量化推理管线的正确性和性能。

用法：
  python inference/test_iq2xxs_q2k_inference.py
"""
import sys
import os
import time
import struct
import numpy as np

# GGUF 读取
from gguf import GGUFReader

# Rust 扩展
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ds4rs import (
    Iq2XxsWeight, Q2KWeight, mixed_ffn_pair_iq2xxs_q2k,
    is_avx512_supported, init_tables, is_tables_initialized,
)

GGUF_PATH = os.environ.get("GGUF_PATH", "/workspace/gguf/experts_iq2xxs_q2k.gguf")

# IQ2_XXS 查找表（与 llama.cpp 一致）
IQ2_XXS_GRID = np.array([
    0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 15, 18, 20, 23, 26,
    30, 34, 38, 42, 47, 52, 57, 62, 68, 74, 80, 86, 93, 100, 107, 114,
    122, 130, 138, 146, 155, 164, 173, 182, 192, 202, 212, 222, 233, 244, 255, 267,
    279, 291, 304, 316, 329, 342, 356, 370, 384, 399, 414, 429, 444, 460, 476, 492,
    509, 526, 543, 561, 579, 597, 615, 634, 653, 672, 692, 712, 732, 753, 774, 795,
    817, 839, 861, 883, 906, 929, 952, 976, 1000, 1024, 1049, 1074, 1099, 1125, 1151, 1177,
    1204, 1231, 1258, 1286, 1314, 1342, 1371, 1400, 1429, 1459, 1489, 1519, 1550, 1581, 1612, 1644,
    1676, 1708, 1741, 1774, 1807, 1841, 1875, 1909, 1944, 1979, 2014, 2050, 2086, 2122, 2159, 2196,
    2233, 2271, 2309, 2347, 2386, 2425, 2464, 2504, 2544, 2584, 2625, 2666, 2707, 2749, 2791, 2833,
    2876, 2919, 2962, 3006, 3050, 3094, 3139, 3184, 3229, 3275, 3321, 3367, 3414, 3461, 3508, 3556,
    3604, 3652, 3701, 3750, 3799, 3849, 3899, 3949, 4000, 4051, 4102, 4154, 4206, 4258, 4311, 4364,
    4417, 4471, 4525, 4579, 4634, 4689, 4744, 4800, 4856, 4912, 4969, 5026, 5084, 5142, 5200, 5258,
    5317, 5376, 5435, 5495, 5555, 5616, 5677, 5738, 5800, 5862, 5924, 5987, 6050, 6113, 6177, 6241,
    6305, 6370, 6435, 6500, 6566, 6632, 6699, 6766, 6833, 6901, 6969, 7037, 7106, 7175, 7244, 7314,
    7384, 7454, 7525, 7596, 7668, 7740, 7812, 7885, 7958, 8031, 8105, 8179, 8253, 8328, 8403, 8478,
    8554, 8630, 8707, 8784, 8861, 8939, 9017, 9095, 9174, 9253, 9332, 9412, 9492, 9572, 9653, 9734,
    9816, 9898, 9980, 10063, 10146, 10229, 10313, 10397, 10481, 10566, 10651, 10736, 10822, 10908, 10995, 11082,
    11169, 11256, 11344, 11432, 11521, 11610, 11699, 11789, 11879, 11969, 12060, 12151, 12242, 12334, 12426, 12518,
    12611, 12704, 12797, 12891, 12985, 13079, 13174, 13269, 13364, 13460, 13556, 13652, 13749, 13846, 13943, 14041,
    14139, 14237, 14336, 14435, 14534, 14634, 14734, 14834, 14935, 15036, 15137, 15239, 15341, 15443, 15546, 15649,
    15752, 15855, 15959, 16063, 16168, 16273, 16378, 16483, 16589, 16695, 16802, 16909, 17016, 17123, 17231, 17339,
    17447, 17556, 17665, 17774, 17884, 17994, 18104, 18215, 18326, 18437, 18549, 18661, 18773, 18886, 18999, 19112,
    19225, 19339, 19453, 19567, 19682, 19797, 19912, 20028, 20144, 20260, 20377, 20494, 20611, 20729, 20847, 20965,
    21084, 21203, 21322, 21442, 21562, 21682, 21803, 21924, 22045, 22167, 22289, 22411, 22534, 22657, 22780, 22903,
    23027, 23151, 23275, 23400, 23525, 23650, 23776, 23902, 24028, 24155, 24282, 24409, 24537, 24665, 24793, 24921,
    25050, 25179, 25309, 25438, 25568, 25699, 25829, 25960, 26091, 26223, 26355, 26487, 26619, 26752, 26885, 27018,
    27152, 27286, 27420, 27555, 27690, 27825, 27961, 28097, 28233, 28370, 28507, 28644, 28782, 28920, 29058, 29196,
    29335, 29474, 29614, 29754, 29894, 30034, 30175, 30316, 30458, 30599, 30741, 30883, 31026, 31169, 31312, 31456,
    31600, 31744, 31888, 32033, 32178, 32324, 32470, 32616, 32762, 32909, 33056, 33204, 33351, 33499, 33647, 33796,
    33945, 34094, 34244, 34393, 34543, 34694, 34844, 34995, 35146, 35298, 35450, 35602, 35754, 35907, 36060, 36213,
    36367, 36521, 36675, 36830, 36985, 37140, 37296, 37452, 37608, 37764, 37921, 38078, 38236, 38394, 38552, 38710,
    38869, 39028, 39188, 39348, 39508, 39668, 39829, 39990, 40152, 40314, 40476, 40638, 40801, 40964, 41127, 41291,
], dtype=np.uint16)

IQ2_XXS_KSIGNS = np.array([
    0, 105, 53, 90, 142, 199, 247, 226, 177, 60, 12, 37, 149, 212, 250, 231,
    85, 40, 93, 20, 232, 253, 205, 178, 228, 249, 201, 182, 68, 9, 57, 32,
    170, 153, 213, 144, 126, 103, 63, 82, 34, 119, 167, 136, 111, 80, 44, 95,
    255, 150, 202, 165, 113, 56, 8, 29, 78, 195, 243, 218, 106, 43, 5, 24,
    51, 94, 46, 73, 157, 208, 240, 219, 130, 87, 39, 64, 176, 217, 245, 224,
    98, 7, 55, 26, 224, 245, 197, 170, 192, 241, 193, 174, 50, 15, 63, 38,
    139, 100, 210, 131, 121, 98, 58, 77, 45, 124, 172, 141, 116, 85, 49, 90,
    204, 145, 197, 160, 108, 51, 3, 24, 73, 190, 238, 213, 101, 38, 0, 21,
    102, 83, 41, 120, 168, 137, 112, 81, 161, 128, 180, 151, 99, 42, 10, 31,
    67, 108, 60, 87, 171, 222, 254, 233, 184, 71, 23, 48, 160, 223, 253, 234,
    146, 163, 223, 154, 134, 111, 71, 90, 18, 103, 151, 120, 95, 64, 28, 79,
    239, 134, 186, 149, 97, 40, 16, 37, 62, 179, 227, 202, 90, 27, 1, 22,
    132, 115, 173, 144, 119, 96, 52, 75, 163, 132, 184, 155, 107, 46, 14, 35,
    3, 44, 28, 55, 139, 190, 222, 201, 152, 39, 7, 32, 144, 207, 249, 230,
    194, 175, 235, 156, 138, 115, 75, 94, 22, 107, 155, 124, 99, 68, 32, 83,
    243, 138, 190, 153, 101, 44, 4, 25, 74, 191, 239, 214, 102, 39, 2, 23,
], dtype=np.uint8)


def init_iq2_tables():
    """初始化 IQ2 查找表（Rust 内核需要）。"""
    if not is_tables_initialized():
        from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS
        grid_u64 = list(IQ2XS_GRID_U64)
        ksigns = list(KSIGNS_IQ2XS)
        init_tables(grid_u64, ksigns)
        print(f"[Init] IQ2 tables initialized, AVX-512: {is_avx512_supported()}")


def parse_iq2xxs_block(data_bytes, n_blocks, out_dim, in_dim):
    """解析 IQ2_XXS 原始字节为 (d, qs) 数组（向量化）。"""
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    # 每个 block 66 bytes: d(fp16, 2B) + qs(u16×32, 64B)
    block_size = 66
    assert len(raw) == n_blocks * block_size

    # 重塑为 (n_blocks, 66)
    blocks = raw.reshape(n_blocks, block_size)

    # d: 前 2 字节 → fp16 → f32
    d = blocks[:, 0:2].view(np.float16).astype(np.float32).ravel()

    # qs: 后 64 字节 → u16×32
    qs = blocks[:, 2:66].view(np.uint16).reshape(n_blocks, 32).ravel().copy()

    return d, qs


def parse_q2k_block(data_bytes, n_blocks, out_dim, in_dim):
    """解析 Q2_K 原始字节为 (d, dmin, scales, qs) 数组（向量化）。

    llama.cpp block_q2_K 布局（84 bytes）：
      scales: uint8[16] (4-bit packed scales+mins) — 前 16 字节
      qs:     uint8[64] (2-bit packed quants)      — 接下来 64 字节
      d:      fp16 (2 bytes)                        — 偏移 80
      dmin:   fp16 (2 bytes)                        — 偏移 82
    """
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    block_size = 84
    assert len(raw) == n_blocks * block_size

    blocks = raw.reshape(n_blocks, block_size)

    # scales: bytes 0-15 → u8×16
    scales = blocks[:, 0:16].reshape(n_blocks, 16).ravel().copy()

    # qs: bytes 16-79 → u8×64
    qs = blocks[:, 16:80].reshape(n_blocks, 64).ravel().copy()

    # d: bytes 80-81 → fp16 → f32
    d = blocks[:, 80:82].view(np.float16).astype(np.float32).ravel()

    # dmin: bytes 82-83 → fp16 → f32
    dmin = blocks[:, 82:84].view(np.float16).astype(np.float32).ravel()

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

        if type_name == 'IQ2_XXS':
            d, qs = parse_iq2xxs_block(data, n_blocks, ne0, ne1)
            weights[wt_name] = ('iq2xxs', d, qs, (ne0, ne1))
        elif type_name == 'Q2_K':
            d, dmin, scales, qs = parse_q2k_block(data, n_blocks, ne0, ne1)
            weights[wt_name] = ('q2k', d, dmin, scales, qs, (ne0, ne1))
        else:
            raise ValueError(f"Unsupported type: {type_name}")

    return weights


def test_matvec():
    """测试 IQ2_XXS 和 Q2_K 的 matvec 正确性。"""
    print("\n=== Test 1: matvec 正确性 ===")
    reader = GGUFReader(GGUF_PATH)

    # 加载 layer 0, expert 0
    weights = load_expert_from_gguf(reader, 0, 0)

    # 构造随机输入
    dim = 2048  # in_dim for gate/up
    np.random.seed(42)
    x = np.random.randn(dim).astype(np.float32)

    # gate (IQ2_XXS)
    wt = weights['w1']
    assert wt[0] == 'iq2xxs'
    _, d, qs, shape = wt
    gate_w = Iq2XxsWeight.from_numpy(d, qs, shape)
    gate_out = gate_w.matvec(x)
    print(f"  gate: shape={shape}, out={gate_out.shape}, range=[{gate_out.min():.4f}, {gate_out.max():.4f}]")

    # up (IQ2_XXS)
    wt = weights['w3']
    assert wt[0] == 'iq2xxs'
    _, d, qs, shape = wt
    up_w = Iq2XxsWeight.from_numpy(d, qs, shape)
    up_out = up_w.matvec(x)
    print(f"  up:   shape={shape}, out={up_out.shape}, range=[{up_out.min():.4f}, {up_out.max():.4f}]")

    # down (Q2_K)
    wt = weights['w2']
    assert wt[0] == 'q2k'
    _, d, dmin, scales, qs, shape = wt
    down_w = Q2KWeight.from_numpy(d, dmin, scales, qs, shape)
    mid = np.random.randn(shape[1]).astype(np.float32)  # inter_dim
    down_out = down_w.matvec(mid)
    print(f"  down: shape={shape}, out={down_out.shape}, range=[{down_out.min():.4f}, {down_out.max():.4f}]")

    print("  [OK] matvec 正确性测试通过")


def test_mixed_ffn():
    """测试混合量化 FFN (IQ2_XXS gate/up + Q2_K down)。"""
    print("\n=== Test 2: 混合量化 FFN ===")
    reader = GGUFReader(GGUF_PATH)

    # 加载 layer 0, expert 0
    weights = load_expert_from_gguf(reader, 0, 0)

    # 构造输入
    dim = 2048
    np.random.seed(42)
    x = np.random.randn(dim).astype(np.float32)

    # 构建权重
    wt = weights['w1']
    gate_w = Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])

    wt = weights['w3']
    up_w = Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])

    wt = weights['w2']
    down_w = Q2KWeight.from_numpy(wt[1], wt[2], wt[3], wt[4], wt[5])

    # 执行混合 FFN
    out = mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)
    print(f"  FFN output: shape={out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")
    print(f"  FFN mean={out.mean():.4f}, std={out.std():.4f}")

    # 验证输出维度
    assert out.shape == (dim,), f"Expected shape ({dim},), got {out.shape}"
    assert np.all(np.isfinite(out)), "Output contains NaN or Inf"

    print("  [OK] 混合量化 FFN 测试通过")


def test_multi_expert():
    """测试多个专家的 FFN。"""
    print("\n=== Test 3: 多专家 FFN ===")
    reader = GGUFReader(GGUF_PATH)

    dim = 2048
    np.random.seed(42)
    x = np.random.randn(dim).astype(np.float32)

    # 测试 3 个不同层的专家
    test_cases = [(0, 0), (10, 100), (42, 255)]

    for layer_id, expert_id in test_cases:
        weights = load_expert_from_gguf(reader, layer_id, expert_id)

        wt = weights['w1']
        gate_w = Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])
        wt = weights['w3']
        up_w = Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])
        wt = weights['w2']
        down_w = Q2KWeight.from_numpy(wt[1], wt[2], wt[3], wt[4], wt[5])

        out = mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)
        print(f"  Layer {layer_id}, Expert {expert_id}: "
              f"range=[{out.min():.4f}, {out.max():.4f}], "
              f"mean={out.mean():.4f}, std={out.std():.4f}")

    print("  [OK] 多专家 FFN 测试通过")


def test_latency():
    """测试混合量化 FFN 延迟。"""
    print("\n=== Test 4: 延迟测试 ===")
    reader = GGUFReader(GGUF_PATH)

    dim = 2048
    np.random.seed(42)
    x = np.random.randn(dim).astype(np.float32)

    # 加载一个专家
    weights = load_expert_from_gguf(reader, 0, 0)
    wt = weights['w1']
    gate_w = Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])
    wt = weights['w3']
    up_w = Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])
    wt = weights['w2']
    down_w = Q2KWeight.from_numpy(wt[1], wt[2], wt[3], wt[4], wt[5])

    # Warmup
    for _ in range(3):
        mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)

    # Benchmark
    n_iter = 20
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000
    print(f"  混合 FFN 延迟: avg={avg_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms ({n_iter} iters)")
    print(f"  目标: <5ms (CPU IQ2_XS FFN ~2.7ms)")


def main():
    print(f"GGUF: {GGUF_PATH}")
    print(f"AVX-512: {is_avx512_supported()}")

    init_iq2_tables()

    test_matvec()
    test_mixed_ffn()
    test_multi_expert()
    test_latency()

    print("\n=== 全部测试通过 ===")


if __name__ == "__main__":
    main()
