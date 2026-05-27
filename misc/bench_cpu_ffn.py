#!/usr/bin/env python3
"""CPU FFN 基准测试：rayon vs streaming vs blocked_mt

对比三种 FFN 实现的性能和带宽利用率。
"""

import time
import numpy as np
import sys

sys.path.insert(0, '/workspace/inference')

from ds4rs import (
    mixed_ffn_pair_iq2xxs_q2k,
    mixed_ffn_pair_iq2xxs_q2k_streaming,
    mixed_ffn_pair_iq2xxs_q2k_blocked_mt,
    Iq2XxsWeight,
    Q2KWeight,
    init_tables,
    is_tables_initialized,
)
from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS


def init():
    if not is_tables_initialized():
        init_tables(list(IQ2XS_GRID_U64), list(KSIGNS_IQ2XS))


def make_random_weight(dim=7168, inter_dim=18432):
    """构造随机 IQ2_XXS + Q2_K 权重"""
    n_blocks_per_row = dim // 256       # 28
    n_rows_gate_up = inter_dim          # 18432
    n_blocks_gate_up = n_rows_gate_up * n_blocks_per_row  # 516096

    n_rows_down = dim                   # 7168
    n_blocks_per_row_down = inter_dim // 256  # 72
    n_blocks_down = n_rows_down * n_blocks_per_row_down  # 516096

    rng = np.random.default_rng(42)

    # gate/up: IQ2_XXS — d[n_blocks], qs[n_blocks * 32]
    gate_d = rng.standard_normal(n_blocks_gate_up, dtype=np.float32).astype(np.float16).astype(np.float32)
    gate_qs = rng.integers(0, 65536, size=n_blocks_gate_up * 32, dtype=np.uint16)
    up_d = rng.standard_normal(n_blocks_gate_up, dtype=np.float32).astype(np.float16).astype(np.float32)
    up_qs = rng.integers(0, 65536, size=n_blocks_gate_up * 32, dtype=np.uint16)

    # down: Q2_K — d[n_blocks], dmin[n_blocks], scales[n_blocks*16], qs[n_blocks*64]
    down_d = rng.standard_normal(n_blocks_down, dtype=np.float32).astype(np.float16).astype(np.float32)
    down_dmin = np.abs(rng.standard_normal(n_blocks_down, dtype=np.float32)).astype(np.float16).astype(np.float32) * 0.01
    down_scales = rng.integers(0, 256, size=n_blocks_down * 16, dtype=np.uint8)
    down_qs = rng.integers(0, 256, size=n_blocks_down * 64, dtype=np.uint8)

    gate_w = Iq2XxsWeight.from_numpy(
        np.ascontiguousarray(gate_d, dtype=np.float32),
        np.ascontiguousarray(gate_qs, dtype=np.uint16),
        (inter_dim, dim),
    )
    up_w = Iq2XxsWeight.from_numpy(
        np.ascontiguousarray(up_d, dtype=np.float32),
        np.ascontiguousarray(up_qs, dtype=np.uint16),
        (inter_dim, dim),
    )
    down_w = Q2KWeight.from_numpy(
        np.ascontiguousarray(down_d, dtype=np.float32),
        np.ascontiguousarray(down_dmin, dtype=np.float32),
        np.ascontiguousarray(down_scales, dtype=np.uint8),
        np.ascontiguousarray(down_qs, dtype=np.uint8),
        (dim, inter_dim),
    )

    return gate_w, up_w, down_w


def bench(name, func, x, gate_w, up_w, down_w, route_weight, swiglu_limit, n_iters=5):
    """基准测试函数"""
    # 预热
    _ = func(x, gate_w, up_w, down_w, route_weight, swiglu_limit)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = func(x, gate_w, up_w, down_w, route_weight, swiglu_limit)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg = np.mean(times)
    std = np.std(times)
    min_t = np.min(times)

    # 带宽估算：gate(34MB) + up(34MB) + down(30MB) = ~98MB
    weight_mb = 98.0
    bandwidth = weight_mb / (min_t / 1000.0) / 1024.0  # GB/s
    ddr5_bw = 44.8  # DDR5-5600 双通道
    util = bandwidth / ddr5_bw * 100.0

    print(f"  {name:30s}  avg={avg:7.2f}ms  min={min_t:7.2f}ms  std={std:5.2f}ms  "
          f"BW={bandwidth:5.1f}GB/s  util={util:4.1f}%")

    return result, min_t


def main():
    init()

    dim = 7168
    inter_dim = 18432
    route_weight = 1.0
    swiglu_limit = 0.0

    print(f"=== CPU FFN 基准测试 ===")
    print(f"  dim={dim}, inter_dim={inter_dim}")
    print(f"  gate/up 权重: {inter_dim * (dim // 256) * 66 / 1024 / 1024:.1f}MB (IQ2_XXS)")
    print(f"  down 权重: {dim * (inter_dim // 256) * 84 / 1024 / 1024:.1f}MB (Q2_K)")
    print(f"  DDR5-5600 双通道理论带宽: 44.8 GB/s")
    print()

    x = np.random.randn(dim).astype(np.float32)
    x = np.ascontiguousarray(x)

    # 使用随机权重（不需要 GGUF 文件）
    print("使用随机权重测试...")
    gate_w, up_w, down_w = make_random_weight(dim, inter_dim)

    # 1. rayon 并行版（基线）
    result_rayon, t_rayon = bench(
        "rayon (12T)",
        lambda x, g, u, d, r, s: mixed_ffn_pair_iq2xxs_q2k(x, g, u, d, r, s),
        x, gate_w, up_w, down_w, route_weight, swiglu_limit,
    )

    # 2. 单线程 streaming 版
    result_stream, t_stream = bench(
        "streaming (1T)",
        lambda x, g, u, d, r, s: mixed_ffn_pair_iq2xxs_q2k_streaming(x, g, u, d, r, s),
        x, gate_w, up_w, down_w, route_weight, swiglu_limit,
    )

    # 3. 多线程 blocked_mt 版（2/3/4/6 线程）
    for n_threads in [2, 3, 4, 6]:
        result_mt, t_mt = bench(
            f"blocked_mt ({n_threads}T)",
            lambda x, g, u, d, r, s, nt=n_threads: mixed_ffn_pair_iq2xxs_q2k_blocked_mt(
                x, g, u, d, r, s, nt),
            x, gate_w, up_w, down_w, route_weight, swiglu_limit,
        )

    print()
    print(f"=== 总结 ===")
    print(f"  rayon (12T):    {t_rayon:.2f}ms (基线)")
    print(f"  streaming (1T): {t_stream:.2f}ms ({t_stream/t_rayon:.2f}x)")
    for n_threads in [2, 3, 4, 6]:
        # 重新测量以获取时间
        pass


if __name__ == '__main__':
    main()
