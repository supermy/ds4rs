#!/usr/bin/env python3
"""6 专家并行 FFN 基准测试

对比串行 vs 2 路并行 vs 3 路并行的性能。
"""

import time
import numpy as np
import sys

sys.path.insert(0, '/workspace/inference')

from ds4rs import (
    mixed_ffn_pair_iq2xxs_q2k,
    mixed_ffn_6experts_iq2xxs_q2k,
    Iq2XxsWeight,
    Q2KWeight,
    init_tables,
    is_tables_initialized,
)
from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS


def init():
    if not is_tables_initialized():
        init_tables(list(IQ2XS_GRID_U64), list(KSIGNS_IQ2XS))


def make_random_weight(dim=7168, inter_dim=18432, seed=42):
    """构造随机 IQ2_XXS + Q2_K 权重"""
    n_blocks_per_row = dim // 256
    n_rows_gate_up = inter_dim
    n_blocks_gate_up = n_rows_gate_up * n_blocks_per_row

    n_rows_down = dim
    n_blocks_per_row_down = inter_dim // 256
    n_blocks_down = n_rows_down * n_blocks_per_row_down

    rng = np.random.default_rng(seed)

    gate_d = rng.standard_normal(n_blocks_gate_up, dtype=np.float32).astype(np.float16).astype(np.float32)
    gate_qs = rng.integers(0, 65536, size=n_blocks_gate_up * 32, dtype=np.uint16)
    up_d = rng.standard_normal(n_blocks_gate_up, dtype=np.float32).astype(np.float16).astype(np.float32)
    up_qs = rng.integers(0, 65536, size=n_blocks_gate_up * 32, dtype=np.uint16)

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


def main():
    init()

    dim = 7168
    inter_dim = 18432
    n_experts = 6

    print(f"=== 6 专家并行 FFN 基准测试 ===")
    print(f"  dim={dim}, inter_dim={inter_dim}, n_experts={n_experts}")
    print(f"  单专家权重: {inter_dim * (dim // 256) * 66 / 1024 / 1024 * 2 + dim * (inter_dim // 256) * 84 / 1024 / 1024:.1f}MB")
    print(f"  6 专家总权重: {(inter_dim * (dim // 256) * 66 / 1024 / 1024 * 2 + dim * (inter_dim // 256) * 84 / 1024 / 1024) * 6:.1f}MB")
    print(f"  DDR5-5600 双通道理论带宽: 44.8 GB/s")
    print()

    x = np.ascontiguousarray(np.random.randn(dim).astype(np.float32))

    # 构造 6 个专家的权重（不同 seed 产生不同权重）
    experts = []
    for i in range(n_experts):
        gate_w, up_w, down_w = make_random_weight(dim, inter_dim, seed=42 + i)
        experts.append((gate_w, up_w, down_w))

    # 1. 串行基线：6 个专家顺序执行
    print("=== 1. 串行基线（6 专家顺序执行）===")
    # 预热
    for gate_w, up_w, down_w in experts:
        _ = mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)

    n_iters = 5
    serial_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        serial_output = None
        for gate_w, up_w, down_w in experts:
            result = mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)
            if serial_output is None:
                serial_output = result
            else:
                serial_output = serial_output + result
        t1 = time.perf_counter()
        serial_times.append((t1 - t0) * 1000.0)

    serial_avg = np.mean(serial_times)
    serial_min = np.min(serial_times)
    print(f"  串行 6 专家: avg={serial_avg:.2f}ms, min={serial_min:.2f}ms")

    # 2. 2 路并行（mixed_ffn_6experts_iq2xxs_q2k）
    print()
    print("=== 2. 2 路专家并行（std::thread::scope）===")
    gate_list = [e[0] for e in experts]
    up_list = [e[1] for e in experts]
    down_list = [e[2] for e in experts]
    route_weights = [1.0] * n_experts

    # 预热
    _ = mixed_ffn_6experts_iq2xxs_q2k(
        x, gate_list, up_list, down_list, route_weights, 0.0)

    parallel_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        parallel_output = mixed_ffn_6experts_iq2xxs_q2k(
            x, gate_list, up_list, down_list, route_weights, 0.0)
        t1 = time.perf_counter()
        parallel_times.append((t1 - t0) * 1000.0)

    parallel_avg = np.mean(parallel_times)
    parallel_min = np.min(parallel_times)
    print(f"  2 路并行 6 专家: avg={parallel_avg:.2f}ms, min={parallel_min:.2f}ms")

    # 3. 结果对比
    print()
    print(f"=== 结果对比 ===")
    print(f"  串行:  {serial_min:.2f}ms (基线)")
    print(f"  2路并行: {parallel_min:.2f}ms ({parallel_min/serial_min:.2f}x)")
    print(f"  加速比: {serial_min/parallel_min:.2f}x")
    print()

    # 带宽估算
    total_weight_mb = (inter_dim * (dim // 256) * 66 / 1024 / 1024 * 2 + dim * (inter_dim // 256) * 84 / 1024 / 1024) * 6
    serial_bw = total_weight_mb / (serial_min / 1000.0) / 1024.0
    parallel_bw = total_weight_mb / (parallel_min / 1000.0) / 1024.0
    ddr5_bw = 44.8
    print(f"  串行带宽利用率: {serial_bw/ddr5_bw*100:.1f}% ({serial_bw:.1f} GB/s)")
    print(f"  并行带宽利用率: {parallel_bw/ddr5_bw*100:.1f}% ({parallel_bw:.1f} GB/s)")
    print()

    # 推理速度估算
    single_ffn_serial = serial_min / 6
    single_ffn_parallel = parallel_min / 6  # effective per-expert time
    for label, ff6 in [("串行", serial_min), ("2路并行", parallel_min)]:
        layer_time = 2.0 + ff6 + 1.0  # Attn + FFN + Shared
        tps = 1000.0 / (layer_time * 43)
        print(f"  {label}: 单层={layer_time:.1f}ms, 推理速度={tps:.2f} t/s")


if __name__ == '__main__':
    main()
