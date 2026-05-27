#!/usr/bin/env python3
"""CPU FFN 时间线及瓶颈分析

dim=7168, inter_dim=18432（DeepSeek V4 Flash 真实维度）
"""
import sys
import time
import numpy as np

sys.path.insert(0, '/workspace/inference')

from ds4rs import (
    Iq2XxsWeight, Q2KWeight,
    mixed_ffn_pair_iq2xxs_q2k,
    init_tables, is_tables_initialized,
    profile_mixed_ffn,
)
from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS


def init():
    if not is_tables_initialized():
        init_tables(list(IQ2XS_GRID_U64), list(KSIGNS_IQ2XS))


def make_weights(dim=7168, inter_dim=18432):
    n_blocks_per_row = dim // 256
    n_rows = inter_dim
    n_blocks = n_rows * n_blocks_per_row

    n_blocks_per_row_down = inter_dim // 256
    n_rows_down = dim
    n_blocks_down = n_rows_down * n_blocks_per_row_down

    rng = np.random.default_rng(42)

    gate_d = rng.standard_normal(n_blocks, dtype=np.float32).astype(np.float16).astype(np.float32)
    gate_qs = rng.integers(0, 65536, size=n_blocks * 32, dtype=np.uint16)
    up_d = rng.standard_normal(n_blocks, dtype=np.float32).astype(np.float16).astype(np.float32)
    up_qs = rng.integers(0, 65536, size=n_blocks * 32, dtype=np.uint16)

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

    print(f"=== CPU FFN 时间线及瓶颈分析 ===")
    print(f"  dim={dim}, inter_dim={inter_dim}")
    print(f"  gate/up 权重: {inter_dim * (dim // 256) * 66 / 1024 / 1024:.1f}MB (IQ2_XXS)")
    print(f"  down 权重: {dim * (inter_dim // 256) * 84 / 1024 / 1024:.1f}MB (Q2_K)")
    print(f"  DDR5-5600 双通道理论带宽: 44.8 GB/s")
    print()

    gate_w, up_w, down_w = make_weights(dim, inter_dim)
    np.random.seed(42)
    x = np.random.randn(dim).astype(np.float32)

    # 1. Rust 内置 profile
    print("=== Rust 内置 profile ===")
    n_profile = 5
    timings = profile_mixed_ffn(x, gate_w, up_w, down_w, 1.0, 0.0, n_profile)
    # timings: [q8_x, gate, up, swiglu, q8_mid, down] × n_iters
    stage_names = ['Q8量化(x)', 'gate(IQ2_XXS)', 'up(IQ2_XXS)', 'SwiGLU', 'Q8量化(mid)', 'down(Q2_K)']
    n_stages = 6
    for s in range(n_stages):
        vals = [timings[s + i * n_stages] for i in range(n_profile)]
        avg = np.mean(vals)
        pct = avg / np.mean(timings) * 100 * n_stages  # rough percentage
        print(f"  {stage_names[s]:20s}  avg={avg:7.2f}ms")
    total_per_iter = sum(np.mean([timings[s + i * n_stages] for i in range(n_profile)]) for s in range(n_stages))
    print(f"  {'总计':20s}  avg={total_per_iter:7.2f}ms")
    print()
    for s in range(n_stages):
        vals = [timings[s + i * n_stages] for i in range(n_profile)]
        avg = np.mean(vals)
        pct = avg / total_per_iter * 100
        print(f"  {stage_names[s]:20s}  {pct:5.1f}%")
    print()

    # 2. Python 侧分阶段计时
    print("=== Python 侧分阶段计时 ===")

    # Warmup
    for _ in range(3):
        _ = mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)

    n_iters = 10
    total_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = mixed_ffn_pair_iq2xxs_q2k(x, gate_w, up_w, down_w, 1.0, 0.0)
        t1 = time.perf_counter()
        total_times.append((t1 - t0) * 1000.0)

    total_avg = np.mean(total_times)
    total_min = np.min(total_times)
    print(f"  总 FFN: avg={total_avg:.2f}ms, min={total_min:.2f}ms")

    # 3. 6 专家串行预估
    print()
    print(f"=== 6 专家串行预估 ===")
    single_min = total_min
    six_serial = single_min * 6
    layer_time = 2.0 + six_serial + 1.0  # Attention + 6 FFN + Shared Expert
    tps = 1000.0 / (layer_time * 43)  # 43 layers
    print(f"  单专家 FFN: {single_min:.2f}ms")
    print(f"  6 专家串行: {six_serial:.1f}ms")
    print(f"  单层总计: {layer_time:.1f}ms (Attn 2.0 + FFN {six_serial:.1f} + Shared 1.0)")
    print(f"  43 层单步: {layer_time * 43:.0f}ms")
    print(f"  推理速度: {tps:.2f} t/s")

    # 4. 带宽利用率
    print()
    print(f"=== 带宽利用率 ===")
    gate_up_mb = inter_dim * (dim // 256) * 66 / 1024 / 1024
    down_mb = dim * (inter_dim // 256) * 84 / 1024 / 1024
    total_weight_mb = gate_up_mb * 2 + down_mb
    actual_bw = total_weight_mb / (single_min / 1000.0) / 1024.0
    ddr5_bw = 44.8
    util = actual_bw / ddr5_bw * 100.0
    print(f"  gate+up 权重: {gate_up_mb:.1f}MB × 2 = {gate_up_mb * 2:.1f}MB")
    print(f"  down 权重: {down_mb:.1f}MB")
    print(f"  总权重: {total_weight_mb:.1f}MB")
    print(f"  实际带宽: {actual_bw:.1f} GB/s")
    print(f"  DDR5-5600 双通道: {ddr5_bw} GB/s")
    print(f"  利用率: {util:.1f}%")

    # 5. 理论下限
    print()
    print(f"=== 理论下限 ===")
    min_time_bw = total_weight_mb / (ddr5_bw * 1024.0) * 1000.0
    print(f"  纯带宽极限: {min_time_bw:.2f}ms")
    print(f"  当前/极限: {single_min / min_time_bw:.1f}x")
    print(f"  计算开销占比: {(1 - min_time_bw / single_min) * 100:.0f}%")


if __name__ == '__main__':
    main()
