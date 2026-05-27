#!/usr/bin/env python3
"""
CPU FFN 性能基准测试

测试 Rust CPU FFN 的性能，覆盖以下场景：
  1. 单行点积（1 行 IQ2_XS 权重 × 向量）
  2. 矩阵向量乘法（完整权重矩阵 × 向量）
  3. 配对 gate/up 点积（gate + up 两次独立 matvec）
  4. 完整 FFN（cpu_expert_ffn_pair，含配对优化 + SwiGLU + down 投影）
  5. 双专家 FFN（cpu_expert_ffn_pair_dual，MoE top-2 场景）

权重维度：7168×14336（DeepSeek V4 MoE 专家尺寸）
  - gate / up 权重：(14336, 7168)
  - down 权重：(7168, 14336)

运行方式：
  docker exec ds4rs-dev bash -c \
    "source ~/.cargo/env && cd /workspace && \
     maturin develop --release -- -C target-cpu=native && \
     python benchmarks/bench_cpu_ffn.py"
"""

import numpy as np
import time
import sys

# ── 常量 ──────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 7168       # DeepSeek V4 hidden dim
INTER_DIM = 14336       # DeepSeek V4 intermediate dim
BLOCK_SIZE = 256        # IQ2_XS block size
N_WARMUP = 3            # 预热迭代次数
N_BENCH = 10            # 基准测试迭代次数


# ── 导入 ──────────────────────────────────────────────────────────────────────
try:
    from ds4rs import (
        init_tables,
        is_tables_initialized,
        is_avx512_supported,
        is_avx2_supported,
        Iq2XsWeight,
        iq2xs_matvec,
        cpu_expert_ffn_pair,
        cpu_expert_ffn_pair_dual,
    )
except ImportError:
    print("错误：无法导入 ds4rs 模块")
    print("请先运行：maturin develop --release -- -C target-cpu=native")
    sys.exit(1)


# ── SIMD 路径检测 ─────────────────────────────────────────────────────────────
def detect_simd_path():
    """检测可用的 SIMD 路径，返回 (avx512, avx2) 布尔元组和路径名称。"""
    avx512 = is_avx512_supported()
    avx2 = is_avx2_supported()

    if avx512:
        path_name = "AVX-512 VNNI (512-bit maddubs+madd)"
    elif avx2:
        path_name = "AVX2 (256-bit maddubs+madd)"
    else:
        path_name = "Scalar (纯标量回退)"

    return avx512, avx2, path_name


# ── 查找表初始化 ──────────────────────────────────────────────────────────────
def init_lookup_tables():
    """初始化 IQ2_XS 查找表（使用随机 grid_u64 和 ksigns）。"""
    if is_tables_initialized():
        print("查找表已初始化，跳过")
        return

    # grid_u64: 512 个 u64 值（IQ2_XS 量化网格）
    grid_u64 = np.random.randint(0, 2**64, size=512, dtype=np.uint64)
    # ksigns: 128 个 u8 值（符号查找表）
    ksigns = np.random.randint(0, 256, size=128, dtype=np.uint8)

    init_tables(grid_u64.tolist(), ksigns.tolist())
    print("查找表初始化完成 (grid_u64=512, ksigns=128)")


# ── 权重构造 ──────────────────────────────────────────────────────────────────
def create_iq2xs_weight(n_rows, n_cols):
    """
    创建随机 IQ2_XS 权重。

    IQ2_XS 布局（每 256 元素一个 block）：
      - d:      每 block 1 个 f32    → 总计 n_rows * (n_cols/256) 个 f32
      - qs:     每 block 32 个 u16   → 总计 n_rows * (n_cols/256) * 32 个 u16
      - scales: 每 block 8 个 u8     → 总计 n_rows * (n_cols/256) * 8 个 u8

    qs 编码：
      - bits [0:9)  → grid index (0..511)
      - bits [9:16) → sign index (0..127)
    """
    assert n_cols % BLOCK_SIZE == 0, f"n_cols={n_cols} 不是 {BLOCK_SIZE} 的倍数"
    blocks_per_row = n_cols // BLOCK_SIZE
    total_blocks = n_rows * blocks_per_row

    # 超块缩放因子
    d = np.random.randn(total_blocks).astype(np.float32)

    # 量化索引：保证 grid_index ∈ [0,511], sign_index ∈ [0,127]
    grid_idx = np.random.randint(0, 512, size=total_blocks * 32, dtype=np.uint16)
    sign_idx = np.random.randint(0, 128, size=total_blocks * 32, dtype=np.uint16)
    qs = (grid_idx | (sign_idx << 9)).astype(np.uint16)

    # 组缩放因子
    scales = np.random.randint(0, 256, size=total_blocks * 8, dtype=np.uint8)

    return Iq2XsWeight(d, qs, scales, (n_rows, n_cols))


# ── 通用计时工具 ──────────────────────────────────────────────────────────────
def benchmark(fn, n_warmup=N_WARMUP, n_iter=N_BENCH, **kwargs):
    """
    通用基准测试：预热 + 多次迭代计时。

    返回 (mean_ms, std_ms, latencies_ms)。
    """
    # 预热
    for _ in range(n_warmup):
        fn(**kwargs)

    # 计时
    latencies = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn(**kwargs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    arr = np.array(latencies)
    return arr.mean(), arr.std(), latencies, result


def print_result(label, mean_ms, std_ms, throughput_info=""):
    """格式化打印基准测试结果。"""
    print(f"  {label:<36s}  {mean_ms:8.2f} ± {std_ms:5.2f} ms  {throughput_info}")


# ── 基准测试场景 ──────────────────────────────────────────────────────────────

def bench_single_row_dot(x_hidden):
    """
    场景 1：单行点积
    - 权重：1 × 7168（1 行 IQ2_XS 权重）
    - 输入：(7168,) f32 向量
    - 操作：1 行 IQ2_XS 解量化 + 点积
    """
    weight_1row = create_iq2xs_weight(1, HIDDEN_DIM)

    def run():
        return iq2xs_matvec(x_hidden, weight_1row)

    mean, std, _, _ = benchmark(run)
    # 吞吐量：单次点积的 GFLOPS ≈ 2 * 7168 / 1e6 (乘加算 2 FLOP)
    gflops = 2.0 * HIDDEN_DIM / (mean * 1e6)
    print_result("单行点积 (1×7168)", mean, std, f"{gflops:.2f} GFLOPS")
    return mean


def bench_matvec(x_hidden):
    """
    场景 2：矩阵向量乘法
    - 权重：14336 × 7168（gate 投影矩阵）
    - 输入：(7168,) f32 向量
    - 操作：完整 IQ2_XS matvec
    """
    weight = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)

    def run():
        return iq2xs_matvec(x_hidden, weight)

    mean, std, _, _ = benchmark(run)
    # 吞吐量：GFLOPS ≈ 2 * 14336 * 7168 / 1e9 / (mean/1000)
    gflops = 2.0 * INTER_DIM * HIDDEN_DIM / 1e9 / (mean / 1000.0)
    print_result(f"矩阵向量乘法 ({INTER_DIM}×{HIDDEN_DIM})", mean, std, f"{gflops:.2f} GFLOPS")
    return mean


def bench_paired_gate_up(x_hidden):
    """
    场景 3：配对 gate/up 点积（两次独立 matvec）
    - gate 权重：14336 × 7168
    - up 权重：14336 × 7168
    - 输入：(7168,) f32 向量
    - 操作：gate_matvec + up_matvec（无配对优化，独立计算）
    """
    gate_weight = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)
    up_weight = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)

    def run():
        gate_out = iq2xs_matvec(x_hidden, gate_weight)
        up_out = iq2xs_matvec(x_hidden, up_weight)
        return gate_out, up_out

    mean, std, _, _ = benchmark(run)
    # 吞吐量：2 次 matvec
    gflops = 2.0 * 2 * INTER_DIM * HIDDEN_DIM / 1e9 / (mean / 1000.0)
    print_result(f"配对 gate/up (2×matvec, {INTER_DIM}×{HIDDEN_DIM})", mean, std, f"{gflops:.2f} GFLOPS")
    return mean


def bench_full_ffn(x_hidden):
    """
    场景 4：完整 FFN（cpu_expert_ffn_pair）
    - gate 权重：14336 × 7168
    - up 权重：14336 × 7168
    - down 权重：7168 × 14336
    - 输入：(7168,) f32 向量
    - 操作：配对 gate/up + SwiGLU + down 投影
    """
    gate_weight = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)
    up_weight = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)
    down_weight = create_iq2xs_weight(HIDDEN_DIM, INTER_DIM)

    def run():
        return cpu_expert_ffn_pair(x_hidden, gate_weight, up_weight, down_weight, 1.0)

    mean, std, _, result = benchmark(run)
    # 吞吐量：FLOPS ≈ 2 * (14336*7168 + 14336*7168 + 7168*14336) = 2 * 3 * 14336 * 7168
    total_flops = 2.0 * 3 * INTER_DIM * HIDDEN_DIM  # 3 次 matvec
    gflops = total_flops / 1e9 / (mean / 1000.0)
    tokens_per_s = 1000.0 / mean
    print_result(f"完整 FFN (pair, {HIDDEN_DIM}→{INTER_DIM}→{HIDDEN_DIM})", mean, std,
                 f"{gflops:.2f} GFLOPS, {tokens_per_s:.1f} tok/s")
    return mean, result


def bench_dual_ffn(x_hidden):
    """
    场景 5：双专家 FFN（cpu_expert_ffn_pair_dual，MoE top-2）
    - 专家 A：gate_a + up_a + down_a
    - 专家 B：gate_b + up_b + down_b
    - 输入：(7168,) f32 向量
    - 操作：rayon::join 并行计算两个专家，加权聚合
    """
    gate_a = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)
    up_a = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)
    down_a = create_iq2xs_weight(HIDDEN_DIM, INTER_DIM)

    gate_b = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)
    up_b = create_iq2xs_weight(INTER_DIM, HIDDEN_DIM)
    down_b = create_iq2xs_weight(HIDDEN_DIM, INTER_DIM)

    def run():
        return cpu_expert_ffn_pair_dual(
            x_hidden,
            gate_a, up_a, down_a,
            gate_b, up_b, down_b,
            0.6, 0.4,
        )

    mean, std, _, result = benchmark(run)
    total_flops = 2.0 * 2 * 3 * INTER_DIM * HIDDEN_DIM  # 2 个专家，各 3 次 matvec
    gflops = total_flops / 1e9 / (mean / 1000.0)
    tokens_per_s = 1000.0 / mean
    print_result(f"双专家 FFN (dual, top-2)", mean, std,
                 f"{gflops:.2f} GFLOPS, {tokens_per_s:.1f} tok/s")
    return mean, result


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  CPU FFN 性能基准测试  —  DeepSeek V4 MoE 专家 (7168×14336)")
    print("=" * 80)

    # 1. 检测 SIMD 路径
    avx512, avx2, path_name = detect_simd_path()
    print(f"\n[SIMD 路径检测]")
    print(f"  AVX-512 VNNI : {'✓ 可用' if avx512 else '✗ 不可用'}")
    print(f"  AVX2          : {'✓ 可用' if avx2 else '✗ 不可用'}")
    print(f"  标量回退      : {'(仅在 AVX2 也不可用时)' if avx2 else '✓ 将使用'}")
    print(f"  → 当前路径    : {path_name}")

    # 2. 初始化查找表
    print(f"\n[查找表初始化]")
    init_lookup_tables()

    # 3. 构造输入向量
    print(f"\n[构造输入数据]")
    x_hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)
    print(f"  输入向量: ({HIDDEN_DIM},) float32, {x_hidden.nbytes / 1024:.1f} KB")

    # 4. 权重内存估算
    def weight_mem_mb(n_rows, n_cols):
        blocks = n_rows * (n_cols // BLOCK_SIZE)
        d_mb = blocks * 4 / 1024 / 1024
        qs_mb = blocks * 32 * 2 / 1024 / 1024
        scales_mb = blocks * 8 / 1024 / 1024
        return d_mb + qs_mb + scales_mb

    gate_up_mb = 2 * weight_mem_mb(INTER_DIM, HIDDEN_DIM)
    down_mb = weight_mem_mb(HIDDEN_DIM, INTER_DIM)
    total_mb = gate_up_mb + down_mb
    print(f"  权重内存估算:")
    print(f"    gate + up (2 × {INTER_DIM}×{HIDDEN_DIM}): {gate_up_mb:.1f} MB")
    print(f"    down     ({HIDDEN_DIM}×{INTER_DIM}):     {down_mb:.1f} MB")
    print(f"    单专家合计:                       {total_mb:.1f} MB")
    print(f"    双专家合计:                       {2 * total_mb:.1f} MB")

    # 5. 运行基准测试
    print(f"\n{'=' * 80}")
    print(f"  基准测试  (预热 {N_WARMUP} 次, 测量 {N_BENCH} 次)")
    print(f"  SIMD 路径: {path_name}")
    print(f"{'=' * 80}")
    print(f"\n  {'场景':<36s}  {'延迟':>16s}  {'吞吐量'}")
    print(f"  {'-' * 36}  {'-' * 16}  {'-' * 30}")

    bench_single_row_dot(x_hidden)
    bench_matvec(x_hidden)
    bench_paired_gate_up(x_hidden)
    ffn_mean, ffn_result = bench_full_ffn(x_hidden)
    dual_mean, dual_result = bench_dual_ffn(x_hidden)

    # 6. 结果验证
    print(f"\n[结果验证]")
    if ffn_result is not None:
        ffn_arr = np.asarray(ffn_result)
        print(f"  FFN 输出: shape={ffn_arr.shape}, "
              f"min={ffn_arr.min():.4f}, max={ffn_arr.max():.4f}, "
              f"mean={ffn_arr.mean():.4f}, finite={np.all(np.isfinite(ffn_arr))}")
    if dual_result is not None:
        dual_arr = np.asarray(dual_result)
        print(f"  Dual FFN 输出: shape={dual_arr.shape}, "
              f"min={dual_arr.min():.4f}, max={dual_arr.max():.4f}, "
              f"mean={dual_arr.mean():.4f}, finite={np.all(np.isfinite(dual_arr))}")

    # 7. 汇总
    print(f"\n{'=' * 80}")
    print(f"  汇总")
    print(f"{'=' * 80}")
    print(f"  SIMD 路径:       {path_name}")
    print(f"  单专家 FFN 延迟: {ffn_mean:.2f} ms  ({1000.0/ffn_mean:.1f} tok/s)")
    print(f"  双专家 FFN 延迟: {dual_mean:.2f} ms  ({1000.0/dual_mean:.1f} tok/s)")
    print(f"  权重维度:        {HIDDEN_DIM} × {INTER_DIM} (IQ2_XS 量化)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
