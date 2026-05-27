#!/usr/bin/env python3
"""
冷热专家推理测试

重点验证 CpuExpertRunner SLRU 缓存机制：
  1. 冷专家（首次访问）vs 热专家（缓存命中）延迟对比
  2. SLRU 晋升：probation → protected
  3. 缓存淘汰：容量满时 LRU 淘汰冷专家
  4. 双专家 FFN：冷/热/混合组合延迟
  5. 命中率统计
  6. Step 保护：当前 step 专家不被淘汰
  7. 层保护：当前层专家不被淘汰

运行方式：
  docker exec ds4rs-dev bash -c \
    "source ~/.cargo/env && cd /workspace && \
     maturin develop --release -- -C target-cpu=native && \
     python benchmarks/test_hot_cold_expert.py"
"""

import numpy as np
import time
import sys

HIDDEN_DIM = 7168
INTER_DIM = 14336
BLOCK_SIZE = 256

try:
    from ds4rs import (
        init_tables,
        is_tables_initialized,
        is_avx512_supported,
        is_avx2_supported,
        Iq2XsWeight,
        CpuExpertRunner,
        cpu_expert_ffn_pair,
    )
except ImportError:
    print("错误：无法导入 ds4rs 模块")
    print("请先运行：maturin develop --release -- -C target-cpu=native")
    sys.exit(1)


def init_lookup_tables():
    if is_tables_initialized():
        return
    grid_u64 = np.random.randint(0, 2**64, size=512, dtype=np.uint64)
    ksigns = np.random.randint(0, 256, size=128, dtype=np.uint8)
    init_tables(grid_u64.tolist(), ksigns.tolist())


def create_weight(n_rows, n_cols, seed=42):
    assert n_cols % BLOCK_SIZE == 0
    rng = np.random.RandomState(seed)
    total_blocks = n_rows * (n_cols // BLOCK_SIZE)
    d = rng.randn(total_blocks).astype(np.float32)
    grid_idx = rng.randint(0, 512, size=total_blocks * 32, dtype=np.uint16)
    sign_idx = rng.randint(0, 128, size=total_blocks * 32, dtype=np.uint16)
    qs = (grid_idx | (sign_idx << 9)).astype(np.uint16)
    scales = rng.randint(0, 256, size=total_blocks * 8, dtype=np.uint8)
    return Iq2XsWeight(d, qs, scales, (n_rows, n_cols))


def create_expert_weights(expert_id, layer_id=0):
    """创建一个专家的三组权重（gate/up/down）"""
    base_seed = expert_id * 1000 + layer_id * 100
    gate = create_weight(INTER_DIM, HIDDEN_DIM, seed=base_seed + 1)
    up = create_weight(INTER_DIM, HIDDEN_DIM, seed=base_seed + 2)
    down = create_weight(HIDDEN_DIM, INTER_DIM, seed=base_seed + 3)
    return gate, up, down


def bench_single(fn, n_warmup=2, n_iter=5):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return arr.mean(), arr.min(), arr.max(), result


def test_cold_vs_hot():
    """测试 1：冷专家 vs 热专家延迟对比"""
    print("\n" + "=" * 80)
    print("  测试 1：冷专家 vs 热专家延迟")
    print("=" * 80)

    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # 添加 1 个专家
    gate, up, down = create_expert_weights(0)
    runner.add_expert(0, 0, gate, up, down)

    # 冷专家：首次计算（权重在 probation 段，无 L3 预热）
    cold_mean, cold_min, cold_max, _ = bench_single(
        lambda: runner.compute_expert(0, 0, x, 1.0), n_warmup=0, n_iter=5
    )

    # 热专家：连续计算（权重已晋升到 protected 段，L3 预热）
    hot_mean, hot_min, hot_max, _ = bench_single(
        lambda: runner.compute_expert(0, 0, x, 1.0), n_warmup=3, n_iter=10
    )

    print(f"  冷专家（首次访问）: avg={cold_mean:.2f}ms, min={cold_min:.2f}ms, max={cold_max:.2f}ms")
    print(f"  热专家（缓存命中）: avg={hot_mean:.2f}ms, min={hot_min:.2f}ms, max={hot_max:.2f}ms")
    print(f"  冷/热比: {cold_mean/hot_mean:.2f}x")
    print(f"  命中率: {runner.hit_rate():.1%}")
    print(f"  内存: {runner.memory_usage_mb():.1f} MB")
    print(f"  专家数: {runner.expert_count()}")


def test_slru_promotion():
    """测试 2：SLRU 晋升 probation → protected"""
    print("\n" + "=" * 80)
    print("  测试 2：SLRU 晋升（probation → protected）")
    print("=" * 80)

    # 小容量缓存，方便观察晋升
    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # 添加 3 个专家（默认 protected_capacity=100, probation_capacity=12）
    for eid in range(3):
        gate, up, down = create_expert_weights(eid)
        runner.add_expert(0, eid, gate, up, down)

    print(f"  初始状态: expert_count={runner.expert_count()}")

    # 访问 expert 0 → 应晋升到 protected
    result = runner.compute_expert(0, 0, x, 1.0)
    assert result is not None, "expert 0 应存在"
    print(f"  访问 expert 0 后: hit_rate={runner.hit_rate():.1%}")

    # 再次访问 expert 0 → 应在 protected 段命中
    result = runner.compute_expert(0, 0, x, 1.0)
    assert result is not None, "expert 0 应仍在缓存"
    print(f"  再次访问 expert 0: hit_rate={runner.hit_rate():.1%}")

    # 访问 expert 1 → 也应晋升
    result = runner.compute_expert(0, 1, x, 1.0)
    assert result is not None, "expert 1 应存在"
    print(f"  访问 expert 1 后: hit_rate={runner.hit_rate():.1%}")

    # expert 2 未被访问，仍在 probation
    result = runner.compute_expert(0, 2, x, 1.0)
    assert result is not None, "expert 2 应存在"
    print(f"  访问 expert 2 后: hit_rate={runner.hit_rate():.1%}")
    print(f"  最终: expert_count={runner.expert_count()}, memory={runner.memory_usage_mb():.1f} MB")


def test_eviction():
    """测试 3：缓存淘汰（容量满时 LRU 淘汰冷专家）"""
    print("\n" + "=" * 80)
    print("  测试 3：缓存淘汰（容量满时淘汰冷专家）")
    print("=" * 80)

    # 创建小容量缓存：protected=2, probation=2
    # 但 CpuExpertRunner::new() 默认 protected=100, probation=12
    # 我们通过不断添加专家来测试淘汰
    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # 添加 15 个专家（超过默认 probation_capacity=12）
    # 注意：add_expert 在容量满时会淘汰 probation 段最冷专家
    for eid in range(15):
        gate, up, down = create_expert_weights(eid)
        runner.add_expert(0, eid, gate, up, down)

    actual_count = runner.expert_count()
    print(f"  添加 15 个专家后: expert_count={actual_count}（容量限制导致部分淘汰）")
    print(f"  内存: {runner.memory_usage_mb():.1f} MB")

    # 检查哪些专家仍在缓存
    surviving = [eid for eid in range(15) if runner.has_expert(0, eid)]
    evicted = [eid for eid in range(15) if not runner.has_expert(0, eid)]
    print(f"  存活的专家: {surviving}")
    print(f"  被淘汰的专家: {evicted}")

    # 访问存活的专家，让它们晋升到 protected
    for eid in surviving[:3]:
        result = runner.compute_expert(0, eid, x, 1.0)
        assert result is not None, f"expert {eid} 应存在"

    print(f"  访问存活专家后: hit_rate={runner.hit_rate():.1%}")


def test_dual_expert():
    """测试 4：双专家 FFN 冷/热/混合组合"""
    print("\n" + "=" * 80)
    print("  测试 4：双专家 FFN 冷/热/混合组合")
    print("=" * 80)

    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # 添加 2 个专家
    for eid in range(2):
        gate, up, down = create_expert_weights(eid)
        runner.add_expert(0, eid, gate, up, down)

    # 冷双专家：首次计算
    cold_mean, cold_min, _, _ = bench_single(
        lambda: runner.compute_dual_expert(0, 0, 1, x, 0.6, 0.4),
        n_warmup=0, n_iter=5
    )

    # 热双专家：连续计算
    hot_mean, hot_min, _, _ = bench_single(
        lambda: runner.compute_dual_expert(0, 0, 1, x, 0.6, 0.4),
        n_warmup=3, n_iter=10
    )

    print(f"  冷双专家（首次）: avg={cold_mean:.2f}ms, min={cold_min:.2f}ms")
    print(f"  热双专家（缓存）: avg={hot_mean:.2f}ms, min={hot_min:.2f}ms")
    print(f"  冷/热比: {cold_mean/hot_mean:.2f}x")
    print(f"  命中率: {runner.hit_rate():.1%}")

    # 与独立计算对比
    runner2 = CpuExpertRunner()
    for eid in range(2):
        gate, up, down = create_expert_weights(eid)
        runner2.add_expert(0, eid, gate, up, down)

    # 预热
    runner2.compute_expert(0, 0, x, 1.0)
    runner2.compute_expert(0, 1, x, 1.0)

    # 顺序计算两个专家
    seq_mean, seq_min, _, _ = bench_single(
        lambda: (
            runner2.compute_expert(0, 0, x, 0.6),
            runner2.compute_expert(0, 1, x, 0.4),
        ),
        n_warmup=3, n_iter=10
    )

    print(f"  顺序两次单专家: avg={seq_mean:.2f}ms, min={seq_min:.2f}ms")
    print(f"  dual/sequential 比: {hot_mean/seq_mean:.2f}x")


def test_step_protection():
    """测试 5：Step 保护（当前 step 专家不被淘汰）"""
    print("\n" + "=" * 80)
    print("  测试 5：Step 保护")
    print("=" * 80)

    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # 添加 15 个专家
    for eid in range(15):
        gate, up, down = create_expert_weights(eid)
        runner.add_expert(0, eid, gate, up, down)

    # 设置 step 保护：保护 expert 0 和 expert 1
    runner.set_step_protected({(0, 0), (0, 1)})
    print(f"  设置 step 保护: expert 0, 1")

    # 访问存活的专家
    surviving = [eid for eid in range(15) if runner.has_expert(0, eid)]
    for eid in surviving:
        result = runner.compute_expert(0, eid, x, 1.0)

    print(f"  访问存活专家后: expert_count={runner.expert_count()}, hit_rate={runner.hit_rate():.1%}")

    # 验证 step 保护的专家仍在
    if runner.has_expert(0, 0):
        print(f"  step 保护的 expert 0 仍在缓存 ✓")
    else:
        print(f"  step 保护的 expert 0 被淘汰（可能在添加阶段就被淘汰了）")
    if runner.has_expert(0, 1):
        print(f"  step 保护的 expert 1 仍在缓存 ✓")
    else:
        print(f"  step 保护的 expert 1 被淘汰（可能在添加阶段就被淘汰了）")

    # 清除 step 保护
    runner.clear_step_protected()
    print(f"  清除 step 保护")


def test_layer_protection():
    """测试 6：层保护（当前层专家不被淘汰）"""
    print("\n" + "=" * 80)
    print("  测试 6：层保护")
    print("=" * 80)

    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # 添加 2 层的专家
    for layer in range(2):
        for eid in range(8):
            gate, up, down = create_expert_weights(eid, layer_id=layer)
            runner.add_expert(layer, eid, gate, up, down)

    print(f"  添加 2 层 × 8 专家: expert_count={runner.expert_count()}")

    # 保护 layer 0
    runner.set_protected_layer(0)
    print(f"  设置层保护: layer 0")

    # 访问存活的专家
    for layer in range(2):
        for eid in range(8):
            if runner.has_expert(layer, eid):
                result = runner.compute_expert(layer, eid, x, 1.0)

    print(f"  访问后: expert_count={runner.expert_count()}, hit_rate={runner.hit_rate():.1%}")

    # layer 0 的存活专家应仍在
    surviving_l0 = [eid for eid in range(8) if runner.has_expert(0, eid)]
    print(f"  layer 0 存活专家: {surviving_l0}")

    runner.set_protected_layer(None)
    print(f"  清除层保护")


def test_freq_persistence():
    """测试 7：LFU 频率持久化"""
    print("\n" + "=" * 80)
    print("  测试 7：LFU 频率持久化")
    print("=" * 80)

    import tempfile
    import os

    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # 添加 3 个专家
    for eid in range(3):
        gate, up, down = create_expert_weights(eid)
        runner.add_expert(0, eid, gate, up, down)

    # 不同频率访问
    for _ in range(10):
        runner.compute_expert(0, 0, x, 1.0)  # 高频
    for _ in range(3):
        runner.compute_expert(0, 1, x, 1.0)  # 中频
    runner.compute_expert(0, 2, x, 1.0)      # 低频

    print(f"  访问频率: expert 0=10次, expert 1=3次, expert 2=1次")
    print(f"  命中率: {runner.hit_rate():.1%}")

    # 保存频率
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        freq_path = f.name
    runner.save_freq(freq_path)
    print(f"  频率已保存到: {freq_path}")

    # 加载频率到新 runner
    runner2 = CpuExpertRunner()
    runner2.load_freq(freq_path)
    print(f"  频率已加载到新 runner")

    # 清理
    os.unlink(freq_path)
    print(f"  临时文件已清理")


def test_repeated_inference():
    """测试 8：重复推理稳定性（热专家连续推理）"""
    print("\n" + "=" * 80)
    print("  测试 8：重复推理稳定性")
    print("=" * 80)

    runner = CpuExpertRunner()
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)

    gate, up, down = create_expert_weights(0)
    runner.add_expert(0, 0, gate, up, down)

    # 预热
    for _ in range(3):
        runner.compute_expert(0, 0, x, 1.0)

    # 连续推理 20 次
    results = []
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        result = runner.compute_expert(0, 0, x, 1.0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        if result is not None:
            results.append(np.array(result))

    times = np.array(times)
    print(f"  20 次推理延迟:")
    print(f"    avg={times.mean():.2f}ms, min={times.min():.2f}ms, max={times.max():.2f}ms")
    print(f"    std={times.std():.2f}ms, p95={np.percentile(times, 95):.2f}ms")

    # 验证结果一致性
    if len(results) >= 2:
        max_diff = np.max(np.abs(results[0] - results[-1]))
        print(f"  首尾结果最大差异: {max_diff:.6f}")
        assert max_diff < 1e-5, f"结果不一致: max_diff={max_diff}"
        print(f"  结果一致性 ✓")

    print(f"  命中率: {runner.hit_rate():.1%}")


def test_simd_path():
    """测试 0：SIMD 路径检测"""
    print("\n" + "=" * 80)
    print("  测试 0：SIMD 路径检测")
    print("=" * 80)

    avx512 = is_avx512_supported()
    avx2 = is_avx2_supported()

    if avx512:
        path = "AVX-512 VNNI (512-bit maddubs+madd)"
    elif avx2:
        path = "AVX2 (256-bit maddubs+madd)"
    else:
        path = "Scalar (纯标量回退)"

    print(f"  AVX-512 VNNI: {'可用' if avx512 else '不可用'}")
    print(f"  AVX2:          {'可用' if avx2 else '不可用'}")
    print(f"  当前路径:      {path}")


def main():
    print("=" * 80)
    print("  冷热专家推理测试  —  CpuExpertRunner SLRU 缓存")
    print(f"  权重维度: {HIDDEN_DIM} × {INTER_DIM} (IQ2_XS)")
    print("=" * 80)

    init_lookup_tables()
    print("  查找表初始化完成")

    test_simd_path()
    test_cold_vs_hot()
    test_slru_promotion()
    test_eviction()
    test_dual_expert()
    test_step_protection()
    test_layer_protection()
    test_freq_persistence()
    test_repeated_inference()

    print("\n" + "=" * 80)
    print("  所有测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
