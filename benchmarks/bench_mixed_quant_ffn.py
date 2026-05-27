"""混合量化 CPU FFN vs GPU FFN 基准测试。

对比 IQ2_XXS+Q2_K 混合量化下 CPU FFN (AVX-512) 和 GPU FFN (TileLang) 的延迟。

用法：
  docker exec ds4rs-dev bash -c "cd /workspace && python benchmarks/bench_mixed_quant_ffn.py"

输出：
  - 单专家 FFN 延迟对比
  - 6 专家串行延迟对比
  - GPU 缓存命中率模拟
  - 理论推理速度估算
"""
import sys
import time
import torch
import numpy as np

sys.path.insert(0, '/workspace/inference')
sys.path.insert(0, '/workspace/tilelang')


def bench_cpu_ffn(mixed_pool, n_iters=10):
    """基准测试 CPU FFN 延迟。"""
    print("\n" + "=" * 60)
    print("CPU FFN 基准测试 (AVX-512, Rust)")
    print("=" * 60)

    # 随机选择几个专家测试
    test_keys = [(0, 0), (0, 1), (0, 2), (10, 0), (20, 0), (42, 255)]
    x = np.random.randn(4096).astype(np.float32)

    latencies = []
    for layer_id, expert_id in test_keys:
        try:
            # 预热
            mixed_pool.compute_ffn(layer_id, expert_id, x, 1.0, 0.0)

            # 计时
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                out = mixed_pool.compute_ffn(layer_id, expert_id, x, 1.0, 0.0)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            avg = np.mean(times)
            latencies.append(avg)
            print(f"  L{layer_id}E{expert_id}: {avg:.2f}ms (std={np.std(times):.2f}ms)")
        except Exception as e:
            print(f"  L{layer_id}E{expert_id}: ERROR {e}")

    if latencies:
        print(f"\n  平均: {np.mean(latencies):.2f}ms")
        print(f"  6 专家串行: {np.mean(latencies) * 6:.2f}ms")


def bench_gpu_ffn(mixed_pool, n_iters=20):
    """基准测试 GPU FFN 延迟。"""
    print("\n" + "=" * 60)
    print("GPU FFN 基准测试 (TileLang mixed_quant_gemm)")
    print("=" * 60)

    test_keys = [(0, 0), (0, 1), (0, 2), (10, 0), (20, 0), (42, 255)]
    x_gpu = torch.randn(4096, dtype=torch.bfloat16, device='cuda')

    latencies = []
    for layer_id, expert_id in test_keys:
        # 上传到 GPU
        try:
            mixed_pool.upload_to_gpu_cache(layer_id, expert_id)
        except Exception:
            continue

        if not mixed_pool.gpu_cache_contains(layer_id, expert_id):
            continue

        try:
            # 预热（JIT 编译）
            x_2d = x_gpu.unsqueeze(0)
            mixed_pool.compute_ffn_gpu(layer_id, expert_id, x_2d, 1.0)
            torch.cuda.synchronize()

            # 计时
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                out = mixed_pool.compute_ffn_gpu(layer_id, expert_id, x_2d, 1.0)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            avg = np.mean(times)
            latencies.append(avg)
            print(f"  L{layer_id}E{expert_id}: {avg:.2f}ms (std={np.std(times):.2f}ms)")
        except Exception as e:
            print(f"  L{layer_id}E{expert_id}: ERROR {e}")

    if latencies:
        print(f"\n  平均: {np.mean(latencies):.2f}ms")
        print(f"  6 专家串行: {np.mean(latencies) * 6:.2f}ms")


def estimate_throughput(cpu_ms, gpu_ms, hit_rate=0.78):
    """估算推理速度。"""
    print("\n" + "=" * 60)
    print("推理速度估算")
    print("=" * 60)

    attn_ms = 2.0
    shared_ms = 1.0

    for hr in [0.0, 0.5, 0.78, 1.0]:
        eff_expert_ms = hr * gpu_ms + (1 - hr) * cpu_ms
        layer_ms = attn_ms + 6 * eff_expert_ms + shared_ms
        step_ms = layer_ms * 43
        tps = 1000 / step_ms if step_ms > 0 else 0
        print(f"  GPU 命中率 {hr:.0%}: "
              f"专家={eff_expert_ms:.2f}ms, "
              f"层={layer_ms:.1f}ms, "
              f"步={step_ms:.0f}ms, "
              f"速度={tps:.2f} t/s")


def main():
    gguf_path = "/workspace/gguf/experts_iq2xxs_q2k.gguf"
    import os
    if not os.path.exists(gguf_path):
        print(f"GGUF 文件不存在: {gguf_path}")
        return

    from rust_cpu_expert import MixedQuantExpertPool

    # 初始化（不启用 GPU FFN，手动测试）
    pool = MixedQuantExpertPool(gguf_path, gpu_ffn=True, gpu_cache_capacity=5000)

    # CPU FFN 基准
    bench_cpu_ffn(pool, n_iters=5)

    # GPU FFN 基准
    bench_gpu_ffn(pool, n_iters=20)

    # 打印 GPU 缓存统计
    pool.print_gpu_stats()

    # 估算推理速度
    estimate_throughput(cpu_ms=4.6, gpu_ms=1.2)

    # 保存频率数据
    pool.save_freq()


if __name__ == "__main__":
    main()
