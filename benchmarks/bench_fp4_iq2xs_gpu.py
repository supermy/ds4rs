#!/usr/bin/env python3
"""四路对比 benchmark：FP4 FFN vs IQ2_XS FFN vs GPU FFN vs GPU Attention

对比维度：
  1. CPU FP4 FFN (Rust AVX-512, FMA 管线)
  2. CPU IQ2_XS FFN (Rust AVX-512, maddubs 管线)
  3. GPU FFN (CUDA, FP4 gemm)
  4. GPU Attention (CUDA, flash attention)

测试场景：DeepSeek-V3 MoE 专家推理
  dim = 7168, inter_dim = 18432
  n_experts_per_token = 8
  n_layers = 61 (MoE layers)
"""

import numpy as np
import time
import sys
import os

# CPU 测试
import ds4rs
from ds4rs import (
    init_tables, is_tables_initialized,
    is_avx512_supported, is_avx2_supported,
    Iq2XsWeight, Fp4Weight, CpuExpertRunner,
    cpu_expert_ffn_pair, cpu_expert_ffn_pair_fp4,
)


def init_lookup_tables():
    if is_tables_initialized():
        return
    grid_u64 = np.random.randint(0, 2**64, size=512, dtype=np.uint64)
    ksigns = np.random.randint(0, 256, size=128, dtype=np.uint8)
    init_tables(grid_u64.tolist(), ksigns.tolist())


init_lookup_tables()

# ============================================================================
# 辅助函数
# ============================================================================
def make_fp4_weight(out_dim, in_dim, seed=42):
    rng = np.random.RandomState(seed)
    packed = rng.randint(0, 256, size=(out_dim, in_dim // 2), dtype=np.uint8)
    # E8M0 scales: random exponents in reasonable range (scale ~ 2^-10 to 2^10)
    scale_exps = rng.randint(117, 138, size=(out_dim, in_dim // 32)).astype(np.uint8)
    return packed.flatten(), scale_exps.flatten(), (out_dim, in_dim)


def make_iq2xs_weight(out_dim, in_dim, seed=42):
    """构造 IQ2_XS 权重（使用合法 grid 索引）"""
    rng = np.random.RandomState(seed)
    n_blocks = out_dim * (in_dim // 256)
    d = rng.uniform(-1.0, 1.0, size=n_blocks).astype(np.float32)
    # qs: 每个 block 8 个 u16，合法范围 0-511 (grid 索引)
    qs = rng.randint(0, 512, size=n_blocks * 8, dtype=np.uint16)
    # scales: 每个 block 4 个 u8
    scales = rng.randint(0, 256, size=n_blocks * 4, dtype=np.uint8)
    return d, qs, scales, (out_dim, in_dim)


# ============================================================================
# DeepSeek-V3 实际尺寸
# ============================================================================
DIM = 7168
INTER_DIM = 18432
ROUTE_WEIGHT = 0.8

print(f"AVX-512: {is_avx512_supported()}")
print(f"AVX2:    {is_avx2_supported()}")
print(f"dim={DIM}, inter_dim={INTER_DIM}")
print()

# ============================================================================
# 1. CPU FP4 FFN benchmark
# ============================================================================
print("=" * 60)
print("1. CPU FP4 FFN (Rust AVX-512, FMA 管线)")
print("=" * 60)

gate_packed, gate_scales, gate_shape = make_fp4_weight(INTER_DIM, DIM, seed=1)
up_packed, up_scales, up_shape = make_fp4_weight(INTER_DIM, DIM, seed=2)
down_packed, down_scales, down_shape = make_fp4_weight(DIM, INTER_DIM, seed=3)

gate_w = Fp4Weight(gate_packed, gate_scales, gate_shape)
up_w = Fp4Weight(up_packed, up_scales, up_shape)
down_w = Fp4Weight(down_packed, down_scales, down_shape)

x = np.random.randn(DIM).astype(np.float32)

# Warmup
_ = cpu_expert_ffn_pair_fp4(x, gate_w, up_w, down_w, ROUTE_WEIGHT)

# Benchmark
n_iter = 10
t0 = time.perf_counter()
for _ in range(n_iter):
    _ = cpu_expert_ffn_pair_fp4(x, gate_w, up_w, down_w, ROUTE_WEIGHT)
fp4_time = (time.perf_counter() - t0) / n_iter

fp4_size = gate_w.size_bytes() + up_w.size_bytes() + down_w.size_bytes()
print(f"  延迟: {fp4_time*1000:.1f} ms/expert")
print(f"  权重: {fp4_size/1024/1024:.1f} MB")
print(f"  吞吐: {fp4_size/1024/1024/fp4_time:.0f} MB/s (权重读取)")
print()

# ============================================================================
# 2. CPU IQ2_XS FFN benchmark
# ============================================================================
print("=" * 60)
print("2. CPU IQ2_XS FFN (Rust AVX-512, maddubs 管线)")
print("=" * 60)

# IQ2_XS 需要真实量化数据，随机数据无法正确使用
# 使用之前 test_hot_cold_expert.py 的实测结果：2.7 ms/expert
iq2xs_time = 2.7e-3  # 实测值
iq2xs_size = (INTER_DIM * DIM * 74 // 256 * 3)
print(f"  延迟: {iq2xs_time*1000:.1f} ms/expert (实测)")
print(f"  权重: {iq2xs_size/1024/1024:.1f} MB (理论)")
print(f"  吞吐: {iq2xs_size/1024/1024/iq2xs_time:.0f} MB/s")
print(f"  注: IQ2_XS 需要真实量化数据，此处使用之前实测值")
print()

# ============================================================================
# 3. GPU FFN benchmark (需要模型加载)
# ============================================================================
print("=" * 60)
print("3. GPU FFN (CUDA, FP4 gemm)")
print("=" * 60)

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")

        # 模拟 GPU FFN：FP4 权重 + FP8 激活的 gemm
        # 实际 GPU 用 cutlass/cublas FP4 gemm，这里用 BF16 近似
        # gate: [inter_dim, dim], up: [inter_dim, dim], down: [dim, inter_dim]
        gate_bf16 = torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16, device='cuda')
        up_bf16 = torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16, device='cuda')
        down_bf16 = torch.randn(DIM, INTER_DIM, dtype=torch.bfloat16, device='cuda')
        x_gpu = torch.randn(1, DIM, dtype=torch.bfloat16, device='cuda')

        # Warmup
        for _ in range(5):
            g = torch.nn.functional.linear(x_gpu, gate_bf16)
            u = torch.nn.functional.linear(x_gpu, up_bf16)
            mid = torch.nn.functional.silu(g) * u
            _ = torch.nn.functional.linear(mid, down_bf16)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            g = torch.nn.functional.linear(x_gpu, gate_bf16)
            u = torch.nn.functional.linear(x_gpu, up_bf16)
            mid = torch.nn.functional.silu(g) * u
            _ = torch.nn.functional.linear(mid, down_bf16)
        torch.cuda.synchronize()
        gpu_ffn_time = (time.perf_counter() - t0) / n_iter

        gpu_bf16_size = (INTER_DIM * DIM * 2 + INTER_DIM * DIM * 2 + DIM * INTER_DIM * 2)
        print(f"  延迟: {gpu_ffn_time*1000:.2f} ms/expert (BF16 近似)")
        print(f"  权重: {gpu_bf16_size/1024/1024:.1f} MB (BF16)")
        print(f"  吞吐: {gpu_bf16_size/1024/1024/gpu_ffn_time:.0f} MB/s")
        print(f"  注: 实际 FP4 gemm 延迟约为 BF16 的 0.5-0.7x")
        has_gpu = True
    else:
        print("  CUDA 不可用")
        has_gpu = False
        gpu_ffn_time = None
except ImportError:
    print("  PyTorch 未安装")
    has_gpu = False
    gpu_ffn_time = None
print()

# ============================================================================
# 4. GPU Attention benchmark
# ============================================================================
print("=" * 60)
print("4. GPU Attention (CUDA, flash attention)")
print("=" * 60)

if has_gpu:
    # DeepSeek-V3 MLA attention 参数
    # n_heads=128, head_dim=128, seq_len=1 (prefill=1 token)
    n_heads = 128
    head_dim = 128
    seq_len = 1
    kv_len = 4096  # 假设 KV cache 长度

    q = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(1, n_heads, kv_len, head_dim, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(1, n_heads, kv_len, head_dim, dtype=torch.bfloat16, device='cuda')

    # Warmup
    for _ in range(5):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter * 10):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    gpu_attn_time = (time.perf_counter() - t0) / (n_iter * 10)

    print(f"  延迟: {gpu_attn_time*1000:.2f} ms (seq_len={seq_len}, kv_len={kv_len})")
    print(f"  n_heads={n_heads}, head_dim={head_dim}")
else:
    gpu_attn_time = None
    print("  CUDA 不可用")
print()

# ============================================================================
# 5. 综合对比
# ============================================================================
print("=" * 60)
print("5. 综合对比")
print("=" * 60)
print(f"  {'组件':<25} {'延迟':>10} {'权重':>10} {'相对CPU FP4':>12}")
print(f"  {'-'*57}")

print(f"  {'CPU FP4 FFN':<25} {fp4_time*1000:>8.1f}ms {fp4_size/1024/1024:>8.1f}MB {'1.00x':>12}")
print(f"  {'CPU IQ2_XS FFN':<25} {iq2xs_time*1000:>8.1f}ms {iq2xs_size/1024/1024:>8.1f}MB {iq2xs_time/fp4_time:>10.2f}x")
if gpu_ffn_time is not None:
    print(f"  {'GPU FFN (BF16)':<25} {gpu_ffn_time*1000:>8.2f}ms {gpu_bf16_size/1024/1024:>8.1f}MB {gpu_ffn_time/fp4_time:>10.2f}x")
if gpu_attn_time is not None:
    print(f"  {'GPU Attention':<25} {gpu_attn_time*1000:>8.2f}ms {'—':>10} {gpu_attn_time/fp4_time:>10.2f}x")

print()

# ============================================================================
# 6. 端到端推理分析
# ============================================================================
print("=" * 60)
print("6. 端到端推理分析 (DeepSeek-V3, 1 token)")
print("=" * 60)

n_moe_layers = 61
n_experts_per_token = 8
n_total_experts = 256

# GPU 热专家：top-8 在 GPU 上计算
# CPU 冷专家：miss 时走 CPU FFN
# 假设 GPU SLRU 命中率 90%（8/256 的专家在 GPU 上）
gpu_hit_rate = 0.9

# GPU attention 延迟（每层）
if gpu_attn_time is not None:
    attn_per_layer = gpu_attn_time
else:
    attn_per_layer = 0.1e-3  # 估计 0.1ms

# GPU FFN 延迟（热专家）
if gpu_ffn_time is not None:
    gpu_ffn_per_expert = gpu_ffn_time
else:
    gpu_ffn_per_expert = 0.05e-3  # 估计 0.05ms

# CPU FFN 延迟（冷专家）
cpu_ffn_per_expert = fp4_time

# 每层 MoE 延迟
# GPU: attention + n_experts_per_token × gpu_ffn (并行)
# CPU: (1-gpu_hit_rate) × n_experts_per_token × cpu_ffn (串行)
gpu_ffn_per_layer = gpu_ffn_per_expert  # 8 个专家并行，取最慢的
cpu_miss_per_layer = (1 - gpu_hit_rate) * n_experts_per_token * cpu_ffn_per_expert

layer_time = attn_per_layer + max(gpu_ffn_per_layer, cpu_miss_per_layer)
total_time = layer_time * n_moe_layers

print(f"  GPU Attention:     {attn_per_layer*1000:.2f} ms/layer")
print(f"  GPU FFN (热专家):  {gpu_ffn_per_expert*1000:.2f} ms/expert (并行)")
print(f"  CPU FP4 FFN (冷):  {cpu_ffn_per_expert*1000:.1f} ms/expert (串行)")
print(f"  GPU SLRU 命中率:   {gpu_hit_rate*100:.0f}%")
print(f"  每层 CPU miss:     {cpu_miss_per_layer*1000:.1f} ms ({(1-gpu_hit_rate)*n_experts_per_token:.1f} 冷专家)")
print(f"  每层总延迟:        {layer_time*1000:.1f} ms")
print(f"  端到端 ({n_moe_layers} MoE层): {total_time*1000:.0f} ms = {total_time:.1f}s")
print(f"  吞吐:              {1/total_time:.1f} tokens/s")
print()

# 对比：纯 GPU vs GPU+CPU
pure_gpu_layer = attn_per_layer + gpu_ffn_per_expert
pure_gpu_total = pure_gpu_layer * n_moe_layers
print(f"  纯 GPU (无 CPU 兜底): {pure_gpu_total*1000:.0f} ms = {pure_gpu_total:.1f}s, {1/pure_gpu_total:.1f} tokens/s")
print(f"  GPU+CPU 混合:        {total_time*1000:.0f} ms = {total_time:.1f}s, {1/total_time:.1f} tokens/s")
print(f"  混合/纯GPU:          {total_time/pure_gpu_total:.2f}x")
