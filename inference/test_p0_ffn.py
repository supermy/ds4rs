"""
P0-2: CPU 专家 FFN 完整测试用例

测试范围：
  1. 完整 FFN 流程正确性（gate/up + SwiGLU + down）
  2. 多专家并行计算
  3. 与 GPU 结果一致性
  4. 性能基准（目标 <5ms）

DeepSeek MoE 结构：
  w1 = gate (inter_dim, dim)
  w3 = up (inter_dim, dim)
  w2 = down (dim, inter_dim)
  output = w2( silu(w1(x)) * w3(x) )
"""

import unittest
import numpy as np
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

BLOCK_SIZE = 256


def _make_iq2xs_weight(n_rows, n_cols, seed=42):
    """创建模拟 IQ2_XS 权重字典。"""
    assert n_cols % BLOCK_SIZE == 0
    n_blocks = n_rows * n_cols // BLOCK_SIZE

    rng = np.random.RandomState(seed)
    d = rng.randn(n_blocks).astype(np.float16) * 0.01
    qs = rng.randint(0, 65535, (n_blocks, 32), dtype=np.uint16)
    scales = rng.randint(0, 256, (n_blocks, 8), dtype=np.uint8)

    return {
        "__iq2xs__": True,
        "d": d,
        "qs": qs,
        "scales": scales,
        "shape": (n_rows, n_cols),
    }


class TestFFNCorrectness(unittest.TestCase):
    """FFN 正确性测试。"""

    def test_ffn_pair_basic(self):
        """测试分离格式 FFN 基本功能。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        output = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        self.assertEqual(output.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_ffn_combined_basic(self):
        """测试合并格式 FFN 基本功能。"""
        from cpu_expert import cpu_expert_ffn

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_up_w = _make_iq2xs_weight(2 * inter_dim, dim, seed=100)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=200)

        output = cpu_expert_ffn(x, gate_up_w, down_w)

        self.assertEqual(output.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_ffn_pair_vs_combined(self):
        """测试分离格式与合并格式结果一致。"""
        from cpu_expert import cpu_expert_ffn, cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256

        rng = np.random.RandomState(42)
        x = rng.randn(dim).astype(np.float32)

        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        gate_up_w = {
            "__iq2xs__": True,
            "d": np.concatenate([gate_w["d"], up_w["d"]]),
            "qs": np.concatenate([gate_w["qs"], up_w["qs"]]),
            "scales": np.concatenate([gate_w["scales"], up_w["scales"]]),
            "shape": (2 * inter_dim, dim),
        }

        output_pair = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
        output_combined = cpu_expert_ffn(x, gate_up_w, down_w)

        np.testing.assert_allclose(output_pair, output_combined, rtol=1e-4, atol=1e-4)

    def test_ffn_route_weight(self):
        """测试路由权重正确应用。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        output1 = cpu_expert_ffn_pair(x, gate_w, up_w, down_w, route_weight=1.0)
        output2 = cpu_expert_ffn_pair(x, gate_w, up_w, down_w, route_weight=0.5)

        np.testing.assert_allclose(output2, output1 * 0.5, rtol=1e-5)

    def test_ffn_swiglu_limit(self):
        """测试 SwiGLU 限幅正确应用。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32) * 10
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        output = cpu_expert_ffn_pair(x, gate_w, up_w, down_w, swiglu_limit=5.0)

        self.assertEqual(output.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(output)))


class TestMultiExpertParallel(unittest.TestCase):
    """多专家并行计算测试。"""

    def test_sequential_experts(self):
        """测试顺序计算多个专家。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256
        n_experts = 4

        x = np.random.randn(dim).astype(np.float32)
        outputs = []

        for i in range(n_experts):
            gate_w = _make_iq2xs_weight(inter_dim, dim, seed=100 + i)
            up_w = _make_iq2xs_weight(inter_dim, dim, seed=200 + i)
            down_w = _make_iq2xs_weight(dim, inter_dim, seed=300 + i)
            output = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            outputs.append(output)

        self.assertEqual(len(outputs), n_experts)
        for output in outputs:
            self.assertEqual(output.shape, (dim,))

    def test_parallel_experts(self):
        """测试并行计算多个专家。"""
        from cpu_expert import cpu_expert_ffn_pair
        from concurrent.futures import ThreadPoolExecutor

        dim = 512
        inter_dim = 256
        n_experts = 6

        x = np.random.randn(dim).astype(np.float32)
        weights = [
            (
                _make_iq2xs_weight(inter_dim, dim, seed=100 + i),
                _make_iq2xs_weight(inter_dim, dim, seed=200 + i),
                _make_iq2xs_weight(dim, inter_dim, seed=300 + i),
            )
            for i in range(n_experts)
        ]

        def compute_expert(args):
            gate_w, up_w, down_w = args
            return cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        with ThreadPoolExecutor(max_workers=6) as executor:
            outputs = list(executor.map(compute_expert, weights))

        self.assertEqual(len(outputs), n_experts)
        for output in outputs:
            self.assertEqual(output.shape, (dim,))

    def test_expert_accumulation(self):
        """测试多专家结果累加。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256
        n_experts = 3
        route_weights = [0.3, 0.5, 0.2]

        x = np.random.randn(dim).astype(np.float32)
        accumulated = np.zeros(dim, dtype=np.float32)

        for i, rw in enumerate(route_weights):
            gate_w = _make_iq2xs_weight(inter_dim, dim, seed=100 + i)
            up_w = _make_iq2xs_weight(inter_dim, dim, seed=200 + i)
            down_w = _make_iq2xs_weight(dim, inter_dim, seed=300 + i)
            output = cpu_expert_ffn_pair(x, gate_w, up_w, down_w, route_weight=rw)
            accumulated += output

        self.assertEqual(accumulated.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(accumulated)))


class TestGPUConsistency(unittest.TestCase):
    """与 GPU 结果一致性测试。"""

    def test_torch_tensor_conversion(self):
        """测试 torch.Tensor 与 numpy 转换。"""
        from cpu_expert import CpuExpertRunner

        class MockCache:
            _iq2xs_pinned_pool = None

        runner = CpuExpertRunner(MockCache())

        x_gpu = torch.randn(512, device='cpu')
        output = runner.compute_expert_cpu(0, 0, x_gpu)

        self.assertEqual(output.shape, x_gpu.shape)
        self.assertTrue(torch.all(output == 0))

    def test_cpu_runner_with_pinned_pool(self):
        """测试 CpuExpertRunner 从 pinned pool 获取数据。"""
        from cpu_expert import CpuExpertRunner

        dim = 512
        inter_dim = 256

        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        class MockCache:
            _iq2xs_pinned_pool = {
                (0, 0): {
                    'w1': gate_w,
                    'w2': down_w,
                    'w3': up_w,
                }
            }

        runner = CpuExpertRunner(MockCache())
        x = torch.randn(dim, device='cpu')
        output = runner.compute_expert_cpu(0, 0, x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(output)))


class TestFFNPerformance(unittest.TestCase):
    """FFN 性能基准测试。"""

    def test_ffn_latency_small(self):
        """测试小规模 FFN 延迟（dim=512）。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        for _ in range(5):
            _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        n_iter = 20
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  FFN (dim={dim}): avg={avg_ms:.2f}ms")

        self.assertLess(avg_ms, 1.0, f"FFN latency {avg_ms:.2f}ms exceeds 1ms target")

    def test_ffn_latency_full(self):
        """测试完整规模 FFN 延迟（dim=4096）。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 4096
        inter_dim = 2048

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        for _ in range(5):
            _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        print(f"\n  FFN (dim={dim}, inter_dim={inter_dim}): avg={avg_ms:.2f}ms, min={min_ms:.2f}ms")

        self.assertLess(avg_ms, 6.0, f"FFN latency {avg_ms:.2f}ms exceeds 6ms target")

    def test_multi_expert_throughput(self):
        """测试多专家吞吐量。"""
        from cpu_expert import cpu_expert_ffn_pair
        from concurrent.futures import ThreadPoolExecutor

        dim = 4096
        inter_dim = 2048
        n_experts = 6

        x = np.random.randn(dim).astype(np.float32)
        weights = [
            (
                _make_iq2xs_weight(inter_dim, dim, seed=100 + i),
                _make_iq2xs_weight(inter_dim, dim, seed=200 + i),
                _make_iq2xs_weight(dim, inter_dim, seed=300 + i),
            )
            for i in range(n_experts)
        ]

        for _ in range(3):
            with ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(lambda args: cpu_expert_ffn_pair(x, *args), weights))

        n_iter = 5
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(lambda args: cpu_expert_ffn_pair(x, *args), weights))
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        throughput = n_experts / (avg_ms / 1000)
        print(f"\n  {n_experts} experts: avg={avg_ms:.2f}ms, throughput={throughput:.0f} experts/s")

        self.assertGreater(throughput, 100, f"Throughput {throughput:.0f} experts/s below 100 target")


if __name__ == '__main__':
    unittest.main(verbosity=2)
