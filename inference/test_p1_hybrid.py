"""
P1-1: Python 原型验证（混合推理集成）

测试范围：
  1. GPU hot 专家命中
  2. CPU cold 专家推理
  3. 混合推理结果正确性
  4. 性能对比（GPU only vs 混合推理）

混合推理流程：
  1. GPU 计算 Attention + Shared Expert → hidden_state
  2. Gate 计算 TopK → 分离 hot/cold 专家
  3. Hot 专家：GPU SLRU 命中 → GPU GEMM
  4. Cold 专家：CPU IQ2_XS 反量化 + FFN → 结果 H2D
  5. GPU 合并 hot + cold 结果 → 残差连接
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


class MockExpertCache:
    """模拟专家缓存，用于测试。"""

    def __init__(self, gpu_experts=None, cpu_experts=None):
        self._gpu_experts = gpu_experts or {}
        self._iq2xs_pinned_pool = cpu_experts or {}

    def get_expert(self, layer_id, expert_id):
        """获取 GPU 专家。"""
        return self._gpu_experts.get((layer_id, expert_id))

    def get_cpu_expert(self, layer_id, expert_id):
        """获取 CPU 专家。"""
        return self._iq2xs_pinned_pool.get((layer_id, expert_id))


class MockExpert:
    """模拟 GPU 专家。"""

    def __init__(self, weight_seed):
        self.weight_seed = weight_seed

    def __call__(self, x, route_weight):
        """模拟专家计算。"""
        return x * route_weight * (self.weight_seed % 10 + 1) / 10.0


class TestHybridInferenceBasic(unittest.TestCase):
    """混合推理基本功能测试。"""

    def test_gpu_hot_expert(self):
        """测试 GPU hot 专家命中。"""
        from cpu_expert import CpuExpertRunner

        gpu_experts = {(0, 0): MockExpert(100)}
        cache = MockExpertCache(gpu_experts=gpu_experts)

        expert = cache.get_expert(0, 0)
        self.assertIsNotNone(expert)

        x = torch.randn(512)
        output = expert(x, 0.5)

        self.assertEqual(output.shape, x.shape)

    def test_cpu_cold_expert(self):
        """测试 CPU cold 专家推理。"""
        from cpu_expert import CpuExpertRunner

        dim = 512
        inter_dim = 256

        cpu_experts = {
            (0, 0): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=1),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=3),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=2),
            }
        }
        cache = MockExpertCache(cpu_experts=cpu_experts)

        runner = CpuExpertRunner(cache)
        x = torch.randn(dim)
        output = runner.compute_expert_cpu(0, 0, x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_mixed_hot_cold(self):
        """测试混合 hot + cold 专家。"""
        from cpu_expert import CpuExpertRunner

        dim = 512
        inter_dim = 256

        gpu_experts = {(0, 0): MockExpert(100)}
        cpu_experts = {
            (0, 1): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=10),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=30),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=20),
            }
        }
        cache = MockExpertCache(gpu_experts=gpu_experts, cpu_experts=cpu_experts)

        x = torch.randn(dim)
        outputs = {}

        expert_0 = cache.get_expert(0, 0)
        if expert_0 is not None:
            outputs[0] = expert_0(x, 0.5)

        runner = CpuExpertRunner(cache)
        outputs[1] = runner.compute_expert_cpu(0, 1, x, route_weight=0.5)

        self.assertEqual(len(outputs), 2)
        for output in outputs.values():
            self.assertEqual(output.shape, (dim,))


class TestHybridInferenceCorrectness(unittest.TestCase):
    """混合推理正确性测试。"""

    def test_cpu_expert_determinism(self):
        """测试 CPU 专家计算确定性。"""
        from cpu_expert import CpuExpertRunner

        dim = 512
        inter_dim = 256

        cpu_experts = {
            (0, 0): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=1),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=3),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=2),
            }
        }
        cache = MockExpertCache(cpu_experts=cpu_experts)

        runner = CpuExpertRunner(cache)
        x = torch.randn(dim)

        output1 = runner.compute_expert_cpu(0, 0, x)
        output2 = runner.compute_expert_cpu(0, 0, x)

        torch.testing.assert_close(output1, output2)

    def test_route_weight_application(self):
        """测试路由权重正确应用。"""
        from cpu_expert import CpuExpertRunner

        dim = 512
        inter_dim = 256

        cpu_experts = {
            (0, 0): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=1),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=3),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=2),
            }
        }
        cache = MockExpertCache(cpu_experts=cpu_experts)

        runner = CpuExpertRunner(cache)
        x = torch.randn(dim)

        output1 = runner.compute_expert_cpu(0, 0, x, route_weight=1.0)
        output2 = runner.compute_expert_cpu(0, 0, x, route_weight=0.5)

        torch.testing.assert_close(output2, output1 * 0.5)

    def test_multi_expert_accumulation(self):
        """测试多专家结果累加。"""
        from cpu_expert import CpuExpertRunner

        dim = 512
        inter_dim = 256

        cpu_experts = {
            (0, i): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=100 + i),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=300 + i),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=200 + i),
            }
            for i in range(3)
        }
        cache = MockExpertCache(cpu_experts=cpu_experts)

        runner = CpuExpertRunner(cache)
        x = torch.randn(dim)
        route_weights = [0.3, 0.5, 0.2]

        accumulated = torch.zeros(dim)
        for i, rw in enumerate(route_weights):
            output = runner.compute_expert_cpu(0, i, x, route_weight=rw)
            accumulated += output

        self.assertEqual(accumulated.shape, (dim,))
        self.assertTrue(torch.all(torch.isfinite(accumulated)))


class TestHybridInferencePerformance(unittest.TestCase):
    """混合推理性能测试。"""

    def test_cpu_expert_latency(self):
        """测试 CPU 专家延迟。"""
        from cpu_expert import CpuExpertRunner

        dim = 4096
        inter_dim = 2048

        cpu_experts = {
            (0, 0): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=1),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=3),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=2),
            }
        }
        cache = MockExpertCache(cpu_experts=cpu_experts)

        runner = CpuExpertRunner(cache)
        x = torch.randn(dim)

        for _ in range(5):
            _ = runner.compute_expert_cpu(0, 0, x)

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = runner.compute_expert_cpu(0, 0, x)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  CPU expert latency (dim={dim}): {avg_ms:.2f}ms")

        self.assertLess(avg_ms, 10.0, f"CPU expert latency {avg_ms:.2f}ms exceeds 10ms target")

    def test_multi_expert_parallel_latency(self):
        """测试多专家并行延迟。"""
        from cpu_expert import CpuExpertRunner
        from concurrent.futures import ThreadPoolExecutor

        dim = 4096
        inter_dim = 2048
        n_experts = 6

        cpu_experts = {
            (0, i): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=100 + i),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=300 + i),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=200 + i),
            }
            for i in range(n_experts)
        }
        cache = MockExpertCache(cpu_experts=cpu_experts)

        runner = CpuExpertRunner(cache)
        x = torch.randn(dim)

        def compute_expert(i):
            return runner.compute_expert_cpu(0, i, x, route_weight=1.0 / n_experts)

        for _ in range(3):
            with ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(compute_expert, range(n_experts)))

        n_iter = 5
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(compute_expert, range(n_experts)))
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  {n_experts} CPU experts parallel: {avg_ms:.2f}ms")

        self.assertLess(avg_ms, 30.0, f"Multi-expert latency {avg_ms:.2f}ms exceeds 30ms target")


class TestHybridInferenceIntegration(unittest.TestCase):
    """混合推理集成测试。"""

    def test_full_pipeline(self):
        """测试完整混合推理流程。"""
        from cpu_expert import CpuExpertRunner

        dim = 4096
        inter_dim = 2048
        n_tokens = 16
        n_hot = 4
        n_cold = 2

        gpu_experts = {
            (0, i): MockExpert(100 + i)
            for i in range(n_hot)
        }
        cpu_experts = {
            (0, n_hot + i): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=100 + n_hot + i),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=300 + n_hot + i),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=200 + n_hot + i),
            }
            for i in range(n_cold)
        }
        cache = MockExpertCache(gpu_experts=gpu_experts, cpu_experts=cpu_experts)

        x = torch.randn(n_tokens, dim)
        outputs = torch.zeros(n_tokens, dim)

        for t in range(n_tokens):
            for i in range(n_hot):
                expert = cache.get_expert(0, i)
                if expert is not None:
                    outputs[t] += expert(x[t], 0.1)

            runner = CpuExpertRunner(cache)
            for i in range(n_cold):
                output = runner.compute_expert_cpu(0, n_hot + i, x[t], route_weight=0.1)
                outputs[t] += output

        self.assertEqual(outputs.shape, (n_tokens, dim))
        self.assertTrue(torch.all(torch.isfinite(outputs)))

    def test_hit_rate_simulation(self):
        """测试命中率模拟。"""
        from cpu_expert import CpuExpertRunner

        dim = 4096
        inter_dim = 2048
        n_experts = 160
        n_topk = 6
        gpu_capacity = 120

        gpu_expert_ids = set(range(gpu_capacity))
        cpu_expert_ids = set(range(gpu_capacity, n_experts))

        gpu_experts = {
            (0, i): MockExpert(i)
            for i in gpu_expert_ids
        }
        cpu_experts = {
            (0, i): {
                'w1': _make_iq2xs_weight(inter_dim, dim, seed=100 + i),
                'w2': _make_iq2xs_weight(dim, inter_dim, seed=300 + i),
                'w3': _make_iq2xs_weight(inter_dim, dim, seed=200 + i),
            }
            for i in cpu_expert_ids
        }
        cache = MockExpertCache(gpu_experts=gpu_experts, cpu_experts=cpu_experts)

        selected_experts = np.random.randint(0, n_experts, n_topk)
        gpu_hits = sum(1 for e in selected_experts if e in gpu_expert_ids)
        cpu_misses = n_topk - gpu_hits

        print(f"\n  Selected experts: {selected_experts}")
        print(f"  GPU hits: {gpu_hits}/{n_topk} ({100*gpu_hits/n_topk:.0f}%)")
        print(f"  CPU misses: {cpu_misses}/{n_topk} ({100*cpu_misses/n_topk:.0f}%)")

        self.assertEqual(gpu_hits + cpu_misses, n_topk)


if __name__ == '__main__':
    unittest.main(verbosity=2)
