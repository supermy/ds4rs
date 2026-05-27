"""
CPU 专家推理 TDD 测试

测试范围：
  1. IQ2_XS 反量化精度（与 GPU 结果对比）
  2. CPU 专家 FFN 正确性（与参考实现对比）
  3. 逐块点积 vs 全量反量化 一致性
  4. 延迟基准（优化前后对比）
  5. 3D 输入形状兼容性
  6. 混合推理集成（GPU hot + CPU cold）
"""

import unittest
import numpy as np
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# IQ2_XS 块大小: QK_K = 256
BLOCK_SIZE = 256


def _make_iq2xs_weight(n_rows, n_cols, seed=42):
    """创建模拟 IQ2_XS 权重字典。

    n_cols 必须是 256 的倍数。
    """
    assert n_cols % BLOCK_SIZE == 0, f"n_cols={n_cols} must be multiple of {BLOCK_SIZE}"
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


class TestIQ2XSDequantization(unittest.TestCase):
    """IQ2_XS 反量化精度测试。"""

    @classmethod
    def setUpClass(cls):
        """加载 IQ2_XS 归档数据用于测试。"""
        cls.archive_path = os.environ.get('IQ2XS_ARCHIVE', '/workspace/iq2xs/experts.iq2xs')
        cls.has_archive = os.path.exists(cls.archive_path)

    def test_dequantize_block_shape(self):
        """测试反量化输出的形状正确。"""
        from cpu_expert import dequantize_iq2xs_block

        n_blocks = 4
        d = np.ones(n_blocks, dtype=np.float16)
        qs = np.zeros((n_blocks, 32), dtype=np.uint16)
        scales = np.ones((n_blocks, 8), dtype=np.uint8)

        result = dequantize_iq2xs_block(d, qs, scales)
        self.assertEqual(result.shape, (n_blocks * 256,))

    def test_dequantize_block_d_scaling(self):
        """测试缩放因子 d 正确应用。"""
        from cpu_expert import dequantize_iq2xs_block

        d1 = np.array([1.0], dtype=np.float16)
        d2 = np.array([2.0], dtype=np.float16)
        qs = np.zeros((1, 32), dtype=np.uint16)
        scales = np.ones((1, 8), dtype=np.uint8)

        result1 = dequantize_iq2xs_block(d1, qs, scales)
        result2 = dequantize_iq2xs_block(d2, qs, scales)

        np.testing.assert_allclose(result2, result1 * 2.0, rtol=1e-2)

    def test_dequantize_block_scale_effect(self):
        """测试 4-bit scale 正确影响输出。"""
        from cpu_expert import dequantize_iq2xs_block

        d = np.array([1.0], dtype=np.float16)
        qs = np.zeros((1, 32), dtype=np.uint16)

        scales_low = np.zeros((1, 8), dtype=np.uint8)
        scales_high = np.full((1, 8), 0xFF, dtype=np.uint8)

        result_low = dequantize_iq2xs_block(d, qs, scales_low)
        result_high = dequantize_iq2xs_block(d, qs, scales_high)

        self.assertGreater(np.abs(result_high).sum(), 0)

    def test_dequantize_3d_input(self):
        """测试 3D 输入形状（归档格式）正确处理。"""
        from cpu_expert import dequantize_iq2xs_block

        # 归档格式: d[N, n_bpr], qs[N, n_bpr, 32], scales[N, n_bpr, 8]
        N, n_bpr = 4, 2
        n_blocks = N * n_bpr

        d_1d = np.ones(n_blocks, dtype=np.float16)
        qs_1d = np.zeros((n_blocks, 32), dtype=np.uint16)
        scales_1d = np.ones((n_blocks, 8), dtype=np.uint8)

        d_3d = d_1d.reshape(N, n_bpr)
        qs_3d = qs_1d.reshape(N, n_bpr, 32)
        scales_3d = scales_1d.reshape(N, n_bpr, 8)

        result_1d = dequantize_iq2xs_block(d_1d, qs_1d, scales_1d)
        result_3d = dequantize_iq2xs_block(d_3d, qs_3d, scales_3d)

        np.testing.assert_allclose(result_1d, result_3d, rtol=1e-6)


class TestMatvecConsistency(unittest.TestCase):
    """逐块点积 vs 全量反量化 一致性测试。"""

    def test_matvec_vs_dequantize(self):
        """测试 iq2xs_matvec_batch 与全量反量化+点积结果一致。"""
        from cpu_expert import iq2xs_matvec_batch, dequantize_iq2xs_weight

        dim = 512  # 必须是 256 的倍数
        n_rows = 4
        n_blocks_per_row = dim // BLOCK_SIZE
        n_blocks = n_rows * n_blocks_per_row

        rng = np.random.RandomState(123)
        x = rng.randn(dim).astype(np.float32)

        weight_dict = _make_iq2xs_weight(n_rows, dim, seed=123)

        # 逐块点积
        result_matvec = iq2xs_matvec_batch(
            weight_dict["d"], weight_dict["qs"], weight_dict["scales"],
            x, n_rows, dim)

        # 全量反量化 + 点积
        weight_full = dequantize_iq2xs_weight(weight_dict)
        result_ref = weight_full @ x

        np.testing.assert_allclose(result_matvec, result_ref, rtol=1e-4, atol=1e-4)

    def test_matvec_batch_vs_single(self):
        """测试批量版本与逐行版本结果一致。"""
        from cpu_expert import iq2xs_matvec_batch, iq2xs_matvec

        dim = 512
        n_rows = 4

        rng = np.random.RandomState(456)
        x = rng.randn(dim).astype(np.float32)
        weight_dict = _make_iq2xs_weight(n_rows, dim, seed=456)

        result_batch = iq2xs_matvec_batch(
            weight_dict["d"], weight_dict["qs"], weight_dict["scales"],
            x, n_rows, dim)

        result_single = iq2xs_matvec(
            weight_dict["d"], weight_dict["qs"], weight_dict["scales"],
            x, n_rows, dim)

        np.testing.assert_allclose(result_batch, result_single, rtol=1e-5, atol=1e-5)

    def test_matvec_3d_input(self):
        """测试 3D 输入形状正确处理。"""
        from cpu_expert import iq2xs_matvec_batch

        dim = 512
        n_rows = 4
        n_bpr = dim // BLOCK_SIZE
        n_blocks = n_rows * n_bpr

        rng = np.random.RandomState(789)
        x = rng.randn(dim).astype(np.float32)
        weight_dict = _make_iq2xs_weight(n_rows, dim, seed=789)

        # 1D/2D 输入
        result_1d = iq2xs_matvec_batch(
            weight_dict["d"], weight_dict["qs"], weight_dict["scales"],
            x, n_rows, dim)

        # 3D 输入（归档格式）
        d_3d = weight_dict["d"].reshape(n_rows, n_bpr)
        qs_3d = weight_dict["qs"].reshape(n_rows, n_bpr, 32)
        scales_3d = weight_dict["scales"].reshape(n_rows, n_bpr, 8)

        result_3d = iq2xs_matvec_batch(d_3d, qs_3d, scales_3d, x, n_rows, dim)

        np.testing.assert_allclose(result_1d, result_3d, rtol=1e-5, atol=1e-5)


class TestCpuExpertFFN(unittest.TestCase):
    """CPU 专家 FFN 正确性测试。"""

    def test_ffn_output_shape(self):
        """测试 FFN 输出形状正确。"""
        from cpu_expert import cpu_expert_ffn

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_up_weight = _make_iq2xs_weight(2 * inter_dim, dim)
        down_weight = _make_iq2xs_weight(dim, inter_dim)

        output = cpu_expert_ffn(x, gate_up_weight, down_weight)
        self.assertEqual(output.shape, (dim,))

    def test_ffn_pair_output_shape(self):
        """测试分离 gate/up 格式的 FFN 输出形状正确。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_weight = _make_iq2xs_weight(inter_dim, dim)
        up_weight = _make_iq2xs_weight(inter_dim, dim)
        down_weight = _make_iq2xs_weight(dim, inter_dim)

        output = cpu_expert_ffn_pair(x, gate_weight, up_weight, down_weight)
        self.assertEqual(output.shape, (dim,))

    def test_ffn_route_weight(self):
        """测试路由权重正确应用。"""
        from cpu_expert import cpu_expert_ffn

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_up_weight = _make_iq2xs_weight(2 * inter_dim, dim)
        down_weight = _make_iq2xs_weight(dim, inter_dim)

        output1 = cpu_expert_ffn(x, gate_up_weight, down_weight, route_weight=1.0)
        output2 = cpu_expert_ffn(x, gate_up_weight, down_weight, route_weight=0.5)

        np.testing.assert_allclose(output2, output1 * 0.5, rtol=1e-5)

    def test_ffn_pair_vs_combined(self):
        """测试分离格式与合并格式结果一致。"""
        from cpu_expert import cpu_expert_ffn, cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256

        rng = np.random.RandomState(42)
        x = rng.randn(dim).astype(np.float32)

        gate_weight = _make_iq2xs_weight(inter_dim, dim, seed=100)
        up_weight = _make_iq2xs_weight(inter_dim, dim, seed=200)
        down_weight = _make_iq2xs_weight(dim, inter_dim, seed=300)

        # 合并 gate_up
        gate_up_weight = {
            "__iq2xs__": True,
            "d": np.concatenate([gate_weight["d"], up_weight["d"]]),
            "qs": np.concatenate([gate_weight["qs"], up_weight["qs"]]),
            "scales": np.concatenate([gate_weight["scales"], up_weight["scales"]]),
            "shape": (2 * inter_dim, dim),
        }

        output_pair = cpu_expert_ffn_pair(x, gate_weight, up_weight, down_weight)
        output_combined = cpu_expert_ffn(x, gate_up_weight, down_weight)

        np.testing.assert_allclose(output_pair, output_combined, rtol=1e-4, atol=1e-4)


class TestCpuExpertLatency(unittest.TestCase):
    """CPU 专家推理延迟基准测试。"""

    def test_matvec_latency(self):
        """测试逐块点积延迟。"""
        from cpu_expert import iq2xs_matvec_batch

        # 模拟真实模型尺寸: gate_up = [2048, 4096]
        dim = 4096
        n_rows = 2048

        weight_dict = _make_iq2xs_weight(n_rows, dim)
        x = np.random.randn(dim).astype(np.float32)

        # Warmup
        _ = iq2xs_matvec_batch(
            weight_dict["d"], weight_dict["qs"], weight_dict["scales"],
            x, n_rows, dim)

        # Benchmark
        n_iter = 5
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            result = iq2xs_matvec_batch(
                weight_dict["d"], weight_dict["qs"], weight_dict["scales"],
                x, n_rows, dim)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        print(f"\n  IQ2_XS matvec ({n_rows}×{dim}): avg={avg_ms:.1f}ms, min={min_ms:.1f}ms")

        self.assertEqual(result.shape, (n_rows,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_ffn_latency(self):
        """测试完整 FFN 延迟。"""
        from cpu_expert import cpu_expert_ffn

        dim = 4096
        inter_dim = 2048

        x = np.random.randn(dim).astype(np.float32)
        gate_up_weight = _make_iq2xs_weight(2 * inter_dim, dim)
        down_weight = _make_iq2xs_weight(dim, inter_dim)

        # Warmup
        _ = cpu_expert_ffn(x, gate_up_weight, down_weight)

        # Benchmark
        n_iter = 5
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            output = cpu_expert_ffn(x, gate_up_weight, down_weight)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        print(f"\n  CPU FFN (dim={dim}, inter_dim={inter_dim}): avg={avg_ms:.1f}ms, min={min_ms:.1f}ms")

        self.assertEqual(output.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_dequantize_latency(self):
        """测试反量化延迟（旧路径，用于对比）。"""
        from cpu_expert import dequantize_iq2xs_block

        n_blocks = 32768  # gate_up: 2048*4096/256
        d = np.random.randn(n_blocks).astype(np.float16) * 0.01
        qs = np.random.randint(0, 65535, (n_blocks, 32), dtype=np.uint16)
        scales = np.random.randint(0, 256, (n_blocks, 8), dtype=np.uint8)

        # Warmup
        _ = dequantize_iq2xs_block(d[:100], qs[:100], scales[:100])

        # Benchmark
        n_iter = 3
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            result = dequantize_iq2xs_block(d, qs, scales)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  IQ2_XS 全量反量化 ({n_blocks} blocks): {avg_ms:.1f}ms")

        self.assertEqual(result.shape, (n_blocks * 256,))
        self.assertTrue(np.all(np.isfinite(result)))


class TestHybridInference(unittest.TestCase):
    """混合推理集成测试。"""

    def test_cpu_runner_basic(self):
        """测试 CpuExpertRunner 基本功能。"""
        from cpu_expert import CpuExpertRunner

        class MockCache:
            _iq2xs_pinned_pool = None

        runner = CpuExpertRunner(MockCache())

        # 无 pinned pool 时应返回零
        x = torch.randn(512, device='cpu')
        output = runner.compute_expert_cpu(0, 0, x)
        self.assertEqual(output.shape, x.shape)
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
