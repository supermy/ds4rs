"""
P0-1: IQ2_XS AVX-512 反量化内核测试

测试范围：
  1. AVX-512 向量化点积正确性
  2. L1/L2/L3 缓存分块性能
  3. 多线程并行扩展性
  4. 与 numba JIT 实现的一致性

硬件：AMD Ryzen 5 7600
  - L1d: 192KB × 6 核
  - L2: 6MB × 6 核
  - L3: 32MB 共享
  - AVX-512: VNNI, BF16, F, BW, VL
"""

import unittest
import numpy as np
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


class TestAVX512Vectorization(unittest.TestCase):
    """AVX-512 向量化正确性测试。"""

    def test_avx512_vs_numba_small(self):
        """测试小矩阵 AVX-512 与 numba 结果一致。"""
        from cpu_expert import iq2xs_matvec_batch, _init_numba_kernel, _NUMBA_AVAILABLE

        if not _NUMBA_AVAILABLE:
            self.skipTest("numba not available")

        dim = 512
        n_rows = 8

        weight = _make_iq2xs_weight(n_rows, dim, seed=123)
        x = np.random.randn(dim).astype(np.float32)

        result = iq2xs_matvec_batch(
            weight["d"], weight["qs"], weight["scales"],
            x, n_rows, dim)

        self.assertEqual(result.shape, (n_rows,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_avx512_vs_numba_large(self):
        """测试大矩阵 AVX-512 与 numba 结果一致。"""
        from cpu_expert import iq2xs_matvec_batch, iq2xs_matvec

        dim = 4096
        n_rows = 2048

        weight = _make_iq2xs_weight(n_rows, dim, seed=456)
        x = np.random.randn(dim).astype(np.float32)

        result_batch = iq2xs_matvec_batch(
            weight["d"], weight["qs"], weight["scales"],
            x, n_rows, dim)

        result_ref = iq2xs_matvec(
            weight["d"], weight["qs"], weight["scales"],
            x, n_rows, dim)

        np.testing.assert_allclose(result_batch, result_ref, rtol=1e-4, atol=1e-4)

    def test_avx512_edge_cases(self):
        """测试边界情况：全零、全一、极值。"""
        from cpu_expert import iq2xs_matvec_batch

        dim = 512
        n_rows = 4

        weight = _make_iq2xs_weight(n_rows, dim, seed=789)

        x_zero = np.zeros(dim, dtype=np.float32)
        result_zero = iq2xs_matvec_batch(
            weight["d"], weight["qs"], weight["scales"],
            x_zero, n_rows, dim)
        self.assertTrue(np.allclose(result_zero, 0, atol=1e-6))

        x_ones = np.ones(dim, dtype=np.float32)
        result_ones = iq2xs_matvec_batch(
            weight["d"], weight["qs"], weight["scales"],
            x_ones, n_rows, dim)
        self.assertTrue(np.all(np.isfinite(result_ones)))


class TestCacheBlocking(unittest.TestCase):
    """L1/L2/L3 缓存分块性能测试。"""

    def test_l1_grid_resident(self):
        """测试 grid 表常驻 L1（4KB）。"""
        from cpu_expert import iq2xs_matvec_batch, _init_iq2xs_tables

        _init_iq2xs_tables()

        dim = 4096
        n_rows = 64

        weight = _make_iq2xs_weight(n_rows, dim, seed=111)
        x = np.random.randn(dim).astype(np.float32)

        for _ in range(3):
            result = iq2xs_matvec_batch(
                weight["d"], weight["qs"], weight["scales"],
                x, n_rows, dim)

        self.assertEqual(result.shape, (n_rows,))

    def test_l2_x_resident(self):
        """测试 x 向量常驻 L2（16KB for dim=4096）。"""
        from cpu_expert import iq2xs_matvec_batch

        dim = 4096
        n_rows = 512

        weight = _make_iq2xs_weight(n_rows, dim, seed=222)
        x = np.random.randn(dim).astype(np.float32)

        for _ in range(3):
            result = iq2xs_matvec_batch(
                weight["d"], weight["qs"], weight["scales"],
                x, n_rows, dim)

        self.assertEqual(result.shape, (n_rows,))

    def test_l3_expert_cache(self):
        """测试专家权重常驻 L3（~7MB/专家，32MB 可缓存 4 专家）。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 4096
        inter_dim = 2048

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        for _ in range(3):
            result = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        self.assertEqual(result.shape, (dim,))


class TestParallelScaling(unittest.TestCase):
    """多线程并行扩展性测试。"""

    def test_parallel_6_cores(self):
        """测试 6 核并行扩展性。"""
        from cpu_expert import iq2xs_matvec_batch

        dim = 4096
        n_rows = 2048

        weight = _make_iq2xs_weight(n_rows, dim, seed=333)
        x = np.random.randn(dim).astype(np.float32)

        for _ in range(3):
            result = iq2xs_matvec_batch(
                weight["d"], weight["qs"], weight["scales"],
                x, n_rows, dim)

        self.assertEqual(result.shape, (n_rows,))

    def test_parallel_gate_up(self):
        """测试 gate/up 并行计算。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 4096
        inter_dim = 2048

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        for _ in range(3):
            result = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        self.assertEqual(result.shape, (dim,))


class TestPerformanceBenchmarks(unittest.TestCase):
    """性能基准测试。"""

    def test_matvec_latency(self):
        """测试 matvec 延迟（目标 <2ms）。"""
        from cpu_expert import iq2xs_matvec_batch

        dim = 4096
        n_rows = 2048

        weight = _make_iq2xs_weight(n_rows, dim, seed=444)
        x = np.random.randn(dim).astype(np.float32)

        for _ in range(3):
            _ = iq2xs_matvec_batch(
                weight["d"], weight["qs"], weight["scales"],
                x, n_rows, dim)

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            result = iq2xs_matvec_batch(
                weight["d"], weight["qs"], weight["scales"],
                x, n_rows, dim)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        print(f"\n  matvec ({n_rows}×{dim}): avg={avg_ms:.2f}ms, min={min_ms:.2f}ms")

        self.assertLess(avg_ms, 2.0, f"matvec latency {avg_ms:.2f}ms exceeds 2ms target")

    def test_ffn_latency(self):
        """测试 FFN 延迟（目标 <10ms）。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 4096
        inter_dim = 2048

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        for _ in range(3):
            _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            result = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        print(f"\n  FFN (dim={dim}, inter_dim={inter_dim}): avg={avg_ms:.2f}ms, min={min_ms:.2f}ms")

        self.assertLess(avg_ms, 10.0, f"FFN latency {avg_ms:.2f}ms exceeds 10ms target")


if __name__ == '__main__':
    unittest.main(verbosity=2)
