"""
Rust 扩展集成测试

测试范围：
  1. Rust vs Python numba 正确性
  2. Rust vs Python numba 性能对比
  3. AVX-512 检测
  4. 多线程并行扩展性
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

BLOCK_SIZE = 256


def _make_iq2xs_weight(n_rows, n_cols, seed=42):
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


class TestRustAvailability(unittest.TestCase):
    def test_rust_available(self):
        from rust_cpu_expert import is_rust_available
        print(f"\n  Rust extension available: {is_rust_available()}")

    def test_avx512_detection(self):
        try:
            from ds4rs import is_avx512_supported
            print(f"\n  AVX-512 supported: {is_avx512_supported()}")
        except ImportError:
            print("\n  Rust extension not available, skipping AVX-512 detection")


class TestRustCorrectness(unittest.TestCase):
    def test_rust_vs_python_small(self):
        from cpu_expert import cpu_expert_ffn_pair
        from rust_cpu_expert import is_rust_available, rust_cpu_expert_ffn_pair

        if not is_rust_available():
            self.skipTest("Rust extension not available")

        dim = 512
        inter_dim = 256

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        py_result = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
        rust_result = rust_cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        self.assertIsNotNone(rust_result)
        np.testing.assert_allclose(py_result, rust_result, rtol=1e-4, atol=1e-4)

    def test_rust_vs_python_large(self):
        from cpu_expert import cpu_expert_ffn_pair
        from rust_cpu_expert import is_rust_available, rust_cpu_expert_ffn_pair

        if not is_rust_available():
            self.skipTest("Rust extension not available")

        dim = 4096
        inter_dim = 2048

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        py_result = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
        rust_result = rust_cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        self.assertIsNotNone(rust_result)
        np.testing.assert_allclose(py_result, rust_result, rtol=1e-4, atol=1e-4)


class TestRustPerformance(unittest.TestCase):
    def test_python_ffn_latency(self):
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
        print(f"\n  Python (numba) FFN: {avg_ms:.2f}ms")

    def test_rust_ffn_latency(self):
        from rust_cpu_expert import is_rust_available, rust_cpu_expert_ffn_pair

        if not is_rust_available():
            self.skipTest("Rust extension not available")

        dim = 4096
        inter_dim = 2048

        x = np.random.randn(dim).astype(np.float32)
        gate_w = _make_iq2xs_weight(inter_dim, dim, seed=1)
        up_w = _make_iq2xs_weight(inter_dim, dim, seed=2)
        down_w = _make_iq2xs_weight(dim, inter_dim, seed=3)

        for _ in range(5):
            _ = rust_cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = rust_cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Rust FFN: {avg_ms:.2f}ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)
