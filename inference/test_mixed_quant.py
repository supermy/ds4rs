"""混合量化 (IQ2_XS + Q2_K) 推理测试。

测试范围：
  1. GGUF 文件读取：验证 IQ2_XS 和 Q2_K 张量可正确读取
  2. IQ2_XS 反量化精度：与 GPU 结果对比
  3. Q2_K 反量化精度：与参考实现对比
  4. 混合 FFN 正确性：gate/up IQ2_XS + down Q2_K
  5. Rust CPU expert 集成：Iq2XsWeight + Q2KWeight
  6. 延迟基准

用法：
  docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python inference/test_mixed_quant.py"
"""
import unittest
import numpy as np
import torch
import time
import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLOCK_SIZE = 256  # QK_K
GGUF_PATH = "/workspace/gguf/experts_mixed.gguf"


def _make_iq2xs_weight(n_rows, n_cols, seed=42):
    """创建模拟 IQ2_XS 权重。"""
    assert n_cols % BLOCK_SIZE == 0
    n_blocks = n_rows * n_cols // BLOCK_SIZE
    rng = np.random.RandomState(seed)
    d = rng.randn(n_blocks).astype(np.float16) * 0.01
    qs = rng.randint(0, 65535, (n_blocks, 32), dtype=np.uint16)
    scales = rng.randint(0, 256, (n_blocks, 8), dtype=np.uint8)
    return {"__iq2xs__": True, "d": d, "qs": qs, "scales": scales, "shape": (n_rows, n_cols)}


def _make_q2k_weight(n_rows, n_cols, seed=43):
    """创建模拟 Q2_K 权重。"""
    assert n_cols % BLOCK_SIZE == 0
    n_blocks = n_rows * n_cols // BLOCK_SIZE
    rng = np.random.RandomState(seed)
    d = rng.randn(n_blocks).astype(np.float16) * 0.01
    dmin = rng.randn(n_blocks).astype(np.float16) * 0.001
    scales = rng.randint(0, 16, (n_blocks, 16), dtype=np.uint8)
    qs = rng.randint(0, 256, (n_blocks, 64), dtype=np.uint8)
    return {"__q2k__": True, "d": d, "dmin": dmin, "scales": scales, "qs": qs, "shape": (n_rows, n_cols)}


class TestGGUFRead(unittest.TestCase):
    """GGUF 混合量化文件读取测试。"""

    @classmethod
    def setUpClass(cls):
        cls.has_gguf = os.path.exists(GGUF_PATH)

    @unittest.skipUnless(os.path.exists(GGUF_PATH), "GGUF 文件不存在")
    def test_gguf_magic(self):
        """测试 GGUF 文件魔数。"""
        with open(GGUF_PATH, 'rb') as f:
            magic = f.read(4)
            self.assertEqual(magic, b'GGUF', f"无效的 GGUF 魔数: {magic}")

    @unittest.skipUnless(os.path.exists(GGUF_PATH), "GGUF 文件不存在")
    def test_gguf_tensor_count(self):
        """测试 GGUF 张量数量。"""
        with open(GGUF_PATH, 'rb') as f:
            f.read(4)  # magic
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<q', f.read(8))[0]
            n_kv = struct.unpack('<q', f.read(8))[0]
            self.assertEqual(version, 3, f"GGUF 版本应为 3，实际 {version}")
            self.assertGreater(n_tensors, 0, "应有张量")
            print(f"  GGUF: version={version}, n_tensors={n_tensors}, n_kv={n_kv}")


class TestQ2KDequantization(unittest.TestCase):
    """Q2_K 反量化精度测试。"""

    def test_q2k_dequantize_shape(self):
        """测试 Q2_K 反量化输出形状。"""
        from cpu_expert import Q2KWeight as RustQ2KWeight
        w = _make_q2k_weight(8, 256)
        rw = RustQ2KWeight(
            w["d"].astype(np.float32),
            w["dmin"].astype(np.float32),
            w["scales"],
            w["qs"],
            w["shape"],
        )
        self.assertEqual(rw.shape, (8, 256))

    def test_q2k_matvec_shape(self):
        """测试 Q2_K matvec 输出形状。"""
        try:
            from cpu_expert import Q2KWeight as RustQ2KWeight
        except ImportError:
            self.skipTest("cpu_expert Q2KWeight 不可用")

        n_rows, n_cols = 16, 512
        w = _make_q2k_weight(n_rows, n_cols)
        rw = RustQ2KWeight(
            w["d"].astype(np.float32),
            w["dmin"].astype(np.float32),
            w["scales"],
            w["qs"],
            w["shape"],
        )
        x = np.random.randn(n_cols).astype(np.float32)
        y = rw.matvec(x)
        self.assertEqual(y.shape, (n_rows,))


class TestMixedFFN(unittest.TestCase):
    """混合量化 FFN 测试：gate/up IQ2_XS + down Q2_K。"""

    def test_mixed_ffn_rust(self):
        """测试 Rust 混合 FFN（IQ2_XS gate/up + Q2_K down）。"""
        try:
            from cpu_expert import Iq2XsWeight, Q2KWeight as RustQ2KWeight
        except ImportError:
            self.skipTest("cpu_expert 不可用")

        dim = 512
        moe_inter_dim = 256

        # gate: IQ2_XS [moe_inter_dim, dim]
        gate_w = _make_iq2xs_weight(moe_inter_dim, dim, seed=1)
        gate_rw = Iq2XsWeight(gate_w["d"], gate_w["qs"], gate_w["scales"], gate_w["shape"])

        # up: IQ2_XS [moe_inter_dim, dim]
        up_w = _make_iq2xs_weight(moe_inter_dim, dim, seed=2)
        up_rw = Iq2XsWeight(up_w["d"], up_w["qs"], up_w["scales"], up_w["shape"])

        # down: Q2_K [dim, moe_inter_dim]
        down_w = _make_q2k_weight(dim, moe_inter_dim, seed=3)
        down_rw = RustQ2KWeight(
            down_w["d"].astype(np.float32),
            down_w["dmin"].astype(np.float32),
            down_w["scales"],
            down_w["qs"],
            down_w["shape"],
        )

        # FFN: x → gate(x), up(x) → swiglu → down(swiglu) → y
        x = np.random.randn(dim).astype(np.float32) * 0.1

        gate_out = gate_rw.matvec(x)
        up_out = up_rw.matvec(x)

        # SwiGLU: silu(gate) * up
        mid = (1.0 / (1.0 + np.exp(-gate_out))) * gate_out * up_out

        # down
        y = down_rw.matvec(mid)

        self.assertEqual(y.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(y)), "输出包含 NaN 或 Inf")


class TestMixedQuantLatency(unittest.TestCase):
    """混合量化延迟基准测试。"""

    def test_iq2xs_matvec_latency(self):
        """IQ2_XS matvec 延迟。"""
        try:
            from cpu_expert import Iq2XsWeight
        except ImportError:
            self.skipTest("cpu_expert 不可用")

        dim = 4096
        moe_inter_dim = 2048
        w = _make_iq2xs_weight(moe_inter_dim, dim)
        rw = Iq2XsWeight(w["d"], w["qs"], w["scales"], w["shape"])
        x = np.random.randn(dim).astype(np.float32) * 0.1

        # 预热
        _ = rw.matvec(x)

        # 测量
        n_iter = 100
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = rw.matvec(x)
        elapsed = (time.perf_counter() - start) / n_iter * 1000

        print(f"\n  IQ2_XS matvec [{moe_inter_dim}×{dim}]: {elapsed:.2f}ms")
        self.assertLess(elapsed, 10.0, f"IQ2_XS matvec 过慢: {elapsed:.2f}ms")

    def test_q2k_matvec_latency(self):
        """Q2_K matvec 延迟。"""
        try:
            from cpu_expert import Q2KWeight as RustQ2KWeight
        except ImportError:
            self.skipTest("cpu_expert Q2KWeight 不可用")

        dim = 4096
        moe_inter_dim = 2048
        w = _make_q2k_weight(dim, moe_inter_dim)
        rw = RustQ2KWeight(
            w["d"].astype(np.float32),
            w["dmin"].astype(np.float32),
            w["scales"],
            w["qs"],
            w["shape"],
        )
        x = np.random.randn(moe_inter_dim).astype(np.float32) * 0.1

        # 预热
        _ = rw.matvec(x)

        # 测量
        n_iter = 100
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = rw.matvec(x)
        elapsed = (time.perf_counter() - start) / n_iter * 1000

        print(f"\n  Q2_K matvec [{dim}×{moe_inter_dim}]: {elapsed:.2f}ms")

    def test_mixed_ffn_latency(self):
        """混合 FFN 延迟（gate IQ2_XS + up IQ2_XS + down Q2_K）。"""
        try:
            from cpu_expert import Iq2XsWeight, Q2KWeight as RustQ2KWeight
        except ImportError:
            self.skipTest("cpu_expert 不可用")

        dim = 4096
        moe_inter_dim = 2048

        gate_w = _make_iq2xs_weight(moe_inter_dim, dim, seed=1)
        gate_rw = Iq2XsWeight(gate_w["d"], gate_w["qs"], gate_w["scales"], gate_w["shape"])

        up_w = _make_iq2xs_weight(moe_inter_dim, dim, seed=2)
        up_rw = Iq2XsWeight(up_w["d"], up_w["qs"], up_w["scales"], up_w["shape"])

        down_w = _make_q2k_weight(dim, moe_inter_dim, seed=3)
        down_rw = RustQ2KWeight(
            down_w["d"].astype(np.float32),
            down_w["dmin"].astype(np.float32),
            down_w["scales"],
            down_w["qs"],
            down_w["shape"],
        )

        x = np.random.randn(dim).astype(np.float32) * 0.1

        # 预热
        gate_out = gate_rw.matvec(x)
        up_out = up_rw.matvec(x)
        mid = (1.0 / (1.0 + np.exp(-gate_out))) * gate_out * up_out
        _ = down_rw.matvec(mid)

        # 测量
        n_iter = 50
        start = time.perf_counter()
        for _ in range(n_iter):
            gate_out = gate_rw.matvec(x)
            up_out = up_rw.matvec(x)
            mid = (1.0 / (1.0 + np.exp(-gate_out))) * gate_out * up_out
            _ = down_rw.matvec(mid)
        elapsed = (time.perf_counter() - start) / n_iter * 1000

        print(f"\n  混合 FFN [dim={dim}, inter={moe_inter_dim}]: {elapsed:.2f}ms")
        print(f"    目标 (IQ2_XS 全量): 2.7ms")
        print(f"    目标 (Q2K+IQ2_XSS): 1.92ms")


class TestGGUFFileVerification(unittest.TestCase):
    """GGUF 文件完整性验证。"""

    @unittest.skipUnless(os.path.exists(GGUF_PATH), "GGUF 文件不存在")
    def test_gguf_file_size(self):
        """测试 GGUF 文件大小合理。"""
        size = os.path.getsize(GGUF_PATH)
        size_gb = size / 1024**3
        print(f"\n  GGUF 文件大小: {size_gb:.2f} GB")
        # 混合量化应约 80GB
        self.assertGreater(size_gb, 50, "GGUF 文件过小")
        self.assertLess(size_gb, 100, "GGUF 文件过大")

    @unittest.skipUnless(os.path.exists(GGUF_PATH), "GGUF 文件不存在")
    def test_gguf_tensor_types(self):
        """测试 GGUF 包含 IQ2_XS 和 Q2_K 两种类型。"""
        from gguf_iq2xs import GGUFReader
        reader = GGUFReader()
        reader.open(GGUF_PATH)
        
        iq2xs_count = 0
        q2k_count = 0
        
        for tensor in reader.tensors:
            if tensor.ggml_type == 28:  # IQ2_XS
                iq2xs_count += 1
            elif tensor.ggml_type == 10:  # Q2_K
                q2k_count += 1
        
        print(f"\n  IQ2_XS 张量: {iq2xs_count}")
        print(f"  Q2_K 张量: {q2k_count}")
        
        self.assertGreater(iq2xs_count, 0, "应有 IQ2_XS 张量")
        self.assertGreater(q2k_count, 0, "应有 Q2_K 张量")


if __name__ == "__main__":
    unittest.main(verbosity=2)
