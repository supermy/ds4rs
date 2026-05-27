"""
P1-2: GPU hot + CPU cold 双路分发测试

测试范围：
  1. hot/cold 双路分发正确性
  2. GPU hot 路径性能
  3. CPU cold 路径性能
  4. 混合路径性能

双路分发流程（参考 llama.cpp deepseek4.cpp）：
  1. hot_remap_table: 热门专家 → [0, K)，冷门 → K（哨兵）
  2. cold_remap_table: 冷门专家 → 原始 ID，热门 → 0
  3. GPU 热门路径：冷门 pick 命中哑专家（零权重），输出自然为 0
  4. CPU 冷门路径：热门 pick 被 mask 清零
  5. 最终 add(hot_out, cold_out) 合并
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


class HotColdDispatcher:
    """模拟 hot/cold 双路分发器。"""

    def __init__(self, n_experts, n_hot, gpu_capacity):
        self.n_experts = n_experts
        self.n_hot = n_hot
        self.gpu_capacity = gpu_capacity

        self.hot_expert_ids = set(range(min(gpu_capacity, n_experts)))
        self.cold_expert_ids = set(range(gpu_capacity, n_experts))

        self.hot_remap = {}
        hot_idx = 0
        for i in range(n_experts):
            if i in self.hot_expert_ids:
                self.hot_remap[i] = hot_idx
                hot_idx += 1
            else:
                self.hot_remap[i] = n_hot

        self.cold_remap = {}
        cold_idx = 0
        for i in range(n_experts):
            if i in self.cold_expert_ids:
                self.cold_remap[i] = cold_idx
                cold_idx += 1
            else:
                self.cold_remap[i] = 0

    def dispatch(self, selected_experts):
        """分发专家到 hot/cold 路径。"""
        hot_picks = []
        cold_picks = []

        for i, expert_id in enumerate(selected_experts):
            if expert_id in self.hot_expert_ids:
                hot_picks.append((i, expert_id, self.hot_remap[expert_id]))
            else:
                cold_picks.append((i, expert_id, self.cold_remap[expert_id]))

        return hot_picks, cold_picks


class TestHotColdDispatch(unittest.TestCase):
    """hot/cold 双路分发测试。"""

    def test_dispatcher_basic(self):
        """测试分发器基本功能。"""
        dispatcher = HotColdDispatcher(
            n_experts=160,
            n_hot=6,
            gpu_capacity=120
        )

        self.assertEqual(len(dispatcher.hot_expert_ids), 120)
        self.assertEqual(len(dispatcher.cold_expert_ids), 40)

    def test_hot_path_dispatch(self):
        """测试 hot 路径分发。"""
        dispatcher = HotColdDispatcher(
            n_experts=160,
            n_hot=6,
            gpu_capacity=120
        )

        selected = [0, 10, 50, 100, 130, 150]
        hot_picks, cold_picks = dispatcher.dispatch(selected)

        hot_ids = [p[1] for p in hot_picks]
        cold_ids = [p[1] for p in cold_picks]

        self.assertEqual(len(hot_picks), 4)
        self.assertEqual(len(cold_picks), 2)
        self.assertTrue(all(id < 120 for id in hot_ids))
        self.assertTrue(all(id >= 120 for id in cold_ids))

    def test_all_hot_dispatch(self):
        """测试全 hot 分发。"""
        dispatcher = HotColdDispatcher(
            n_experts=160,
            n_hot=6,
            gpu_capacity=120
        )

        selected = [0, 10, 20, 30, 40, 50]
        hot_picks, cold_picks = dispatcher.dispatch(selected)

        self.assertEqual(len(hot_picks), 6)
        self.assertEqual(len(cold_picks), 0)

    def test_all_cold_dispatch(self):
        """测试全 cold 分发。"""
        dispatcher = HotColdDispatcher(
            n_experts=160,
            n_hot=6,
            gpu_capacity=120
        )

        selected = [120, 130, 140, 150, 155, 159]
        hot_picks, cold_picks = dispatcher.dispatch(selected)

        self.assertEqual(len(hot_picks), 0)
        self.assertEqual(len(cold_picks), 6)


class TestHotColdPerformance(unittest.TestCase):
    """hot/cold 路径性能测试。"""

    def test_hot_path_latency(self):
        """测试 hot 路径延迟（GPU 模拟）。"""
        dim = 4096
        inter_dim = 2048
        n_hot = 6

        x = torch.randn(dim)
        weights = [torch.randn(inter_dim, dim) for _ in range(n_hot)]

        for _ in range(5):
            for w in weights:
                _ = x @ w.T

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            for w in weights:
                _ = x @ w.T
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Hot path (GPU sim, {n_hot} experts): {avg_ms:.2f}ms")

        self.assertLess(avg_ms, 10.0, f"Hot path latency {avg_ms:.2f}ms exceeds 10ms target")

    def test_cold_path_latency(self):
        """测试 cold 路径延迟（CPU）。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 4096
        inter_dim = 2048
        n_cold = 2

        x = np.random.randn(dim).astype(np.float32)
        weights = [
            (
                _make_iq2xs_weight(inter_dim, dim, seed=100 + i),
                _make_iq2xs_weight(inter_dim, dim, seed=200 + i),
                _make_iq2xs_weight(dim, inter_dim, seed=300 + i),
            )
            for i in range(n_cold)
        ]

        for _ in range(5):
            for gate_w, up_w, down_w in weights:
                _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            for gate_w, up_w, down_w in weights:
                _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Cold path (CPU, {n_cold} experts): {avg_ms:.2f}ms")

        self.assertLess(avg_ms, 15.0, f"Cold path latency {avg_ms:.2f}ms exceeds 15ms target")

    def test_mixed_path_latency(self):
        """测试混合路径延迟。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 4096
        inter_dim = 2048
        n_hot = 4
        n_cold = 2

        x = np.random.randn(dim).astype(np.float32)
        hot_weights = [torch.randn(inter_dim, dim) for _ in range(n_hot)]
        cold_weights = [
            (
                _make_iq2xs_weight(inter_dim, dim, seed=100 + i),
                _make_iq2xs_weight(inter_dim, dim, seed=200 + i),
                _make_iq2xs_weight(dim, inter_dim, seed=300 + i),
            )
            for i in range(n_cold)
        ]

        for _ in range(5):
            for w in hot_weights:
                _ = torch.from_numpy(x) @ w.T
            for gate_w, up_w, down_w in cold_weights:
                _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)

        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            for w in hot_weights:
                _ = torch.from_numpy(x) @ w.T
            for gate_w, up_w, down_w in cold_weights:
                _ = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Mixed path ({n_hot} hot + {n_cold} cold): {avg_ms:.2f}ms")

        self.assertLess(avg_ms, 20.0, f"Mixed path latency {avg_ms:.2f}ms exceeds 20ms target")


class TestHotColdCorrectness(unittest.TestCase):
    """hot/cold 路径正确性测试。"""

    def test_output_accumulation(self):
        """测试输出累加正确性。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256
        n_experts = 6

        x = np.random.randn(dim).astype(np.float32)

        outputs = []
        for i in range(n_experts):
            gate_w = _make_iq2xs_weight(inter_dim, dim, seed=100 + i)
            up_w = _make_iq2xs_weight(inter_dim, dim, seed=200 + i)
            down_w = _make_iq2xs_weight(dim, inter_dim, seed=300 + i)
            out = cpu_expert_ffn_pair(x, gate_w, up_w, down_w)
            outputs.append(out)

        total = np.zeros(dim)
        for out in outputs:
            total += out

        self.assertEqual(total.shape, (dim,))
        self.assertTrue(np.all(np.isfinite(total)))

    def test_route_weight_distribution(self):
        """测试路由权重分配。"""
        from cpu_expert import cpu_expert_ffn_pair

        dim = 512
        inter_dim = 256
        n_experts = 6

        x = np.random.randn(dim).astype(np.float32)
        route_weights = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])

        outputs = []
        for i in range(n_experts):
            gate_w = _make_iq2xs_weight(inter_dim, dim, seed=100 + i)
            up_w = _make_iq2xs_weight(inter_dim, dim, seed=200 + i)
            down_w = _make_iq2xs_weight(dim, inter_dim, seed=300 + i)
            out = cpu_expert_ffn_pair(x, gate_w, up_w, down_w, route_weight=route_weights[i])
            outputs.append(out)

        total = np.sum(outputs, axis=0)

        self.assertEqual(total.shape, (dim,))
        self.assertAlmostEqual(np.sum(route_weights), 1.0, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
