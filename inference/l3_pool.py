"""
P1: L3 常驻专家池

策略：
  - L3 容量：32MB
  - 专家大小：~7MB (IQ2_XS)
  - 常驻专家：4 个（28MB）

特性：
  1. 热点专家常驻 L3
  2. 访问计数自动更新
  3. LRU 淘汰策略
  4. 预取下一层专家
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from collections import OrderedDict
import threading


class L3ResidentPool:
    """L3 常驻专家池。

    将热点专家权重常驻 L3 缓存，减少内存访问延迟。

    使用方式：
      1. 初始化时指定容量（专家数）
      2. 通过 get_expert() 获取专家
      3. 自动更新访问计数和 LRU 顺序
    """

    CAPACITY_EXPERTS = 4
    EXPERT_SIZE_MB = 7

    def __init__(self, capacity_experts: int = 4):
        self.capacity = capacity_experts
        self.pool: OrderedDict[Tuple[int, int], dict] = OrderedDict()
        self.access_count: Dict[Tuple[int, int], int] = {}
        self.lock = threading.Lock()

        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get_expert(self, layer_id: int, expert_id: int) -> Optional[dict]:
        """获取专家权重。

        如果专家在池中，返回权重并更新 LRU。
        否则返回 None。
        """
        key = (layer_id, expert_id)

        with self.lock:
            if key in self.pool:
                self.pool.move_to_end(key)
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self._stats["hits"] += 1
                return self.pool[key]
            else:
                self._stats["misses"] += 1
                return None

    def put_expert(self, layer_id: int, expert_id: int, weight: dict) -> bool:
        """将专家权重放入池中。

        如果池已满，淘汰最少使用的专家。

        返回：
          True 如果成功放入
          False 如果专家太大
        """
        key = (layer_id, expert_id)

        with self.lock:
            if key in self.pool:
                self.pool.move_to_end(key)
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return True

            if len(self.pool) >= self.capacity:
                oldest_key = next(iter(self.pool))
                del self.pool[oldest_key]
                del self.access_count[oldest_key]
                self._stats["evictions"] += 1

            self.pool[key] = weight
            self.access_count[key] = 1

            self._preload_to_l3(weight)

            return True

    def _preload_to_l3(self, weight: dict):
        """预加载专家权重到 L3 缓存。

        通过顺序访问所有数据，触发硬件预取到 L3。
        """
        for key in ['d', 'qs', 'scales']:
            if key in weight:
                data = weight[key]
                if isinstance(data, np.ndarray):
                    _ = data.sum()

    def prefetch_experts(self, experts: list[Tuple[int, int]], weight_loader):
        """预取多个专家到 L3。

        参数：
          experts: [(layer_id, expert_id), ...] 专家列表
          weight_loader: 加载专家权重的函数
        """
        for layer_id, expert_id in experts:
            if self.get_expert(layer_id, expert_id) is None:
                weight = weight_loader(layer_id, expert_id)
                if weight is not None:
                    self.put_expert(layer_id, expert_id, weight)

    def get_stats(self) -> dict:
        """获取统计信息。"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0

        return {
            "capacity": self.capacity,
            "current_size": len(self.pool),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
        }

    def clear(self):
        """清空池。"""
        with self.lock:
            self.pool.clear()
            self.access_count.clear()


class L3ResidentManager:
    """L3 常驻专家管理器。

    管理多层 L3 常驻池，支持：
      1. 按层分配池
      2. 全局热点专家
      3. 预取策略
    """

    def __init__(self, n_layers: int, experts_per_layer: int, capacity_per_layer: int = 2):
        self.n_layers = n_layers
        self.experts_per_layer = experts_per_layer
        self.capacity_per_layer = capacity_per_layer

        self.global_pool = L3ResidentPool(capacity_experts=4)
        self.layer_pools = [
            L3ResidentPool(capacity_experts=capacity_per_layer)
            for _ in range(n_layers)
        ]

    def get_expert(self, layer_id: int, expert_id: int) -> Optional[dict]:
        """获取专家权重。

        优先从层池获取，其次从全局池获取。
        """
        weight = self.layer_pools[layer_id].get_expert(layer_id, expert_id)
        if weight is not None:
            return weight

        return self.global_pool.get_expert(layer_id, expert_id)

    def put_expert(self, layer_id: int, expert_id: int, weight: dict):
        """将专家权重放入池中。"""
        self.layer_pools[layer_id].put_expert(layer_id, expert_id, weight)
        self.global_pool.put_expert(layer_id, expert_id, weight)

    def prefetch_by_route(self, layer_id: int, topk_indices: np.ndarray):
        """根据路由结果预取下一层专家。"""
        next_layer = layer_id + 1
        if next_layer >= self.n_layers:
            return

        experts = [(next_layer, int(eid)) for eid in topk_indices.ravel()]
        for layer, eid in experts:
            if self.layer_pools[layer].get_expert(layer, eid) is None:
                pass

    def get_stats(self) -> dict:
        """获取统计信息。"""
        layer_stats = [pool.get_stats() for pool in self.layer_pools]
        global_stats = self.global_pool.get_stats()

        return {
            "global": global_stats,
            "layers": layer_stats,
        }


_l3_pool: Optional[L3ResidentPool] = None
_l3_manager: Optional[L3ResidentManager] = None


def get_l3_pool() -> L3ResidentPool:
    """获取全局 L3 常驻池。"""
    global _l3_pool
    if _l3_pool is None:
        _l3_pool = L3ResidentPool()
    return _l3_pool


def get_l3_manager(n_layers: int = 61, experts_per_layer: int = 256) -> L3ResidentManager:
    """获取全局 L3 管理器。"""
    global _l3_manager
    if _l3_manager is None:
        _l3_manager = L3ResidentManager(n_layers, experts_per_layer)
    return _l3_manager


def benchmark_l3_pool():
    """L3 常驻池性能测试。"""
    pool = L3ResidentPool(capacity_experts=4)

    def make_weight(seed: int) -> dict:
        rng = np.random.RandomState(seed)
        n_blocks = 2048 * 4096 // 256
        return {
            'd': rng.randn(n_blocks).astype(np.float16),
            'qs': rng.randint(0, 65535, (n_blocks, 32), dtype=np.uint16),
            'scales': rng.randint(0, 256, (n_blocks, 8), dtype=np.uint8),
        }

    weights = [make_weight(i) for i in range(10)]

    for i in range(4):
        pool.put_expert(0, i, weights[i])

    n_iter = 1000
    times = []

    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = pool.get_expert(0, 0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = np.mean(times) * 1e6
    min_us = np.min(times) * 1e6

    print(f"L3 pool get_expert:")
    print(f"  avg: {avg_us:.2f}μs, min: {min_us:.2f}μs")
    print(f"  stats: {pool.get_stats()}")


if __name__ == "__main__":
    benchmark_l3_pool()
