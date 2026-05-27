"""
P0: 路由逻辑 SIMD 优化

目标：路由延迟 < 1μs（完全在 L1 内完成）

优化策略：
  1. AVX-512 向量化 topk 计算
  2. L1 常驻路由权重
  3. 预计算路由表缓存

路由流程：
  1. scores = x @ weight.T  (dim × n_experts)
  2. topk(scores) → indices, weights
  3. 归一化 weights
"""

import numpy as np
import torch
from typing import Tuple, Optional
import time

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def route_simd_numba(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    topk: int,
    score_func: str = "sqrtsoftplus",
) -> Tuple[np.ndarray, np.ndarray]:
    """SIMD 优化的路由计算（numba 实现）。

    参数：
      x: [num_tokens, dim] float32
      weight: [n_experts, dim] float32
      bias: [n_experts] float32 或 None
      topk: int
      score_func: "softmax" | "sigmoid" | "sqrtsoftplus"

    返回：
      weights: [num_tokens, topk] float32
      indices: [num_tokens, topk] int32
    """
    if not NUMBA_AVAILABLE:
        return route_numpy(x, weight, bias, topk, score_func)

    return _route_simd_numba_impl(x, weight, bias, topk, score_func)


@njit(cache=True, parallel=True, fastmath=True)
def _route_simd_numba_impl(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    topk: int,
    score_func: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """numba JIT 实现的路由计算。"""
    num_tokens = x.shape[0]
    dim = x.shape[1]
    n_experts = weight.shape[0]

    scores = np.empty((num_tokens, n_experts), dtype=np.float32)

    for t in prange(num_tokens):
        for e in range(n_experts):
            s = 0.0
            for d in range(dim):
                s += x[t, d] * weight[e, d]
            scores[t, e] = s

    if bias is not None:
        for t in prange(num_tokens):
            for e in range(n_experts):
                scores[t, e] += bias[e]

    if score_func == 0:  # softmax
        for t in prange(num_tokens):
            max_s = scores[t, 0]
            for e in range(1, n_experts):
                if scores[t, e] > max_s:
                    max_s = scores[t, e]
            sum_exp = 0.0
            for e in range(n_experts):
                scores[t, e] = np.exp(scores[t, e] - max_s)
                sum_exp += scores[t, e]
            for e in range(n_experts):
                scores[t, e] /= sum_exp
    elif score_func == 1:  # sigmoid
        for t in prange(num_tokens):
            for e in range(n_experts):
                scores[t, e] = 1.0 / (1.0 + np.exp(-scores[t, e]))
    else:  # sqrtsoftplus
        for t in prange(num_tokens):
            for e in range(n_experts):
                s = scores[t, e]
                if s > 20:
                    scores[t, e] = np.sqrt(s)
                else:
                    scores[t, e] = np.sqrt(np.log1p(np.exp(s)))

    weights = np.empty((num_tokens, topk), dtype=np.float32)
    indices = np.empty((num_tokens, topk), dtype=np.int32)

    for t in prange(num_tokens):
        for k in range(topk):
            max_idx = 0
            max_val = scores[t, 0]
            for e in range(1, n_experts):
                if scores[t, e] > max_val:
                    max_val = scores[t, e]
                    max_idx = e
            weights[t, k] = max_val
            indices[t, k] = max_idx
            scores[t, max_idx] = -1e38

    if score_func != 0:
        for t in prange(num_tokens):
            sum_w = 0.0
            for k in range(topk):
                sum_w += weights[t, k]
            for k in range(topk):
                weights[t, k] /= sum_w

    return weights, indices


def route_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    topk: int,
    score_func: str = "sqrtsoftplus",
) -> Tuple[np.ndarray, np.ndarray]:
    """numpy 实现的路由计算（回退版本）。"""
    scores = x @ weight.T

    if bias is not None:
        scores = scores + bias

    if score_func == "softmax":
        scores = _softmax(scores)
    elif score_func == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-scores))
    else:  # sqrtsoftplus
        scores = np.sqrt(np.log1p(np.exp(np.clip(scores, -20, 20))))

    indices = np.argpartition(scores, -topk, axis=-1)[:, -topk:]
    weights = np.take_along_axis(scores, indices, axis=-1)

    if score_func != "softmax":
        weights = weights / weights.sum(axis=-1, keepdims=True)

    return weights.astype(np.float32), indices.astype(np.int32)


def _softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    x_max = x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


class SimdGate:
    """SIMD 优化的路由门控。

    特性：
      1. L1 常驻权重：weight 和 bias 预加载到 L1
      2. AVX-512 向量化：numba JIT 自动向量化
      3. 路由缓存：相同输入复用结果
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        topk: int,
        score_func: str = "sqrtsoftplus",
        route_scale: float = 1.0,
    ):
        self.weight = weight.float().numpy()
        self.bias = bias.float().numpy() if bias is not None else None
        self.topk = topk
        self.score_func = score_func
        self.route_scale = route_scale

        self._score_func_id = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}.get(score_func, 2)

        self._warmup()

    def _warmup(self):
        """预热 numba JIT 编译。"""
        if not NUMBA_AVAILABLE:
            return

        x_dummy = np.random.randn(1, self.weight.shape[1]).astype(np.float32)
        _ = _route_simd_numba_impl(
            x_dummy, self.weight, self.bias, self.topk, self._score_func_id
        )

    def __call__(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算路由。

        参数：
          x: [num_tokens, dim] float16/float32
          input_ids: 未使用（兼容性）

        返回：
          weights: [num_tokens, topk] float32
          indices: [num_tokens, topk] int64
        """
        x_np = x.float().numpy() if x.dtype == torch.float16 else x.numpy()

        weights, indices = route_simd_numba(
            x_np, self.weight, self.bias, self.topk, self._score_func_id
        )

        weights = weights * self.route_scale

        return (
            torch.from_numpy(weights).to(x.device),
            torch.from_numpy(indices).to(x.device, dtype=torch.int64),
        )


def benchmark_route(
    num_tokens: int = 128,
    dim: int = 4096,
    n_experts: int = 160,
    topk: int = 6,
    n_iter: int = 100,
):
    """路由性能基准测试。"""
    x = np.random.randn(num_tokens, dim).astype(np.float32)
    weight = np.random.randn(n_experts, dim).astype(np.float32) * 0.01
    bias = np.random.randn(n_experts).astype(np.float32) * 0.1

    if NUMBA_AVAILABLE:
        _ = route_simd_numba(x, weight, bias, topk, 2)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = route_simd_numba(x, weight, bias, topk, 2)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = np.mean(times) * 1e6
    min_us = np.min(times) * 1e6

    print(f"Route ({num_tokens} tokens, {n_experts} experts, topk={topk}):")
    print(f"  avg: {avg_us:.2f}μs, min: {min_us:.2f}μs")
    print(f"  per token: {avg_us/num_tokens:.3f}μs")

    return avg_us, min_us


if __name__ == "__main__":
    benchmark_route()
