"""
Rust CPU 专家推理 Python 绑定

通过 PyO3 暴露 Rust 实现的 AVX2 优化内核。
使用 numpy buffer protocol 零拷贝传输，避免 .tolist() 开销。
"""

import numpy as np
from typing import Optional

try:
    from ds4rs import (
        cpu_expert_ffn_pair as _rust_ffn_pair,
        cpu_expert_ffn as _rust_ffn,
        CpuExpertRunner as _RustCpuExpertRunner,
        Iq2XsWeight as _Iq2XsWeight,
        Iq2XxsWeight as _Iq2XxsWeight,
        Q2KWeight as _Q2KWeight,
        mixed_ffn_pair_iq2xxs_q2k as _rust_mixed_ffn,
        init_tables as _init_tables,
        is_tables_initialized as _is_tables_initialized,
        is_avx512_supported as _is_avx512_supported,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def is_rust_available() -> bool:
    return RUST_AVAILABLE


def init_rust_tables():
    """初始化 Rust 查找表。"""
    if not RUST_AVAILABLE:
        return

    if _is_tables_initialized():
        return

    from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS

    grid_u64 = list(IQ2XS_GRID_U64)
    ksigns = list(KSIGNS_IQ2XS)

    _init_tables(grid_u64, ksigns)


def _to_rust_weight(w: dict) -> '_Iq2XsWeight':
    """Convert weight dict to Rust Iq2XsWeight using numpy zero-copy."""
    d = np.ascontiguousarray(w['d'].ravel(), dtype=np.float32)
    qs = np.ascontiguousarray(w['qs'].ravel(), dtype=np.uint16)
    scales = np.ascontiguousarray(w['scales'].ravel(), dtype=np.uint8)
    return _Iq2XsWeight.from_numpy(d, qs, scales, tuple(w['shape']))


def rust_cpu_expert_ffn_pair(
    x: np.ndarray,
    gate_weight: dict,
    up_weight: dict,
    down_weight: dict,
    route_weight: float = 1.0,
) -> Optional[np.ndarray]:
    if not RUST_AVAILABLE:
        return None

    init_rust_tables()

    gate_w = _to_rust_weight(gate_weight)
    up_w = _to_rust_weight(up_weight)
    down_w = _to_rust_weight(down_weight)

    x_f32 = np.ascontiguousarray(x.ravel(), dtype=np.float32)
    result = _rust_ffn_pair(x_f32, gate_w, up_w, down_w, route_weight)

    return np.array(result, dtype=np.float32)


def rust_cpu_expert_ffn(
    x: np.ndarray,
    gate_up_weight: dict,
    down_weight: dict,
    route_weight: float = 1.0,
    swiglu_limit: float = 0.0,
) -> Optional[np.ndarray]:
    """Rust CPU expert FFN with fused gate_up weight."""
    if not RUST_AVAILABLE:
        return None

    init_rust_tables()

    gate_up_w = _to_rust_weight(gate_up_weight)
    down_w = _to_rust_weight(down_weight)

    x_f32 = np.ascontiguousarray(x.ravel(), dtype=np.float32)
    result = _rust_ffn(x_f32, gate_up_w, down_w, route_weight, swiglu_limit)

    return np.array(result, dtype=np.float32)


class RustCpuExpertRunner:
    def __init__(self):
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust extension not available")
        init_rust_tables()
        self._runner = _RustCpuExpertRunner()

    def add_expert(
        self,
        layer_id: int,
        expert_id: int,
        gate_weight: dict,
        up_weight: dict,
        down_weight: dict,
    ):
        self._runner.add_expert(
            layer_id, expert_id,
            _to_rust_weight(gate_weight),
            _to_rust_weight(up_weight),
            _to_rust_weight(down_weight),
        )

    def add_expert_protected(
        self,
        layer_id: int,
        expert_id: int,
        gate_weight: dict,
        up_weight: dict,
        down_weight: dict,
    ):
        """添加专家到 protected 段（GPU 热专家同步专用，跳过 probation 准入）"""
        self._runner.add_expert_protected(
            layer_id, expert_id,
            _to_rust_weight(gate_weight),
            _to_rust_weight(up_weight),
            _to_rust_weight(down_weight),
        )

    def compute_expert(
        self,
        layer_id: int,
        expert_id: int,
        x: np.ndarray,
        route_weight: float = 1.0,
    ) -> Optional[np.ndarray]:
        x_f32 = np.ascontiguousarray(x.ravel(), dtype=np.float32)
        result = self._runner.compute_expert(layer_id, expert_id, x_f32, route_weight)
        if result is None:
            return None
        return np.array(result, dtype=np.float32)

    def has_expert(self, layer_id: int, expert_id: int) -> bool:
        return self._runner.has_expert(layer_id, expert_id)

    def record_access(self, layer_id: int, expert_id: int):
        """记录专家访问（不加载权重，只更新频率和访问时间）

        用于统一频率统计：无论专家走 GPU 还是 CPU，Gate 选中就计数。
        """
        self._runner.record_access(layer_id, expert_id)

    def protected_keys(self) -> list[tuple]:
        """返回 Rust SLRU protected 段的专家 key 列表"""
        return self._runner.protected_keys()

    def probation_keys(self) -> list[tuple]:
        """返回 Rust SLRU probation 段的专家 key 列表"""
        return self._runner.probation_keys()

    def warmup_layer(self, layer_id: int):
        """L3 权重预热：将指定层的热专家权重数据预取到 L3 缓存"""
        self._runner.warmup_layer(layer_id)

    def warmup_layer_targeted(
        self,
        layer_id: int,
        gpu_keys: set[tuple],
        gpu_topn: int,
        warmup_m: int,
    ):
        """L3 权重预热（分层预加载版）：只预热 CPU 负责的 TopN+1 ~ TopN+M 专家

        Args:
            layer_id: 要预热的层
            gpu_keys: GPU SLRU 中已有的专家 key 集合
            gpu_topn: GPU 负责的 Top-N 数量
            warmup_m: CPU L3 预热的专家数量
        """
        self._runner.warmup_layer_targeted(layer_id, gpu_keys, gpu_topn, warmup_m)

    def layer_freq_rank(self, layer_id: int) -> list[tuple]:
        """返回指定层的专家按频率降序排名列表

        Returns:
            [(expert_id, frequency), ...] 按频率降序
        """
        return self._runner.layer_freq_rank(layer_id)

    def hit_rate(self) -> float:
        return self._runner.hit_rate()

    def memory_usage_mb(self) -> float:
        return self._runner.memory_usage_mb()

    def expert_count(self) -> int:
        return self._runner.expert_count()


# ============================================================================
# IQ2_XXS + Q2_K 混合量化专家池
# ============================================================================

class MixedQuantExpertPool:
    """GGUF 格式的 IQ2_XXS+Q2_K 混合量化专家池。

    从 experts_iq2xxs_q2k.gguf 加载专家权重，缓存为 Rust 权重对象，
    通过 mixed_ffn_pair_iq2xxs_q2k 执行 CPU FFN。

    数据流：GGUF mmap → numpy 解析 → Rust 权重对象 → CPU FFN
    """

    def __init__(self, gguf_path: str):
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust extension not available")
        init_rust_tables()

        from gguf import GGUFReader
        self._reader = GGUFReader(gguf_path)
        self._cache: dict[tuple, tuple] = {}  # (layer_id, expert_id) → (gate_w, up_w, down_w)
        self._raw_cache: dict[tuple, dict] = {}  # (layer_id, expert_id) → {w1/w2/w3: raw data}

        # 预构建张量索引
        self._tensor_index: dict[str, object] = {}
        for t in self._reader.tensors:
            self._tensor_index[t.name] = t

        print(f"[MixedQuant] Loaded GGUF: {gguf_path}, {len(self._tensor_index)} tensors")

    def _parse_iq2xxs(self, data_bytes, n_blocks):
        """解析 IQ2_XXS 块：d(fp16,2B) + qs(u16×32,64B) = 66B/block"""
        raw = np.frombuffer(data_bytes, dtype=np.uint8)
        blocks = raw.reshape(n_blocks, 66)
        d = blocks[:, 0:2].view(np.float16).astype(np.float32).ravel()
        qs = blocks[:, 2:66].view(np.uint16).reshape(n_blocks, 32).ravel().copy()
        return d, qs

    def _parse_q2k(self, data_bytes, n_blocks):
        """解析 Q2_K 块：scales(16B) + qs(64B) + d(fp16,2B) + dmin(fp16,2B) = 84B/block"""
        raw = np.frombuffer(data_bytes, dtype=np.uint8)
        blocks = raw.reshape(n_blocks, 84)
        scales = blocks[:, 0:16].reshape(n_blocks, 16).ravel().copy()
        qs = blocks[:, 16:80].reshape(n_blocks, 64).ravel().copy()
        d = blocks[:, 80:82].view(np.float16).astype(np.float32).ravel()
        dmin = blocks[:, 82:84].view(np.float16).astype(np.float32).ravel()
        return d, dmin, scales, qs

    def _load_raw(self, layer_id: int, expert_id: int) -> dict:
        """从 GGUF 加载单个专家的原始权重数据。"""
        key = (layer_id, expert_id)
        if key in self._raw_cache:
            return self._raw_cache[key]

        weights = {}
        for wt_name, gguf_wt in [('w1', 'w1'), ('w3', 'w3'), ('w2', 'w2')]:
            tensor_name = f"layers.{layer_id}.experts.{expert_id}.{gguf_wt}"
            t = self._tensor_index.get(tensor_name)
            if t is None:
                raise ValueError(f"Tensor {tensor_name} not found in GGUF")

            data = t.data.ravel().tobytes()
            ne0, ne1 = [int(x) for x in t.shape]
            n_blocks = ne0 * ne1 // 256
            type_name = t.tensor_type.name

            # GGUF shape [ne0, ne1] 中 ne0=in_features, ne1=out_features
            # 逻辑矩阵形状为 (ne1, ne0) = (out_features, in_features)
            logical_shape = (ne1, ne0)

            if type_name == 'IQ2_XXS':
                d, qs = self._parse_iq2xxs(data, n_blocks)
                weights[wt_name] = ('iq2xxs', d, qs, logical_shape)
            elif type_name == 'Q2_K':
                d, dmin, scales, qs = self._parse_q2k(data, n_blocks)
                weights[wt_name] = ('q2k', d, dmin, scales, qs, logical_shape)
            else:
                raise ValueError(f"Unsupported type: {type_name}")

        self._raw_cache[key] = weights
        return weights

    def get_rust_weights(self, layer_id: int, expert_id: int):
        """获取 Rust 权重对象（带缓存）。"""
        key = (layer_id, expert_id)
        if key in self._cache:
            return self._cache[key]

        raw = self._load_raw(layer_id, expert_id)

        # gate (IQ2_XXS)
        wt = raw['w1']
        assert wt[0] == 'iq2xxs', f"Expected IQ2_XXS for gate, got {wt[0]}"
        gate_w = _Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])

        # up (IQ2_XXS)
        wt = raw['w3']
        assert wt[0] == 'iq2xxs', f"Expected IQ2_XXS for up, got {wt[0]}"
        up_w = _Iq2XxsWeight.from_numpy(wt[1], wt[2], wt[3])

        # down (Q2_K)
        wt = raw['w2']
        assert wt[0] == 'q2k', f"Expected Q2_K for down, got {wt[0]}"
        down_w = _Q2KWeight.from_numpy(wt[1], wt[2], wt[3], wt[4], wt[5])

        self._cache[key] = (gate_w, up_w, down_w)
        # 释放原始数据，节省内存
        if key in self._raw_cache:
            del self._raw_cache[key]

        return (gate_w, up_w, down_w)

    def compute_ffn(self, layer_id: int, expert_id: int,
                    x: np.ndarray, route_weight: float = 1.0,
                    swiglu_limit: float = 0.0) -> np.ndarray:
        """执行混合量化 FFN：IQ2_XXS gate/up + Q2_K down。"""
        gate_w, up_w, down_w = self.get_rust_weights(layer_id, expert_id)
        x_f32 = np.ascontiguousarray(x.ravel(), dtype=np.float32)
        result = _rust_mixed_ffn(x_f32, gate_w, up_w, down_w, route_weight, swiglu_limit)
        return np.array(result, dtype=np.float32)

    def has_expert(self, layer_id: int, expert_id: int) -> bool:
        return (layer_id, expert_id) in self._cache or (layer_id, expert_id) in self._raw_cache

    def cached_count(self) -> int:
        return len(self._cache)
