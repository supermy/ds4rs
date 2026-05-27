"""
Rust CPU 专家推理 Python 绑定

通过 PyO3 暴露 Rust 实现的 AVX2 优化内核。
使用 numpy buffer protocol 零拷贝传输，避免 .tolist() 开销。
"""

import os
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

    支持 GPU FFN：量化格式数据直传 GPU，使用 TileLang mixed_quant_gemm 推理。
    GPU SLRU 缓存：一次命中准入 + 异步预取 + 频率持久化。

    数据流：
      CPU: GGUF mmap → numpy 解析 → Rust 权重对象 → CPU FFN (AVX-512)
      GPU: GGUF mmap → numpy 解析 → 量化格式 GPU tensor → TileLang GEMM
    """

    QK_K = 256

    # IQ2_XXS Grid — 与 Rust tables.rs 中 get_iq2xxs_grid() 一致
    _IQ2XXS_GRID_U64 = [
        0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
        0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
        0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
        0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
        0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
        0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
        0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
        0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
        0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
        0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
        0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
        0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
        0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
        0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
        0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
        0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
        0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
        0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
        0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
        0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
        0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
        0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
        0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
        0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
        0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
        0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
        0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
        0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
        0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
        0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
        0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
        0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
        0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
        0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
        0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
        0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
        0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
        0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
        0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
        0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
        0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
        0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
        0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
        0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
        0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
        0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
        0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
        0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
        0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
        0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
        0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
        0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
        0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
        0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
        0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
        0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
        0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
        0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
        0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
        0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
        0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
        0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
        0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
        0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
    ]

    _KSIGNS_IQ2XXS = np.array([
          0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
        144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
        160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
         48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
        192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
         80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
         96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
        240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
    ], dtype=np.uint8)

    def __init__(self, gguf_path: str, gpu_ffn: bool = False, gpu_cache_capacity: int = 0):
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

        # GPU FFN 支持
        self._gpu_ffn = gpu_ffn
        self._gpu_cache: dict[tuple, dict] = {}  # (layer_id, expert_id) → 量化格式 GPU tensors
        self._gpu_freq: dict[tuple, int] = {}  # 频率计数（驱动淘汰排序）
        self._gpu_last_access: dict[tuple, int] = {}  # LRU 淘汰：最后访问时间
        self._gpu_access_counter: int = 0  # 全局访问计数器
        # 每专家量化格式 GPU 占用 ~6.75MB（实测：633专家=4272MB）
        # gate/up: qs [N, 28, 32] uint16 + d [N, 28] fp16
        # down: qs [N, 28, 64] uint8 + scales [N, 28, 16] uint8 + d/dmin [N, 28] fp16
        self._expert_gpu_mb = 7.0  # 保守估计
        if gpu_cache_capacity > 0:
            self._gpu_cache_capacity = gpu_cache_capacity
        elif gpu_ffn:
            # 自动计算：预留 2GB 给推理，其余用于缓存
            import torch
            total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            used_vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
            free_vram_mb = total_vram_mb - used_vram_mb
            cache_budget_mb = max(free_vram_mb - 2048, 512)  # 至少 512MB
            self._gpu_cache_capacity = int(cache_budget_mb / self._expert_gpu_mb)
            print(f"[MixedQuant GPU] VRAM: total={total_vram_mb:.0f}MB, used={used_vram_mb:.0f}MB, "
                  f"cache_budget={cache_budget_mb:.0f}MB, capacity={self._gpu_cache_capacity} experts")
        else:
            self._gpu_cache_capacity = 0

        # 异步预取：CUDA Stream + 线程池
        self._prefetch_stream = None  # torch.cuda.Stream, 延迟创建
        self._prefetch_executor = None  # ThreadPoolExecutor, 延迟创建
        self._prefetch_pending: set[tuple] = set()  # 正在预取的专家 key
        self._prefetch_lock = __import__('threading').Lock()

        # GPU FFN 统计
        self._gpu_ffn_hits = 0
        self._gpu_ffn_misses = 0
        self._gpu_upload_count = 0
        self._gpu_ffn_time = 0.0   # GPU FFN 累计时间（秒）
        self._cpu_ffn_time = 0.0   # CPU FFN 累计时间（秒）

        # 专家延迟：异步 CPU FFN 线程池
        from concurrent.futures import ThreadPoolExecutor
        self._cpu_ffn_executor = ThreadPoolExecutor(max_workers=4)

        # 预测性预取配置
        self.prefetch_count = 50   # 每层预取最大专家数
        self.prefetch_layers = 1   # 向前预取层数

        # 预计算 IQ2_XXS grid
        self._iq2xxs_grid = np.zeros((256, 8), dtype=np.int8)
        for _i, _g in enumerate(self._IQ2XXS_GRID_U64):
            _bytes = _g.to_bytes(8, 'little')
            for _j, _b in enumerate(_bytes):
                self._iq2xxs_grid[_i, _j] = np.int8(_b if _b < 128 else _b - 256)

        # 频率持久化：热启动时直接加载频率数据
        # 如果 GGUF 路径不可写（如只读挂载），保存到 /tmp 下
        freq_candidate = gguf_path + ".gpu_freq.json"
        freq_dir = os.path.dirname(freq_candidate)
        if not os.access(freq_dir, os.W_OK):
            freq_candidate = "/tmp/" + os.path.basename(gguf_path) + ".gpu_freq.json"
        self._freq_path = freq_candidate
        self._load_freq()

        print(f"[MixedQuant] Loaded GGUF: {gguf_path}, {len(self._tensor_index)} tensors, "
              f"gpu_ffn={gpu_ffn}, capacity={self._gpu_cache_capacity}")

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
        """从 GGUF 加载单个专家的原始权重数据。

        支持两种 GGUF 格式：
        1. 逐专家独立：layers.L.experts.E.w1/w2/w3（shape [ne0, ne1]）
        2. 按层打包（imatrix）：blk.L.ffn_gate/up/down_exps.weight（shape [ne0, ne1, 256]）
        """
        key = (layer_id, expert_id)
        if key in self._raw_cache:
            return self._raw_cache[key]

        weights = {}

        # 格式 1：逐专家独立张量
        for wt_name, gguf_wt in [('w1', 'w1'), ('w3', 'w3'), ('w2', 'w2')]:
            tensor_name = f"layers.{layer_id}.experts.{expert_id}.{gguf_wt}"
            t = self._tensor_index.get(tensor_name)
            if t is not None:
                data = t.data.ravel().tobytes()
                ne0, ne1 = [int(x) for x in t.shape]
                n_blocks = ne0 * ne1 // 256
                type_name = t.tensor_type.name
                logical_shape = (ne1, ne0)

                if type_name == 'IQ2_XXS':
                    d, qs = self._parse_iq2xxs(data, n_blocks)
                    weights[wt_name] = ('iq2xxs', d, qs, logical_shape)
                elif type_name == 'Q2_K':
                    d, dmin, scales, qs = self._parse_q2k(data, n_blocks)
                    weights[wt_name] = ('q2k', d, dmin, scales, qs, logical_shape)
                else:
                    raise ValueError(f"Unsupported type: {type_name}")

        # 格式 2：按层打包（imatrix GGUF）
        # 数据布局：data.shape=(256, n_super_blocks, bytes_per_super_block)
        # GGUF super-block = 16 个 IQ2_XXS block 或 8 个 Q2_K block
        # 实际 n_blocks = n_super_blocks × (bytes_per_super_block / bytes_per_quant_block)
        if not weights:
            role_map = {'w1': 'gate', 'w3': 'up', 'w2': 'down'}
            for wt_name, role in role_map.items():
                tensor_name = f"blk.{layer_id}.ffn_{role}_exps.weight"
                t = self._tensor_index.get(tensor_name)
                if t is None:
                    raise ValueError(f"Tensor {tensor_name} not found in GGUF")

                raw_data = t.data  # numpy memmap, shape=(256, n_super_blocks, bytes_per_super_block)
                n_experts = raw_data.shape[0]
                assert n_experts == 256, f"Expected 256 experts, got {n_experts}"

                # 切片出第 expert_id 个专家的数据
                expert_raw = raw_data[expert_id]  # shape: (n_super_blocks, bytes_per_super_block)
                data = expert_raw.ravel().tobytes()

                # 从 GGUF 报告的 shape 推断逻辑形状（t.shape 可能返回 numpy 类型，需转 int）
                ne0 = int(t.shape[0])  # 4096
                ne1 = int(t.shape[1])  # 2048
                logical_shape = (ne1, ne0)

                # 计算实际量化 block 数
                type_name = t.tensor_type.name
                if type_name == 'IQ2_XXS':
                    bytes_per_quant_block = 66  # d(2B) + qs(64B)
                    n_blocks = len(data) // bytes_per_quant_block
                    d, qs = self._parse_iq2xxs(data, n_blocks)
                    weights[wt_name] = ('iq2xxs', d, qs, logical_shape)
                elif type_name == 'Q2_K':
                    bytes_per_quant_block = 84  # d(4B) + dmin(4B) + scales(32B) + qs(44B) → 实际 scales(16B)+qs(64B)+d(2B)+dmin(2B)
                    n_blocks = len(data) // bytes_per_quant_block
                    d, dmin, scales, qs = self._parse_q2k(data, n_blocks)
                    weights[wt_name] = ('q2k', d, dmin, scales, qs, logical_shape)
                else:
                    raise ValueError(f"Unsupported type: {type_name}")

        if not weights:
            raise ValueError(f"No expert weights found for L{layer_id}E{expert_id}")

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
        import time
        gate_w, up_w, down_w = self.get_rust_weights(layer_id, expert_id)
        x_f32 = np.ascontiguousarray(x.ravel(), dtype=np.float32)
        _t0 = time.time()
        result = _rust_mixed_ffn(x_f32, gate_w, up_w, down_w, route_weight, swiglu_limit)
        self._cpu_ffn_time += time.time() - _t0
        return np.array(result, dtype=np.float32)

    def compute_ffn_async(self, layer_id: int, expert_id: int,
                          x_cpu: np.ndarray, route_weight: float = 1.0,
                          swiglu_limit: float = 0.0):
        """异步 CPU FFN：提交到线程池，返回 Future。

        专家延迟机制使用此方法，CPU FFN 与下一层 GPU Attention 重叠。
        x_cpu 必须是已经从 GPU 传输到 CPU 的 numpy 数组。
        """
        future = self._cpu_ffn_executor.submit(
            self.compute_ffn, layer_id, expert_id, x_cpu, route_weight, swiglu_limit)
        return future

    def has_expert(self, layer_id: int, expert_id: int) -> bool:
        return (layer_id, expert_id) in self._cache or (layer_id, expert_id) in self._raw_cache

    def cached_count(self) -> int:
        return len(self._cache)

    # ========================================================================
    # GPU FFN 支持
    # ========================================================================

    def _dequantize_iq2xxs(self, d_flat, qs_flat, ne0, ne1):
        """反量化 IQ2_XXS 权重为 BF16 矩阵 [ne1, ne0]。NumPy 向量化实现。"""
        N = ne1
        K = ne0
        n_blocks = K // self.QK_K

        d_2d = d_flat.reshape(N, n_blocks)
        # qs: 每 ib32 子块 4 个 uint16 → (N, n_blocks, 8_ib32, 4_qs)
        qs_4d = qs_flat.reshape(N, n_blocks, 8, 4)

        grid = self._iq2xxs_grid      # (256, 8) int8
        ksigns = self._KSIGNS_IQ2XXS  # (256,) uint8

        # 构建 aux32 对: q0|q1<<16, q2|q3<<16
        q0 = qs_4d[..., 0].astype(np.uint32)
        q1 = qs_4d[..., 1].astype(np.uint32)
        q2 = qs_4d[..., 2].astype(np.uint32)
        q3 = qs_4d[..., 3].astype(np.uint32)
        aux32_0 = q0 | (q1 << 16)     # (N, n_blocks, 8)
        aux32_1 = q2 | (q3 << 16)     # (N, n_blocks, 8)

        # 提取 ls: 2*((aux32_1>>28)&0xF)+1
        ls = (2.0 * ((aux32_1 >> 28) & 0xF).astype(np.float32) + 1.0)  # (N, n_blocks, 8)

        # 提取 grid 索引 (N, n_blocks, 8, 4): aux32_0 的 4 个字节
        gi = np.empty(aux32_0.shape + (4,), dtype=np.intp)
        gi[..., 0] = aux32_0 & 0xFF
        gi[..., 1] = (aux32_0 >> 8) & 0xFF
        gi[..., 2] = (aux32_0 >> 16) & 0xFF
        gi[..., 3] = (aux32_0 >> 24) & 0xFF

        # 提取 sign 索引 (N, n_blocks, 8, 4): aux32_1 的 4 个 7-bit 组
        si = np.empty(aux32_1.shape + (4,), dtype=np.intp)
        si[..., 0] = aux32_1 & 0x7F
        si[..., 1] = (aux32_1 >> 7) & 0x7F
        si[..., 2] = (aux32_1 >> 14) & 0x7F
        si[..., 3] = (aux32_1 >> 21) & 0x7F

        # Grid 查表: (N, n_blocks, 8, 4, 8)
        grid_vals = grid[gi].astype(np.float32)

        # Sign 查表并展开: (N, n_blocks, 8, 4) → (N, n_blocks, 8, 4, 8)
        sign_byte = ksigns[si]
        j_range = np.arange(8, dtype=np.uint8)
        sign_bits = ((sign_byte[..., np.newaxis] >> j_range) & 1).astype(np.float32)
        sign_vals = 1.0 - 2.0 * sign_bits

        # 计算权重: d * ls * grid * sign * 0.125 → (N, n_blocks, 8, 4, 8)
        result = (d_2d[..., np.newaxis, np.newaxis, np.newaxis] *
                  ls[..., np.newaxis, np.newaxis] *
                  grid_vals * sign_vals * 0.125)

        # 重排: (N, n_blocks, ib32, l, j) → (N, n_blocks, 256) → (N, K)
        return result.reshape(N, K).astype(np.float16)

    def _dequantize_q2k(self, d_flat, dmin_flat, scales_flat, qs_flat, ne0, ne1):
        """反量化 Q2_K 权重为 BF16 矩阵 [ne1, ne0]。NumPy 向量化实现。"""
        N = ne1
        K = ne0
        n_blocks = K // self.QK_K

        d_2d = d_flat.reshape(N, n_blocks)
        dmin_2d = dmin_flat.reshape(N, n_blocks)
        scales_3d = scales_flat.reshape(N, n_blocks, 16)
        qs_3d = qs_flat.reshape(N, n_blocks, 64)

        # 输出布局: (N, n_blocks, k_half, _j, sub, pos) → (N, K)
        # 索引 = k_half*128 + _j*32 + sub*16 + pos
        weight = np.empty((N, n_blocks, 2, 4, 2, 16), dtype=np.float32)

        for k_half in range(2):
            qs_half = qs_3d[:, :, k_half*32:k_half*32+32]   # (N, n_blocks, 32)
            sc_half = scales_3d[:, :, k_half*8:k_half*8+8]  # (N, n_blocks, 8)

            for _j in range(4):
                shift = _j * 2
                # 提取 2-bit 量化值: 同一 qs 字节的不同 2-bit 段对应不同 _j
                quant = (qs_half >> shift) & 3               # (N, n_blocks, 32)
                quant_lo = quant[:, :, :16].astype(np.float32)   # sub-group 0
                quant_hi = quant[:, :, 16:].astype(np.float32)   # sub-group 1

                # 每个 _j 对应 2 个 scale 字节: sc[_j*2] → sub0, sc[_j*2+1] → sub1
                sc_lo = sc_half[:, :, _j*2]                  # (N, n_blocks)
                sc_hi = sc_half[:, :, _j*2+1]                # (N, n_blocks)

                scale_lo = (sc_lo & 0xF).astype(np.float32)
                min_lo = (sc_lo >> 4).astype(np.float32)
                scale_hi = (sc_hi & 0xF).astype(np.float32)
                min_hi = (sc_hi >> 4).astype(np.float32)

                # d*scale*quant - dmin*min
                weight[:, :, k_half, _j, 0, :] = (
                    d_2d[..., np.newaxis] * scale_lo[..., np.newaxis] * quant_lo
                    - dmin_2d[..., np.newaxis] * min_lo[..., np.newaxis])
                weight[:, :, k_half, _j, 1, :] = (
                    d_2d[..., np.newaxis] * scale_hi[..., np.newaxis] * quant_hi
                    - dmin_2d[..., np.newaxis] * min_hi[..., np.newaxis])

        return weight.reshape(N, K).astype(np.float16)

    def _upload_expert_to_gpu(self, layer_id: int, expert_id: int) -> dict:
        """将专家权重量化格式数据直接上传到 GPU（不反量化）。

        上传格式与 mixed_quant_gemm.py 的 kernel 接口一致：
          - IQ2_XXS: d [N, n_blocks] fp16, qs [N, n_blocks, 32] uint16
          - Q2_K: d [N, n_blocks] fp16, dmin [N, n_blocks] fp16,
                  scales [N, n_blocks, 16] uint8, qs [N, n_blocks, 64] uint8

        每专家 ~0.5MB（vs BF16 ~24MB），显存占用极小。
        """
        import torch

        raw = self._load_raw(layer_id, expert_id)

        # gate (IQ2_XXS): raw = ('iq2xxs', d_flat, qs_flat, logical_shape)
        wt = raw['w1']
        assert wt[0] == 'iq2xxs'
        gate_ne0, gate_ne1 = wt[-1][1], wt[-1][0]  # (K, N)
        gate_n_blocks = gate_ne0 // self.QK_K
        gate_d = torch.from_numpy(wt[1].reshape(gate_ne1, gate_n_blocks).astype(np.float16)).cuda()
        gate_qs = torch.from_numpy(wt[2].reshape(gate_ne1, gate_n_blocks, 32).copy()).cuda()

        # up (IQ2_XXS)
        wt = raw['w3']
        assert wt[0] == 'iq2xxs'
        up_ne0, up_ne1 = wt[-1][1], wt[-1][0]
        up_n_blocks = up_ne0 // self.QK_K
        up_d = torch.from_numpy(wt[1].reshape(up_ne1, up_n_blocks).astype(np.float16)).cuda()
        up_qs = torch.from_numpy(wt[2].reshape(up_ne1, up_n_blocks, 32).copy()).cuda()

        # down (Q2_K): raw = ('q2k', d_flat, dmin_flat, scales_flat, qs_flat, logical_shape)
        wt = raw['w2']
        assert wt[0] == 'q2k'
        down_ne0, down_ne1 = wt[-1][1], wt[-1][0]
        down_n_blocks = down_ne0 // self.QK_K
        down_d = torch.from_numpy(wt[1].reshape(down_ne1, down_n_blocks).astype(np.float16)).cuda()
        down_dmin = torch.from_numpy(wt[2].reshape(down_ne1, down_n_blocks).astype(np.float16)).cuda()
        down_scales = torch.from_numpy(wt[3].reshape(down_ne1, down_n_blocks, 16).copy()).cuda()
        down_qs = torch.from_numpy(wt[4].reshape(down_ne1, down_n_blocks, 64).copy()).cuda()

        return {
            'gate_d': gate_d, 'gate_qs': gate_qs,
            'up_d': up_d, 'up_qs': up_qs,
            'down_d': down_d, 'down_dmin': down_dmin,
            'down_scales': down_scales, 'down_qs': down_qs,
        }

    def gpu_cache_contains(self, layer_id: int, expert_id: int) -> bool:
        """检查专家是否在 GPU 缓存中。"""
        return (layer_id, expert_id) in self._gpu_cache

    def record_gpu_access(self, layer_id: int, expert_id: int) -> bool:
        """记录 GPU 访问频率，一次命中准入。返回 True 表示准入。

        首次 Gate 选中即上传到 GPU（一次命中准入），
        利用马太效应：热专家越用越热，首次命中即缓存可最大化命中率。
        """
        if not self._gpu_ffn:
            return False
        key = (layer_id, expert_id)
        self._gpu_freq[key] = self._gpu_freq.get(key, 0) + 1
        self._gpu_access_counter += 1
        self._gpu_last_access[key] = self._gpu_access_counter
        # 一次命中准入：只要不在缓存中就准入
        admitted = key not in self._gpu_cache
        return admitted

    def get_gpu_weights(self, layer_id: int, expert_id: int) -> dict | None:
        """获取 GPU 缓存的专家权重。"""
        return self._gpu_cache.get((layer_id, expert_id))

    def upload_to_gpu_cache(self, layer_id: int, expert_id: int):
        """将专家上传到 GPU 缓存（一次命中准入后调用）。

        每专家量化格式 ~0.5MB（IQ2_XXS+Q2_K），远小于 BF16 ~24MB。
        OOM 时自动淘汰频率最低 + 最久未访问的专家（LFU + LRU）。
        """
        if not self._gpu_ffn:
            return
        key = (layer_id, expert_id)
        if key in self._gpu_cache:
            return

        # 自动计算容量：根据可用显存估算
        if self._gpu_cache_capacity <= 0:
            import torch
            free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
            # 每专家 ~7MB 量化数据，预留 500MB 给计算
            self._gpu_cache_capacity = max(10, int((free_mb - 500) / 7.0))

        # 淘汰：LFU 优先，同频率时 LRU（最久未访问）
        while len(self._gpu_cache) >= self._gpu_cache_capacity:
            if not self._gpu_cache:
                break
            cache_keys = list(self._gpu_cache.keys())
            # LFU + LRU：先按频率升序，同频率按最后访问时间升序
            min_key = min(cache_keys,
                          key=lambda k: (self._gpu_freq.get(k, 0), self._gpu_last_access.get(k, 0)))
            del self._gpu_cache[min_key]

        try:
            gpu_weights = self._upload_expert_to_gpu(layer_id, expert_id)
            self._gpu_cache[key] = gpu_weights
            self._gpu_upload_count += 1
            if self._gpu_upload_count <= 5 or self._gpu_upload_count % 100 == 0:
                print(f"[MixedQuant GPU] uploaded L{layer_id}E{expert_id}, "
                      f"cache_size={len(self._gpu_cache)}/{self._gpu_cache_capacity}, "
                      f"vram={self.gpu_cache_vram_mb():.1f}MB")
        except RuntimeError as e:
            import torch
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                # OOM 时淘汰一半缓存重试
                n_evict = max(1, len(self._gpu_cache) // 2)
                keys = sorted(self._gpu_cache, key=lambda k: (self._gpu_freq.get(k, 0), self._gpu_last_access.get(k, 0)))
                for k in keys[:n_evict]:
                    del self._gpu_cache[k]
                try:
                    gpu_weights = self._upload_expert_to_gpu(layer_id, expert_id)
                    self._gpu_cache[key] = gpu_weights
                    self._gpu_upload_count += 1
                except Exception as e2:
                    print(f"[MixedQuant GPU] upload retry failed L{layer_id}E{expert_id}: {e2}")
            else:
                print(f"[MixedQuant GPU] upload failed L{layer_id}E{expert_id}: {e}")
        except Exception as e:
            print(f"[MixedQuant GPU] upload failed L{layer_id}E{expert_id}: {e}")

    def compute_ffn_gpu(self, layer_id: int, expert_id: int,
                        x_gpu: 'torch.Tensor', route_weight: float = 1.0) -> 'torch.Tensor':
        """GPU FFN：量化权重已在 GPU 上，使用 mixed_quant_gemm 融合反量化+GEMM。"""
        import time
        import torch
        import sys
        sys.path.insert(0, '/workspace/tilelang')
        from mixed_quant_gemm import mixed_quant_ffn

        gpu_w = self._gpu_cache.get((layer_id, expert_id))
        if gpu_w is None:
            return None

        # x_gpu: [1, K] BF16 → mixed_quant_ffn 期望 [M, K]
        x_2d = x_gpu.unsqueeze(0) if x_gpu.dim() == 1 else x_gpu

        _t0 = time.time()
        y = mixed_quant_ffn(
            x_2d,
            gpu_w['gate_d'], gpu_w['gate_qs'],
            gpu_w['up_d'], gpu_w['up_qs'],
            gpu_w['down_d'], gpu_w['down_dmin'], gpu_w['down_scales'], gpu_w['down_qs'],
        )
        self._gpu_ffn_time += time.time() - _t0

        return (y.squeeze(0) if y.shape[0] == 1 else y) * route_weight

    def gpu_cache_size(self) -> int:
        """GPU 缓存中的专家数量。"""
        return len(self._gpu_cache)

    def gpu_cache_vram_mb(self) -> float:
        """GPU 缓存占用显存 (MB)。"""
        total = 0
        for gpu_w in list(self._gpu_cache.values()):  # list() 防止迭代期间字典修改
            for v in gpu_w.values():
                total += v.nelement() * v.element_size()
        return total / 1024 / 1024

    # ========================================================================
    # 预测性缓存预热：统一 warmup + 异步预取 + 一次命中准入
    # ========================================================================
    #
    # 三种触发方式共享同一套上传基础设施：
    #   1. Warmup（启动时）：批量预取所有层 Top-N 热专家
    #   2. Predictive prefetch（层间）：L 层计算时预取 L+1 层预测专家
    #   3. One-hit admission（访问时）：首次 Gate 选中触发预取
    #
    # 所有上传都通过异步线程池执行，不阻塞推理热路径。
    # 频率数据持久化，热启动直接加载历史频率。

    def _ensure_prefetch_resources(self):
        """延迟创建线程池。"""
        if self._prefetch_executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

    def _async_upload(self, layer_id: int, expert_id: int):
        """异步上传专家到 GPU（非阻塞，线程池执行）。

        所有上传统一走此方法，避免推理热路径中的同步上传。
        """
        self._ensure_prefetch_resources()
        key = (layer_id, expert_id)

        with self._prefetch_lock:
            if key in self._prefetch_pending:
                return  # 已在预取队列中
            self._prefetch_pending.add(key)

        def _do_upload():
            try:
                self.upload_to_gpu_cache(layer_id, expert_id)
            except Exception:
                pass
            finally:
                with self._prefetch_lock:
                    self._prefetch_pending.discard((layer_id, expert_id))

        self._prefetch_executor.submit(_do_upload)

    def record_gpu_access(self, layer_id: int, expert_id: int):
        """记录 GPU 访问频率 + 二次触发准入。

        二次触发准入：第 2 次 Gate 选中时才触发异步上传。
        减少同步频次：首次选中的冷专家大概率不再被选中，
        避免浪费上传带宽和缓存空间。

        如果专家已在缓存中（来自 warmup/prefetch），直接命中。
        """
        if not self._gpu_ffn:
            return
        key = (layer_id, expert_id)
        self._gpu_freq[key] = self._gpu_freq.get(key, 0) + 1
        self._gpu_access_counter += 1
        self._gpu_last_access[key] = self._gpu_access_counter

        # 二次触发准入：第 2 次命中且不在缓存中 → 异步上传
        if self._gpu_freq[key] >= 2 and key not in self._gpu_cache:
            self._async_upload(layer_id, expert_id)

    def predictive_prefetch(self, layer_id: int, activated_ids: list[int],
                            n_layers: int = 43, prefetch_count: int = None, prefetch_layers: int = None):
        """延迟感知预测性预取未来层专家到 GPU。

        在 L 层 GPU 计算期间启动，未来层需要的专家在后台上传。
        预测策略：当前层激活专家 + 延迟感知排序 → 未来层差集 Top-N。

        P3 延迟感知：排序因子 = freq × latency_saving
        - freq: 专家被 Gate 选中的频率（越高越值得预取）
        - latency_saving: GPU hit 节省的延迟（cpu_ffn_ms - gpu_ffn_ms）
        - 效果：优先预取"高频 + 延迟差大"的专家，而非单纯高频

        关键约束：缓存容量有限（535 专家），预取总量不能超过剩余空间，
        否则会导致缓存颠簸（evict→upload→evict）和内存压力。

        Args:
            layer_id: 当前层 ID
            activated_ids: 当前层激活的专家 ID（用于预测）
            n_layers: 总层数（边界检查）
            prefetch_count: 每层预取的最大专家数（None=使用 self.prefetch_count）
            prefetch_layers: 向前预取的层数（None=使用 self.prefetch_layers）
        """
        if not self._gpu_ffn or self._gpu_cache_capacity <= 0:
            return

        if prefetch_count is None:
            prefetch_count = self.prefetch_count
        if prefetch_layers is None:
            prefetch_layers = self.prefetch_layers

        # P3: 延迟感知因子
        # GPU hit 节省的延迟 = CPU FFN 延迟 - GPU FFN 延迟
        gpu_avg_ms = self._gpu_ffn_time / self._gpu_ffn_hits * 1000 if self._gpu_ffn_hits > 0 else 0.5
        cpu_avg_ms = self._cpu_ffn_time / self._gpu_ffn_misses * 1000 if self._gpu_ffn_misses > 0 else 2.7
        latency_saving = max(cpu_avg_ms - gpu_avg_ms, 0.1)  # 至少 0.1ms，避免零值

        # 预取总量上限：缓存剩余空间（避免颠簸）
        free_slots = self._gpu_cache_capacity - len(self._gpu_cache)
        total_budget = max(free_slots, 0) + 5  # 允许少量淘汰，但不超过 5 个
        total_uploaded = 0

        for offset in range(1, prefetch_layers + 1):
            if total_uploaded >= total_budget:
                break

            target_layer = layer_id + offset
            if target_layer < 0 or target_layer >= n_layers:
                continue

            # 预测目标层需要的专家：
            # 1. 当前层激活的专家（相邻层专家激活高度相关）
            predicted = set(activated_ids)

            # 2. 目标层按延迟感知排序的 Top-N
            # P3: 排序因子 = freq × latency_saving（而非单纯 freq）
            layer_score = [(eid, self._gpu_freq.get((target_layer, eid), 0) * latency_saving)
                           for eid in range(256)]
            layer_score.sort(key=lambda x: -x[1])
            for eid, _ in layer_score[:prefetch_count]:
                predicted.add(eid)

            # 差集 = 不在 GPU 缓存中的 → 异步上传
            # P3: 按延迟感知分数排序（而非频率）
            predicted_sorted = sorted(predicted,
                                      key=lambda eid: -self._gpu_freq.get((target_layer, eid), 0) * latency_saving)
            for eid in predicted_sorted:
                if total_uploaded >= total_budget:
                    break
                key = (target_layer, eid)
                if key not in self._gpu_cache:
                    self._async_upload(target_layer, eid)
                    total_uploaded += 1

    def warmup_gpu_cache(self, n_layers: int = 43, top_n_per_layer: int = 6):
        """启动时根据频率数据批量预热 GPU 缓存。

        利用持久化的频率数据，热启动时跳过冷启动阶段，
        直接将高频专家上传到 GPU 缓存。

        冷启动（无频率数据）时，上传每层前 top_n_per_layer 个专家作为初始缓存。

        Args:
            n_layers: 层数
            top_n_per_layer: 每层预热的专家数
        """
        if not self._gpu_ffn:
            return

        uploaded = 0
        for layer_id in range(n_layers):
            if self._gpu_freq:
                # 热启动：按频率降序
                layer_freq = [(eid, self._gpu_freq.get((layer_id, eid), 0))
                              for eid in range(256)]
                layer_freq.sort(key=lambda x: -x[1])
                candidates = [(eid, freq) for eid, freq in layer_freq[:top_n_per_layer] if freq > 0]
            else:
                # 冷启动：上传前 N 个专家
                candidates = [(eid, 0) for eid in range(min(top_n_per_layer, 256))]

            for eid, _ in candidates:
                if (layer_id, eid) not in self._gpu_cache:
                    try:
                        self.upload_to_gpu_cache(layer_id, eid)
                        uploaded += 1
                    except Exception:
                        pass

        if uploaded > 0:
            print(f"[MixedQuant GPU] Warmup: uploaded {uploaded} experts, "
                  f"cache_size={len(self._gpu_cache)}, vram={self.gpu_cache_vram_mb():.1f}MB")

    # ========================================================================
    # 频率持久化：热启动时直接加载频率数据
    # ========================================================================

    def _load_freq(self):
        """从 JSON 文件加载频率数据。"""
        import json
        if not os.path.exists(self._freq_path):
            return
        try:
            with open(self._freq_path, 'r') as f:
                data = json.load(f)
            for key_str, freq in data.items():
                parts = key_str.split(',')
                if len(parts) == 2:
                    self._gpu_freq[(int(parts[0]), int(parts[1]))] = freq
            print(f"[MixedQuant GPU] Loaded freq data: {len(self._gpu_freq)} entries from {self._freq_path}")
        except Exception as e:
            print(f"[MixedQuant GPU] Failed to load freq: {e}")

    def save_freq(self):
        """保存频率数据到 JSON 文件。"""
        import json
        if not self._gpu_freq:
            return
        try:
            data = {f"{k[0]},{k[1]}": v for k, v in self._gpu_freq.items()}
            with open(self._freq_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[MixedQuant GPU] Failed to save freq: {e}")

    # ========================================================================
    # GPU FFN 统计
    # ========================================================================

    def record_gpu_ffn_hit(self):
        """记录 GPU FFN 命中。"""
        self._gpu_ffn_hits += 1

    def record_gpu_ffn_miss(self):
        """记录 GPU FFN miss。"""
        self._gpu_ffn_misses += 1

    def get_gpu_stats(self) -> dict:
        """获取 GPU FFN 统计信息。"""
        total = self._gpu_ffn_hits + self._gpu_ffn_misses
        hit_rate = self._gpu_ffn_hits / total * 100 if total > 0 else 0
        gpu_avg_ms = self._gpu_ffn_time / self._gpu_ffn_hits * 1000 if self._gpu_ffn_hits > 0 else 0
        cpu_avg_ms = self._cpu_ffn_time / self._gpu_ffn_misses * 1000 if self._gpu_ffn_misses > 0 else 0
        return {
            'hits': self._gpu_ffn_hits,
            'misses': self._gpu_ffn_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._gpu_cache),
            'cache_capacity': self._gpu_cache_capacity,
            'vram_mb': self.gpu_cache_vram_mb(),
            'uploads': self._gpu_upload_count,
            'gpu_ffn_time_s': self._gpu_ffn_time,
            'cpu_ffn_time_s': self._cpu_ffn_time,
            'gpu_avg_ms': gpu_avg_ms,
            'cpu_avg_ms': cpu_avg_ms,
        }

    def print_gpu_stats(self):
        """打印 GPU FFN 统计信息。"""
        stats = self.get_gpu_stats()
        print(f"[MixedQuant GPU] hits={stats['hits']}, misses={stats['misses']}, "
              f"hit_rate={stats['hit_rate']:.1f}%, "
              f"cache={stats['cache_size']}/{stats['cache_capacity']}, "
              f"vram={stats['vram_mb']:.1f}MB, uploads={stats['uploads']}, "
              f"gpu_ffn={stats['gpu_avg_ms']:.2f}ms/hit, cpu_ffn={stats['cpu_avg_ms']:.2f}ms/miss, "
              f"gpu_total={stats['gpu_ffn_time_s']:.3f}s, cpu_total={stats['cpu_ffn_time_s']:.3f}s")
