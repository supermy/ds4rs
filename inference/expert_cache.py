"""
ExpertCache v8: GPU SLRU + CPU SLRU + SSD5

简化策略（AGENTS.md L16-20）：
  GPU：采用SLRU；如果缓存不中从CPU DMA获取；
  CPU：采用SLRU；如果缓存不中从SSD5获取；
  SSD5全量路由专家；
  GPU/CPU M层N专家编号+频率到磁盘，支持热启动。

三级缓存架构：
  L1 GPU:  SLRU（分段LRU，热点稳定保留）
           protected段(多次访问) + probation段(首次访问)
           容量由VRAM动态决定，最少12专家/层
           DMA异步预取L+2层TopN热点专家

  L2 CPU:  SLRU（分段LRU，区分热点/冷门）
           protected段(多次访问) + probation段(首次访问)
           容量由RAM动态决定

  L3 SSD:  safetensors 直接 I/O
"""

import gc
import json
import os
import struct
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from iq2xs_archive import IQ2XSArchiveReader


class GpuLFU:
    """GPU 缓存：LFU 策略 + 频率衰减。

    频率驱动淘汰：淘汰频率最低的专家，保留热点专家。
    频率衰减：定期减半，适应访问模式变化。
    """

    def __init__(self, capacity: int = 250):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()  # key → (params, freq)
        self._access_count = 0
        self._decay_interval = 2000

    def get(self, key: tuple) -> dict | None:
        if key in self.cache:
            params, freq = self.cache[key]
            self.cache[key] = (params, freq + 1)
            self.cache.move_to_end(key)
            self._access_count += 1
            if self._access_count >= self._decay_interval:
                self._decay_frequencies()
            return params
        return None

    def put_force(self, key: tuple, params: dict) -> tuple | None:
        """强制插入，满时淘汰频率最低的专家，返回被淘汰的 (key, params)。"""
        if key in self.cache:
            _, freq = self.cache[key]
            self.cache[key] = (params, freq + 1)
            self.cache.move_to_end(key)
            return None

        evicted = None
        if len(self.cache) >= self.capacity:
            evicted = self._evict_min_freq()
        self.cache[key] = (params, 1)
        return evicted

    def _evict_min_freq(self) -> tuple | None:
        """淘汰频率最低的专家。"""
        min_key = None
        min_freq = float('inf')
        for key, (_, freq) in self.cache.items():
            if freq < min_freq:
                min_freq = freq
                min_key = key
        if min_key is not None:
            params, _ = self.cache.pop(min_key)
            return (min_key, params)
        return None

    def _decay_frequencies(self):
        """频率衰减：减半，为0则移除。"""
        to_remove = []
        for key in list(self.cache.keys()):
            params, freq = self.cache[key]
            new_freq = freq // 2
            if new_freq == 0:
                to_remove.append(key)
            else:
                self.cache[key] = (params, new_freq)
        for key in to_remove:
            del self.cache[key]
        self._access_count = 0

    def contains(self, key: tuple) -> bool:
        return key in self.cache

    def total_entries(self) -> int:
        return len(self.cache)

    def all_items(self):
        for key, (params, _) in self.cache.items():
            yield key, params

    def clear(self):
        self.cache.clear()


class GpuSLRU:
    """GPU 缓存：SLRU（Segmented LRU）策略。

    protected 段: 热点专家，LRU 驱动
    probation 段: 新准入专家，LRU 驱动
    新插入 → probation；probation 命中 → 提升 protected

    优势：
    - 热点专家稳定保留在 protected 段
    - 新专家有机会进入 probation 段
    - 防止冷专家挤占热点空间
    """

    def __init__(self, capacity: int = 250):
        self.capacity = capacity
        self.protected: OrderedDict = OrderedDict()
        self.probation: OrderedDict = OrderedDict()
        # protected 40% / probation 60%：GPU 显存宝贵，protected 占比更高
        self.protected_capacity = int(capacity * 0.4)
        self.probation_capacity = capacity - self.protected_capacity

    def get(self, key: tuple) -> dict | None:
        """获取专家，命中 probation 则提升到 protected。"""
        if key in self.protected:
            self.protected.move_to_end(key)
            return self.protected[key]
        if key in self.probation:
            params = self.probation.pop(key)
            self._promote_to_protected(key, params)
            return self.protected.get(key, params)
        return None

    def put_force(self, key: tuple, params: dict) -> tuple | None:
        """强制插入，满时淘汰 probation 段 LRU，返回被淘汰的 (key, params)。"""
        if key in self.protected:
            self.protected[key] = params
            self.protected.move_to_end(key)
            return None
        if key in self.probation:
            params_old = self.probation.pop(key)
            self._promote_to_protected(key, params)
            return None

        evicted = None
        # 先尝试插入 probation
        if self.probation_capacity > 0:
            if len(self.probation) >= self.probation_capacity:
                evicted = self.probation.popitem(last=False)
            self.probation[key] = params
        elif self.protected_capacity > 0:
            # probation 容量为 0，直接插入 protected
            if len(self.protected) >= self.protected_capacity:
                evicted = self.protected.popitem(last=False)
            self.protected[key] = params
        return evicted

    def _promote_to_protected(self, key: tuple, params: dict):
        """将专家从 probation 提升到 protected。"""
        if self.protected_capacity <= 0:
            return
        if len(self.protected) >= self.protected_capacity:
            # protected 满，demote LRU 到 probation
            demote_key, demote_params = self.protected.popitem(last=False)
            if self.probation_capacity > 0:
                if len(self.probation) >= self.probation_capacity:
                    self.probation.popitem(last=False)
                self.probation[demote_key] = demote_params
        self.protected[key] = params

    def contains(self, key: tuple) -> bool:
        return key in self.protected or key in self.probation

    def total_entries(self) -> int:
        return len(self.protected) + len(self.probation)

    def all_items(self):
        yield from self.protected.items()
        yield from self.probation.items()

    def clear(self):
        self.protected.clear()
        self.probation.clear()


class CpuSLRU:
    """CPU 缓存：SLRU（Segmented LRU）策略。

    protected 段: 已被访问 2 次及以上，LRU 驱动
    probation 段: 仅被访问 1 次，LRU 驱动
    新插入 → probation；probation 命中 → 提升 protected
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.protected: OrderedDict = OrderedDict()
        self.probation: OrderedDict = OrderedDict()
        # protected 30% / probation 70%：probation 段大，容纳更多新准入专家
        self.protected_capacity = int(capacity * 0.3)
        self.probation_capacity = capacity - self.protected_capacity

    def get(self, key: tuple, record_freq: bool = True) -> dict | None:
        if key in self.protected:
            self.protected.move_to_end(key)
            return self.protected[key]
        if key in self.probation:
            if record_freq:
                # 访问频率记录：从 probation 移除并尝试晋升到 protected
                params = self.probation.pop(key)
                self._promote_to_protected(key, params)
                if key in self.protected:
                    return self.protected[key]
                # 晋升失败（protected 满且降级失败），放回 probation
                self.probation[key] = params
                return params
            else:
                # 仅读取，不从缓存中移除（避免预取偷走 probation 条目）
                self.probation.move_to_end(key)
                return self.probation[key]
        return None

    def put(self, key: tuple, params: dict):
        if key in self.protected:
            self.protected[key] = params
            self.protected.move_to_end(key)
            return
        if key in self.probation:
            self.probation[key] = params
            self.probation.move_to_end(key)
            return
        if self.probation_capacity > 0:
            if len(self.probation) >= self.probation_capacity:
                self.probation.popitem(last=False)
            self.probation[key] = params
        elif self.protected_capacity > 0:
            if len(self.protected) >= self.protected_capacity:
                self._demote_protected()
            self.protected[key] = params

    def _promote_to_protected(self, key: tuple, params: dict):
        if self.protected_capacity <= 0:
            if self.probation_capacity > 0:
                if len(self.probation) >= self.probation_capacity:
                    self.probation.popitem(last=False)
                self.probation[key] = params
            return
        if len(self.protected) >= self.protected_capacity:
            self._demote_protected()
        self.protected[key] = params

    def _demote_protected(self):
        if not self.protected:
            return
        key, params = self.protected.popitem(last=False)
        if self.probation_capacity > 0:
            if len(self.probation) >= self.probation_capacity:
                self.probation.popitem(last=False)
            self.probation[key] = params

    def contains(self, key: tuple) -> bool:
        return key in self.protected or key in self.probation

    def total_entries(self) -> int:
        return len(self.protected) + len(self.probation)

    def all_items(self):
        yield from self.protected.items()
        yield from self.probation.items()

    def clear(self):
        self.protected.clear()
        self.probation.clear()


class ExpertCache:
    """
    三级缓存管理器：GPU SLRU → CPU SLRU → SSD。

    L1 GPU: SLRU（分段LRU，热点稳定保留）
    L2 CPU: SLRU（分段LRU，区分热点/冷门）
    L3 SSD: safetensors 直接 I/O
    """

    MEM_SAFE_LIMIT_GB = 20  # 预留系统+模型+推理buffer+scale
    MIN_EXPERTS_PER_LAYER = 12

    EXPERT_MB = 12.0  # 仅 weight (I8)，scale 从 RAM 读取
    VRAM_TOTAL_MB = 15.5 * 1024
    VRAM_RESERVE_MB = 800
    VRAM_COMPUTE_OVERHEAD_MB = 1000
    # IQ2_XS 反量化创建 float32 中间张量（~112MB/权重），需额外 VRAM 预留
    # 峰值：1个权重 float32 (~112MB) + 预取可能同时1个 = ~224MB
    # 实测需要更多，设为 500MB
    IQ2_XS_VRAM_OVERHEAD_MB = 500
    CACHE_MISS_RESERVE_EXPERTS = 4

    _DTYPE_MAP = {
        "BF16": (torch.bfloat16, 2),
        "FP8_E4M3": (torch.float8_e4m3fn, 1),
        "F8_E4M3": (torch.float8_e4m3fn, 1),
        "FP8_E8M0": (torch.float8_e8m0fnu, 1),
        "F8_E8M0": (torch.float8_e8m0fnu, 1),
        "INT8": (torch.int8, 1),
        "I8": (torch.int8, 1),
        "I64": (torch.int64, 8),
        "F32": (torch.float32, 4),
        "F16": (torch.float16, 2),
    }

    def __init__(self, expert_key_map: dict, top_n: int = 12, window_m: int = 3,
                 cpu_cache_size: int = 2000, cpu_top_n: int = 100,
                 iq2xs_dir: str = ""):
        """
        Args:
            iq2xs_dir: IQ2_XS 归档文件目录（如 /models_iq2xs）
        """
        self.expert_key_map = expert_key_map
        self.top_n = top_n
        self.window_m = window_m
        self.cpu_cache_size = cpu_cache_size
        self._cpu_top_n = cpu_top_n
        self._iq2xs_dir = iq2xs_dir

        # IQ2_XS 归档读取器（mmap）
        self._iq2xs_archive: Optional[IQ2XSArchiveReader] = None

        # 检测归档文件
        if self._iq2xs_dir:
            archive_path = os.path.join(self._iq2xs_dir, "experts.iq2xs")
            if os.path.exists(archive_path):
                try:
                    self._iq2xs_archive = IQ2XSArchiveReader(archive_path)
                    print(f"[IQ2_XS] 使用归档文件: {archive_path}")
                except Exception as e:
                    print(f"[IQ2_XS] 归档打开失败: {e}")

        # GPU 缓存：SLRU（热点稳定保留）
        self._gpu_cache = GpuSLRU(capacity=250)
        self._cpu_cache = CpuSLRU(capacity=cpu_cache_size)

        self._current_layer = -1
        self._n_layers = 0
        self._vram_adjusted = False

        self._layer_stats: Dict[int, dict] = defaultdict(
            lambda: {"gpu": 0, "cpu": 0, "ssd": 0, "ram": 0, "total": 0})

        self._param_path_cache: Dict[str, tuple] = {}

        self._stats = {"gpu_hits": 0, "cpu_hits": 0, "ssd_hits": 0,
                       "ram_hits": 0, "gpu_evictions": 0,
                       "prefetch_hits": 0, "prefetch_misses": 0}

        self._shard_metadata: Dict[str, Dict] = {}

        self._transfer_stream = torch.cuda.Stream()
        self._prefetch_pending: Dict[int, Dict[int, dict]] = {}

        self._cpu_prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu_prefetch")
        self._cpu_prefetch_pending: set = set()  # (layer_id, expert_id) 正在预取

        self._ram_weight_cache: Dict[str, bytes] = {}
        self._ram_scale_cache: Dict[str, bytes] = {}
        self._ram_scale_shapes: Dict[str, tuple] = {}
        self._ram_cache_loaded = False
        self._model = None
        self._cache_dir: str = ""

        self._layer_freq: Dict[int, dict] = defaultdict(dict)

    def setup(self, model, n_layers: int, cache_dir: str = ""):
        self._n_layers = n_layers
        self._cache_dir = cache_dir
        self._model = model

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value: int):
        self._n_layers = value

    @property
    def cpu_top_n(self) -> int:
        return self._cpu_top_n

    def total_freq_entries(self) -> int:
        return sum(len(f) for f in self._layer_freq.values())

    def get_expert_gpu_params(self, layer_id: int, expert_id: int, topk_info: list = None) -> dict | None:
        """公共 API: 获取专家 GPU 参数，支持路由预测预取。

        查找顺序: GPU SLRU → DMA 预取结果 → CPU SLRU → SSD

        参数:
            layer_id: 层 ID
            expert_id: 专家 ID
            topk_info: 当前层激活的 (expert_id, score) 列表，用于路由预测预取
        """
        key = (layer_id, expert_id)
        self._record_access(layer_id, expert_id)

        # L1: GPU SLRU
        gpu_params = self._gpu_cache.get(key)
        if gpu_params is not None:
            self._stats["gpu_hits"] += 1
            self._layer_stats[layer_id]["gpu"] += 1
            self._layer_stats[layer_id]["total"] += 1
            self._validate_layer(gpu_params, layer_id, expert_id)
            # 路由预测预取
            if topk_info is not None:
                self.prefetch_by_route_prediction(layer_id, topk_info)
            return gpu_params

        # L1.5: DMA 预取结果
        if layer_id in self._prefetch_pending and expert_id in self._prefetch_pending[layer_id]:
            self._stats["prefetch_hits"] += 1
            gpu_params = self._prefetch_pending[layer_id].pop(expert_id)
            self._validate_layer(gpu_params, layer_id, expert_id)
            self._gpu_put(key, gpu_params)
            # 路由预测预取
            if topk_info is not None:
                self.prefetch_by_route_prediction(layer_id, topk_info)
            return gpu_params

        self._stats["prefetch_misses"] += 1

        # L2: CPU SLRU
        cpu_params = self._cpu_cache.get(key)
        if cpu_params is not None:
            self._stats["cpu_hits"] += 1
            self._layer_stats[layer_id]["cpu"] += 1
            self._layer_stats[layer_id]["total"] += 1
            # CPU 缓存只有 weight，需要从 RAM 读取 scale
            # IQ2_XS 反量化为 FP8 后自带 scale，跳过 RAM scale
            full_params = dict(cpu_params)
            has_iq2_xs = any(isinstance(v, dict) and v.get("__iq2xs__") for v in full_params.values())
            if not has_iq2_xs:
                if layer_id in self.expert_key_map and expert_id in self.expert_key_map[layer_id]:
                    for skey, shard_path in self.expert_key_map[layer_id][expert_id]:
                        if 'scale' in skey and skey not in full_params:
                            full_params[skey] = self._read_tensor_no_mmap(shard_path, skey, count_ram=True)
            gpu_params = self._pinned_to_gpu(full_params, layer_id)
            self._gpu_put(key, gpu_params)
            # 路由预测预取
            if topk_info is not None:
                self.prefetch_by_route_prediction(layer_id, topk_info)
            return gpu_params

        # L3: SSD
        self._stats["ssd_hits"] += 1
        self._layer_stats[layer_id]["ssd"] += 1
        self._layer_stats[layer_id]["total"] += 1

        # IQ2_XS 模式：优先从归档读取 IQ2_XS 数据（而非 safetensors 的 FP4）
        if self._iq2xs_archive is not None:
            cpu_params = {}
            for weight_type, weight_name in enumerate(['w1', 'w2', 'w3']):
                result = self._iq2xs_archive.get_expert(layer_id, expert_id, weight_type)
                if result is None:
                    continue
                d, qs, scales, shape = result
                skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{weight_name}.weight"
                cpu_params[skey] = {
                    "__iq2xs__": True,
                    "d": torch.from_numpy(d),
                    "qs": torch.from_numpy(qs),
                    "scales": torch.from_numpy(scales),
                    "shape": shape,
                }
            if cpu_params:
                self._cpu_cache.put(key, cpu_params)
                gpu_params = self._pinned_to_gpu(cpu_params, layer_id)
                if gpu_params:
                    self._gpu_put(key, gpu_params)
                    if topk_info is not None:
                        self.prefetch_by_route_prediction(layer_id, topk_info)
                    return gpu_params

        # FP4 回退：从 safetensors 读取
        raw_tensors = {}
        ram_hit_count = 0
        total_count = 0
        if layer_id in self.expert_key_map and expert_id in self.expert_key_map[layer_id]:
            for skey, shard_path in self.expert_key_map[layer_id][expert_id]:
                total_count += 1
                before_ram = self._stats["ram_hits"]
                raw_tensors[skey] = self._read_tensor_no_mmap(shard_path, skey)
                if self._stats["ram_hits"] > before_ram:
                    ram_hit_count += 1

        if ram_hit_count > 0 and ram_hit_count == total_count:
            self._stats["ssd_hits"] -= 1
            self._stats["ram_hits"] -= total_count
            self._stats["ram_hits"] += 1
            self._layer_stats[layer_id]["ssd"] -= 1
            self._layer_stats[layer_id]["ram"] += 1

        if not raw_tensors:
            return None

        self._put_cpu_cache(key, raw_tensors)
        gpu_params = self._raw_to_gpu(raw_tensors, layer_id)
        if not gpu_params:
            return None
        self._gpu_put(key, gpu_params)
        # 路由预测预取
        if topk_info is not None:
            self.prefetch_by_route_prediction(layer_id, topk_info)
        return gpu_params

    def on_layer_start(self, layer_id: int, model=None):
        """层计算开始：提升预取结果。"""
        self._current_layer = layer_id
        self._promote_prefetch(layer_id)

    def prefetch_next_layer(self, current_layer: int, activated_indices: list):
        """多级异步预取流水线：GPU←CPU←SSD。

        GPU 计算 L 层时：
          - CPU 异步预取 L+2 层热点专家到 GPU（隐藏 PCIe 延迟）
          - SSD 异步预取 L+5 层热点专家到 CPU（隐藏 SSD I/O 延迟）
        """
        if self._n_layers <= 0:
            return

        # L+2: CPU → GPU 预取
        gpu_prefetch_layer = current_layer + 2
        if gpu_prefetch_layer < self._n_layers:
            self._prefetch_gpu_from_cpu(gpu_prefetch_layer)

        # L+5: SSD → CPU 预取
        cpu_prefetch_layer = current_layer + 5
        if cpu_prefetch_layer < self._n_layers:
            self._prefetch_cpu_from_ssd(cpu_prefetch_layer)

    def _prefetch_gpu_from_cpu(self, layer_id: int):
        """CPU → GPU 异步预取：将热点专家从 CPU 传输到 GPU。"""
        vram_free = self._vram_free_mb()
        prefetch_budget = int((vram_free - self.VRAM_RESERVE_MB) / self.EXPERT_MB)
        if prefetch_budget <= 0:
            return

        prefetched = 0
        top_experts = self._get_top_n_experts_for_layer(layer_id)
        for eid in top_experts:
            if prefetched >= prefetch_budget:
                break
            key = (layer_id, eid)
            if self._gpu_cache.contains(key):
                continue
            if layer_id in self._prefetch_pending and eid in self._prefetch_pending[layer_id]:
                continue
            if layer_id not in self.expert_key_map or eid not in self.expert_key_map[layer_id]:
                continue
            self._async_prefetch_expert(layer_id, eid)
            prefetched += 1

    def _prefetch_cpu_from_ssd(self, layer_id: int):
        """SSD → CPU 异步预取：将热点专家从 SSD 加载到 CPU 缓存。"""
        top_experts = self._get_top_n_experts_for_layer(layer_id)
        for eid in top_experts:
            key = (layer_id, eid)
            if self._cpu_cache.contains(key):
                continue
            if key in self._cpu_prefetch_pending:
                continue
            if layer_id not in self.expert_key_map or eid not in self.expert_key_map[layer_id]:
                continue
            self._cpu_prefetch_pending.add(key)
            self._cpu_prefetch_executor.submit(self._async_load_to_cpu, layer_id, eid)

    def _async_load_to_cpu(self, layer_id: int, expert_id: int):
        """异步加载专家到 CPU 缓存（线程池执行）。"""
        key = (layer_id, expert_id)
        try:
            if layer_id not in self.expert_key_map or expert_id not in self.expert_key_map[layer_id]:
                return
            raw_tensors = {}
            for skey, shard_path in self.expert_key_map[layer_id][expert_id]:
                if 'weight' in skey:
                    raw_tensors[skey] = self._read_tensor_no_mmap(shard_path, skey, count_ram=False)
            if raw_tensors:
                self._put_cpu_cache(key, raw_tensors)
        finally:
            self._cpu_prefetch_pending.discard(key)

    def clear_gpu_cache(self):
        self._gpu_cache.clear()
        self._prefetch_pending.clear()
        self._cpu_prefetch_pending.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def finalize_prefetch(self):
        if not self._prefetch_pending:
            return
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        if self._current_layer >= 0:
            self._promote_prefetch(self._current_layer)

    def adjust_window_for_vram(self):
        """根据 VRAM 动态调整 GPU 缓存容量。"""
        if self._vram_adjusted or self._n_layers <= 0:
            return

        vram_free_mb = self._vram_free_mb()
        # IQ2_XS 模式需额外 VRAM 预留（float32 中间张量）
        compute_overhead = self.VRAM_COMPUTE_OVERHEAD_MB
        if self._iq2xs_archive is not None:
            compute_overhead += self.IQ2_XS_VRAM_OVERHEAD_MB
        safe_budget_mb = vram_free_mb - self.VRAM_RESERVE_MB - compute_overhead
        if safe_budget_mb < 200:
            safe_budget_mb = 200

        cache_miss_mb = self.CACHE_MISS_RESERVE_EXPERTS * self.EXPERT_MB
        expert_budget_mb = safe_budget_mb - cache_miss_mb
        if expert_budget_mb < 100:
            expert_budget_mb = 100

        max_experts = int(expert_budget_mb / self.EXPERT_MB)
        self._gpu_cache.capacity = max_experts

        self.top_n = max(max_experts // self._n_layers, self.MIN_EXPERTS_PER_LAYER)
        self._vram_adjusted = True

        coverage = self.top_n / 256 * 100
        vram_needed = max_experts * self.EXPERT_MB
        print(f"[Cache] VRAM adjust: GPU capacity={max_experts} experts, "
              f"per_layer_coverage={coverage:.0f}%, "
              f"VRAM={vram_needed:.0f}+{cache_miss_mb:.0f}MB(miss), "
              f"budget={safe_budget_mb:.0f}MB")

    def check_memory_pressure(self) -> bool:
        used, available, swap_used = self.get_mem_info()
        if available < self.MEM_SAFE_LIMIT_GB:
            print(f"[Cache] 内存紧张! 可用: {available:.1f}GB < {self.MEM_SAFE_LIMIT_GB}GB, Swap: {swap_used:.1f}GB")
            self._emergency_cleanup()
            return True
        return False

    def get_stats(self) -> str:
        total = (self._stats["gpu_hits"] + self._stats["cpu_hits"] +
                 self._stats["ssd_hits"] + self._stats["ram_hits"])
        if total == 0:
            return "No cache accesses yet"
        gpu_pct = self._stats["gpu_hits"] / total * 100
        cpu_pct = self._stats["cpu_hits"] / total * 100
        ssd_pct = self._stats["ssd_hits"] / total * 100
        ram_pct = self._stats["ram_hits"] / total * 100
        pf_total = self._stats["prefetch_hits"] + self._stats["prefetch_misses"]
        pf_pct = self._stats["prefetch_hits"] / pf_total * 100 if pf_total > 0 else 0
        cpu_accesses = self._stats["cpu_hits"] + self._stats["ssd_hits"]
        cpu_hit_rate = self._stats["cpu_hits"] / cpu_accesses * 100 if cpu_accesses > 0 else 0
        gpu_hit_rate = self._stats["gpu_hits"] / total * 100 if total > 0 else 0
        gpu_total = self._gpu_cache.total_entries()
        cpu_prot = len(self._cpu_cache.protected)
        cpu_prob = len(self._cpu_cache.probation)
        return (f"L1 GPU({gpu_total}): {self._stats['gpu_hits']} ({gpu_pct:.0f}%), "
                f"L2 CPU({cpu_prot}prot+{cpu_prob}prob): {self._stats['cpu_hits']} ({cpu_pct:.0f}%), "
                f"CPU hit: {cpu_hit_rate:.0f}%, GPU hit: {gpu_hit_rate:.0f}%, "
                f"L3 SSD: {self._stats['ssd_hits']} ({ssd_pct:.0f}%), "
                f"RAM: {self._stats['ram_hits']} ({ram_pct:.0f}%), "
                f"PF: {self._stats['prefetch_hits']} ({pf_pct:.0f}%), "
                f"evict: {self._stats['gpu_evictions']}")

    # ==================== GPU 缓存内部 ====================

    def _gpu_put(self, key: tuple, gpu_params: dict):
        """插入 GPU 缓存，满时淘汰最冷专家转存 CPU。

        当前 pass 活跃层的专家不会被淘汰（避免 pass 内换入换出）。
        """
        evicted = self._gpu_cache.put_force(key, gpu_params)
        if evicted is not None:
            evict_key, evict_params = evicted
            if evict_params:
                self._transfer_gpu_to_cpu(evict_key, evict_params)
                self._stats["gpu_evictions"] += 1
            effective_model = self._model
            if effective_model is not None:
                layer_id, expert_id = evict_key
                moe = effective_model.layers[layer_id].ffn
                if expert_id < len(moe.experts) and moe.experts[expert_id] is not None:
                    moe.experts[expert_id] = None

    def _clear_gpu_to_cpu(self, model=None):
        """新 pass 开始时，将 GPU 缓存全部转存到 CPU。"""
        effective_model = model or self._model
        for key, gpu_params in list(self._gpu_cache.all_items()):
            if gpu_params:
                self._transfer_gpu_to_cpu(key, gpu_params)
                self._stats["gpu_evictions"] += 1
            if effective_model is not None:
                layer_id, expert_id = key
                moe = effective_model.layers[layer_id].ffn
                if expert_id < len(moe.experts) and moe.experts[expert_id] is not None:
                    moe.experts[expert_id] = None
        self._gpu_cache.clear()
        torch.cuda.empty_cache()

    def _promote_prefetch(self, current_layer: int):
        """将预取结果提升到 GPU 缓存。过期预取转存 CPU。"""
        if not self._prefetch_pending:
            return
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        for lid in list(self._prefetch_pending.keys()):
            if lid < current_layer:
                pending = self._prefetch_pending.pop(lid)
                for eid, params in pending.items():
                    key = (lid, eid)
                    self._transfer_gpu_to_cpu(key, params)
            else:
                pending = self._prefetch_pending.pop(lid)
                for eid, gpu_params in pending.items():
                    key = (lid, eid)
                    self._gpu_put(key, gpu_params)

    # ==================== IQ2_XS 量化/反量化 ====================

    # IQ2_XS 非均匀量化级别（适配解码后的 FP4 值范围）
    _FP4_TABLE = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32)

    # 量化级别：覆盖 [-6, 6] 范围，重要区域更精细
    _IQ2_XS_LEVELS = torch.tensor([
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25,
        0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    ], dtype=torch.float32)
    _IQ2_XS_QK = 256  # 块大小

    # ==================== CPU 缓存内部 ====================

    def _put_cpu_cache(self, key: tuple, raw_tensors: dict):
        """CPU 缓存只存 weight，scale 从 RAM 读取（已在 _ram_scale_cache）。"""
        if self._cpu_cache.capacity <= 0:
            return
        cpu_params = {}
        for skey, tensor in raw_tensors.items():
            # 跳过 scale，已在 RAM 缓存
            if 'scale' in skey:
                continue
            try:
                if not tensor.is_pinned():
                    tensor = tensor.pin_memory()
            except RuntimeError:
                pass
            cpu_params[skey] = tensor
        if cpu_params:
            self._cpu_cache.put(key, cpu_params)

    def _transfer_gpu_to_cpu(self, key: tuple, gpu_params: dict):
        """GPU 淘汰专家转存到 CPU 缓存（只存 weight，scale 从 RAM 读取）。"""
        if not gpu_params:
            return
        layer_id, expert_id = key
        try:
            cpu_params = {}
            for param_name, tensor in gpu_params.items():
                if param_name.startswith("__"):
                    continue
                # 跳过 scale，已在 RAM 缓存
                if param_name.endswith('.scale'):
                    continue
                cpu_tensor = tensor.to("cpu", non_blocking=False)
                if not cpu_tensor.is_pinned():
                    try:
                        cpu_tensor = cpu_tensor.pin_memory()
                    except RuntimeError:
                        pass
                skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{param_name}"
                cpu_params[skey] = cpu_tensor
            if cpu_params:
                self._cpu_cache.put(key, cpu_params)
        except Exception:
            pass

    # ==================== DMA 异步预取 ====================

    def _async_prefetch_expert(self, layer_id: int, expert_id: int):
        """DMA 异步预取：优先从 CPU 缓存传输到 GPU。"""
        vram_free_mb = self._vram_free_mb()
        if vram_free_mb < 200:
            return

        key = (layer_id, expert_id)
        cpu_params = self._cpu_cache.get(key, record_freq=False)

        if cpu_params is not None:
            full_params = dict(cpu_params)
            # IQ2_XS 反量化为 FP8 后自带 scale，跳过 RAM scale
            has_iq2_xs = any(isinstance(v, dict) and v.get("__iq2xs__") for v in full_params.values())
            if not has_iq2_xs:
                if layer_id in self.expert_key_map and expert_id in self.expert_key_map[layer_id]:
                    for skey, shard_path in self.expert_key_map[layer_id][expert_id]:
                        if 'scale' in skey and skey not in full_params:
                            full_params[skey] = self._read_tensor_no_mmap(shard_path, skey, count_ram=False)
            gpu_params = {}
            try:
                with torch.cuda.stream(self._transfer_stream):
                    for skey, tensor in full_params.items():
                        param_parts = self._parse_param_name(skey)
                        if param_parts is None:
                            continue
                        # IQ2_XS: 直接传原始数据到 GPU，不反量化
                        # 由 model.py 的 linear 函数调用 iq2xs_gemm_optimized 处理
                        if isinstance(tensor, dict) and tensor.get("__iq2xs__"):
                            # 归档格式：直接传 GPU
                            gpu_params[param_parts] = {
                                "__iq2xs__": True,
                                "d": tensor["d"].to("cuda", non_blocking=True),
                                "qs": tensor["qs"].to("cuda", non_blocking=True),
                                "scales": tensor["scales"].to("cuda", non_blocking=True),
                                "shape": tensor["shape"],
                            }
                            continue
                        is_fp4_packed = tensor.dtype == torch.int8
                        gpu_tensor = tensor.to("cuda", non_blocking=True)
                        if is_fp4_packed:
                            gpu_tensor = gpu_tensor.view(dtype=torch.float4_e2m1fn_x2)
                        gpu_params[param_parts] = gpu_tensor
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return
            gpu_params["__layer__"] = layer_id
            if layer_id not in self._prefetch_pending:
                self._prefetch_pending[layer_id] = {}
            self._prefetch_pending[layer_id][expert_id] = gpu_params
            return

        if layer_id not in self.expert_key_map or expert_id not in self.expert_key_map[layer_id]:
            return

        # IQ2_XS 模式：从归档读取
        if self._iq2xs_archive is not None:
            cpu_params = {}
            for weight_type, weight_name in enumerate(['w1', 'w2', 'w3']):
                result = self._iq2xs_archive.get_expert(layer_id, expert_id, weight_type)
                if result is None:
                    continue
                d, qs, scales, shape = result
                skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{weight_name}.weight"
                cpu_params[skey] = {
                    "__iq2xs__": True,
                    "d": torch.from_numpy(d),
                    "qs": torch.from_numpy(qs),
                    "scales": torch.from_numpy(scales),
                    "shape": shape,
                }
            if cpu_params:
                self._cpu_cache.put(key, cpu_params)
                gpu_params = {}
                try:
                    with torch.cuda.stream(self._transfer_stream):
                        for skey, tensor in cpu_params.items():
                            param_parts = self._parse_param_name(skey)
                            if param_parts is None:
                                continue
                            if isinstance(tensor, dict) and tensor.get("__iq2xs__"):
                                gpu_params[param_parts] = {
                                    "__iq2xs__": True,
                                    "d": tensor["d"].to("cuda", non_blocking=True),
                                    "qs": tensor["qs"].to("cuda", non_blocking=True),
                                    "scales": tensor["scales"].to("cuda", non_blocking=True),
                                    "shape": tensor["shape"],
                                }
                                continue
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    return
                gpu_params["__layer__"] = layer_id
                if layer_id not in self._prefetch_pending:
                    self._prefetch_pending[layer_id] = {}
                self._prefetch_pending[layer_id][expert_id] = gpu_params
                return

        # FP4 回退：从 safetensors 读取
        raw_tensors = {}
        for skey, shard_path in self.expert_key_map[layer_id][expert_id]:
            raw_tensors[skey] = self._read_tensor_no_mmap(shard_path, skey, count_ram=False)

        gpu_params = {}
        try:
            with torch.cuda.stream(self._transfer_stream):
                for skey, tensor in raw_tensors.items():
                    param_parts = self._parse_param_name(skey)
                    if param_parts is None:
                        continue
                    is_fp4_packed = tensor.dtype == torch.int8
                    if not tensor.is_pinned():
                        tensor = tensor.pin_memory()
                    gpu_tensor = tensor.to("cuda", non_blocking=True)
                    if is_fp4_packed:
                        gpu_tensor = gpu_tensor.view(dtype=torch.float4_e2m1fn_x2)
                    gpu_params[param_parts] = gpu_tensor
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._put_cpu_cache(key, raw_tensors)
            return

        if layer_id not in self._prefetch_pending:
            self._prefetch_pending[layer_id] = {}
        gpu_params["__layer__"] = layer_id
        self._prefetch_pending[layer_id][expert_id] = gpu_params
        self._put_cpu_cache(key, raw_tensors)

    # ==================== 频率统计 ====================

    def _record_access(self, layer_id: int, expert_id: int):
        freq = self._layer_freq[layer_id]
        freq[expert_id] = freq.get(expert_id, 0) + 1

    def _get_top_n_experts_for_layer(self, layer_id: int) -> list:
        if layer_id not in self.expert_key_map:
            return []
        freq = self._layer_freq.get(layer_id, {})
        if freq:
            sorted_eids = sorted(freq.keys(), key=lambda e: -freq.get(e, 0))
            return sorted_eids[:self._cpu_top_n]
        eids = list(self.expert_key_map[layer_id].keys())
        return eids[:self._cpu_top_n]

    # ==================== GPU 传输 ====================

    def _pinned_to_gpu(self, cpu_params: dict, layer_id: int = -1) -> dict:
        """CPU 参数传输到 GPU。

        IQ2_XS 数据：直接传原始数据到 GPU，不反量化，由 model.py 的 linear 函数调用 iq2xs_gemm_optimized 处理。
        FP4 数据：传 int8 到 GPU，view 为 float4_e2m1fn_x2。
        """
        gpu_params = {}
        with torch.cuda.stream(self._transfer_stream):
            for skey, tensor in cpu_params.items():
                param_parts = self._parse_param_name(skey)
                if param_parts is None:
                    continue
                # IQ2_XS: 直接传原始数据到 GPU，不反量化
                # 由 model.py 的 linear 函数调用 iq2xs_gemm_optimized 处理
                if isinstance(tensor, dict) and tensor.get("__iq2xs__"):
                    # 归档格式：直接传 GPU
                    gpu_params[param_parts] = {
                        "__iq2xs__": True,
                        "d": tensor["d"].to("cuda", non_blocking=True),
                        "qs": tensor["qs"].to("cuda", non_blocking=True),
                        "scales": tensor["scales"].to("cuda", non_blocking=True),
                        "shape": tensor["shape"],
                    }
                    continue
                is_fp4_packed = tensor.dtype == torch.int8
                gpu_tensor = tensor.to("cuda", non_blocking=True)
                if is_fp4_packed:
                    gpu_tensor = gpu_tensor.view(dtype=torch.float4_e2m1fn_x2)
                gpu_params[param_parts] = gpu_tensor
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        if layer_id >= 0:
            gpu_params["__layer__"] = layer_id
        return gpu_params

    def _raw_to_gpu(self, raw_tensors: dict, layer_id: int = -1) -> dict:
        gpu_params = {}
        with torch.cuda.stream(self._transfer_stream):
            for skey, tensor in raw_tensors.items():
                param_parts = self._parse_param_name(skey)
                if param_parts is None:
                    continue
                is_fp4_packed = tensor.dtype == torch.int8
                gpu_tensor = tensor.to("cuda", non_blocking=True)
                if is_fp4_packed:
                    gpu_tensor = gpu_tensor.view(dtype=torch.float4_e2m1fn_x2)
                gpu_params[param_parts] = gpu_tensor
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        if layer_id >= 0:
            gpu_params["__layer__"] = layer_id
        return gpu_params

    @staticmethod
    def _validate_layer(gpu_params: dict, layer_id: int, expert_id: int):
        tagged = gpu_params.get("__layer__")
        if tagged is not None and tagged != layer_id:
            raise RuntimeError(
                f"Expert layer mismatch: requested layer={layer_id} expert={expert_id}, "
                f"but params tagged layer={tagged}.")

    def _set_expert_params(self, expert, gpu_params: dict):
        # 先设置 weight，再设置 scale（scale 依赖 weight 已存在）
        sorted_items = sorted(gpu_params.items(), key=lambda x: (1 if x[0].endswith('.scale') else 0, x[0]))
        for param_name, tensor in sorted_items:
            if param_name.startswith("__"):
                continue
            if param_name not in self._param_path_cache:
                self._param_path_cache[param_name] = param_name.split(".")
            parts = self._param_path_cache[param_name]
            module = expert
            for attr in parts[:-1]:
                module = getattr(module, attr)
            final_attr = parts[-1]

            # IQ2_XS: dict 格式（含 d/qs/scales/__iq2xs__），直接 setattr 而非包装为 Parameter
            # model.py 的 linear() 函数通过 isinstance(weight, dict) 检测 IQ2_XS 并调用 iq2xs_gemm_optimized
            if isinstance(tensor, dict) and tensor.get("__iq2xs__"):
                setattr(module, final_attr, tensor)
                continue

            requires_grad = tensor.is_floating_point() and tensor.dtype not in (
                torch.float8_e8m0fnu, torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
            param = nn.Parameter(tensor, requires_grad=requires_grad)
            setattr(module, final_attr, param)
            if final_attr == "weight" and hasattr(module, 'scale') and module.scale is not None:
                # FP8 权重的 scale 会通过单独的 "scale" 条目设置，跳过
                if tensor.dtype != torch.float8_e4m3fn:
                    module.weight.scale = module.scale
            elif final_attr == "scale" and hasattr(module, 'weight'):
                module.weight.scale = param
                module.scale = param

    # ==================== SSD 读取 ====================

    def get_shard_metadata(self, shard_path: str) -> dict:
        if shard_path in self._shard_metadata:
            return self._shard_metadata[shard_path]
        with open(shard_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        metadata = {}
        for key, info in header.items():
            dtype_str = info["dtype"]
            shape = info["shape"]
            data_start, data_end = info["data_offsets"]
            if dtype_str in self._DTYPE_MAP:
                dtype, _ = self._DTYPE_MAP[dtype_str]
            else:
                dtype = torch.uint8
            metadata[key] = (dtype, shape, 8 + header_size + data_start, data_end - data_start)
        self._shard_metadata[shard_path] = metadata
        return metadata

    def _read_tensor_no_mmap(self, shard_path: str, key: str, count_ram: bool = True) -> torch.Tensor:
        if key in self._ram_weight_cache:
            raw_bytes = self._ram_weight_cache[key]
            metadata = self.get_shard_metadata(shard_path)
            dtype, shape, _, _ = metadata[key]
            tensor = torch.frombuffer(raw_bytes, dtype=dtype).reshape(shape).clone()
            if count_ram:
                self._stats["ram_hits"] += 1
            return tensor

        if key in self._ram_scale_cache:
            import zipnn
            z = zipnn.ZipNN()
            compressed = self._ram_scale_cache[key]
            decompressed = z.decompress(compressed)
            shape = self._ram_scale_shapes[key]
            tensor = torch.frombuffer(decompressed, dtype=torch.float8_e8m0fnu).reshape(shape).clone()
            if count_ram:
                self._stats["ram_hits"] += 1
            return tensor

        metadata = self.get_shard_metadata(shard_path)
        if key not in metadata:
            raise KeyError(f"Key {key} not found in {shard_path}")
        dtype, shape, offset, length = metadata[key]
        with open(shard_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        tensor = torch.frombuffer(data, dtype=dtype).reshape(shape).clone()
        return tensor

    @staticmethod
    def _parse_param_name(skey: str) -> str | None:
        """从 safetensors key 提取参数路径。

        输入格式: model.layers.{L}.ffn.experts.{E}.{module}.{param}
        输出格式: {module}.{param}  (如 'w1.weight' 或 'scale')
        """
        # 找到 'experts' 后面的部分
        idx = skey.find(".experts.")
        if idx < 0:
            return None
        after_experts = skey[idx + len(".experts."):]  # "{E}.{module}.{param}"
        parts = after_experts.split(".", 1)  # ["{E}", "{module}.{param}"]
        if len(parts) == 2:
            return parts[1]
        return None

    # ==================== 内存管理 ====================

    def _vram_free_mb(self) -> float:
        return self.VRAM_TOTAL_MB - torch.cuda.memory_allocated() / (1024**2)

    @staticmethod
    def get_mem_info() -> tuple:
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 0) / (1024**2)
            available = info.get("MemAvailable", 0) / (1024**2)
            swap_total = info.get("SwapTotal", 0) / (1024**2)
            swap_free = info.get("SwapFree", 0) / (1024**2)
            used = total - available
            swap_used = swap_total - swap_free
            return used, available, swap_used
        except Exception:
            return 0, 999, 0

    def _emergency_cleanup(self):
        self._shard_metadata.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================== RAM 预加载 ====================

    def preload_experts_to_ram(self):
        """预加载全量 FP8 scale 到 RAM（ZipNN 压缩后仅 ~1.4GB）。"""
        import zipnn
        z = zipnn.ZipNN()

        used, available, swap_used = self.get_mem_info()
        if available < 10:
            print(f"[RAM Cache] 可用 RAM 不足 ({available:.1f}GB), 跳过预加载")
            return

        print(f"[RAM Cache] 开始预加载 FP8 scale (ZipNN): 可用 {available:.1f}GB")

        total_scale_bytes_raw = 0
        total_scale_bytes_compressed = 0
        total_scales = 0
        total_scales_all = 0

        for layer_id in sorted(self.expert_key_map.keys()):
            for expert_id in sorted(self.expert_key_map[layer_id].keys()):
                for skey, shard_path in self.expert_key_map[layer_id][expert_id]:
                    if skey in self._ram_scale_cache:
                        continue
                    metadata = self.get_shard_metadata(shard_path)
                    if skey not in metadata:
                        continue
                    dtype, shape, offset, length = metadata[skey]
                    if dtype != torch.float8_e8m0fnu:
                        continue
                    total_scales_all += 1

                    with open(shard_path, "rb") as f:
                        f.seek(offset)
                        raw = f.read(length)

                    compressed = z.compress(raw)
                    self._ram_scale_cache[skey] = compressed
                    self._ram_scale_shapes[skey] = shape
                    total_scale_bytes_raw += len(raw)
                    total_scale_bytes_compressed += len(compressed)
                    total_scales += 1

        self._ram_cache_loaded = True
        scale_ratio = total_scale_bytes_raw / total_scale_bytes_compressed if total_scale_bytes_compressed > 0 else 1
        scale_raw_gb = total_scale_bytes_raw / (1024**3)
        scale_comp_gb = total_scale_bytes_compressed / (1024**3)
        saved_gb = scale_raw_gb - scale_comp_gb
        print(f"[RAM Cache] 完成: {total_scales}/{total_scales_all} FP8 scale, "
              f"{scale_raw_gb:.2f}GB→{scale_comp_gb:.2f}GB ({scale_ratio:.1f}x, 节省{saved_gb:.2f}GB)")

        used, available, _ = self.get_mem_info()
        print(f"[RAM Cache] RAM: {used:.1f}GB used, {available:.1f}GB avail, swap={swap_used:.1f}GB")

        self._adjust_cpu_cache_capacity()

    def _adjust_cpu_cache_capacity(self):
        """根据 RAM 资源动态调整 CPU 缓存容量。"""
        used, available, _ = self.get_mem_info()
        safe_mb = (available - self.MEM_SAFE_LIMIT_GB) * 1024
        if safe_mb < 1000:
            safe_mb = 1000

        max_experts = int(safe_mb / self.EXPERT_MB)
        self._cpu_cache.capacity = max_experts
        self._cpu_cache.protected_capacity = int(max_experts * 0.3)
        self._cpu_cache.probation_capacity = max_experts - self._cpu_cache.protected_capacity
        self.cpu_cache_size = max_experts

        print(f"[Cache] CPU SLRU: capacity={max_experts} experts ({max_experts * self.EXPERT_MB / 1024:.1f}GB), "
              f"protected={self._cpu_cache.protected_capacity}/probation={self._cpu_cache.probation_capacity}, "
              f"可用RAM={available:.1f}GB")

    # ==================== 热启动 ====================

    CACHE_FILENAME = "expert_cache_state.json"

    def save_cache_state(self, cache_dir: str = ""):
        save_dir = cache_dir or self._cache_dir
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)

        layer_data = {}
        for layer_id, freq in self._layer_freq.items():
            freq_serializable = {str(eid): count for eid, count in freq.items()}
            layer_data[str(layer_id)] = {"freq": freq_serializable}

        gpu_prot_keys = [f"{k[0]},{k[1]}" for k in self._gpu_cache.protected.keys()]
        gpu_prob_keys = [f"{k[0]},{k[1]}" for k in self._gpu_cache.probation.keys()]
        cpu_prot_keys = [f"{k[0]},{k[1]}" for k in self._cpu_cache.protected.keys()]
        cpu_prob_keys = [f"{k[0]},{k[1]}" for k in self._cpu_cache.probation.keys()]

        state = {
            "version": 4,
            "timestamp": time.time(),
            "top_n": self.top_n,
            "cpu_top_n": self._cpu_top_n,
            "n_layers": self._n_layers,
            "gpu_capacity": self._gpu_cache.capacity,
            "cpu_capacity": self._cpu_cache.capacity,
            "layer_data": layer_data,
            "gpu_prot_keys": gpu_prot_keys,
            "gpu_prob_keys": gpu_prob_keys,
            "cpu_prot_keys": cpu_prot_keys,
            "cpu_prob_keys": cpu_prob_keys,
        }

        path = os.path.join(save_dir, self.CACHE_FILENAME)
        with open(path, "w") as f:
            json.dump(state, f)

        total_freq = sum(len(f) for f in self._layer_freq.values())
        print(f"[Cache] State saved (v4): {total_freq} freq, "
              f"GPU prot={len(gpu_prot_keys)}/prob={len(gpu_prob_keys)}, "
              f"CPU prot={len(cpu_prot_keys)}/prob={len(cpu_prob_keys)} → {path}")

    def load_cache_state(self, cache_dir: str = "") -> bool:
        load_dir = cache_dir or self._cache_dir
        if not load_dir:
            return False

        path = os.path.join(load_dir, self.CACHE_FILENAME)
        if not os.path.exists(path):
            return False

        try:
            with open(path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        version = state.get("version", 1)

        if state.get("n_layers") != self._n_layers:
            return False

        if version >= 2:
            layer_data = state.get("layer_data", {})
            for layer_id_str, data in layer_data.items():
                layer_id = int(layer_id_str)
                freq_data = data.get("freq", {}) if isinstance(data, dict) else {}
                for eid_str, count in freq_data.items():
                    self._layer_freq[layer_id][int(eid_str)] = count
            total_freq = sum(len(f) for f in self._layer_freq.values())
            print(f"[Cache] State loaded (v{version}): {total_freq} freq entries")
        elif version == 1:
            freq_data = state.get("freq", {})
            for key_str, count in freq_data.items():
                parts = key_str.split(",")
                if len(parts) == 2:
                    layer_id, expert_id = int(parts[0]), int(parts[1])
                    self._layer_freq[layer_id][expert_id] = count
            print(f"[Cache] State loaded (v1 compat): {len(freq_data)} freq entries")

        cpu_top_n = state.get("cpu_top_n")
        if cpu_top_n is not None:
            self._cpu_top_n = cpu_top_n

        return True

    def warmup_iq2xs_mmap(self) -> None:
        """预热 IQ2_XS mmap 页面缓存（全部载入内存）。

        通过顺序读取整个归档文件，触发 OS 预读，
        将所有页面缓存到内存中。
        """
        if self._iq2xs_archive is None:
            return

        archive = self._iq2xs_archive
        if archive._mmap is None:
            return

        print(f"[IQ2_XS] 预热 mmap 页面缓存...")

        # 顺序读取整个文件，触发 OS 预读
        # 使用 madvise(MADV_WILLNEED) 提示 OS 预读
        try:
            import mmap
            archive._mmap.madvise(mmap.MADV_WILLNEED)
        except (AttributeError, OSError):
            pass

        # 顺序访问所有索引条目，触发数据页加载
        total_size = len(archive._mmap)
        chunk_size = 1024 * 1024  # 1MB
        for offset in range(0, total_size, chunk_size):
            _ = archive._mmap[offset:offset + chunk_size]

        print(f"[IQ2_XS] mmap 预热完成: {total_size / 1024**3:.2f} GB 已载入内存")

    def load_iq2xs_to_cpu(self, top_n_per_layer: int = 100):
        """从 IQ2_XS 归档加载热点专家到 CPU 缓存。

        Args:
            top_n_per_layer: 每层加载的热点专家数（按频率统计）
        """
        if self._iq2xs_archive is None:
            print("[IQ2_XS] 无归档文件，跳过 CPU 加载")
            return
        self._load_iq2xs_from_archive(top_n_per_layer)

    def _load_iq2xs_from_archive(self, top_n_per_layer: int):
        """从归档文件加载热点专家到 CPU 缓存。

        使用 mmap 读取归档文件，OS 自动管理页面缓存。
        """
        archive = self._iq2xs_archive
        if archive is None:
            return

        print(f"[IQ2_XS] 从归档加载热点专家 (top_n={top_n_per_layer})")

        loaded = 0
        for layer_id in range(self._n_layers):
            # 获取热点专家
            freq = self._layer_freq.get(layer_id, {})
            top_experts = sorted(freq.items(), key=lambda x: -x[1])[:top_n_per_layer]
            if not top_experts:
                # 无频率数据，加载前 N 个
                top_experts = [(eid, 0) for eid in range(min(top_n_per_layer, 256))]

            for expert_id, _ in top_experts:
                key = (layer_id, expert_id)
                if self._cpu_cache.contains(key):
                    continue

                cpu_params = {}
                for weight_type, weight_name in enumerate(['w1', 'w2', 'w3']):
                    result = archive.get_expert(layer_id, expert_id, weight_type)
                    if result is None:
                        continue

                    d, qs, scales, shape = result

                    # 构建缓存键
                    skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{weight_name}.weight"

                    # 存储到 CPU 缓存
                    iq2xs_data = {
                        "__iq2xs__": True,
                        "d": torch.from_numpy(d),
                        "qs": torch.from_numpy(qs),
                        "scales": torch.from_numpy(scales),
                        "shape": shape,
                    }

                    cpu_params[skey] = iq2xs_data

                if cpu_params:
                    self._cpu_cache.put(key, cpu_params)
                    loaded += 1

        print(f"[IQ2_XS] 从归档加载: {loaded} 个权重")

    def warmup_from_cache(self, model):
        """热启动: 从频率统计加载热点专家到 CPU + GPU。"""
        if self._n_layers <= 0:
            return

        effective_model = model or self._model
        if effective_model is None:
            return

        # Phase 1: 全层 TopN → CPU SLRU
        # 根据 CPU 缓存容量限制加载数量，避免加载后立即被淘汰
        phase1_loaded = 0
        cpu_capacity = self._cpu_cache.capacity
        # 每层最多加载 top_n 个，但总加载量不超过 CPU 容量
        per_layer_budget = max(cpu_capacity // self._n_layers, 1)
        effective_top_n = min(self._cpu_top_n, per_layer_budget)

        for layer_id in range(self._n_layers):
            top_experts = self._get_top_n_experts_for_layer(layer_id)
            if not top_experts:
                n_experts = min(effective_top_n,
                               len(self.expert_key_map.get(layer_id, {})))
                top_experts = list(range(n_experts))
            # 只取 effective_top_n 个
            top_experts = top_experts[:effective_top_n]
            for eid in top_experts:
                key = (layer_id, eid)
                if self._cpu_cache.contains(key):
                    phase1_loaded += 1
                    continue
                if layer_id not in self.expert_key_map or eid not in self.expert_key_map[layer_id]:
                    continue
                raw_tensors = {}
                for skey, shard_path in self.expert_key_map[layer_id][eid]:
                    raw_tensors[skey] = self._read_tensor_no_mmap(shard_path, skey, count_ram=False)
                cpu_params = {}
                for skey, tensor in raw_tensors.items():
                    try:
                        if not tensor.is_pinned():
                            tensor = tensor.pin_memory()
                    except RuntimeError:
                        pass
                    cpu_params[skey] = tensor
                self._cpu_cache.put(key, cpu_params)
                phase1_loaded += 1

        print(f"[Cache] Phase1 全层 Top{self._cpu_top_n}→CPU: "
              f"{phase1_loaded} experts ({self._n_layers} layers)")

        # Phase 2: 前 N 层 TopN → GPU LFU
        # 直接从 SSD 加载到 GPU（不依赖 CPU 缓存，避免 CPU 容量不足）
        phase2_loaded = 0
        vram_free = self._vram_free_mb()
        budget = int((vram_free - self.VRAM_RESERVE_MB) / self.EXPERT_MB)

        for layer_id in range(min(3, self._n_layers)):
            if phase2_loaded >= budget:
                break
            top_experts = self._get_top_n_experts_for_layer(layer_id)
            for eid in top_experts:
                if phase2_loaded >= budget:
                    break
                key = (layer_id, eid)
                if self._gpu_cache.contains(key):
                    continue
                # 优先从 CPU 缓存获取
                cpu_params = self._cpu_cache.get(key, record_freq=False)
                if cpu_params is not None:
                    try:
                        gpu_params = self._pinned_to_gpu(cpu_params, layer_id)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        break
                else:
                    # CPU 缓存未命中，直接从 SSD 加载
                    if layer_id not in self.expert_key_map or eid not in self.expert_key_map[layer_id]:
                        continue
                    raw_tensors = {}
                    for skey, shard_path in self.expert_key_map[layer_id][eid]:
                        raw_tensors[skey] = self._read_tensor_no_mmap(shard_path, skey, count_ram=False)
                    if not raw_tensors:
                        continue
                    try:
                        gpu_params = self._raw_to_gpu(raw_tensors, layer_id)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        break
                    if not gpu_params:
                        continue
                    # 同时放入 CPU 缓存
                    self._put_cpu_cache(key, raw_tensors)
                self._gpu_put(key, gpu_params)
                moe = effective_model.layers[layer_id].ffn
                if eid < len(moe.experts) and moe.experts[eid] is None:
                    with torch.device('cpu'):
                        moe._ensure_expert(eid)
                    expert = moe.experts[eid]
                    if expert is not None:
                        self._set_expert_params(expert, gpu_params)
                phase2_loaded += 1

        print(f"[Cache] Phase2 GPU preload: {phase2_loaded} experts from CPU → GPU")

        gpu_total = self._gpu_cache.total_entries()
        cpu_total = self._cpu_cache.total_entries()
        print(f"[Cache] Warmup done: GPU={gpu_total}, CPU={cpu_total}")

    def prefetch_by_route_prediction(
        self,
        current_layer: int,
        topk_experts: list[tuple[int, float]],
        prefetch_layers: int = 2,
    ) -> None:
        """根据路由预测预取下一层专家。

        MoE 推理中，当前层的 topk 专家往往与下一层的 topk 专家高度相关。
        利用此特性预取下一层的专家，减少缓存未命中。

        参数:
            current_layer: 当前层 ID
            topk_experts: 当前层激活的 (expert_id, score) 列表
            prefetch_layers: 预取层数（默认 2 层）
        """
        if not topk_experts:
            return

        # 预取下一层的专家
        for layer_offset in range(1, prefetch_layers + 1):
            next_layer = current_layer + layer_offset
            if next_layer >= self._n_layers:
                break

            # 预测：下一层的 topk 专家与当前层相似
            # 按分数排序，优先预取高分专家
            sorted_experts = sorted(topk_experts, key=lambda x: -x[1])

            prefetch_count = len(topk_experts)
            prefetched = 0

            for expert_id, score in sorted_experts:
                key = (next_layer, expert_id)

                # 检查是否已在 GPU 缓存
                if self._gpu_cache.contains(key):
                    continue

                # 检查是否已在预取队列
                if next_layer in self._prefetch_pending:
                    if expert_id in self._prefetch_pending[next_layer]:
                        continue

                # 异步预取
                self._async_prefetch_expert(next_layer, expert_id)
                prefetched += 1

                # 限制预取数量（每层最多预取 topk 数量）
                if prefetched >= prefetch_count:
                    break
