"""
ExpertCache v9: GPU SLRU + CPU SLRU + SSD5

缓存策略：
  GPU：SLRU（分段LRU，90% protected + 10% probation）
       Step 级别保护：当前 step 已访问专家不被淘汰
       层保护：当前层专家不被淘汰
       W-TinyLFU 实测不如 SLRU：MoE 访问模式稳定，准入策略无意义

  CPU：SLRU（分段LRU，区分热点/冷门）
       IQ2_XS 模式：全量 pinned pool，CPU 命中率 100%

  SSD5：全量路由专家

三级缓存架构：
  L1 GPU:  SLRU（90% protected / 10% probation）
           容量由VRAM动态决定
           Step 保护 + 层保护双重防淘汰
           DMA异步预取L+1层差集

  L2 CPU:  SLRU / Pinned Pool（IQ2_XS 模式 100% 命中）
           容量由RAM动态决定

  L3 SSD:  safetensors / IQ2_XS 归档直接 I/O
"""

import gc
import json
import os
import struct
import threading
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

    def evict_half(self):
        """淘汰频率最低的一半专家，用于 OOM 时紧急释放显存。"""
        n_evict = len(self.cache) // 2
        if n_evict == 0:
            return
        # 按频率排序，淘汰最低的
        sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])
        for key in sorted_keys[:n_evict]:
            del self.cache[key]


class CountMinSketch:
    """Count-Min Sketch 频率估计器。

    O(1) 频率更新和查询，空间固定（与缓存容量无关）。
    周期性衰减（减半）适应访问模式变化。

    用于 W-TinyLFU 的准入决策：新专家的估计频率必须高于
    Main 段 LRU 候选的估计频率，才能准入。
    """

    def __init__(self, width: int = 128, depth: int = 4, seed: int = 0x9e3779b9):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)
        # 使用不同种子生成多组哈希函数
        self._seeds = [(seed * (i + 1)) & 0xFFFFFFFF for i in range(depth)]
        self._total_accesses = 0
        self._decay_interval = 2000  # 每 2000 次访问衰减一次

    def _hash(self, key: tuple, i: int) -> int:
        """第 i 个哈希函数，使用 FNV-1a 变体。"""
        h = self._seeds[i]
        for part in key:
            h ^= (part & 0xFF)
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h % self.width

    def add(self, key: tuple, count: int = 1):
        """增加 key 的频率估计。"""
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.table[i, idx] += count
        self._total_accesses += count
        # 周期性衰减
        if self._total_accesses >= self._decay_interval:
            self._decay()

    def estimate(self, key: tuple) -> int:
        """估计 key 的频率（取所有哈希函数的最小值）。"""
        return min(self.table[i, self._hash(key, i)] for i in range(self.depth))

    def _decay(self):
        """频率减半，适应访问模式变化。"""
        self.table //= 2
        self._total_accesses = 0

    def reset(self):
        """重置所有计数。"""
        self.table.fill(0)
        self._total_accesses = 0


class WTinyLFU:
    """W-TinyLFU 缓存策略（Caffeine/Guava Cache 同款算法）。

    架构：
      Window (1%): 新准入专家，LRU 驱动
      Main (99%): SLRU，protected(90%) + probation(10%)

    准入策略：
      新专家进入 Window；Window 满时，如果新专家的 CMS 频率 >
      Main 段 probation LRU 候选的频率，则淘汰候选，新专家进入 Main；
      否则淘汰 Window LRU。

    优势（针对 MoE 场景）：
      1. 防缓存污染：一次访问的冷专家只进 Window，不会挤占 Main 热点
      2. O(1) 淘汰：Frequency-Bucket + SLRU LRU，无需遍历找 min_freq
      3. 热点自适应：CMS 周期衰减，对话切换时旧热点自然淘汰
      4. 与 SLRU 兼容：Main 段就是 SLRU，warmup 逻辑不变
    """

    def __init__(self, capacity: int = 250):
        self.capacity = capacity

        # Window 段：5% 容量（最少 4 个，最多 100 个）
        # MoE 场景工作集接近缓存容量，Window 需要更大以避免过度淘汰
        # Caffeine 默认 1%，但 MoE 场景需要 5% 缓冲
        self.window_capacity = max(4, min(100, int(capacity * 0.05)))
        self.window: OrderedDict = OrderedDict()

        # Main 段：95% 容量，SLRU 结构
        main_capacity = capacity - self.window_capacity
        self.main = GpuSLRU(capacity=main_capacity)

        # Count-Min Sketch 频率估计器
        # width=128 足够覆盖 11000+ 专家（43层×256专家）的频率估计
        self.cms = CountMinSketch(width=128, depth=4)

    def get(self, key: tuple, protect_layer: int = -1) -> dict | None:
        """获取专家。命中 Window 则提升到 Main protected。"""
        # 先查 Main
        result = self.main.get(key, protect_layer)
        if result is not None:
            self.cms.add(key)
            return result

        # 再查 Window
        if key in self.window:
            params = self.window.pop(key)
            # Window 命中 → 提升到 Main protected
            self.cms.add(key)
            self._insert_to_main(key, params, protect_layer)
            return params

        return None

    def put_force(self, key: tuple, params: dict,
                  protect_layer: int = -1,
                  step_protected_keys: set = None) -> tuple | None:
        """强制插入缓存。新专家进入 Window，Window 满时执行准入策略。

        W-TinyLFU 准入流程：
        1. 新专家进入 Window
        2. Window 满时，Window LRU 候选与 Main probation LRU 候选比较频率
        3. 频率高的留下（进入 Main），频率低的被淘汰
        """
        self.cms.add(key)

        # 已在 Main 中：更新
        if key in self.main.protected or key in self.main.probation:
            if key in self.main.protected:
                self.main.protected[key] = params
                self.main.protected.move_to_end(key)
            else:
                self.main.probation[key] = params
                self.main.probation.move_to_end(key)
            return None

        # 已在 Window 中：更新
        if key in self.window:
            self.window[key] = params
            self.window.move_to_end(key)
            return None

        # 新专家：直接进入 Window
        evicted = None
        if len(self.window) < self.window_capacity:
            self.window[key] = params
            return None

        # Window 满：执行 W-TinyLFU 准入策略
        # 获取 Window LRU 候选（最久未访问的）
        window_victim_key = next(iter(self.window))
        window_victim_params = self.window[window_victim_key]

        # Main 未满：Window LRU 直接移入 Main，不触发淘汰
        if self.main.total_entries() < self.main.capacity:
            del self.window[window_victim_key]
            self._insert_to_main(window_victim_key, window_victim_params,
                                 protect_layer, step_protected_keys)
            self.window[key] = params
            return None

        # Main 已满：执行准入决策
        # Window LRU 候选频率 vs Main probation LRU 候选频率
        main_victim = self._find_main_victim(protect_layer, step_protected_keys)

        if main_victim is not None:
            main_victim_key, main_victim_params = main_victim
            window_freq = self.cms.estimate(window_victim_key)
            main_victim_freq = self.cms.estimate(main_victim_key)

            if window_freq > main_victim_freq:
                # Window LRU 值得保留：进入 Main，淘汰 Main 候选
                self._remove_from_main(main_victim_key)
                evicted = (main_victim_key, main_victim_params)

                # Window LRU 移到 Main
                del self.window[window_victim_key]
                self._insert_to_main(window_victim_key, window_victim_params,
                                     protect_layer, step_protected_keys)
            else:
                # Window LRU 不值得保留：直接淘汰
                del self.window[window_victim_key]
                evicted = (window_victim_key, window_victim_params)
        else:
            # Main 无可淘汰（全部受保护）：淘汰 Window LRU
            del self.window[window_victim_key]
            evicted = (window_victim_key, window_victim_params)

        # 新专家进入 Window
        self.window[key] = params
        return evicted

    def put_force_protected(self, key: tuple, params: dict,
                            protect_layer: int = -1,
                            step_protected_keys: set = None) -> tuple | None:
        """直接插入缓存（warmup 专用），优先 Main protected，溢出到 probation/window。

        warmup 时按 LFU 频率排序加载热点专家，应填满整个缓存容量，
        而非仅 protected 容量。
        """
        self.cms.add(key)

        # 已在 Window 中，移除
        if key in self.window:
            del self.window[key]

        # 已在 Main 中，更新
        if key in self.main.protected:
            self.main.protected[key] = params
            self.main.protected.move_to_end(key)
            return None
        if key in self.main.probation:
            self.main.probation[key] = params
            self.main.probation.move_to_end(key)
            return None

        # 新专家：优先 protected，溢出到 probation，再溢出到 window
        evicted = None
        if len(self.main.protected) < self.main.protected_capacity:
            self.main.protected[key] = params
        elif len(self.main.probation) < self.main.probation_capacity:
            self.main.probation[key] = params
        elif len(self.window) < self.window_capacity:
            self.window[key] = params
        else:
            # 全部满：淘汰 Main probation LRU（跳过受保护的）
            evicted = self.main._evict_lru_skip_layer(
                self.main.probation, protect_layer, step_protected_keys)
            if evicted is not None:
                self.main.probation[key] = params
            else:
                # probation 全部受保护，淘汰 protected LRU
                evicted = self.main._evict_lru_skip_layer(
                    self.main.protected, protect_layer, step_protected_keys)
                if evicted is not None:
                    self.main.protected[key] = params
                else:
                    # 全部受保护，淘汰 Window LRU
                    if self.window:
                        window_victim_key = next(iter(self.window))
                        window_victim_params = self.window.pop(window_victim_key)
                        evicted = (window_victim_key, window_victim_params)
                        self.window[key] = params
        return evicted

    def _insert_to_main(self, key: tuple, params: dict,
                        protect_layer: int = -1,
                        step_protected_keys: set = None):
        """将专家插入 Main 段。

        W-TinyLFU 准入通过后，Window LRU 进入 Main。
        优先进入 protected，满时进入 probation。
        不再触发 Main 内部的准入决策（避免递归淘汰）。
        """
        # 优先进入 protected
        if len(self.main.protected) < self.main.protected_capacity:
            self.main.protected[key] = params
        elif len(self.main.probation) < self.main.probation_capacity:
            self.main.probation[key] = params
        else:
            # Main 也满了，需要淘汰一个 Main LRU
            # 优先淘汰 probation LRU
            evicted_key = None
            for k in list(self.main.probation.keys()):
                if k[0] != protect_layer and (step_protected_keys is None or k not in step_protected_keys):
                    evicted_key = k
                    break
            if evicted_key is not None:
                del self.main.probation[evicted_key]
                self.main.probation[key] = params
            else:
                # probation 全部受保护，尝试 protected LRU
                for k in list(self.main.protected.keys()):
                    if k[0] != protect_layer and (step_protected_keys is None or k not in step_protected_keys):
                        evicted_key = k
                        break
                if evicted_key is not None:
                    # demote protected LRU 到 probation
                    demote_params = self.main.protected.pop(evicted_key)
                    if len(self.main.probation) < self.main.probation_capacity:
                        self.main.probation[evicted_key] = demote_params
                    self.main.protected[key] = params
                else:
                    # 全部受保护，无法插入，放入 Window 作为最后手段
                    if len(self.window) < self.window_capacity:
                        self.window[key] = params
                    # 否则丢弃（极端情况）

    def _find_main_victim(self, protect_layer: int = -1,
                          step_protected_keys: set = None) -> tuple | None:
        """找 Main 段的淘汰候选（probation LRU，跳过受保护的）。"""
        # 先找 probation 段
        for k in list(self.main.probation.keys()):
            if k[0] != protect_layer and (step_protected_keys is None or k not in step_protected_keys):
                return (k, self.main.probation[k])
        # probation 全部受保护，找 protected 段
        for k in list(self.main.protected.keys()):
            if k[0] != protect_layer and (step_protected_keys is None or k not in step_protected_keys):
                return (k, self.main.protected[k])
        return None

    def _remove_from_main(self, key: tuple):
        """从 Main 段移除专家。"""
        if key in self.main.probation:
            del self.main.probation[key]
        elif key in self.main.protected:
            del self.main.protected[key]

    def contains(self, key: tuple) -> bool:
        return (key in self.window or
                key in self.main.protected or
                key in self.main.probation)

    def total_entries(self) -> int:
        return len(self.window) + self.main.total_entries()

    def all_items(self):
        yield from self.window.items()
        yield from self.main.protected.items()
        yield from self.main.probation.items()

    def clear(self):
        self.window.clear()
        self.main.clear()
        self.cms.reset()

    def evict_half(self):
        """淘汰一半专家（优先 Window → probation → protected），用于 OOM 时紧急释放。"""
        n_total = self.total_entries()
        n_evict = n_total // 2
        evicted = 0
        # 先淘汰 Window
        while self.window and evicted < n_evict:
            self.window.popitem(last=False)
            evicted += 1
        # 再淘汰 Main probation
        while self.main.probation and evicted < n_evict:
            self.main.probation.popitem(last=False)
            evicted += 1
        # 最后淘汰 Main protected
        while self.main.protected and evicted < n_evict:
            self.main.protected.popitem(last=False)
            evicted += 1


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
        # protected 90% / probation 10%：最大化热点专家保留
        # 工作集约 258 专家/step，热点系数 3-4x → 800-1000 专家需常驻
        # 90% protected 确保热点专家不被冷专家挤出
        self.protected_capacity = int(capacity * 0.90)
        self.probation_capacity = capacity - self.protected_capacity

    def get(self, key: tuple, protect_layer: int = -1) -> dict | None:
        """获取专家，命中 probation 则提升到 protected。"""
        if key in self.protected:
            self.protected.move_to_end(key)
            return self.protected[key]
        if key in self.probation:
            params = self.probation.pop(key)
            self._promote_to_protected(key, params, protect_layer)
            return self.protected.get(key, params)
        return None

    def put_force(self, key: tuple, params: dict,
                  protect_layer: int = -1,
                  step_protected_keys: set = None) -> tuple | None:
        """强制插入，满时淘汰 probation 段 LRU，返回被淘汰的 (key, params)。

        protect_layer: 当前层 ID，该层专家不会被淘汰。
        step_protected_keys: 当前 step 已访问的专家 key 集合，这些专家不会被淘汰。

        SLRU 语义：新专家进入 probation，二次访问后提升到 protected。
        不做自动 promotion（避免一次访问的专家占据 protected）。
        """
        if key in self.protected:
            self.protected[key] = params
            self.protected.move_to_end(key)
            return None
        if key in self.probation:
            params_old = self.probation.pop(key)
            self._promote_to_protected(key, params, protect_layer, step_protected_keys)
            return None

        evicted = None
        # 先尝试插入 probation
        if self.probation_capacity > 0:
            if len(self.probation) >= self.probation_capacity:
                # probation 满：淘汰 probation LRU（跳过 protect_layer 和 step_protected_keys）
                evicted = self._evict_lru_skip_layer(self.probation, protect_layer, step_protected_keys)
            self.probation[key] = params
        elif self.protected_capacity > 0:
            # probation 容量为 0，直接插入 protected
            if len(self.protected) >= self.protected_capacity:
                evicted = self._evict_lru_skip_layer(self.protected, protect_layer, step_protected_keys)
            self.protected[key] = params
        return evicted

    def put_force_protected(self, key: tuple, params: dict,
                            protect_layer: int = -1,
                            step_protected_keys: set = None) -> tuple | None:
        """直接插入 protected 段（warmup 专用）。

        warmup 时按 LFU 频率排序加载热点专家，直接放入 protected，
        避免 probation 容量限制导致只能加载少量专家。
        """
        if key in self.protected:
            self.protected[key] = params
            self.protected.move_to_end(key)
            return None
        if key in self.probation:
            params_old = self.probation.pop(key)

        evicted = None
        if len(self.protected) >= self.protected_capacity:
            # protected 满，淘汰 LRU（跳过 protect_layer 和 step_protected_keys）
            evicted = self._evict_lru_skip_layer(self.protected, protect_layer, step_protected_keys)
        self.protected[key] = params
        return evicted

    def _evict_lru_skip_layer(self, od: OrderedDict, protect_layer: int,
                               step_protected_keys: set = None) -> tuple | None:
        """淘汰 OrderedDict 的 LRU 条目，跳过 protect_layer 和 step_protected_keys 的专家。

        返回 (key, value) 或 None。
        如果所有条目都受保护，则强制淘汰最旧的。
        """
        # 先尝试找非 protect_layer 且非 step_protected 的 LRU 条目
        for k in list(od.keys()):
            if k[0] != protect_layer and (step_protected_keys is None or k not in step_protected_keys):
                v = od.pop(k)
                return (k, v)
        # 全部受保护，强制淘汰最旧的
        if od:
            return od.popitem(last=False)
        return None

    def _promote_to_protected(self, key: tuple, params: dict,
                              protect_layer: int = -1,
                              step_protected_keys: set = None):
        """将专家从 probation 提升到 protected。"""
        if self.protected_capacity <= 0:
            return
        if len(self.protected) >= self.protected_capacity:
            # protected 满，demote LRU 到 probation（跳过 protect_layer 和 step_protected_keys）
            demote_key, demote_params = self._evict_lru_skip_layer(
                self.protected, protect_layer, step_protected_keys)
            if demote_key is not None and self.probation_capacity > 0:
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

    def evict_half(self):
        """淘汰一半专家（优先淘汰 probation），用于 OOM 时紧急释放显存。"""
        n_total = len(self.protected) + len(self.probation)
        n_evict = n_total // 2
        evicted = 0
        # 先淘汰 probation 段
        while self.probation and evicted < n_evict:
            self.probation.popitem(last=False)
            evicted += 1
        # 还需要淘汰，从 protected 段 LRU 端淘汰
        while self.protected and evicted < n_evict:
            self.protected.popitem(last=False)
            evicted += 1


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
    MEM_SAFE_LIMIT_GB_FP4 = 6  # FP4 模式：更激进的内存使用，预留更少
    MIN_EXPERTS_PER_LAYER = 12

    EXPERT_MB = 12.0  # 仅 weight (I8)，scale 从 RAM 读取
    EXPERT_MB_IQ2XS = 7.0  # IQ2_XS: GPU 上实际占用 (d+qs+scales × 3 weights + 对齐)
    # VRAM 预留：推理时非专家占用的显存
    # 实测：模型常驻 ~10GB + CUDA context ~200MB + KV cache ~350MB + activations ~150MB
    # head logits ~256MB + 碎片 ~200MB = ~800MB
    # 但 mem_get_info() 返回的 free 已扣除模型常驻，所以只需预留推理 buffer
    VRAM_RESERVE_MB = 800
    # 计算开销：GEMM workspace + 临时张量 + head logits 分块 (~256MB)
    VRAM_COMPUTE_OVERHEAD_MB = 400
    # IQ2_XS GEMM 计算开销：workspace + 临时张量
    IQ2_XS_VRAM_OVERHEAD_MB = 50
    CACHE_MISS_RESERVE_EXPERTS = 2

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

        # IQ2_XS 归档读取器（延迟打开，避免 mmap 77GB 与 safetensors 加载竞争页缓存）
        self._iq2xs_archive: Optional[IQ2XSArchiveReader] = None
        self._iq2xs_archive_path: Optional[str] = None

        # 记录归档路径，但不立即打开（mmap 77GB 会与 safetensors 加载竞争页缓存）
        if self._iq2xs_dir:
            archive_path = os.path.join(self._iq2xs_dir, "experts.iq2xs")
            if os.path.exists(archive_path):
                self._iq2xs_archive_path = archive_path
                print(f"[IQ2_XS] 归档文件已检测: {archive_path}（延迟打开）")

        # GPU 缓存：SLRU（热点稳定保留）
        # W-TinyLFU 实测不如 SLRU：MoE 访问模式稳定，准入策略无意义，Window 浪费容量
        self._gpu_cache = GpuSLRU(capacity=250)
        self._cpu_cache = CpuSLRU(capacity=cpu_cache_size)

        self._current_layer = -1
        self._n_layers = 0
        self._vram_adjusted = False

        self._layer_stats: Dict[int, dict] = defaultdict(
            lambda: {"gpu": 0, "cpu": 0, "ssd": 0, "ram": 0, "total": 0})

        self._param_path_cache: Dict[str, tuple] = {}

        self._stats = {"gpu_hits": 0, "cpu_hits": 0, "ssd_hits": 0,
                       "ram_hits": 0, "pinned_pool_hits": 0,
                       "gpu_evictions": 0,
                       "prefetch_hits": 0, "prefetch_misses": 0}

        self._shard_metadata: Dict[str, Dict] = {}

        self._transfer_stream = torch.cuda.Stream()
        self._prefetch_pending: Dict[int, Dict[int, dict]] = {}

        # Step 级别专家保护：当前 step 已访问的专家不被淘汰
        # 解决跨层淘汰问题：处理层 L 时，层 0~L-1 的专家仍受保护
        self._step_protected_keys: set = set()
        self._step_count: int = 0

        self._cpu_prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu_prefetch")
        self._cpu_prefetch_pending: set = set()  # (layer_id, expert_id) 正在预取

        # IQ2_XS 归档预取线程池：将 SSD I/O 移到后台线程，与 GPU 计算重叠
        self._iq2xs_prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="iq2xs_prefetch")
        self._iq2xs_prefetch_pending: set = set()  # (layer_id, expert_id) 正在预取
        self._prefetch_lock = threading.Lock()  # 保护 _prefetch_pending 的并发写入

        # IQ2_XS pinned memory pool: 预加载专家到 pinned RAM，DMA 直传 GPU 无 page fault
        # {(layer_id, expert_id): {"d": pinned_tensor, "qs": pinned_tensor, "scales": pinned_tensor, "shape": tuple}}
        self._iq2xs_pinned_pool: Dict[tuple, dict] = {}
        self._iq2xs_pinned_pool_lock = threading.Lock()  # 保护 pinned pool 的并发访问

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

        查找顺序: GPU W-TinyLFU → DMA 预取结果 → Pinned Pool → SSD

        参数:
            layer_id: 层 ID
            expert_id: 专家 ID
            topk_info: 当前层激活的 (expert_id, score) 列表，用于路由预测预取
        """
        key = (layer_id, expert_id)
        # _layer_freq 已在 _load_activated_experts Phase 0 统一递增，此处不再重复

        # 将当前专家加入 step 保护集合（防止当前 step 内被淘汰）
        if self._step_protected_keys is not None:
            self._step_protected_keys.add(key)

        # L1: GPU W-TinyLFU
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
        with self._prefetch_lock:
            if layer_id in self._prefetch_pending and expert_id in self._prefetch_pending[layer_id]:
                self._stats["prefetch_hits"] += 1
                gpu_params = self._prefetch_pending[layer_id].pop(expert_id)
                if not self._prefetch_pending[layer_id]:
                    del self._prefetch_pending[layer_id]
            else:
                gpu_params = None
        if gpu_params is not None:
            self._validate_layer(gpu_params, layer_id, expert_id)
            self._gpu_put(key, gpu_params)
            # 路由预测预取
            if topk_info is not None:
                self.prefetch_by_route_prediction(layer_id, topk_info)
            return gpu_params

        self._stats["prefetch_misses"] += 1

        # IQ2_XS pinned pool: 优先从 pinned memory pool 获取（无 page fault，无 SSD I/O）
        if self._iq2xs_archive_path is not None or self._iq2xs_archive is not None:
            with self._iq2xs_pinned_pool_lock:
                pool_entry = self._iq2xs_pinned_pool.get(key)
            if pool_entry is not None:
                self._stats["pinned_pool_hits"] += 1
                self._layer_stats[layer_id]["ram"] += 1
                self._layer_stats[layer_id]["total"] += 1
                # 从 pinned pool 构建 cpu_params，数据已是 pinned，无需再 pin_memory()
                cpu_params = {}
                for weight_name, entry in pool_entry.items():
                    skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{weight_name}.weight"
                    cpu_params[skey] = {
                        "__iq2xs__": True,
                        "d": entry["d"],
                        "qs": entry["qs"],
                        "scales": entry["scales"],
                        "shape": entry["shape"],
                        "__from_pinned_pool__": True,  # 标记已 pinned，跳过冗余 pin_memory()
                    }
                if cpu_params:
                    gpu_params = self._pinned_to_gpu(cpu_params, layer_id)
                    if gpu_params:
                        self._gpu_put(key, gpu_params)
                        if topk_info is not None:
                            self.prefetch_by_route_prediction(layer_id, topk_info)
                        return gpu_params
                    else:
                        # OOM: 淘汰 GPU 缓存腾出空间后重试
                        self._gpu_cache.evict_half()
                        torch.cuda.empty_cache()

        # IQ2_XS 模式：从 mmap 归档读取（pinned pool 未命中时回退）
        # 归档文件通过 mmap + OS 页缓存已充当 CPU 缓存，无需重复存储
        if self._iq2xs_archive_path is not None and self._iq2xs_archive is None:
            self._ensure_iq2xs_archive()
        if self._iq2xs_archive is not None:
            self._stats["ssd_hits"] += 1
            self._layer_stats[layer_id]["ssd"] += 1
            self._layer_stats[layer_id]["total"] += 1
            cpu_params = {}
            for weight_type, weight_name in enumerate(['w1', 'w2', 'w3']):
                result = self._iq2xs_archive.get_expert(layer_id, expert_id, weight_type)
                if result is None:
                    continue
                d, qs, scales, shape = result
                skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{weight_name}.weight"
                # v2 归档: numpy 数组与 mmap 共享内存（零拷贝视图）
                # pin_memory() 会拷贝到 pinned memory，避免 GPU 传输时 page fault
                cpu_params[skey] = {
                    "__iq2xs__": True,
                    "d": torch.from_numpy(d).pin_memory(),
                    "qs": torch.from_numpy(qs).pin_memory(),
                    "scales": torch.from_numpy(scales).pin_memory(),
                    "shape": shape,
                }
            if cpu_params:
                gpu_params = self._pinned_to_gpu(cpu_params, layer_id)
                if gpu_params:
                    self._gpu_put(key, gpu_params)
                    if topk_info is not None:
                        self.prefetch_by_route_prediction(layer_id, topk_info)
                    return gpu_params
                else:
                    # OOM: 淘汰 GPU 缓存腾出空间
                    self._gpu_cache.evict_half()
                    torch.cuda.empty_cache()

        # FP4 模式：L2 CPU SLRU
        cpu_params = self._cpu_cache.get(key)
        if cpu_params is not None:
            self._stats["cpu_hits"] += 1
            self._layer_stats[layer_id]["cpu"] += 1
            self._layer_stats[layer_id]["total"] += 1
            # CPU 缓存只有 weight，需要从 RAM 读取 scale
            full_params = dict(cpu_params)
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

        # L3: SSD (FP4 模式)
        self._stats["ssd_hits"] += 1
        self._layer_stats[layer_id]["ssd"] += 1
        self._layer_stats[layer_id]["total"] += 1

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

    def on_step_start(self, step: int):
        """Step 开始：重置 step 级别保护集合。

        注意：当缓存容量 < 工作集时，step 保护会阻止新专家进入缓存，
        导致缓存抖动。因此仅在缓存容量足够时启用 step 保护。
        """
        self._step_count = step
        # 仅当缓存容量 ≥ 2× 工作集时启用 step 保护
        # 工作集 = n_layers × n_activated_experts = 43 × 6 = 258
        working_set = self._n_layers * 6
        if self._gpu_cache.capacity >= working_set * 2:
            self._step_protected_keys = set()
        else:
            # 缓存容量不足，不启用 step 保护（避免阻止新专家进入）
            self._step_protected_keys = None

    def on_step_end(self):
        """Step 结束：清理 step 级别保护。"""
        # 保持 on_step_start 设置的启用/禁用语义
        # None → None（禁用），set() → set()（清空但保持启用）
        if self._step_protected_keys is not None:
            self._step_protected_keys.clear()

    def prefetch_next_layer(self, current_layer: int, activated_indices: list):
        """多级异步预取流水线：GPU←PinnedPool←SSD。

        100% pinned pool 模式：
          - L+1: Pinned Pool → GPU 预取（隐藏 PCIe DMA 延迟）
          - L+2: Pinned Pool → GPU 预取（更积极地填充缓存）
          - 使用当前层的激活专家预测下一层

        部分 pinned pool 模式：
          - L+2: CPU → GPU 预取
          - L+5: SSD → CPU 预取
        """
        if self._n_layers <= 0:
            return

        # 100% pinned pool 模式：预取 L+1 和 L+2 层差集
        # 使用当前层的激活专家作为下一层的预测
        has_full_pinned_pool = len(self._iq2xs_pinned_pool) > 0 and \
            len(self._iq2xs_pinned_pool) >= self._n_layers * (self._iq2xs_archive.n_experts if self._iq2xs_archive else 0) * 0.9

        if has_full_pinned_pool:
            # 100% pinned pool: 只预取 L+1 层差集
            # 减少预取激进程度，避免缓存抖动
            next_layer = current_layer + 1
            if next_layer < self._n_layers:
                self._prefetch_layer_diffset(next_layer, activated_indices)
        else:
            # L+2: CPU → GPU 预取
            gpu_prefetch_layer = current_layer + 2
            if gpu_prefetch_layer < self._n_layers:
                self._prefetch_gpu_from_cpu(gpu_prefetch_layer)

            # L+5: SSD → CPU 预取
            cpu_prefetch_layer = current_layer + 5
            if cpu_prefetch_layer < self._n_layers:
                self._prefetch_cpu_from_ssd(cpu_prefetch_layer)

    def _prefetch_layer_diffset(self, layer_id: int, predicted_experts: list):
        """预取指定层的差集专家（基于预测的激活专家列表）。

        在 100% pinned pool 模式下，从 pinned pool DMA 到 GPU。
        """
        for expert_id in predicted_experts:
            key = (layer_id, expert_id)
            # 已在 GPU 缓存，跳过
            if self._gpu_cache.contains(key):
                continue
            # 已在预取队列，跳过
            with self._prefetch_lock:
                if key in self._iq2xs_prefetch_pending:
                    continue
                if layer_id in self._prefetch_pending and expert_id in self._prefetch_pending[layer_id]:
                    continue
            # 提交异步预取
            self._async_prefetch_expert(layer_id, expert_id)

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
        with self._prefetch_lock:
            self._prefetch_pending.clear()
        self._cpu_prefetch_pending.clear()
        self._iq2xs_prefetch_pending.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def finalize_prefetch(self):
        if not self._prefetch_pending:
            return
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        if self._current_layer >= 0:
            self._promote_prefetch(self._current_layer)

    def adjust_window_for_vram(self):
        """根据 VRAM 动态调整 GPU 缓存容量。

        使用初始 VRAM（模型加载后的空闲量）计算预算，
        而非当前 VRAM（可能已被 warmup 专家占用）。

        目标：最大化 GPU 缓存容量，使活跃专家全部常驻 GPU。
        """
        if self._n_layers <= 0:
            return

        # 使用初始 VRAM（模型加载后、专家缓存加载前）
        # 如果已记录初始值，使用它；否则使用当前值
        vram_free_mb = getattr(self, '_initial_vram_free_mb', None)
        if vram_free_mb is None:
            vram_free_mb = self._vram_free_mb()

        # IQ2_XS 模式需额外 VRAM 预留
        compute_overhead = self.VRAM_COMPUTE_OVERHEAD_MB
        expert_mb = self.EXPERT_MB
        if self._iq2xs_archive is not None:
            compute_overhead += self.IQ2_XS_VRAM_OVERHEAD_MB
            expert_mb = self.EXPERT_MB_IQ2XS
        safe_budget_mb = vram_free_mb - self.VRAM_RESERVE_MB - compute_overhead
        if safe_budget_mb < 200:
            safe_budget_mb = 200

        cache_miss_mb = self.CACHE_MISS_RESERVE_EXPERTS * expert_mb
        expert_budget_mb = safe_budget_mb - cache_miss_mb
        if expert_budget_mb < 100:
            expert_budget_mb = 100

        max_experts = int(expert_budget_mb / expert_mb)
        self._gpu_cache.capacity = max_experts
        # SLRU: 90% protected / 10% probation
        self._gpu_cache.protected_capacity = int(max_experts * 0.90)
        self._gpu_cache.probation_capacity = max_experts - self._gpu_cache.protected_capacity

        self.top_n = max(max_experts // self._n_layers, self.MIN_EXPERTS_PER_LAYER)
        self._vram_adjusted = True

        coverage = self.top_n / 256 * 100
        vram_needed = max_experts * expert_mb
        print(f"[Cache] VRAM adjust: GPU capacity={max_experts} experts "
              f"(prot={self._gpu_cache.protected_capacity}/prob={self._gpu_cache.probation_capacity}), "
              f"per_layer_coverage={coverage:.0f}%, "
              f"VRAM={vram_needed:.0f}+{cache_miss_mb:.0f}MB(miss), "
              f"budget={safe_budget_mb:.0f}MB (free={vram_free_mb:.0f}MB)")

    def check_memory_pressure(self) -> bool:
        used, available, swap_used = self.get_mem_info()
        if available < self.MEM_SAFE_LIMIT_GB:
            print(f"[Cache] 内存紧张! 可用: {available:.1f}GB < {self.MEM_SAFE_LIMIT_GB}GB, Swap: {swap_used:.1f}GB")
            self._emergency_cleanup()
            return True
        return False

    def get_stats(self) -> str:
        total = (self._stats["gpu_hits"] + self._stats["cpu_hits"] +
                 self._stats["ssd_hits"] + self._stats["ram_hits"] +
                 self._stats["pinned_pool_hits"])
        if total == 0:
            return "No cache accesses yet"
        gpu_pct = self._stats["gpu_hits"] / total * 100
        cpu_pct = self._stats["cpu_hits"] / total * 100
        ssd_pct = self._stats["ssd_hits"] / total * 100
        ram_pct = self._stats["ram_hits"] / total * 100
        pp_pct = self._stats["pinned_pool_hits"] / total * 100
        pf_total = self._stats["prefetch_hits"] + self._stats["prefetch_misses"]
        pf_pct = self._stats["prefetch_hits"] / pf_total * 100 if pf_total > 0 else 0
        cpu_accesses = self._stats["cpu_hits"] + self._stats["ssd_hits"]
        cpu_hit_rate = self._stats["cpu_hits"] / cpu_accesses * 100 if cpu_accesses > 0 else 0
        gpu_hit_rate = self._stats["gpu_hits"] / total * 100 if total > 0 else 0
        gpu_total = self._gpu_cache.total_entries()
        gpu_prot = len(self._gpu_cache.protected)
        gpu_prob = len(self._gpu_cache.probation)
        cpu_prot = len(self._cpu_cache.protected)
        cpu_prob = len(self._cpu_cache.probation)
        step_prot = len(self._step_protected_keys)
        return (f"L1 GPU({gpu_total}: p{gpu_prot}/b{gpu_prob}): "
                f"{self._stats['gpu_hits']} ({gpu_pct:.0f}%), "
                f"L2 CPU({cpu_prot}prot+{cpu_prob}prob): {self._stats['cpu_hits']} ({cpu_pct:.0f}%), "
                f"CPU hit: {cpu_hit_rate:.0f}%, GPU hit: {gpu_hit_rate:.0f}%, "
                f"step_prot={step_prot}, "
                f"L3 SSD: {self._stats['ssd_hits']} ({ssd_pct:.0f}%), "
                f"RAM: {self._stats['ram_hits']} ({ram_pct:.0f}%), "
                f"PinnedPool: {self._stats['pinned_pool_hits']} ({pp_pct:.0f}%), "
                f"PF: {self._stats['prefetch_hits']} ({pf_pct:.0f}%), "
                f"evict: {self._stats['gpu_evictions']}")

    # ==================== GPU 缓存内部 ====================

    def _gpu_put(self, key: tuple, gpu_params: dict):
        """插入 GPU 缓存，满时淘汰最冷专家转存 CPU。

        当前层活跃专家和当前 step 已访问专家不会被淘汰。
        """
        evicted = self._gpu_cache.put_force(key, gpu_params,
                                            protect_layer=self._current_layer,
                                            step_protected_keys=self._step_protected_keys)
        self._handle_eviction(evicted)

    def _gpu_put_protected(self, key: tuple, gpu_params: dict):
        """直接插入 GPU 缓存 Main protected 段（warmup 专用）。

        warmup 时按 LFU 频率排序加载热点专家，直接放入 protected，
        填满整个缓存容量而非仅 probation 容量。
        """
        evicted = self._gpu_cache.put_force_protected(key, gpu_params,
                                                       protect_layer=self._current_layer,
                                                       step_protected_keys=self._step_protected_keys)
        self._handle_eviction(evicted)

    def _handle_eviction(self, evicted):
        """处理被淘汰的专家：释放 GPU 内存，清除模型引用。"""
        if evicted is not None:
            evict_key, evict_params = evicted
            if evict_params:
                self._transfer_gpu_to_cpu(evict_key, evict_params)
                # IQ2_XS 模式：_transfer_gpu_to_cpu 跳过，需显式释放 GPU 张量
                del evict_params
                self._stats["gpu_evictions"] += 1
            # 仅对非当前层专家清除 Expert 对象引用
            effective_model = self._model
            if effective_model is not None:
                evict_layer, expert_id = evict_key
                if evict_layer != self._current_layer:
                    moe = effective_model.layers[evict_layer].ffn
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
        with self._prefetch_lock:
            if not self._prefetch_pending:
                return
            # 快照当前 pending，避免长时间持锁
            pending_snapshot = {lid: dict(pending) for lid, pending in self._prefetch_pending.items()}
            self._prefetch_pending.clear()
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        for lid, pending in pending_snapshot.items():
            if lid < current_layer:
                for eid, params in pending.items():
                    key = (lid, eid)
                    self._transfer_gpu_to_cpu(key, params)
            else:
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
        """GPU 淘汰专家转存到 CPU 缓存（只存 weight，scale 从 RAM 读取）。

        IQ2_XS 模式下跳过，归档文件已充当 CPU 缓存。
        """
        if not gpu_params:
            return
        # IQ2_XS 模式：不转存 CPU SLRU，归档文件通过 mmap 可直接读取
        if self._iq2xs_archive is not None:
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
        if vram_free_mb < 100:
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

        # IQ2_XS 模式：从归档读取（后台线程，隐藏 SSD I/O 延迟）
        if self._iq2xs_archive is not None:
            pf_key = (layer_id, expert_id)
            with self._prefetch_lock:
                if pf_key in self._iq2xs_prefetch_pending:
                    return  # 已在预取中
                self._iq2xs_prefetch_pending.add(pf_key)
            self._iq2xs_prefetch_executor.submit(
                self._iq2xs_prefetch_worker, layer_id, expert_id)
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

    def _iq2xs_prefetch_worker(self, layer_id: int, expert_id: int):
        """IQ2_XS 归档预取工作线程：优先从 pinned pool 获取，回退到 SSD I/O + GPU 传输。

        Phase 1 优化：
        1. 优先从 pinned pool 获取（无 SSD I/O，无 page fault）
        2. pinned pool 命中时，数据已是 pinned，直接 DMA 到 GPU
        3. pinned pool 未命中时，回退到 mmap 归档读取

        在后台线程执行，与主推理线程的 GPU 计算重叠。
        """
        pf_key = (layer_id, expert_id)
        try:
            # 优先从 pinned pool 获取
            with self._iq2xs_pinned_pool_lock:
                pool_entry = self._iq2xs_pinned_pool.get(pf_key)

            if pool_entry is not None:
                # pinned pool 命中：数据已是 pinned，直接 DMA 到 GPU
                cpu_params = {}
                for weight_name, entry in pool_entry.items():
                    skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{weight_name}.weight"
                    cpu_params[skey] = {
                        "__iq2xs__": True,
                        "d": entry["d"],
                        "qs": entry["qs"],
                        "scales": entry["scales"],
                        "shape": entry["shape"],
                        "__from_pinned_pool__": True,
                    }
            else:
                # pinned pool 未命中：回退到 mmap 归档读取
                cpu_params = {}
                for weight_type, weight_name in enumerate(['w1', 'w2', 'w3']):
                    result = self._iq2xs_archive.get_expert(layer_id, expert_id, weight_type)
                    if result is None:
                        continue
                    d, qs, scales, shape = result
                    skey = f"model.layers.{layer_id}.ffn.experts.{expert_id}.{weight_name}.weight"
                    # v2 归档: numpy 数组与 mmap 共享内存（零拷贝视图）
                    # pin_memory() 拷贝到 pinned memory，避免 GPU 传输时 page fault
                    cpu_params[skey] = {
                        "__iq2xs__": True,
                        "d": torch.from_numpy(d).pin_memory(),
                        "qs": torch.from_numpy(qs).pin_memory(),
                        "scales": torch.from_numpy(scales).pin_memory(),
                        "shape": shape,
                    }

            if not cpu_params:
                return

            # GPU 异步传输
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
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return

            if not gpu_params:
                return

            gpu_params["__layer__"] = layer_id
            with self._prefetch_lock:
                if layer_id not in self._prefetch_pending:
                    self._prefetch_pending[layer_id] = {}
                self._prefetch_pending[layer_id][expert_id] = gpu_params
        except Exception:
            pass  # 预取失败不影响主推理流程
        finally:
            with self._prefetch_lock:
                self._iq2xs_prefetch_pending.discard(pf_key)

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
        try:
            with torch.cuda.stream(self._transfer_stream):
                for skey, tensor in cpu_params.items():
                    param_parts = self._parse_param_name(skey)
                    if param_parts is None:
                        continue
                    # IQ2_XS: 直接传原始数据到 GPU，不反量化
                    # 由 model.py 的 linear 函数调用 iq2xs_gemm_optimized 处理
                    if isinstance(tensor, dict) and tensor.get("__iq2xs__"):
                        # Phase 1 优化：如果数据来自 pinned pool，已是 pinned，跳过冗余 pin_memory()
                        from_pinned_pool = tensor.get("__from_pinned_pool__", False)
                        d_tensor = tensor["d"]
                        qs_tensor = tensor["qs"]
                        scales_tensor = tensor["scales"]
                        if from_pinned_pool:
                            # 数据已是 pinned，直接 DMA 到 GPU
                            gpu_params[param_parts] = {
                                "__iq2xs__": True,
                                "d": d_tensor.to("cuda", non_blocking=True),
                                "qs": qs_tensor.to("cuda", non_blocking=True),
                                "scales": scales_tensor.to("cuda", non_blocking=True),
                                "shape": tensor["shape"],
                            }
                        else:
                            # 数据未 pinned，先 pin_memory() 再 DMA
                            gpu_params[param_parts] = {
                                "__iq2xs__": True,
                                "d": d_tensor.pin_memory().to("cuda", non_blocking=True),
                                "qs": qs_tensor.pin_memory().to("cuda", non_blocking=True),
                                "scales": scales_tensor.pin_memory().to("cuda", non_blocking=True),
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
            return None
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        if layer_id >= 0:
            gpu_params["__layer__"] = layer_id
        return gpu_params

    def _raw_to_gpu(self, raw_tensors: dict, layer_id: int = -1) -> dict:
        gpu_params = {}
        try:
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
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
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
            # 注意：nn.Module.__setattr__ 不允许将已注册的 Parameter 替换为 dict，
            # 需要先删除旧 Parameter 再设置新值
            if isinstance(tensor, dict) and tensor.get("__iq2xs__"):
                if hasattr(module, final_attr):
                    # 先从 _parameters 中移除，避免 __setattr__ 类型检查
                    module._parameters.pop(final_attr, None)
                object.__setattr__(module, final_attr, tensor)
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
        """计算 VRAM 空闲量，使用实际设备可用内存（含 CUDA context 等非 PyTorch 分配）。"""
        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info()
        return free / (1024**2)

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
        """预加载全量 FP8 scale 到 RAM（ZipNN 压缩后仅 ~1.4GB）。

        仅 FP4 模式需要：路由专家 FP4 GEMM 使用 FP8 scale。
        IQ2_XS 模式：路由专家使用归档中的 d/qs/scales，不需要 FP8 scale。
        非路由专家（含共享专家）从 SSD5 加载后常驻 GPU，不走 CPU SLRU。
        """
        # IQ2_XS 模式：不需要 FP8 scale
        if self._iq2xs_archive_path is not None or self._iq2xs_archive is not None:
            print("[RAM Cache] IQ2_XS 模式：跳过 FP8 scale 预加载（路由专家使用归档 d/qs/scales）")
            self._ram_cache_loaded = True
            self._adjust_cpu_cache_capacity()
            return

        import zipnn
        z = zipnn.ZipNN()

        used, available, swap_used = self.get_mem_info()
        if available < 10:
            print(f"[RAM Cache] 可用 RAM 不足 ({available:.1f}GB), 跳过预加载")
            return

        print(f"[RAM Cache] 开始预加载 FP8 scale (ZipNN, FP4模式): 可用 {available:.1f}GB")

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

    def _ensure_iq2xs_archive(self):
        """延迟打开 IQ2XS 归档文件。在常驻权重加载完成后调用，避免与 safetensors 竞争页缓存。"""
        if self._iq2xs_archive is not None:
            return True
        if self._iq2xs_archive_path is None:
            return False
        try:
            self._iq2xs_archive = IQ2XSArchiveReader(self._iq2xs_archive_path)
            print(f"[IQ2XSArchive] 打开归档: {self._iq2xs_archive_path}")
            print(f"  层数: {self._iq2xs_archive.n_layers}, 专家数: {self._iq2xs_archive.n_experts}, 条目数: {len(self._iq2xs_archive._index)}")
            return True
        except Exception as e:
            print(f"[IQ2_XS] 归档打开失败: {e}")
            self._iq2xs_archive_path = None
            return False

    def _adjust_cpu_cache_capacity(self):
        """根据 RAM 资源动态调整 CPU 缓存容量。

        IQ2_XS 模式：归档文件通过 mmap + OS 页缓存充当 CPU 缓存，CPU SLRU 容量极小。
        FP4 模式：CPU SLRU 缓存热点专家，容量按可用内存最大化计算。
        """
        used, available, _ = self.get_mem_info()

        # IQ2_XS 模式：归档文件通过 mmap + OS 页缓存充当 CPU 缓存
        # 检查 _iq2xs_archive_path（归档可能延迟打开，_iq2xs_archive 还是 None）
        if self._iq2xs_archive_path is not None or self._iq2xs_archive is not None:
            max_experts = 0  # 不使用 CPU SLRU，mmap 归档充当 CPU 缓存
            print(f"[Cache] CPU SLRU (IQ2_XS模式): capacity=0 (mmap归档充当CPU缓存)")
        else:
            # FP4 模式：大量 CPU SLRU 缓存，预留更少内存
            safe_mb = (available - self.MEM_SAFE_LIMIT_GB_FP4) * 1024
            if safe_mb < 1000:
                safe_mb = 1000
            max_experts = int(safe_mb / self.EXPERT_MB)
            print(f"[Cache] CPU SLRU (FP4模式): capacity={max_experts} experts ({max_experts * self.EXPERT_MB / 1024:.1f}GB)")

        self._cpu_cache.capacity = max_experts
        self._cpu_cache.protected_capacity = int(max_experts * 0.3)
        self._cpu_cache.probation_capacity = max_experts - self._cpu_cache.protected_capacity
        self.cpu_cache_size = max_experts

        print(f"[Cache] CPU SLRU: protected={self._cpu_cache.protected_capacity}/probation={self._cpu_cache.probation_capacity}, 可用RAM={available:.1f}GB")

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
            "version": 5,
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
        """IQ2_XS 专家预加载策略。

        策略：
        1. 先尝试全量 pinned pool 加载（无 mmap 预热，无双缓冲）
        2. 如果 RAM 装不下全部专家，再 mmap 预热 + 部分 pinned pool
        """
        if self._iq2xs_archive is None:
            return

        archive = self._iq2xs_archive
        if archive._mmap is None:
            return

        # 计算全量加载所需内存
        n_layers = self._n_layers
        n_experts = archive.n_experts
        total_experts = n_layers * n_experts
        # 每专家约 6.8GB (d + qs + scales)
        est_bytes_per_expert = 6.8 * 1024**2  # ~6.8MB
        total_pinned_gb = total_experts * est_bytes_per_expert / 1024**3

        # 检查是否可以全量加载
        used, available, swap_used = self.get_mem_info()
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            mem_free_gb = meminfo.get("MemFree", 0) / (1024**2)
            mem_total_gb = meminfo.get("MemTotal", 0) / (1024**2)
        except Exception:
            mem_free_gb = available
            mem_total_gb = available + used

        reserved_gb = 10  # 系统+模型+推理buffer
        can_load_all = (mem_total_gb - reserved_gb) >= total_pinned_gb

        if can_load_all:
            print(f"[IQ2_XS] 全量 pinned pool 模式: {total_experts} 专家 ≈ {total_pinned_gb:.0f}GB, "
                  f"RAM {mem_total_gb:.0f}GB (无需 mmap 预热)")
            # 直接全量 pinned，跳过 mmap 预热（无双缓冲）
            self._preload_iq2xs_to_pinned_pool()
        else:
            print(f"[IQ2_XS] 部分 pinned pool 模式: {total_experts} 专家 ≈ {total_pinned_gb:.0f}GB, "
                  f"RAM {mem_total_gb:.0f}GB (需要 mmap 预热回退)")
            # 先 mmap 预热，再部分 pinned pool
            self._warmup_mmap_sequential()
            self._preload_iq2xs_to_pinned_pool()

    def _warmup_mmap_sequential(self) -> None:
        """mmap 顺序预热：将归档文件全部载入 OS 页缓存。

        仅在 RAM 装不下全部专家时使用，为未缓存专家提供快速回退路径。
        """
        archive = self._iq2xs_archive
        if archive is None or archive._mmap is None:
            return

        print(f"[IQ2_XS] 预热 mmap 页面缓存（MADV_SEQUENTIAL）...")

        try:
            import mmap
            archive._mmap.madvise(mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass

        total_size = len(archive._mmap)
        chunk_size = 1024 * 1024  # 1MB
        for offset in range(0, total_size, chunk_size):
            _ = archive._mmap[offset:offset + chunk_size]

        print(f"[IQ2_XS] mmap 预热完成: {total_size / 1024**3:.2f} GB 已载入内存")

        try:
            import mmap
            archive._mmap.madvise(mmap.MADV_NORMAL)
        except (AttributeError, OSError):
            pass

    def _preload_iq2xs_to_pinned_pool(self):
        """将 IQ2_XS 专家预加载到 pinned memory pool。

        全量模式：RAM 足够装下所有专家，无需 mmap 预热，无双缓冲。
        部分模式：RAM 不足，先 mmap 预热，再加载尽可能多的专家。

        pinned pool 中的数据可直接 DMA 到 GPU，无需 pin_memory() 和 page fault。
        """
        if self._iq2xs_archive is None:
            return

        # 确保归档已打开
        if not self._ensure_iq2xs_archive():
            return

        archive = self._iq2xs_archive
        n_layers = self._n_layers
        n_experts = archive.n_experts
        total_experts = n_layers * n_experts

        # 计算可用 RAM 用于 pinned pool
        used, available, swap_used = self.get_mem_info()
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            mem_free_gb = meminfo.get("MemFree", 0) / (1024**2)
            mem_total_gb = meminfo.get("MemTotal", 0) / (1024**2)
            cached_gb = meminfo.get("Cached", 0) / (1024**2)
        except Exception:
            mem_free_gb = available
            mem_total_gb = available + used
            cached_gb = 0

        # 预留系统+模型+推理buffer
        reserved_gb = 10

        # 判断是否为全量模式（mmap 预热是否已跳过）
        est_total_gb = total_experts * 6.8 / 1024  # 每专家 ~6.8MB
        full_mode = (mem_total_gb - reserved_gb) >= est_total_gb

        if full_mode:
            # 全量模式：使用 MemTotal - reserved，确保装下全部专家
            available_for_pool_gb = mem_total_gb - reserved_gb
        else:
            # 部分模式：使用 MemFree + Cached*0.9（OS 会回收 page cache）
            available_for_pool_gb = mem_free_gb + cached_gb * 0.9 - reserved_gb

        available_for_pool_gb = max(available_for_pool_gb, 0)
        available_for_pool_bytes = int(available_for_pool_gb * 1024**3)

        if available_for_pool_gb < 5:
            print(f"[IQ2_XS PinnedPool] 可用 RAM 不足 ({available_for_pool_gb:.1f}GB), 跳过预加载")
            return

        print(f"[IQ2_XS PinnedPool] 开始预加载: 可用 {available_for_pool_gb:.1f}GB, "
              f"{n_layers} 层 × {n_experts} 专家")

        # 构建优先加载的层序列：首尾各10层优先
        priority_head = list(range(min(10, n_layers)))
        priority_tail = list(range(max(10, n_layers - 10), n_layers))
        priority_layers = priority_head + [l for l in priority_tail if l not in priority_head]
        other_layers = [l for l in range(n_layers) if l not in priority_layers]
        load_order = priority_layers + other_layers

        loaded = 0
        loaded_bytes = 0
        t0 = time.time()
        last_madvise_layer = -1

        for layer_id in load_order:
            # 检查内存预算
            if loaded_bytes >= available_for_pool_bytes:
                print(f"[IQ2_XS PinnedPool] 内存预算已满，停止加载 "
                      f"({loaded_bytes / 1024**3:.1f}GB / {available_for_pool_gb:.1f}GB)")
                break

            for expert_id in range(n_experts):
                key = (layer_id, expert_id)

                # 检查是否已在 pool 中
                with self._iq2xs_pinned_pool_lock:
                    if key in self._iq2xs_pinned_pool:
                        continue

                # 从归档读取
                cpu_params = {}
                entry_bytes = 0
                for weight_type, weight_name in enumerate(['w1', 'w2', 'w3']):
                    result = archive.get_expert(layer_id, expert_id, weight_type)
                    if result is None:
                        continue
                    d, qs, scales, shape = result
                    # pin_memory() 拷贝到 pinned memory，脱离 mmap 共享内存
                    d_tensor = torch.from_numpy(d).pin_memory()
                    qs_tensor = torch.from_numpy(qs).pin_memory()
                    scales_tensor = torch.from_numpy(scales).pin_memory()
                    entry_bytes += d_tensor.nbytes + qs_tensor.nbytes + scales_tensor.nbytes
                    cpu_params[weight_name] = {
                        "d": d_tensor,
                        "qs": qs_tensor,
                        "scales": scales_tensor,
                        "shape": shape,
                    }

                if not cpu_params:
                    continue

                # 检查内存预算
                if loaded_bytes + entry_bytes > available_for_pool_bytes:
                    # 超出预算，释放已分配的 pinned tensors
                    del cpu_params
                    break

                # 存入 pinned pool
                with self._iq2xs_pinned_pool_lock:
                    self._iq2xs_pinned_pool[key] = cpu_params

                loaded += 1
                loaded_bytes += entry_bytes

                # 进度日志：每 1000 个专家打印一次
                if loaded % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"[IQ2_XS PinnedPool] 进度: {loaded} 专家, "
                          f"{loaded_bytes / 1024**3:.1f}GB, "
                          f"{elapsed:.1f}s")

        elapsed = time.time() - t0
        total_experts = n_layers * n_experts
        coverage = loaded / total_experts * 100 if total_experts > 0 else 0

        # 全量加载完成：释放 mmap page cache（pinned pool 已包含所有数据）
        if coverage >= 100 and archive._mmap is not None:
            try:
                import mmap as _mmap
                archive._mmap.madvise(_mmap.MADV_DONTNEED)
                print(f"[IQ2_XS PinnedPool] 全量加载完成，释放 mmap page cache")
            except (AttributeError, OSError):
                pass

        used, available, swap_used = self.get_mem_info()
        print(f"[IQ2_XS PinnedPool] 完成: {loaded}/{total_experts} 专家 "
              f"({coverage:.0f}%), {loaded_bytes / 1024**3:.1f}GB, "
              f"{elapsed:.1f}s")
        print(f"[IQ2_XS PinnedPool] RAM: {used:.1f}GB used, {available:.1f}GB avail, "
              f"swap={swap_used:.1f}GB")

    def load_iq2xs_to_cpu(self, top_n_per_layer: int = 100):
        """IQ2_XS 模式下预加载专家到 pinned memory pool。

        Phase 1 优化：将专家数据加载到 pinned memory pool，
        而非跳过。pinned pool 中的数据可直接 DMA 到 GPU，无 page fault。
        如果 pinned pool 已有数据（warmup_iq2xs_mmap 已预加载），则跳过。
        """
        if self._iq2xs_archive is None:
            print("[IQ2_XS] 无归档文件，跳过")
            return

        # 检查 pinned pool 是否已有数据
        with self._iq2xs_pinned_pool_lock:
            pool_size = len(self._iq2xs_pinned_pool)

        if pool_size > 0:
            print(f"[IQ2_XS] pinned pool 已有 {pool_size} 专家，跳过重复加载")
            return

        # 确保 mmap 已预热（预热过程中会调用 _preload_iq2xs_to_pinned_pool）
        self.warmup_iq2xs_mmap()

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
                        "d": torch.from_numpy(d).pin_memory(),
                        "qs": torch.from_numpy(qs).pin_memory(),
                        "scales": torch.from_numpy(scales).pin_memory(),
                        "shape": shape,
                    }

                    cpu_params[skey] = iq2xs_data

                if cpu_params:
                    self._cpu_cache.put(key, cpu_params)
                    loaded += 1

        print(f"[IQ2_XS] 从归档加载: {loaded} 个权重")

    def warmup_from_cache(self, model):
        """热启动: 从频率统计加载热点专家到 GPU。"""
        if self._n_layers <= 0:
            return

        effective_model = model or self._model
        if effective_model is None:
            return

        # IQ2_XS 模式：直接从归档加载热点专家到 GPU，跳过 CPU SLRU
        if self._iq2xs_archive is not None:
            self._warmup_iq2xs_from_archive(effective_model)
            return

        # FP4 模式：Phase1 → CPU SLRU, Phase2 → GPU
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

    def _warmup_iq2xs_from_archive(self, model):
        """IQ2_XS 模式热启动：从 pinned pool 加载专家到 GPU，尽可能填满缓存。

        策略：
        1. 优先从 pinned pool 加载（零拷贝 DMA，无 SSD I/O）
        2. 按 LFU 频率排序，优先加载热点专家
        3. 填满 GPU 缓存容量，使活跃专家全部常驻 GPU
        """
        import time as _time
        t0 = _time.time()

        # 使用 GPU 缓存容量作为预算
        budget = self._gpu_cache.capacity
        if budget <= 0:
            return

        loaded = 0
        oom = False

        # 收集所有专家按 LFU 频率排序
        # 优先加载热点专家，确保高频访问的专家常驻 GPU
        sorted_experts = []
        for layer_id in range(self._n_layers):
            freq = self._layer_freq.get(layer_id, {})
            if freq:
                for eid, count in freq.items():
                    sorted_experts.append((layer_id, eid, count))
            else:
                # 无频率数据，加载前 30 个专家（覆盖 top-6 × 5 层的典型工作集）
                for eid in range(min(30, 256)):
                    sorted_experts.append((layer_id, eid, 0))

        # 按频率降序排序
        sorted_experts.sort(key=lambda x: -x[2])

        for layer_id, eid, _ in sorted_experts:
            if loaded >= budget or oom:
                break
            key = (layer_id, eid)
            if self._gpu_cache.contains(key):
                continue

            # 优先从 pinned pool 获取
            cpu_params = None
            with self._iq2xs_pinned_pool_lock:
                pool_entry = self._iq2xs_pinned_pool.get(key)
            if pool_entry is not None:
                cpu_params = {}
                for weight_name, entry in pool_entry.items():
                    skey = f"model.layers.{layer_id}.ffn.experts.{eid}.{weight_name}.weight"
                    cpu_params[skey] = {
                        "__iq2xs__": True,
                        "d": entry["d"],
                        "qs": entry["qs"],
                        "scales": entry["scales"],
                        "shape": entry["shape"],
                        "__from_pinned_pool__": True,
                    }

            # pinned pool 未命中，从归档读取
            if cpu_params is None and self._iq2xs_archive is not None:
                cpu_params = {}
                for weight_type, weight_name in enumerate(['w1', 'w2', 'w3']):
                    result = self._iq2xs_archive.get_expert(layer_id, eid, weight_type)
                    if result is None:
                        continue
                    d, qs, scales, shape = result
                    skey = f"model.layers.{layer_id}.ffn.experts.{eid}.{weight_name}.weight"
                    cpu_params[skey] = {
                        "__iq2xs__": True,
                        "d": torch.from_numpy(d).pin_memory(),
                        "qs": torch.from_numpy(qs).pin_memory(),
                        "scales": torch.from_numpy(scales).pin_memory(),
                        "shape": shape,
                    }

            if not cpu_params:
                continue

            try:
                gpu_params = self._pinned_to_gpu(cpu_params, layer_id)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom = True
                break

            if gpu_params:
                # SLRU: 先填满 protected（热点专家），再填满 probation
                if len(self._gpu_cache.protected) < self._gpu_cache.protected_capacity:
                    self._gpu_put_protected(key, gpu_params)
                else:
                    self._gpu_put(key, gpu_params)
                moe = model.layers[layer_id].ffn
                if eid < len(moe.experts) and moe.experts[eid] is None:
                    with torch.device('cpu'):
                        moe._ensure_expert(eid)
                    expert = moe.experts[eid]
                    if expert is not None:
                        self._set_expert_params(expert, gpu_params)
                loaded += 1

        elapsed = _time.time() - t0
        gpu_total = self._gpu_cache.total_entries()
        gpu_prot = len(self._gpu_cache.protected)
        gpu_prob = len(self._gpu_cache.probation)
        # 释放被淘汰专家的 GPU 内存
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Cache] IQ2_XS warmup: {loaded} experts → GPU "
              f"(prot={gpu_prot}/prob={gpu_prob}/total={gpu_total}), "
              f"budget={budget}, {elapsed:.1f}s")

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

    def prefetch_experts_batch(self, expert_keys: list[tuple[int, int]]) -> None:
        """Phase 2: 批量并行预取专家（非阻塞）。

        提交所有不在 GPU 缓存且未在预取中的专家到预取线程池。
        调用后立即返回，预取在后台线程并行执行。
        通过 collect_prefetch_results() 收集结果。

        参数:
            expert_keys: 需要预取的 (layer_id, expert_id) 列表
        """
        for layer_id, expert_id in expert_keys:
            key = (layer_id, expert_id)

            # 已在 GPU 缓存，跳过
            if self._gpu_cache.contains(key):
                continue

            # 已在预取队列，跳过
            with self._prefetch_lock:
                if layer_id in self._prefetch_pending and expert_id in self._prefetch_pending[layer_id]:
                    continue

            # 已在 IQ2_XS 预取队列，跳过
            with self._prefetch_lock:
                if key in self._iq2xs_prefetch_pending:
                    continue

            # 提交异步预取
            self._async_prefetch_expert(layer_id, expert_id)

    def collect_prefetch_results(self, expert_keys: list[tuple[int, int]],
                                  timeout: float = 2.0) -> Dict[tuple, dict]:
        """Phase 2: 收集预取结果（带超时）。

        等待指定专家的预取完成，返回已完成的 GPU 参数。
        不阻塞未完成的预取。

        参数:
            expert_keys: 需要收集的 (layer_id, expert_id) 列表
            timeout: 最大等待时间（秒）

        返回:
            {(layer_id, expert_id): gpu_params} 已完成的专家参数
        """
        results = {}
        deadline = time.time() + timeout

        for layer_id, expert_id in expert_keys:
            key = (layer_id, expert_id)

            # 已在 GPU 缓存，直接获取
            gpu_params = self._gpu_cache.get(key)
            if gpu_params is not None:
                results[key] = gpu_params
                continue

            # 检查预取结果
            with self._prefetch_lock:
                if layer_id in self._prefetch_pending and expert_id in self._prefetch_pending[layer_id]:
                    results[key] = self._prefetch_pending[layer_id].pop(expert_id)
                    if not self._prefetch_pending[layer_id]:
                        del self._prefetch_pending[layer_id]

            # 超时检查
            if time.time() > deadline:
                break

        return results
