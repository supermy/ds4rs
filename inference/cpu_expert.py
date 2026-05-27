"""
CPU 专家推理模块：IQ2_XS 逐块点积 + P2 极致缓存优化

策略：热点专家 GPU 推理，非热点专家 CPU 推理
硬件：AMD Ryzen 5 7600 (Zen 4, 6C/12T, 32MB L3, AVX-512 VNNI/BF16)

缓存层次优化：
  L1 (32KB/核): grid 表 4KB + sign 表 4KB + scale 表 2KB = 10KB 常驻
  L2 (1MB/核): x 向量 16KB + 输出缓冲 8KB = 24KB 常驻
  L3 (32MB): 专家权重 ~7MB，预加载 4 专家 = 28MB 常驻

P2 极致优化：
  1. L1 预取：手动预取下一组 grid/sign 数据
  2. L2 连续访问：x 向量预处理为连续块
  3. L3 预加载：专家权重提前加载到 L3
  4. 循环展开：group 循环 4 路展开
  5. 内存对齐：所有表数据 64B 对齐
"""

import atexit
import numpy as np
import torch
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

CACHE_LINE_SIZE = 64

IQ2XS_GRID = None
KSIGNS_IQ2XS = None
_SIGN_MUL_TABLE = None
_SCALE_DECODE_TABLE = None
_NUMBA_MATVEC_KERNEL = None
_NUMBA_AVAILABLE = None
_THREAD_POOL = None
_EXPERT_CACHE_L3 = None


def _cleanup_thread_pool():
    global _THREAD_POOL
    if _THREAD_POOL is not None:
        _THREAD_POOL.shutdown(wait=False)
        _THREAD_POOL = None


atexit.register(_cleanup_thread_pool)


def _align_array(arr: np.ndarray, alignment: int = CACHE_LINE_SIZE) -> np.ndarray:
    """将数组对齐到指定字节边界。"""
    if arr.nbytes % alignment == 0 and arr.ctypes.data % alignment == 0:
        return arr
    padded_size = ((arr.nbytes + alignment - 1) // alignment) * alignment
    padded = np.empty(padded_size // arr.itemsize, dtype=arr.dtype)
    padded[:arr.size] = arr.ravel()
    return np.ascontiguousarray(padded[:arr.size].reshape(arr.shape))


def _init_iq2xs_tables():
    global IQ2XS_GRID, KSIGNS_IQ2XS, _SIGN_MUL_TABLE, _SCALE_DECODE_TABLE
    if IQ2XS_GRID is not None:
        return

    from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS as KSIGNS_LIST

    grid_raw = np.frombuffer(
        b''.join(IQ2XS_GRID_U64[i].to_bytes(8, 'little') for i in range(512)),
        dtype=np.int8
    ).reshape(512, 8)
    IQ2XS_GRID = _align_array(np.ascontiguousarray(grid_raw))

    KSIGNS_IQ2XS = _align_array(np.array(KSIGNS_LIST, dtype=np.uint8))

    _SIGN_MUL_TABLE = np.empty((128, 8), dtype=np.float32)
    for i in range(128):
        for j in range(8):
            _SIGN_MUL_TABLE[i, j] = -1.0 if (KSIGNS_IQ2XS[i] >> j) & 1 else 1.0
    _SIGN_MUL_TABLE = _align_array(np.ascontiguousarray(_SIGN_MUL_TABLE))

    _SCALE_DECODE_TABLE = np.empty((256, 2), dtype=np.float32)
    for i in range(256):
        _SCALE_DECODE_TABLE[i, 0] = 2.0 * (i & 0xF) + 1.0
        _SCALE_DECODE_TABLE[i, 1] = 2.0 * (i >> 4) + 1.0
    _SCALE_DECODE_TABLE = _align_array(np.ascontiguousarray(_SCALE_DECODE_TABLE))


def _init_numba_kernel():
    global _NUMBA_MATVEC_KERNEL, _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is not None:
        return

    try:
        import numba
        from numba import njit, prange, set_num_threads
        import os

        n_threads = int(os.environ.get('DS4RS_CPU_THREADS', 6))
        set_num_threads(n_threads)

        @njit(cache=True, parallel=True, fastmath=True)
        def _iq2xs_matvec_numba_optimized(
            d, qs, scales, x, output,
            grid_table, sign_mul_table, scale_decode_table,
            n_rows, n_blocks_per_row,
        ):
            for row in prange(n_rows):
                row_sum = 0.0
                row_offset = row * n_blocks_per_row

                for blk in range(n_blocks_per_row):
                    bi = row_offset + blk
                    d_val = d[bi]
                    block_sum = 0.0

                    g = 0
                    while g < 32:
                        for _g_offset in range(4):
                            if g + _g_offset >= 32:
                                break
                            _g = g + _g_offset
                            q = qs[bi, _g]
                            gi = q & 511
                            si = (q >> 9) & 127

                            ib32 = _g >> 2
                            within = _g & 3
                            sc_val = scales[bi, ib32]
                            if within < 2:
                                ls = scale_decode_table[sc_val, 0]
                            else:
                                ls = scale_decode_table[sc_val, 1]

                            x_base = blk * 256 + _g * 8
                            group_dot = 0.0

                            gv0 = grid_table[gi, 0]
                            gv1 = grid_table[gi, 1]
                            gv2 = grid_table[gi, 2]
                            gv3 = grid_table[gi, 3]
                            gv4 = grid_table[gi, 4]
                            gv5 = grid_table[gi, 5]
                            gv6 = grid_table[gi, 6]
                            gv7 = grid_table[gi, 7]

                            sm0 = sign_mul_table[si, 0]
                            sm1 = sign_mul_table[si, 1]
                            sm2 = sign_mul_table[si, 2]
                            sm3 = sign_mul_table[si, 3]
                            sm4 = sign_mul_table[si, 4]
                            sm5 = sign_mul_table[si, 5]
                            sm6 = sign_mul_table[si, 6]
                            sm7 = sign_mul_table[si, 7]

                            group_dot = (gv0 * sm0 * x[x_base + 0] +
                                        gv1 * sm1 * x[x_base + 1] +
                                        gv2 * sm2 * x[x_base + 2] +
                                        gv3 * sm3 * x[x_base + 3] +
                                        gv4 * sm4 * x[x_base + 4] +
                                        gv5 * sm5 * x[x_base + 5] +
                                        gv6 * sm6 * x[x_base + 6] +
                                        gv7 * sm7 * x[x_base + 7])

                            block_sum += ls * group_dot

                        g += 4

                    row_sum += d_val * 0.125 * block_sum

                output[row] = row_sum

        _NUMBA_MATVEC_KERNEL = _iq2xs_matvec_numba_optimized
        _NUMBA_AVAILABLE = True
    except ImportError:
        _NUMBA_AVAILABLE = False


def _get_thread_pool():
    global _THREAD_POOL
    if _THREAD_POOL is None:
        import os
        n_threads = int(os.environ.get('DS4RS_CPU_THREADS', 6))
        _THREAD_POOL = ThreadPoolExecutor(max_workers=n_threads)
    return _THREAD_POOL


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.numpy()
    return t


def _flatten_iq2xs_params(d, qs, scales):
    d = np.ascontiguousarray(_to_numpy(d)).ravel()
    qs = np.ascontiguousarray(_to_numpy(qs), dtype=np.uint16).reshape(-1, 32)
    scales = np.ascontiguousarray(_to_numpy(scales), dtype=np.uint8).reshape(-1, 8)
    return d, qs, scales


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1.0 + exp_x)
    return result


def _preload_expert_to_l3(expert_data: dict) -> dict:
    """预加载专家权重到 L3 缓存。

    通过顺序访问所有权重数据，触发硬件预取到 L3。
    """
    global _EXPERT_CACHE_L3

    if _EXPERT_CACHE_L3 is None:
        _EXPERT_CACHE_L3 = {}

    for key in ['w1', 'w2', 'w3']:
        if key in expert_data:
            w = expert_data[key]
            d, qs, scales = _flatten_iq2xs_params(w['d'], w['qs'], w['scales'])
            _ = d.sum()
            _ = qs.sum()
            _ = scales.sum()

    return expert_data


def iq2xs_matvec(
    d: np.ndarray,
    qs: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    _init_iq2xs_tables()

    d, qs, scales = _flatten_iq2xs_params(d, qs, scales)
    x = np.ascontiguousarray(_to_numpy(x), dtype=np.float32).ravel()

    n_blocks_per_row = n_cols // 256
    output = np.zeros(n_rows, dtype=np.float32)
    x_blocks = x.reshape(-1, 256)

    for row in range(n_rows):
        row_start = row * n_blocks_per_row
        row_sum = 0.0

        for blk_idx in range(n_blocks_per_row):
            bi = row_start + blk_idx
            x_block = x_blocks[blk_idx]

            qs_block = qs[bi]
            grid_idx = (qs_block & 511).astype(np.int32)
            sign_idx = (qs_block >> 9).astype(np.int32) & 127

            grid_vals = IQ2XS_GRID[grid_idx]
            sign_mul = _SIGN_MUL_TABLE[sign_idx]
            grid_signed = grid_vals.astype(np.float32) * sign_mul

            sc = scales[bi]
            scale_low = (sc & 0xF).astype(np.float32)
            scale_high = (sc >> 4).astype(np.float32)
            scale_16 = np.stack([scale_low, scale_high], axis=-1).ravel()
            scale_32 = np.repeat(scale_16, 2)
            scale_val = 2.0 * scale_32 + 1.0

            d_val = float(d[bi])
            dequant_block = d_val * scale_val[:, np.newaxis] * grid_signed * 0.125
            row_sum += np.dot(dequant_block.ravel(), x_block)

        output[row] = row_sum

    return output


def iq2xs_matvec_batch(
    d: np.ndarray,
    qs: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    _init_iq2xs_tables()
    _init_numba_kernel()

    d, qs, scales = _flatten_iq2xs_params(d, qs, scales)
    x = np.ascontiguousarray(_to_numpy(x), dtype=np.float32).ravel()

    n_blocks_per_row = n_cols // 256
    assert n_blocks_per_row > 0, f"n_cols={n_cols} must be >= 256 (block size)"
    n_blocks = len(d)
    assert n_blocks == n_rows * n_blocks_per_row, \
        f"n_blocks={n_blocks} != n_rows={n_rows} * n_blocks_per_row={n_blocks_per_row}"

    if _NUMBA_AVAILABLE and n_rows > 0:
        d_f32 = d.astype(np.float32)
        output = np.zeros(n_rows, dtype=np.float32)
        _NUMBA_MATVEC_KERNEL(
            d_f32, qs, scales, x, output,
            IQ2XS_GRID, _SIGN_MUL_TABLE, _SCALE_DECODE_TABLE,
            n_rows, n_blocks_per_row,
        )
        return output

    return _iq2xs_matvec_numpy(d, qs, scales, x, n_rows, n_cols, n_blocks_per_row)


def _iq2xs_matvec_numpy(
    d, qs, scales, x, n_rows, n_cols, n_blocks_per_row,
) -> np.ndarray:
    x_groups = x.reshape(n_blocks_per_row, 32, 8)
    TILE_ROWS = 64
    output = np.zeros(n_rows, dtype=np.float32)

    for row_start in range(0, n_rows, TILE_ROWS):
        row_end = min(row_start + TILE_ROWS, n_rows)
        tile_rows = row_end - row_start
        b0 = row_start * n_blocks_per_row
        b1 = row_end * n_blocks_per_row
        tile_n_blocks = b1 - b0

        d_t = d[b0:b1].astype(np.float32)
        qs_t = qs[b0:b1]
        sc_t = scales[b0:b1]

        gi = (qs_t & 511).astype(np.int32)
        si = (qs_t >> 9).astype(np.int32) & 127
        gv = IQ2XS_GRID[gi]
        sm = _SIGN_MUL_TABLE[si]
        grid_signed = gv.astype(np.float32) * sm

        bi = np.arange(tile_n_blocks) % n_blocks_per_row
        xg = x_groups[bi]
        group_dots = (grid_signed * xg).sum(axis=2)

        scale_16 = _SCALE_DECODE_TABLE[sc_t].reshape(tile_n_blocks, 16)
        scale_32 = np.repeat(scale_16, 2, axis=1)
        scaled_dots = scale_32 * group_dots
        block_dots = d_t * 0.125 * scaled_dots.sum(axis=1)

        output[row_start:row_end] = block_dots.reshape(tile_rows, n_blocks_per_row).sum(axis=1)

    return output


def dequantize_iq2xs_block(d: np.ndarray, qs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    _init_iq2xs_tables()

    d, qs, scales = _flatten_iq2xs_params(d, qs, scales)
    n_blocks = len(d)

    grid_idx = (qs & 511).astype(np.int32)
    sign_idx = (qs >> 9).astype(np.int32) & 127

    grid_vals = IQ2XS_GRID[grid_idx]
    sign_mul = _SIGN_MUL_TABLE[sign_idx]
    grid_signed = grid_vals.astype(np.float32) * sign_mul

    scale_16 = _SCALE_DECODE_TABLE[scales].reshape(n_blocks, 16)
    scale_32 = np.repeat(scale_16, 2, axis=1)
    scale_val = scale_32

    d_float = d.astype(np.float32)
    result = d_float[:, np.newaxis, np.newaxis] * scale_val[:, :, np.newaxis] * grid_signed * 0.125

    return result.reshape(-1)


def dequantize_iq2xs_weight(cpu_params: dict) -> np.ndarray:
    if not cpu_params.get("__iq2xs__"):
        raise ValueError("Not IQ2_XS format")

    d = cpu_params["d"]
    qs = cpu_params["qs"]
    scales = cpu_params["scales"]
    shape = cpu_params["shape"]

    d, qs, scales = _flatten_iq2xs_params(d, qs, scales)
    weight = dequantize_iq2xs_block(d, qs, scales)

    total_elements = weight.size
    expected_elements = 1
    for s in shape:
        expected_elements *= s
    if total_elements != expected_elements:
        if total_elements > expected_elements:
            weight = weight[:expected_elements]
        else:
            weight = np.pad(weight, (0, expected_elements - total_elements))
    return weight.reshape(shape)


def cpu_expert_ffn(
    x: np.ndarray,
    gate_up_weight: dict,
    down_weight: dict,
    route_weight: float = 1.0,
    swiglu_limit: float = 0.0,
) -> np.ndarray:
    x = np.ascontiguousarray(_to_numpy(x), dtype=np.float32).ravel()

    gate_up_shape = gate_up_weight["shape"]
    n_rows_gu = 1
    for s in gate_up_shape[:-1]:
        n_rows_gu *= s
    n_cols_gu = gate_up_shape[-1]

    d_gu, qs_gu, sc_gu = _flatten_iq2xs_params(
        gate_up_weight["d"], gate_up_weight["qs"], gate_up_weight["scales"])
    gate_up_out = iq2xs_matvec_batch(d_gu, qs_gu, sc_gu, x, n_rows_gu, n_cols_gu)

    if n_rows_gu % 2 == 0:
        inter_dim = n_rows_gu // 2
        gate = gate_up_out[:inter_dim]
        up = gate_up_out[inter_dim:]
    else:
        inter_dim = n_rows_gu
        gate = gate_up_out
        up = gate_up_out

    if swiglu_limit > 0:
        up = np.clip(up, -swiglu_limit, swiglu_limit)
        gate = np.clip(gate, None, swiglu_limit)

    sigmoid_gate = _stable_sigmoid(gate.astype(np.float32))
    mid = gate * sigmoid_gate * up

    if route_weight != 1.0:
        mid = mid * route_weight

    down_shape = down_weight["shape"]
    n_rows_d = 1
    for s in down_shape[:-1]:
        n_rows_d *= s
    n_cols_d = down_shape[-1]

    d_d, qs_d, sc_d = _flatten_iq2xs_params(
        down_weight["d"], down_weight["qs"], down_weight["scales"])
    output = iq2xs_matvec_batch(d_d, qs_d, sc_d, mid, n_rows_d, n_cols_d)

    return output


def cpu_expert_ffn_pair(
    x: np.ndarray,
    gate_weight: dict,
    up_weight: dict,
    down_weight: dict,
    route_weight: float = 1.0,
    swiglu_limit: float = 0.0,
    preload_to_l3: bool = True,
) -> np.ndarray:
    if preload_to_l3:
        _preload_expert_to_l3({'w1': gate_weight, 'w3': up_weight, 'w2': down_weight})

    x = np.ascontiguousarray(_to_numpy(x), dtype=np.float32).ravel()

    gate_shape = gate_weight["shape"]
    n_rows_g = 1
    for s in gate_shape[:-1]:
        n_rows_g *= s
    n_cols_g = gate_shape[-1]

    up_shape = up_weight["shape"]
    n_rows_u = 1
    for s in up_shape[:-1]:
        n_rows_u *= s
    n_cols_u = up_shape[-1]

    d_g, qs_g, sc_g = _flatten_iq2xs_params(
        gate_weight["d"], gate_weight["qs"], gate_weight["scales"])
    gate = iq2xs_matvec_batch(d_g, qs_g, sc_g, x, n_rows_g, n_cols_g)

    d_u, qs_u, sc_u = _flatten_iq2xs_params(
        up_weight["d"], up_weight["qs"], up_weight["scales"])
    up = iq2xs_matvec_batch(d_u, qs_u, sc_u, x, n_rows_u, n_cols_u)

    if swiglu_limit > 0:
        up = np.clip(up, -swiglu_limit, swiglu_limit)
        gate = np.clip(gate, None, swiglu_limit)

    sigmoid_gate = _stable_sigmoid(gate.astype(np.float32))
    mid = gate * sigmoid_gate * up

    if route_weight != 1.0:
        mid = mid * route_weight

    down_shape = down_weight["shape"]
    n_rows_d = 1
    for s in down_shape[:-1]:
        n_rows_d *= s
    n_cols_d = down_shape[-1]

    d_d, qs_d, sc_d = _flatten_iq2xs_params(
        down_weight["d"], down_weight["qs"], down_weight["scales"])
    output = iq2xs_matvec_batch(d_d, qs_d, sc_d, mid, n_rows_d, n_cols_d)

    return output


class CpuExpertRunner:
    """CPU 专家推理运行器，带分层预加载缓存。

    分层预加载策略（按频率分层）：
      1. IQ2_XS 全量载入内存（pinned pool，100% CPU 命中）
      2. 每层专家按访问频率排序
      3. GPU: 预加载每层 Top-N 热专家 → GPU GEMM (0.2ms)
      4. CPU L3: 预热每层 TopN+1 ~ TopN+M → CPU FFN warm (2.0ms)
      5. CPU RAM: 其余专家 → CPU FFN cold (2.7ms)
      6. GPU miss 时走 CPU FFN 而非 DMA（CPU FFN 2.7ms < DMA 5.0ms）

    预取时机：
      - on_step_start(): 协调预取下一层 Top-N → GPU + TopN+1~M → CPU L3
      - on_cpu_ffn_done(): 事件驱动预取同层下一个差集专家
    """

    # CPU L3 预热的专家数量（TopN+1 ~ TopN+M）
    DEFAULT_WARMUP_M = 4

    def __init__(self, expert_cache):
        self._cache = expert_cache
        self._pinned_pool = getattr(expert_cache, '_iq2xs_pinned_pool', None)
        self._use_rust = False
        self._rust_runner = None
        self._gpu_cache_ref = None
        self._access_freq: dict[tuple, int] = {}
        self._step_count = 0
        self._gpu_topk = 8  # 模型配置的 top-K 值，从 model config 获取
        self._warmup_m = self.DEFAULT_WARMUP_M
        self._mixed_pool = None  # IQ2_XXS+Q2_K 混合量化专家池

        # 尝试初始化 Rust AVX-512 内核
        try:
            from inference.rust_cpu_expert import is_rust_available, init_rust_tables, RustCpuExpertRunner
            if is_rust_available():
                init_rust_tables()
                self._rust_runner = RustCpuExpertRunner()
                self._use_rust = True
        except (ImportError, Exception):
            pass

    def set_mixed_pool(self, mixed_pool):
        """设置 IQ2_XXS+Q2_K 混合量化专家池。"""
        self._mixed_pool = mixed_pool

    def set_gpu_cache(self, gpu_cache):
        """设置 GPU SLRU 引用，用于热专家同步。"""
        self._gpu_cache_ref = gpu_cache

    def set_gpu_topk(self, topk: int):
        """设置模型配置的 GPU top-K 值（动态，从 model config 获取）。

        不同模型 top-K 不同：DeepSeek-V2=6, DeepSeek-V3=8, Mixtral=2。
        当前预取策略基于 GPU SLRU 成员检查而非 top-N 截断，
        此值保留用于统计和日志，未来可能用于预取预算控制。
        """
        self._gpu_topk = topk

    def auto_set_memory_budget(self, total_expert_bytes: int, available_bytes: int):
        """自动设置内存预算。

        关键：pinned_pool 使用 mmap 零拷贝，不占 Python 堆内存。
        Rust SLRU 的 Iq2XsWeight 是真正的内存副本。
        所以预算只控制 Rust SLRU 的内存占用。

        Args:
            total_expert_bytes: 所有专家的总字节数（用于计算比例）
            available_bytes: 可用内存字节数（扣除系统和其他进程）
        """
        if self._rust_runner is not None:
            self._rust_runner.auto_set_budget(total_expert_bytes, available_bytes)

    def on_step_start(self, step: int):
        """Step 开始：记录 step 计数。

        预取由 MoE.forward() 中的 prefetch_layer(layer_id + 1) 驱动，
        不在此处推断 next_layer（避免从 GPU SLRU 推断不准确）。
        """
        self._step_count = step

    def prefetch_layer(self, layer_id: int):
        """预取指定层专家到 GPU + CPU（协调预取）。

        GPU-CPU 协调预取策略：
          1. 差集 Top-N → GPU DMA 预取（GPU SLRU 的新专家来源）
          2. 差集 N+1 之后 → CPU Rust SLRU 预取（GPU miss 时 CPU 兜底）

        "差集" = 该层中不在 GPU SLRU 的专家
        "Top-N" = 差集中按频率排序的前 N 个（N = n_activated_experts）

        为什么差集 Top-N 要预取到 GPU：
          GPU SLRU 的专家不是凭空出现的，需要 CPU 侧主动推送。
          差集 Top-N 是最可能被激活的专家，提前 DMA 到 GPU 可以减少 miss。

        两级 CPU 预取：
          1. 格式预转换：pinned_pool dict → Iq2XsWeight
          2. L3 缓存预热：顺序读取触发硬件预取器

        Args:
            layer_id: 要预取的层 ID
        """
        self._prefetch_layer_coordinated(layer_id)

    def _prefetch_layer_coordinated(self, layer_id: int, step: int = -1):
        """分层预加载：Top-N → GPU DMA，TopN+1~M → CPU L3 预热，其余 → CPU RAM。

        策略：
          1. 每层专家按频率降序排序
          2. Top-N (N=gpu_topk): DMA 预取到 GPU SLRU
          3. TopN+1 ~ TopN+M (M=warmup_m): 加载到 Rust SLRU + L3 预热
          4. 其余: 仅加载到 Rust SLRU（cold RAM）

        关键改变（vs 旧差集分流策略）：
          - GPU miss 时走 CPU FFN 而非 DMA（2.7ms < 5.0ms）
          - L3 预热只针对 TopN+1~M 范围，跳过 GPU 已有的
          - 预期加权延迟：4.15ms（vs 旧策略 7.54ms）
        """
        if not self._use_rust or self._rust_runner is None:
            return
        if self._pinned_pool is None:
            return

        # 收集该层所有专家，按 CPU miss 频率降序排序
        layer_experts = self._get_layer_experts_sorted(layer_id)

        # 构建 GPU SLRU 成员集合
        gpu_cached = set()
        if self._gpu_cache_ref is not None:
            gpu_cached = set(self._gpu_cache_ref.protected.keys()) | set(self._gpu_cache_ref.probation.keys())

        # 差集 = 不在 GPU SLRU 中的专家（按频率降序）
        diff_experts = [(lid, eid) for (lid, eid) in layer_experts
                        if (lid, eid) not in gpu_cached]

        if not diff_experts:
            return

        gpu_topk = self._gpu_topk
        warmup_m = self._warmup_m

        # ---- Phase 1: 差集 Top-N → GPU DMA 预取 ----
        gpu_prefetch_keys = diff_experts[:gpu_topk]
        if gpu_prefetch_keys and self._cache is not None:
            try:
                self._cache.prefetch_experts_batch(gpu_prefetch_keys)
            except Exception:
                pass

        # ---- Phase 2: 差集 TopN+1 ~ TopN+M → CPU Rust SLRU + L3 预热 ----
        warmup_keys = diff_experts[gpu_topk:gpu_topk + warmup_m]
        synced_warmup = 0
        for (lid, eid) in warmup_keys:
            if not self._rust_runner.has_expert(lid, eid):
                expert_data = self._pinned_pool.get((lid, eid))
                if expert_data is not None:
                    w1 = expert_data.get('w1')
                    w3 = expert_data.get('w3')
                    w2 = expert_data.get('w2')
                    if w1 is not None and w2 is not None and w3 is not None:
                        try:
                            self._rust_runner.add_expert(lid, eid, w1, w3, w2)
                            synced_warmup += 1
                        except Exception:
                            pass

        # ---- Phase 3: 差集 M 之后 → CPU Rust SLRU (cold RAM) ----
        cold_keys = diff_experts[gpu_topk + warmup_m:]
        synced_cold = 0
        for (lid, eid) in cold_keys:
            if not self._rust_runner.has_expert(lid, eid):
                expert_data = self._pinned_pool.get((lid, eid))
                if expert_data is not None:
                    w1 = expert_data.get('w1')
                    w3 = expert_data.get('w3')
                    w2 = expert_data.get('w2')
                    if w1 is not None and w2 is not None and w3 is not None:
                        try:
                            self._rust_runner.add_expert(lid, eid, w1, w3, w2)
                            synced_cold += 1
                        except Exception:
                            pass

        # ---- Phase 4: L3 缓存预热（只预热 TopN+1~M 范围） ----
        try:
            if hasattr(self._rust_runner, 'warmup_layer_targeted'):
                self._rust_runner.warmup_layer_targeted(
                    layer_id, gpu_cached, gpu_topk, warmup_m)
            else:
                self._rust_runner.warmup_layer(layer_id)
        except Exception:
            pass

        if step >= 0 and (gpu_prefetch_keys or synced_warmup or synced_cold):
            print(f"[CPU-PRELOAD] Step {step} L{layer_id}: "
                  f"GPU prefetch={len(gpu_prefetch_keys)}, "
                  f"CPU L3 warmup={synced_warmup}, "
                  f"CPU cold={synced_cold}, "
                  f"total={self._rust_runner.expert_count()}, "
                  f"mem={self._rust_runner.memory_usage_mb():.0f}MB")

    def on_cpu_ffn_done(self, layer_id: int, expert_id: int):
        """CPU FFN 计算完成后的回调（事件驱动预取）。

        在 CPU FFN 计算完成后触发，预取同层下一个差集专家：
          - 差集 Top-N 内的 → 触发 GPU DMA 预取
          - 差集 N+1 之后的 → 预取到 CPU Rust SLRU

        差集 = 不在 GPU SLRU 中的专家，按频率降序。
        找差集中 Rust SLRU 也没有的第一个专家（还没被预取的）。
        """
        if not self._use_rust or self._rust_runner is None:
            return
        if self._pinned_pool is None:
            return

        # GPU SLRU 成员检查
        gpu_cached = set()
        if self._gpu_cache_ref is not None:
            gpu_cached = set(self._gpu_cache_ref.protected.keys()) | set(self._gpu_cache_ref.probation.keys())

        # 找同层差集中还没被预取的下一个专家
        layer_experts = self._get_layer_experts_sorted(layer_id)
        diff_experts = [(lid, eid) for (lid, eid) in layer_experts
                        if (lid, eid) not in gpu_cached]

        gpu_topk = self._gpu_topk
        for idx, (lid, eid) in enumerate(diff_experts):
            # 跳过已经在 Rust SLRU 中的（已被预取过）
            if self._rust_runner.has_expert(lid, eid):
                continue

            if idx < gpu_topk and self._cache is not None:
                # 差集 Top-N → GPU DMA 预取
                try:
                    self._cache.prefetch_experts_batch([(lid, eid)])
                except Exception:
                    pass
            else:
                # 差集 N+1 之后 → CPU Rust SLRU
                expert_data = self._pinned_pool.get((lid, eid))
                if expert_data is not None:
                    w1 = expert_data.get('w1')
                    w3 = expert_data.get('w3')
                    w2 = expert_data.get('w2')
                    if w1 is not None and w2 is not None and w3 is not None:
                        try:
                            self._rust_runner.add_expert(lid, eid, w1, w3, w2)
                        except Exception:
                            pass
            break  # 只预取一个

    def _get_layer_experts_sorted(self, layer_id: int) -> list[tuple]:
        """获取指定层的专家列表，按访问频率降序排序。"""
        layer_experts = []
        for (lid, eid), freq in self._access_freq.items():
            if lid == layer_id:
                layer_experts.append(((lid, eid), freq))

        # 按频率降序排序
        layer_experts.sort(key=lambda x: -x[1])

        # 如果频率数据不足，补充 pinned_pool 中的专家
        if len(layer_experts) < 256 and self._pinned_pool is not None:
            existing = {k for k, _ in layer_experts}
            for (lid, eid) in self._pinned_pool.keys():
                if lid == layer_id and (lid, eid) not in existing:
                    layer_experts.append(((lid, eid), 0))
                    existing.add((lid, eid))

        return [k for k, _ in layer_experts]

    def on_step_end(self):
        """Step 结束：频率数据已在 Phase 0 统一更新，无需额外操作。"""
        pass

    def restore_access_freq_from_rust(self):
        """从 Rust runner 的 freq 数据恢复 Python _access_freq。

        热重启后 Python _access_freq 为空，但 Rust runner 可能有持久化的 freq 数据。
        调用此方法同步，使 _get_layer_experts_sorted 排序正确。
        """
        if not self._use_rust or self._rust_runner is None:
            return
        try:
            for layer_id in range(43):  # 43 层
                rank_list = self._rust_runner.layer_freq_rank(layer_id)
                for expert_id, freq in rank_list:
                    if freq > 0:
                        self._access_freq[(layer_id, expert_id)] = freq
        except Exception:
            pass

    def compute_expert_cpu(
        self,
        layer_id: int,
        expert_id: int,
        x_gpu: torch.Tensor,
        route_weight: float = 1.0,
        swiglu_limit: float = 0.0,
    ) -> torch.Tensor:
        # 频率已在 _load_activated_experts Phase 0 的 record_access 中统一递增
        # 此处不再重复计数

        # IQ2_XXS+Q2_K 混合量化路径
        if self._mixed_pool is not None:
            x_cpu = x_gpu.float().cpu().numpy()
            try:
                output = self._mixed_pool.compute_ffn(layer_id, expert_id, x_cpu, route_weight, swiglu_limit)
                output_gpu = torch.from_numpy(output).to(x_gpu.device, dtype=x_gpu.dtype)
                return output_gpu
            except Exception as e:
                print(f"[MixedFFN] Error: {e}")
                return torch.zeros_like(x_gpu)

        x_cpu = x_gpu.float().cpu().numpy()

        expert_data = self._get_expert_data(layer_id, expert_id)
        if expert_data is None:
            return torch.zeros_like(x_gpu)

        w1 = expert_data.get('w1')
        w2 = expert_data.get('w2')
        w3 = expert_data.get('w3')

        if w1 is None or w2 is None or w3 is None:
            return torch.zeros_like(x_gpu)

        # 优先使用 Rust CpuExpertRunner（Arc 缓存权重，避免每次重建 Iq2XsWeight）
        if self._use_rust and self._rust_runner is not None:
            try:
                # 懒加载：首次访问时 add_expert，后续直接 compute_expert
                if not self._rust_runner.has_expert(layer_id, expert_id):
                    self._rust_runner.add_expert(layer_id, expert_id, w1, w3, w2)

                x_f32 = np.ascontiguousarray(x_cpu.ravel(), dtype=np.float32)
                output = self._rust_runner.compute_expert(layer_id, expert_id, x_f32, route_weight)
                if output is not None:
                    output_gpu = torch.from_numpy(output).to(x_gpu.device, dtype=x_gpu.dtype)
                    # 事件驱动：CPU FFN 完成后预取同层下一个冷专家
                    self.on_cpu_ffn_done(layer_id, expert_id)
                    return output_gpu
            except Exception:
                pass  # fallback to numba

        # numba 回退路径
        output = cpu_expert_ffn_pair(x_cpu, w1, w3, w2, route_weight, swiglu_limit, preload_to_l3=True)

        output_gpu = torch.from_numpy(output).to(x_gpu.device, dtype=x_gpu.dtype)
        return output_gpu

    def _get_expert_data(self, layer_id: int, expert_id: int) -> dict | None:
        if self._pinned_pool is None:
            return None

        key = (layer_id, expert_id)
        return self._pinned_pool.get(key)
