"""Python ctypes 封装调用 C IQ2_XS 量化函数。

根据 AGENTS.md 的量化路径：
解码 FP4 → TileLang(float32 → IQ2_XS 量化) → 缓存 → 传 GPU → TileLang IQ2_XS GEMM kernel

此模块提供：
- C 量化函数的 Python 封装
- GPU 张量与 C 内存的高效转换
- 与 PyTorch 的无缝集成
"""
import ctypes
import numpy as np
import torch
import os
from typing import Tuple, Optional

# ============================================================================
# C 库加载
# ============================================================================

_lib = None

def _get_lib_path() -> str:
    """查找 libiq2_xs_c.so 路径"""
    for root, dirs, files in os.walk('/workspace/target/release/build'):
        for f in files:
            if f == 'libiq2_xs_c.so':
                return os.path.join(root, f)
    raise RuntimeError("找不到 libiq2_xs_c.so，请先运行 cargo build --release")

def _init_lib():
    """初始化 C 库"""
    global _lib
    if _lib is not None:
        return _lib
    
    lib_path = _get_lib_path()
    _lib = ctypes.CDLL(lib_path)
    
    # 设置函数签名
    _lib.iq2xs_init_ffi.argtypes = []
    _lib.iq2xs_init_ffi.restype = None
    
    _lib.iq2xs_free_ffi.argtypes = []
    _lib.iq2xs_free_ffi.restype = None
    
    _lib.quantize_iq2_xs_ffi.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # src
        ctypes.POINTER(ctypes.c_uint8),  # dst
        ctypes.c_int64,                  # nrow
        ctypes.c_int64,                  # n_per_row
    ]
    _lib.quantize_iq2_xs_ffi.restype = ctypes.c_size_t
    
    _lib.dequantize_iq2_xs_ffi.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # blocks
        ctypes.POINTER(ctypes.c_float),  # dst
        ctypes.c_int64,                  # n
    ]
    _lib.dequantize_iq2_xs_ffi.restype = None
    
    _lib.iq2_xs_block_size_ffi.argtypes = []
    _lib.iq2_xs_block_size_ffi.restype = ctypes.c_size_t
    
    _lib.iq2_xs_qk_k_ffi.argtypes = []
    _lib.iq2_xs_qk_k_ffi.restype = ctypes.c_int
    
    # 初始化运行时
    _lib.iq2xs_init_ffi()
    
    return _lib

# ============================================================================
# 常量
# ============================================================================

def get_block_size() -> int:
    """获取 block_iq2_xs 大小（74 字节）"""
    lib = _init_lib()
    return lib.iq2_xs_block_size_ffi()

def get_qk_k() -> int:
    """获取 QK_K 常量（256）"""
    lib = _init_lib()
    return lib.iq2_xs_qk_k_ffi()

# ============================================================================
# 量化函数
# ============================================================================

def quantize_iq2_xs_cpu(
    x: np.ndarray,
    nrow: int = 1,
) -> np.ndarray:
    """
    使用 C CPU 实现 IQ2_XS 量化。
    
    参数:
        x: float32 数组，形状 [nrow * n_per_row]，n_per_row 必须是 256 的倍数
        nrow: 行数（默认 1）
    
    返回:
        uint8 数组，形状 [n_blocks * block_size]，n_blocks = x.size / 256
    """
    lib = _init_lib()
    block_size = get_block_size()
    qk_k = get_qk_k()
    
    n_elements = x.size
    n_per_row = n_elements // nrow
    
    if n_elements % qk_k != 0:
        raise ValueError(f"元素数 {n_elements} 必须是 {qk_k} 的倍数")
    
    n_blocks = n_elements // qk_k
    dst_size = n_blocks * block_size
    
    # 确保输入是连续的 float32
    x_cont = np.ascontiguousarray(x, dtype=np.float32)
    
    # 分配输出
    dst = np.zeros(dst_size, dtype=np.uint8)
    
    # 调用 C 函数
    written = lib.quantize_iq2_xs_ffi(
        x_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int64(nrow),
        ctypes.c_int64(n_per_row),
    )
    
    if written != dst_size:
        raise RuntimeError(f"C 量化写入 {written} 字节，预期 {dst_size}")
    
    return dst

def quantize_iq2_xs_gpu_via_cpu(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU 张量 → CPU 量化 → GPU 结果。
    
    用于批量量化：GPU 加载数据，CPU 量化，结果传回 GPU。
    
    参数:
        x: GPU float32 张量，形状 [N, 256] 或 [N * 256]
    
    返回:
        d:   GPU float16 张量，形状 [N]，全局缩放因子
        qs:  GPU uint16 张量，形状 [N, 32]，grid 索引 + 符号索引
        scales: GPU uint8 张量，形状 [N, 8]，4-bit 打包的子块缩放
    """
    block_size = get_block_size()
    qk_k = get_qk_k()
    
    # 处理输入形状
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    N = x.shape[0]
    n_per_block = x.shape[1] if x.dim() > 1 else qk_k
    
    if n_per_block != qk_k:
        raise ValueError(f"每块元素数 {n_per_block} 必须是 {qk_k}")
    
    # GPU → CPU
    x_cpu = x.contiguous().cpu().numpy().astype(np.float32)
    
    # C 量化
    quantized = quantize_iq2_xs_cpu(x_cpu.flatten(), nrow=N)
    
    # 解析 block_iq2_xs 结构
    # struct block_iq2_xs {
    #     __half d;        // 2 bytes
    #     uint16_t qs[32]; // 64 bytes
    #     uint8_t scales[8]; // 8 bytes
    # }  // total 74 bytes
    
    quantized = quantized.reshape(N, block_size)
    
    d_np = np.frombuffer(quantized[:, 0:2].tobytes(), dtype=np.float16).reshape(N)
    qs_np = np.frombuffer(quantized[:, 2:66].tobytes(), dtype=np.uint16).reshape(N, 32)
    scales_np = quantized[:, 66:74].copy()
    
    # CPU → GPU
    d = torch.from_numpy(d_np).to(x.device)
    qs = torch.from_numpy(qs_np).to(x.device)
    scales = torch.from_numpy(scales_np).to(x.device)
    
    return d, qs, scales

def dequantize_iq2_xs_cpu(
    quantized: np.ndarray,
) -> np.ndarray:
    """
    使用 C CPU 实现 IQ2_XS 反量化。
    
    参数:
        quantized: uint8 数组，形状 [n_blocks * 74]
    
    返回:
        float32 数组，形状 [n_blocks * 256]
    """
    lib = _init_lib()
    block_size = get_block_size()
    qk_k = get_qk_k()
    
    if quantized.size % block_size != 0:
        raise ValueError(f"量化数据大小 {quantized.size} 必须是 {block_size} 的倍数")
    
    n_blocks = quantized.size // block_size
    n_elements = n_blocks * qk_k
    
    # 确保输入是连续的 uint8
    quantized_cont = np.ascontiguousarray(quantized, dtype=np.uint8)
    
    # 分配输出
    dst = np.zeros(n_elements, dtype=np.float32)
    
    # 调用 C 函数
    lib.dequantize_iq2_xs_ffi(
        quantized_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(n_elements),
    )
    
    return dst

# ============================================================================
# 批量量化（用于离线处理）
# ============================================================================

def quantize_batch_via_cpu(
    batches: torch.Tensor,
    progress_callback: Optional[callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量量化多个 256 元素块。
    
    参数:
        batches: GPU float32 张量，形状 [N, 256]
        progress_callback: 可选的进度回调函数
    
    返回:
        d, qs, scales: 同 quantize_iq2_xs_gpu_via_cpu
    """
    N = batches.shape[0]
    block_size = get_block_size()
    qk_k = get_qk_k()
    
    # 分批处理以控制内存
    batch_size = min(1024, N)  # 每批最多 1024 个块
    n_batches = (N + batch_size - 1) // batch_size
    
    all_d = []
    all_qs = []
    all_scales = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, N)
        chunk = batches[start:end]
        
        d, qs, scales = quantize_iq2_xs_gpu_via_cpu(chunk)
        all_d.append(d)
        all_qs.append(qs)
        all_scales.append(scales)
        
        if progress_callback:
            progress_callback(i + 1, n_batches, end - start)
    
    return (
        torch.cat(all_d, dim=0),
        torch.cat(all_qs, dim=0),
        torch.cat(all_scales, dim=0),
    )

# ============================================================================
# 清理
# ============================================================================

def cleanup():
    """释放 C 运行时资源"""
    global _lib
    if _lib is not None:
        _lib.iq2xs_free_ffi()
        _lib = None

import atexit
atexit.register(cleanup)
