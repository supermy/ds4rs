"""GGUF 格式 IQ2_XS 专家权重存储。

GGUF 文件格式：
================================================================================
1. 文件头:
   - magic: "GGUF" (4 字节)
   - version: uint32 (当前版本 3)
   - n_tensors: int64 (张量数量)
   - n_kv: int64 (KV 对数量)

2. KV 对 (元数据):
   - key: string (长度 + 内容)
   - type: int32
   - value: 根据 type

3. 张量信息:
   - name: string
   - n_dims: uint32
   - dims: int64[]
   - type: int32 (GGML_TYPE)
   - offset: uint64

4. 张量数据 (对齐到 32 字节)
================================================================================

IQ2_XS 张量存储：
- 每个 block (256 元素) 需要 74 字节：
  - d: 2 字节 (float16)
  - qs: 64 字节 (32 × uint16)
  - scales: 8 字节 (uint8)
"""
import struct
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, BinaryIO
import numpy as np
import mmap

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF 类型
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML 类型 (仅定义需要的)
GGML_TYPE_F16 = 1
GGML_TYPE_IQ2_XS = 28  # llama.cpp 中的定义

# IQ2_XS block 大小
IQ2_XS_BLOCK_SIZE = 256  # 每个block 256 个元素
IQ2_XS_BLOCK_BYTES = 74  # 每个block 74 字节 (d:2 + qs:64 + scales:8)


def _write_string(f: BinaryIO, s: str) -> None:
    """写入 GGUF 字符串。"""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _read_string(f: BinaryIO) -> str:
    """读取 GGUF 字符串。"""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _pad_to_alignment(f: BinaryIO, alignment: int) -> None:
    """填充到对齐边界。"""
    pos = f.tell()
    padded = (pos + alignment - 1) // alignment * alignment
    if padded > pos:
        f.write(b"\x00" * (padded - pos))


@dataclass
class GGUFTensorInfo:
    """张量信息。"""
    name: str
    dims: List[int]
    ggml_type: int
    offset: int  # 数据区偏移（相对于数据区起始）
    
    def n_elements(self) -> int:
        result = 1
        for d in self.dims:
            result *= d
        return result
    
    def n_blocks(self) -> int:
        """IQ2_XS block 数量。"""
        return (self.n_elements() + IQ2_XS_BLOCK_SIZE - 1) // IQ2_XS_BLOCK_SIZE
    
    def data_size(self) -> int:
        """张量数据大小（字节）。"""
        if self.ggml_type == GGML_TYPE_IQ2_XS:
            return self.n_blocks() * IQ2_XS_BLOCK_BYTES
        elif self.ggml_type == GGML_TYPE_F16:
            return self.n_elements() * 2
        else:
            raise ValueError(f"不支持的 GGML 类型: {self.ggml_type}")


class GGUFWriter:
    """GGUF 文件写入器（简化版，仅支持 IQ2_XS）。"""
    
    def __init__(self, alignment: int = GGUF_DEFAULT_ALIGNMENT):
        self.alignment = alignment
        self.kv_pairs: Dict[str, Tuple[int, bytes]] = {}  # key -> (type, value)
        self.tensors: List[GGUFTensorInfo] = []
        self.tensor_data: Dict[str, bytes] = {}  # name -> data
        self._current_data_offset = 0
    
    def set_kv_string(self, key: str, value: str) -> None:
        """设置字符串 KV 对。"""
        encoded = value.encode("utf-8")
        data = struct.pack("<Q", len(encoded)) + encoded
        self.kv_pairs[key] = (GGUF_TYPE_STRING, data)
    
    def set_kv_uint32(self, key: str, value: int) -> None:
        """设置 uint32 KV 对。"""
        self.kv_pairs[key] = (GGUF_TYPE_UINT32, struct.pack("<I", value))
    
    def set_kv_int32(self, key: str, value: int) -> None:
        """设置 int32 KV 对。"""
        self.kv_pairs[key] = (GGUF_TYPE_INT32, struct.pack("<i", value))
    
    def add_iq2xs_tensor(self, name: str, d: np.ndarray, qs: np.ndarray, 
                         scales: np.ndarray, dims: List[int]) -> None:
        """添加 IQ2_XS 张量。
        
        参数:
            name: 张量名称 (如 "layers.0.ffn.experts.0.w1.weight")
            d: float16 数组, 形状 [n_blocks]
            qs: uint16 数组, 形状 [n_blocks, 32]
            scales: uint8 数组, 形状 [n_blocks, 8]
            dims: 原始形状 [out_dim, in_dim]
        """
        n_blocks = d.size
        
        # 打包数据: d (2B) + qs (64B) + scales (8B) = 74B per block
        # 向量化交织: 避免逐 block Python 循环
        d_flat = d.ravel().astype(np.float16)
        qs_flat = qs.reshape(n_blocks, 32).astype(np.uint16)
        scales_flat = scales.reshape(n_blocks, 8).astype(np.uint8)
        
        block_arr = np.zeros((n_blocks, IQ2_XS_BLOCK_BYTES), dtype=np.uint8)
        block_arr[:, 0:2] = d_flat.view(np.uint8).reshape(n_blocks, 2)
        block_arr[:, 2:66] = qs_flat.view(np.uint8).reshape(n_blocks, 64)
        block_arr[:, 66:74] = scales_flat
        data = block_arr.tobytes()
        
        tensor_info = GGUFTensorInfo(
            name=name,
            dims=dims,
            ggml_type=GGML_TYPE_IQ2_XS,
            offset=self._current_data_offset,
        )
        
        self.tensors.append(tensor_info)
        self.tensor_data[name] = bytes(data)
        self._current_data_offset += len(data)
        
        # 对齐到 32 字节
        padding = (self.alignment - (len(data) % self.alignment)) % self.alignment
        self._current_data_offset += padding
    
    def write(self, filepath: str) -> None:
        """写入 GGUF 文件。"""
        n_tensors = len(self.tensors)
        n_kv = len(self.kv_pairs)
        
        # 计算元数据大小
        meta_size = 4 + 4 + 8 + 8  # magic + version + n_tensors + n_kv
        
        # KV 对大小
        for key, (kv_type, value) in self.kv_pairs.items():
            meta_size += 8 + len(key.encode("utf-8"))  # key
            meta_size += 4  # type
            meta_size += len(value)  # value
        
        # 张量信息大小
        for tensor in self.tensors:
            meta_size += 8 + len(tensor.name.encode("utf-8"))  # name
            meta_size += 4  # n_dims
            meta_size += 8 * len(tensor.dims)  # dims
            meta_size += 4  # type
            meta_size += 8  # offset
        
        # 数据区偏移（对齐）
        data_offset = (meta_size + self.alignment - 1) // self.alignment * self.alignment
        
        with open(filepath, "wb") as f:
            # 文件头
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<q", n_tensors))
            f.write(struct.pack("<q", n_kv))
            
            # KV 对
            for key, (kv_type, value) in self.kv_pairs.items():
                _write_string(f, key)
                f.write(struct.pack("<i", kv_type))
                f.write(value)
            
            # 张量信息
            for tensor in self.tensors:
                _write_string(f, tensor.name)
                f.write(struct.pack("<I", len(tensor.dims)))
                for dim in tensor.dims:
                    f.write(struct.pack("<q", dim))
                f.write(struct.pack("<i", tensor.ggml_type))
                f.write(struct.pack("<Q", tensor.offset))
            
            # 填充到数据区
            _pad_to_alignment(f, self.alignment)
            
            # 张量数据
            for tensor in self.tensors:
                f.write(self.tensor_data[tensor.name])
                _pad_to_alignment(f, self.alignment)
        
        total_size = os.path.getsize(filepath)
        print(f"[GGUF] 写入完成: {filepath}")
        print(f"  张量数: {n_tensors}, KV 对数: {n_kv}")
        print(f"  文件大小: {total_size / 1024**3:.2f} GB")


class GGUFReader:
    """GGUF 文件读取器（简化版，仅支持 IQ2_XS）。"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        
        self.version: int = 0
        self.alignment: int = GGUF_DEFAULT_ALIGNMENT
        self.kv_pairs: Dict[str, Tuple[int, bytes]] = {}
        self.tensors: Dict[str, GGUFTensorInfo] = {}
        self._data_offset: int = 0
        
        self._open()
    
    def _open(self) -> None:
        """打开 GGUF 文件。"""
        self._file = open(self.filepath, "rb")
        
        # 读取文件头
        magic = self._file.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"无效的 GGUF 文件: magic={magic}")
        
        self.version = struct.unpack("<I", self._file.read(4))[0]
        n_tensors = struct.unpack("<q", self._file.read(8))[0]
        n_kv = struct.unpack("<q", self._file.read(8))[0]
        
        # 读取 KV 对
        for _ in range(n_kv):
            key = _read_string(self._file)
            kv_type = struct.unpack("<i", self._file.read(4))[0]
            
            if kv_type == GGUF_TYPE_STRING:
                length = struct.unpack("<Q", self._file.read(8))[0]
                value = self._file.read(length)
            elif kv_type == GGUF_TYPE_UINT32:
                value = self._file.read(4)
            elif kv_type == GGUF_TYPE_INT32:
                value = self._file.read(4)
            elif kv_type == GGUF_TYPE_ARRAY:
                arr_type = struct.unpack("<i", self._file.read(4))[0]
                arr_len = struct.unpack("<Q", self._file.read(8))[0]
                type_size = {GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4}.get(arr_type, 1)
                value = self._file.read(arr_len * type_size)
            else:
                raise ValueError(f"不支持的 KV 类型: {kv_type}")
            
            self.kv_pairs[key] = (kv_type, value)
        
        # 检查对齐设置
        if "general.alignment" in self.kv_pairs:
            self.alignment = struct.unpack("<I", self.kv_pairs["general.alignment"][1])[0]
        
        # 读取张量信息
        for _ in range(n_tensors):
            name = _read_string(self._file)
            n_dims = struct.unpack("<I", self._file.read(4))[0]
            dims = [struct.unpack("<q", self._file.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<i", self._file.read(4))[0]
            offset = struct.unpack("<Q", self._file.read(8))[0]
            
            self.tensors[name] = GGUFTensorInfo(
                name=name,
                dims=dims,
                ggml_type=ggml_type,
                offset=offset,
            )
        
        # 计算数据区偏移
        self._data_offset = (self._file.tell() + self.alignment - 1) // self.alignment * self.alignment
        
        # 创建 mmap
        self._file.seek(0, 2)
        file_size = self._file.tell()
        self._mmap = mmap.mmap(self._file.fileno(), file_size, access=mmap.ACCESS_READ)
        
        print(f"[GGUF] 打开文件: {self.filepath}")
        print(f"  版本: {self.version}, 张量数: {len(self.tensors)}, KV 对数: {len(self.kv_pairs)}")
    
    def get_kv_string(self, key: str) -> Optional[str]:
        """获取字符串 KV 值。"""
        if key not in self.kv_pairs:
            return None
        kv_type, data = self.kv_pairs[key]
        if kv_type != GGUF_TYPE_STRING:
            return None
        # data 已经是字符串内容（不包含长度前缀）
        return data.decode("utf-8")
    
    def get_kv_uint32(self, key: str) -> Optional[int]:
        """获取 uint32 KV 值。"""
        if key not in self.kv_pairs:
            return None
        kv_type, data = self.kv_pairs[key]
        if kv_type != GGUF_TYPE_UINT32:
            return None
        return struct.unpack("<I", data)[0]
    
    def get_iq2xs_tensor(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]]:
        """读取 IQ2_XS 张量。
        
        返回:
            (d, qs, scales, dims) 或 None
        """
        if name not in self.tensors:
            return None
        
        tensor = self.tensors[name]
        if tensor.ggml_type != GGML_TYPE_IQ2_XS:
            raise ValueError(f"张量 {name} 不是 IQ2_XS 类型")
        
        n_blocks = tensor.n_blocks()
        offset = self._data_offset + tensor.offset
        
        # 读取数据
        data = self._mmap[offset:offset + n_blocks * IQ2_XS_BLOCK_BYTES]
        
        # 向量化解析: 避免逐 block Python 循环
        raw_arr = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, IQ2_XS_BLOCK_BYTES)
        d = raw_arr[:, 0:2].copy().view(np.float16).reshape(n_blocks)
        qs = raw_arr[:, 2:66].copy().view(np.uint16).reshape(n_blocks, 32)
        scales = raw_arr[:, 66:74].copy()
        
        return d, qs, scales, tensor.dims
    
    def close(self) -> None:
        """关闭文件。"""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
