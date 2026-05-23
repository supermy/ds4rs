"""统一专家量化文件格式 — 支持所有专家打包到单个文件，mmap 读取。

文件格式设计 (v2 分离存储):
================================================================================
文件头 (固定 64 字节):
  magic:           4B  "IQ2X"
  version:         4B  uint32 (当前版本 2)
  n_layers:        4B  uint32
  n_experts:       4B  uint32 (每层专家数)
  n_weights:       4B  uint32 (每专家权重数，通常 3 = w1/w2/w3)
  index_offset:    8B  uint64 (索引表偏移)
  index_size:      8B  uint64 (索引表大小)
  data_offset:     8B  uint64 (数据区偏移)
  reserved:       20B  保留

索引表 (n_layers * n_experts * n_weights 条目，每条目 32 字节):
  layer_id:        4B  uint32
  expert_id:       4B  uint32
  weight_type:     4B  uint32 (0=w1, 1=w2, 2=w3)
  n_blocks:        4B  uint32
  out_dim:         4B  uint32
  in_dim:          4B  uint32
  offset:          8B  uint64 (数据在文件中的偏移)

数据区 (每个专家权重, v2 分离存储):
  d:               n_blocks * 2B  float16 (全局缩放因子, 连续存储)
  qs:              n_blocks * 32 * 2B  uint16 (grid 索引 + 符号索引, 连续存储)
  scales:          n_blocks * 8B  uint8 (4-bit 打包的子块缩放, 连续存储)
  总大小:          n_blocks * 74B (与 v1 交织存储相同)

v1 交织存储 (兼容):
  每个 block: d(2B) + qs(64B) + scales(8B) = 74B
================================================================================

优势：
- 单文件便于管理和传输
- mmap 读取，OS 自动管理页面缓存
- 索引表支持 O(1) 查找任意专家
- v2 分离存储: 读取时零拷贝，无需 .copy() 消除列切片非连续问题
- 顺序存储优化预读性能
"""
import os
import struct
import mmap
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

MAGIC = b"IQ2X"
VERSION = 2  # v2: 分离存储 d/qs/scales（消除读取时的 .copy()）
HEADER_SIZE = 64
INDEX_ENTRY_SIZE = 32

@dataclass
class ExpertIndex:
    """专家索引条目"""
    layer_id: int
    expert_id: int
    weight_type: int  # 0=w1, 1=w2, 2=w3
    n_blocks: int
    out_dim: int
    in_dim: int
    offset: int

class IQ2XSArchiveWriter:
    """IQ2_XS 归档写入器 — 将所有专家打包到单个文件"""
    
    def __init__(self, filepath: str, n_layers: int, n_experts: int, n_weights: int = 3):
        """
        参数:
            filepath: 输出文件路径
            n_layers: 层数
            n_experts: 每层专家数
            n_weights: 每专家权重数（默认 3 = w1/w2/w3）
        """
        self.filepath = filepath
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_weights = n_weights
        
        # 索引表
        self.index: List[ExpertIndex] = []
        
        # 数据缓冲
        self.data_buffers: List[bytes] = []
        self.current_offset = 0
        
    def add_expert(
        self,
        layer_id: int,
        expert_id: int,
        weight_type: int,
        d: np.ndarray,
        qs: np.ndarray,
        scales: np.ndarray,
        out_dim: int,
        in_dim: int,
    ) -> None:
        """
        添加一个专家权重到归档。
        
        参数:
            layer_id: 层 ID
            expert_id: 专家 ID
            weight_type: 权重类型 (0=w1, 1=w2, 2=w3)
            d: float16 数组，形状 [n_blocks]
            qs: uint16 数组，形状 [n_blocks, 32]
            scales: uint8 数组，形状 [n_blocks, 8]
            out_dim: 输出维度
            in_dim: 输入维度
        """
        n_blocks = d.shape[0]

        # v2 分离存储: d + qs + scales 各自连续存储
        # 优势: 读取时 np.frombuffer 可直接创建连续视图，无需 .copy()
        # 总大小与 v1 交织存储相同: n_blocks * (2 + 64 + 8) = n_blocks * 74B
        d_data = d.astype(np.float16).ravel().tobytes()
        qs_data = qs.astype(np.uint16).reshape(n_blocks, 32).tobytes()
        scales_data = scales.astype(np.uint8).reshape(n_blocks, 8).tobytes()
        data = d_data + qs_data + scales_data
        
        # 添加索引
        self.index.append(ExpertIndex(
            layer_id=layer_id,
            expert_id=expert_id,
            weight_type=weight_type,
            n_blocks=n_blocks,
            out_dim=out_dim,
            in_dim=in_dim,
            offset=self.current_offset,
        ))
        
        self.data_buffers.append(data)
        self.current_offset += len(data)
    
    def write(self) -> None:
        """写入归档文件"""
        # 计算偏移
        n_entries = len(self.index)
        index_offset = HEADER_SIZE
        index_size = n_entries * INDEX_ENTRY_SIZE
        data_offset = index_offset + index_size
        
        # 写入文件
        with open(self.filepath, 'wb') as f:
            # 文件头
            f.write(MAGIC)
            f.write(struct.pack('<I', VERSION))
            f.write(struct.pack('<I', self.n_layers))
            f.write(struct.pack('<I', self.n_experts))
            f.write(struct.pack('<I', self.n_weights))
            f.write(struct.pack('<Q', index_offset))
            f.write(struct.pack('<Q', index_size))
            f.write(struct.pack('<Q', data_offset))
            f.write(b'\x00' * 20)  # reserved (64 - 44 = 20)
            
            # 索引表
            for entry in self.index:
                f.write(struct.pack('<III', entry.layer_id, entry.expert_id, entry.weight_type))
                f.write(struct.pack('<III', entry.n_blocks, entry.out_dim, entry.in_dim))
                f.write(struct.pack('<Q', data_offset + entry.offset))
            
            # 数据区
            for data in self.data_buffers:
                f.write(data)
        
        total_size = os.path.getsize(self.filepath)
        print(f"[IQ2XSArchive] 写入完成: {self.filepath}")
        print(f"  层数: {self.n_layers}, 专家数: {self.n_experts}, 权重数: {self.n_weights}")
        print(f"  条目数: {n_entries}, 文件大小: {total_size / 1024**3:.2f} GB")


class IQ2XSArchiveReader:
    """IQ2_XS 归档读取器 — mmap 读取，支持随机访问"""
    
    def __init__(self, filepath: str):
        """
        参数:
            filepath: 归档文件路径
        """
        self.filepath = filepath
        self._file = None
        self._mmap = None
        self._index: Dict[Tuple[int, int, int], ExpertIndex] = {}
        self._version = VERSION  # 归档版本

        self._open()
    
    def _open(self):
        """打开归档文件并构建索引"""
        self._file = open(self.filepath, 'rb')
        
        # 读取文件头
        header = self._file.read(HEADER_SIZE)
        magic = header[0:4]
        if magic != MAGIC:
            raise ValueError(f"无效的 IQ2XS 归档文件: magic={magic}")
        
        version, = struct.unpack('<I', header[4:8])
        if version not in (1, 2):
            raise ValueError(f"不支持的版本: {version}")
        self._version = version
        
        self.n_layers, = struct.unpack('<I', header[8:12])
        self.n_experts, = struct.unpack('<I', header[12:16])
        self.n_weights, = struct.unpack('<I', header[16:20])
        index_offset, = struct.unpack('<Q', header[20:28])
        index_size, = struct.unpack('<Q', header[28:36])
        self._data_offset, = struct.unpack('<Q', header[36:44])
        
        # mmap 映射
        self._file.seek(0, 2)
        file_size = self._file.tell()
        self._mmap = mmap.mmap(self._file.fileno(), file_size, access=mmap.ACCESS_READ)
        
        # 读取索引表
        n_entries = index_size // INDEX_ENTRY_SIZE
        for i in range(n_entries):
            entry_offset = index_offset + i * INDEX_ENTRY_SIZE
            entry_data = self._mmap[entry_offset:entry_offset + INDEX_ENTRY_SIZE]
            
            layer_id, expert_id, weight_type = struct.unpack('<III', entry_data[0:12])
            n_blocks, out_dim, in_dim = struct.unpack('<III', entry_data[12:24])
            data_offset, = struct.unpack('<Q', entry_data[24:32])
            
            # 跳过重复键（保留第一个）
            key = (layer_id, expert_id, weight_type)
            if key in self._index:
                continue
            
            # 注意：写入器存储的是 data_offset + entry.offset（绝对偏移）
            # 所以 data_offset 已经是绝对偏移，直接使用
            entry = ExpertIndex(
                layer_id=layer_id,
                expert_id=expert_id,
                weight_type=weight_type,
                n_blocks=n_blocks,
                out_dim=out_dim,
                in_dim=in_dim,
                offset=data_offset,
            )
            
            self._index[key] = entry
        
        print(f"[IQ2XSArchive] 打开归档: {self.filepath}")
        print(f"  层数: {self.n_layers}, 专家数: {self.n_experts}, 条目数: {n_entries}")
    
    def get_expert(
        self,
        layer_id: int,
        expert_id: int,
        weight_type: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]]:
        """
        读取专家权重的 IQ2_XS 数据。
        
        参数:
            layer_id: 层 ID
            expert_id: 专家 ID
            weight_type: 权重类型 (0=w1, 1=w2, 2=w3)
        
        返回:
            (d, qs, scales, shape) 或 None
        """
        key = (layer_id, expert_id, weight_type)
        entry = self._index.get(key)
        if entry is None:
            return None
        
        # 从 mmap 读取数据
        offset = entry.offset
        n_blocks = entry.n_blocks

        if self._version == 2:
            # v2 分离存储: d + qs + scales 各自连续
            # np.frombuffer(mmap, offset=...) 零拷贝，直接映射 mmap 页面
            d = np.frombuffer(self._mmap, dtype=np.float16,
                              count=n_blocks, offset=offset)
            d_bytes = n_blocks * 2
            qs = np.frombuffer(self._mmap, dtype=np.uint16,
                               count=n_blocks * 32,
                               offset=offset + d_bytes).reshape(n_blocks, 32)
            qs_bytes = n_blocks * 64
            scales = np.frombuffer(self._mmap, dtype=np.uint8,
                                   count=n_blocks * 8,
                                   offset=offset + d_bytes + qs_bytes).reshape(n_blocks, 8)
        else:
            # v1 交织存储: d(2)+qs(64)+scales(8)=74B per block
            # 列切片不连续，必须 .copy() 才能 .view() 为目标 dtype
            BLOCK_BYTES = 74
            raw_arr = np.frombuffer(self._mmap, dtype=np.uint8,
                                    count=n_blocks * BLOCK_BYTES,
                                    offset=offset).reshape(n_blocks, BLOCK_BYTES)
            d = raw_arr[:, 0:2].copy().view(np.float16).reshape(n_blocks)
            qs = raw_arr[:, 2:66].copy().view(np.uint16).reshape(n_blocks, 32)
            scales = raw_arr[:, 66:74].copy()

        # reshape 为 IQ2_XS GEMM kernel 期望的形状
        # kernel 期望: d[N, n_blocks_per_row], qs[N, n_blocks_per_row, 32], scales[N, n_blocks_per_row, 8]
        # 其中 N = out_dim, n_blocks_per_row = in_dim / 256
        N = entry.out_dim
        K = entry.in_dim
        n_blocks_per_row = K // 256  # QK_K = 256

        # d: [n_blocks] → [N, n_blocks_per_row]
        d = d.reshape(N, n_blocks_per_row)

        # qs: [n_blocks, 32] → [N, n_blocks_per_row, 32]
        qs = qs.reshape(N, n_blocks_per_row, 32)

        # scales: [n_blocks * 8] → [N, n_blocks_per_row, 8]
        scales = scales.reshape(N, n_blocks_per_row, 8)

        return d, qs, scales, (entry.out_dim, entry.in_dim)
    
    def get_expert_weight_names(self, layer_id: int, expert_id: int) -> List[str]:
        """获取专家的所有权重名称"""
        names = []
        for wt in range(self.n_weights):
            if (layer_id, expert_id, wt) in self._index:
                names.append(['w1', 'w2', 'w3'][wt])
        return names
    
    def close(self):
        """关闭归档文件"""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
