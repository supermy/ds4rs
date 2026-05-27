"""混合量化脚本：IQ2_XXS (gate, up) + Q2_K (down) - CUDA 量化优化版。

专家 FFN 权重使用混合量化策略：
  - gate_proj (w1): IQ2_XS — 2.0625 bpw（使用 CUDA kernel）
  - up_proj (w3):   IQ2_XS — 2.0625 bpw（使用 CUDA kernel）
  - down_proj (w2): Q2_K    — 2.5625 bpw（CPU 量化）

优化策略（参考 prequant_iq2xs.py）：
  1. CUDA kernel 直接在 GPU 上量化 IQ2_XS（~50-100 e/s vs C CPU ~0.3 e/s）
  2. GPU 批量解码 FP4 + 应用 scale
  3. 双缓冲流水线（CPU 加载与 GPU 量化并行）
  4. CUDA stream 异步操作
  5. 向量化解析输出

用法：
  python inference/prequant_mixed_iq2xxs_q2k.py --ckpt-path /models --imatrix /models/imatrix.dat --output /workspace/gguf/experts_mixed.gguf
"""
import os
import sys
import time
import argparse
import ctypes
import struct
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QK_K = 256

FP4_TABLE_NP = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)

FP4_TABLE_F32 = torch.tensor(FP4_TABLE_NP, dtype=torch.float32)


def load_imatrix_dat(path: str) -> Dict[str, np.ndarray]:
    """加载 llama.cpp 格式的 imatrix.dat 文件。"""
    imatrix = {}
    with open(path, 'rb') as f:
        n_entries = struct.unpack('<I', f.read(4))[0]
        print(f"[imatrix] 加载 {n_entries} 个条目")
        for i in range(n_entries):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = []
            for _ in range(n_dims):
                dim = struct.unpack('<I', f.read(4))[0]
                dims.append(dim)
            n_elements = 1
            for d in dims:
                n_elements *= d
            data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
            imatrix[name] = data
            if (i + 1) % 20 == 0:
                print(f"\r  [imatrix] 已加载 {i+1}/{n_entries} 个条目", end='', flush=True)
        print()
    return imatrix


def _parse_iq2xs_batch(output_cpu: np.ndarray, n_blocks: int,
                        blocks_per_expert: int) -> list:
    """向量化解析批量 block_iq2xs 输出。"""
    n_experts = n_blocks // blocks_per_expert
    raw = output_cpu.reshape(n_blocks, 74)
    d_all = raw[:, :2].copy().view(np.float16).reshape(n_blocks)
    qs_all = raw[:, 2:66].copy().view(np.uint16).reshape(n_blocks, 32)
    scales_all = raw[:, 66:74].copy().reshape(n_blocks, 8)
    results = []
    for e in range(n_experts):
        start = e * blocks_per_expert
        end = start + blocks_per_expert
        results.append((
            d_all[start:end].copy(),
            qs_all[start:end].copy(),
            scales_all[start:end].copy(),
        ))
    return results


def quantize_q2k_gpu(f32_gpu: torch.Tensor, n_blocks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU 端 Q2_K 量化，数据全程不离开 GPU，只在最后 D2H 一次。
    
    Q2_K 格式: block_q2_K = { d(fp16), dmin(fp16), scales[16](uint8), qs[64](uint8) }
    每 block 256 元素，16 个子块各 16 元素，2-bit 量化。
    """
    blocks = f32_gpu.reshape(n_blocks, QK_K)  # (n_blocks, 256)
    
    # super-block: d, dmin
    block_max = blocks.max(dim=1).values   # (n_blocks,)
    block_min = blocks.min(dim=1).values   # (n_blocks,)
    d = ((block_max - block_min) / 3.0).half()  # fp16
    dmin = block_min.half()
    
    # 子块: (n_blocks, 16, 16)
    sub_blocks = blocks.reshape(n_blocks, 16, 16)
    sub_max = sub_blocks.max(dim=2).values   # (n_blocks, 16)
    sub_min = sub_blocks.min(dim=2).values   # (n_blocks, 16)
    
    # 子块 scale: clip(scale / d * 8 + 8, 0, 15)
    d_safe = d.float().abs().clamp(min=1e-8)  # (n_blocks,)
    sub_scale = (sub_max - sub_min) / 3.0
    scales = (sub_scale / d_safe.unsqueeze(1) * 8 + 8).clamp(0, 15).to(torch.uint8)  # (n_blocks, 16)
    
    # 2-bit 量化: (val - sub_min) / (sub_max - sub_min) * 3, clip [0, 3]
    sub_range = (sub_max - sub_min).clamp(min=1e-8)  # (n_blocks, 16)
    quant_2bit = ((sub_blocks - sub_min.unsqueeze(2)) / sub_range.unsqueeze(2) * 3).clamp(0, 3).to(torch.uint8)
    
    # 打包 2-bit: (n_blocks, 16, 16) -> (n_blocks, 16, 4, 4)
    quant_2bit = quant_2bit.reshape(n_blocks, 16, 4, 4)
    # bit packing: byte = q[0] | (q[1]<<2) | (q[2]<<4) | (q[3]<<6)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=f32_gpu.device).reshape(1, 1, 4, 1)
    qs = (quant_2bit << shifts).sum(dim=3).to(torch.uint8)  # (n_blocks, 16, 4)
    qs = qs.reshape(n_blocks, 64)
    
    # 一次性 D2H
    return d.cpu().numpy(), dmin.cpu().numpy(), scales.cpu().numpy(), qs.cpu().numpy()


class MixedQuantWriter:
    """混合量化 GGUF 写入器 — GGUFWriter 已内置增量写入，无需额外分片。"""
    
    def __init__(self, output_path: str, n_layers: int, n_experts: int):
        self.output_path = output_path
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.total_iq2xxs_bytes = 0
        self.total_q2k_bytes = 0
        self.total_iq2xxs_tensors = 0
        self.total_q2k_tensors = 0
        
        from gguf_iq2xs import GGUFWriter as BaseGGUFWriter
        self._gguf_writer = BaseGGUFWriter()
        self._gguf_writer.set_kv_string("general.name", "deepseek-v4-experts-mixed")
        self._gguf_writer.set_kv_string("general.quantization", "iq2xs_q2k")
        self._gguf_writer.set_kv_uint32("general.n_layers", self.n_layers)
        self._gguf_writer.set_kv_uint32("general.n_experts", self.n_experts)
    
    def add_iq2xxs_tensor(self, name: str, d: np.ndarray, qs: np.ndarray, scales: np.ndarray, dims: List[int]):
        n_blocks = d.size
        self._gguf_writer.add_iq2xs_tensor(name, d, qs, scales, dims)
        self.total_iq2xxs_bytes += n_blocks * 74
        self.total_iq2xxs_tensors += 1
    
    def add_q2k_tensor(self, name: str, d: np.ndarray, dmin: np.ndarray, scales: np.ndarray, qs: np.ndarray, dims: List[int]):
        n_blocks = d.size
        self._gguf_writer.add_q2k_tensor(name, d, dmin, scales, qs, dims)
        self.total_q2k_bytes += n_blocks * 84
        self.total_q2k_tensors += 1
    
    def shard_done(self):
        """一个 shard 处理完成（无需额外操作，GGUFWriter 已增量写入）。"""
        pass
    
    def write(self):
        """最终写入。"""
        self._gguf_writer.write(self.output_path)
        
        print(f"\n[统计]")
        print(f"  IQ2_XS 张量: {self.total_iq2xxs_tensors}")
        print(f"  Q2_K 张量: {self.total_q2k_tensors}")
        print(f"  IQ2_XS 大小: {self.total_iq2xxs_bytes / 1024**3:.2f} GB")
        print(f"  Q2_K 大小: {self.total_q2k_bytes / 1024**3:.2f} GB")
        print(f"  总大小: {(self.total_iq2xxs_bytes + self.total_q2k_bytes) / 1024**3:.2f} GB")


def prequant_mixed(
    ckpt_path: str,
    output_path: str,
    imatrix_path: Optional[str] = None,
    batch_size: int = 64,
):
    """预量化专家权重为混合格式：IQ2_XS (gate/up) + Q2_K (down)。
    
    IQ2_XS 量化使用 CUDA kernel（参考 prequant_iq2xs.py 的 GPU 路径），
    Q2_K 量化使用 CPU（权重较小，CPU 量化可接受）。
    """
    print("=" * 70)
    print("混合量化：IQ2_XS (gate, up) + Q2_K (down) - CUDA 量化版")
    print("=" * 70)
    print(f"输入: {ckpt_path}")
    print(f"输出: {output_path}")
    print(f"imatrix: {imatrix_path}")
    print(f"batch_size: {batch_size}")
    
    # 加载 imatrix
    imatrix_data: Dict[str, np.ndarray] = {}
    if imatrix_path and os.path.exists(imatrix_path):
        if imatrix_path.endswith('.dat'):
            imatrix_data = load_imatrix_dat(imatrix_path)
        else:
            data = np.load(imatrix_path)
            imatrix_data = {k: data[k] for k in data.files}
        print(f"加载 imatrix: {len(imatrix_data)} 个张量")
    
    # 扫描 shard
    shard_files = sorted(Path(ckpt_path).glob("model-*.safetensors"))
    print(f"找到 {len(shard_files)} 个 shard")
    
    # 收集专家信息
    experts: Dict[str, List] = defaultdict(list)
    all_layer_ids = set()
    all_expert_ids = set()
    
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt") as sf:
            for name in sf.keys():
                if '.ffn.experts.' in name and '.weight' in name:
                    parts = name.split('.')
                    layer_id = int(parts[1])
                    expert_id = int(parts[4])
                    weight_name = parts[5]
                    shape = sf.get_tensor(name).shape
                    experts[str(shard_path)].append((layer_id, expert_id, weight_name, name, shape))
                    all_layer_ids.add(layer_id)
                    all_expert_ids.add(expert_id)
    
    n_layers = max(all_layer_ids) + 1
    n_experts = max(all_expert_ids) + 1
    total_experts = sum(len(v) for v in experts.values())
    
    print(f"n_layers={n_layers}, n_experts={n_experts}")
    print(f"总专家权重数: {total_experts}")
    
    writer = MixedQuantWriter(output_path, n_layers, n_experts)
    
    # 检测 CUDA 量化是否可用
    cuda_quant_available = False
    try:
        from iq2xs_cuda_quant import _find_lib
        lib = _find_lib()
        if lib is not None:
            ret = lib.iq2xs_quantize_init()
            if ret == 0:
                cuda_quant_available = True
                print("CUDA IQ2_XS 量化: 可用")
            else:
                print("CUDA IQ2_XS 量化: 初始化失败")
        else:
            print("CUDA IQ2_XS 量化: libiq2xs_quantize.so 未找到")
    except Exception as e:
        print(f"CUDA IQ2_XS 量化: 不可用 ({e})")
    
    start_time = time.time()
    quantized = 0
    last_progress_time = start_time
    progress_interval = 2.0
    
    # ========================================================================
    # CUDA 量化路径（IQ2_XS 用 CUDA，Q2_K 用 CPU）
    # ========================================================================
    if cuda_quant_available:
        from iq2xs_cuda_quant import _find_lib
        lib = _find_lib()
        
        stream_compute = torch.cuda.Stream()
        stream_xfer = torch.cuda.Stream()
        compute_done_event = torch.cuda.Event(enable_timing=False)
        cpu_executor = ThreadPoolExecutor(max_workers=1)
        
        class BatchBuffer:
            def __init__(self):
                self.clear()
            def clear(self):
                self.f32_gpu = None
                self.output_gpu = None
                self.meta_list = []
                self.n_experts = 0
                self.blocks_per_expert = 0
                self.total_blocks = 0
                self.ready = False
                self.weight_type = None  # 'iq2xs' or 'q2k'
                self.result_d = None
                self.result_qs = None
                self.result_scales = None
                self.result_dmin = None
        
        buf = [BatchBuffer(), BatchBuffer()]
        
        def load_batch_cpu(tensor_cache, batch, weight_type):
            packed_parts = [tensor_cache[name] for _, _, _, name, _ in batch]
            scale_parts = []
            for _, _, _, name, _ in batch:
                scale_name = name.replace('.weight', '.scale')
                if scale_name in tensor_cache:
                    scale_parts.append(tensor_cache[scale_name])
                else:
                    scale_parts.append(None)
            return torch.stack(packed_parts, dim=0), scale_parts, weight_type
        
        def prepare_batch(buf_idx, packed_cpu, scale_parts, expert_list, shape_key, weight_type):
            b = buf[buf_idx]
            b.clear()
            
            out_dim, packed_in_dim = shape_key
            in_dim = packed_in_dim * 2
            blocks_per_expert = out_dim * in_dim // QK_K
            n = len(expert_list)
            
            with torch.cuda.stream(stream_xfer):
                packed_gpu = packed_cpu.cuda(non_blocking=True)
                del packed_cpu
                
                # GPU 批量解码 FP4（原地操作减少峰值显存）
                arr = packed_gpu.view(torch.uint8).long()
                table = FP4_TABLE_F32.to(arr.device)
                # 合并 lo/hi 解码为一步，减少中间张量
                lo_val = table[arr & 0x0F]       # (N, out_dim, packed_in_dim)
                hi_val = table[(arr >> 4) & 0x0F]
                # 交织 lo/hi: [lo0, hi0, lo1, hi1, ...] -> reshape
                decoded = torch.stack([lo_val, hi_val], dim=-1)  # (..., packed_in_dim, 2)
                del lo_val, hi_val, packed_gpu
                decoded = decoded.reshape(n * out_dim, in_dim).contiguous().float()
                
                # 应用 FP4 scale
                if scale_parts[0] is not None:
                    scale_scales = torch.stack(scale_parts, dim=0)
                    scale_gpu = scale_scales.cuda(non_blocking=True).view(torch.float8_e8m0fnu).float()
                    scale_expanded = scale_gpu.reshape(n * out_dim, -1).repeat_interleave(32, dim=1)
                    decoded = decoded * scale_expanded
                    del scale_scales, scale_gpu, scale_expanded
                
                b.f32_gpu = decoded
                b.total_blocks = n * blocks_per_expert
                b.weight_type = weight_type
                
                # IQ2_XS 需要预分配输出缓冲
                if weight_type == 'iq2xs':
                    b.output_gpu = torch.empty(b.total_blocks * 74, dtype=torch.uint8, device='cuda')
            
            b.meta_list = [
                (lid, eid, wn, out_dim, in_dim)
                for lid, eid, wn, _, _ in expert_list
            ]
            b.n_experts = n
            b.blocks_per_expert = blocks_per_expert
            b.ready = False
        
        def launch_quantize(buf_idx):
            b = buf[buf_idx]
            with torch.cuda.stream(stream_compute):
                stream_compute.wait_stream(stream_xfer)
                
                if b.weight_type == 'iq2xs':
                    # CUDA kernel 直接在 GPU 上量化
                    nrow = b.f32_gpu.shape[0]  # n * out_dim
                    n_per_row = b.f32_gpu.shape[1]  # in_dim
                    src_ptr = ctypes.c_void_p(b.f32_gpu.data_ptr())
                    dst_ptr = ctypes.c_void_p(b.output_gpu.data_ptr())
                    lib.iq2xs_quantize_ffi(src_ptr, dst_ptr, nrow, n_per_row, ctypes.c_void_p(0))
                else:
                    # Q2_K: GPU 量化（数据不离开 GPU）
                    d, dmin, scales, qs = quantize_q2k_gpu(b.f32_gpu, b.total_blocks)
                    b.result_d = d
                    b.result_dmin = dmin
                    b.result_scales = scales
                    b.result_qs = qs
                
                compute_done_event.record(stream_compute)
            b.ready = True
        
        def consume_batch(buf_idx):
            b = buf[buf_idx]
            if not b.ready:
                return 0
            
            compute_done_event.synchronize()
            count = 0
            n_blocks_per_expert = b.blocks_per_expert
            
            if b.weight_type == 'iq2xs':
                # 解析 CUDA 量化输出
                output_cpu = b.output_gpu.cpu().numpy()
                results = _parse_iq2xs_batch(output_cpu, b.total_blocks, n_blocks_per_expert)
                
                for i, (lid, eid, wn, od, idim) in enumerate(b.meta_list):
                    d_np, qs_np, scales_np = results[i]
                    writer.add_iq2xxs_tensor(
                        f"model.{lid}.ffn.experts.{eid}.{wn}.weight",
                        d_np, qs_np, scales_np, [od, idim]
                    )
                    count += 1
            else:
                # 解析 Q2_K 向量化量化输出（按 n_blocks 索引）
                for i, (lid, eid, wn, od, idim) in enumerate(b.meta_list):
                    start_block = i * n_blocks_per_expert
                    end_block = (i + 1) * n_blocks_per_expert
                    
                    d_np = b.result_d[start_block:end_block]
                    dmin_np = b.result_dmin[start_block:end_block]
                    scales_np = b.result_scales[start_block:end_block]  # (n_blocks_per_expert, 16)
                    qs_np = b.result_qs[start_block:end_block]  # (n_blocks_per_expert, 64)
                    
                    writer.add_q2k_tensor(
                        f"model.{lid}.ffn.experts.{eid}.{wn}.weight",
                        d_np, dmin_np, scales_np, qs_np, [od, idim]
                    )
                    count += 1
            
            b.clear()
            return count
        
        # 逐 shard 处理
        for shard_idx, (shard_path, shard_experts) in enumerate(experts.items()):
            print(f"\n[Shard {shard_idx+1}/{len(experts)}] {os.path.basename(shard_path)}")
            
            with safe_open(shard_path, framework="pt") as sf:
                # 预加载
                t_preload = time.time()
                tensor_cache = {}
                for layer_id, expert_id, weight_name, tensor_name, shape in shard_experts:
                    tensor_cache[tensor_name] = sf.get_tensor(tensor_name)
                    scale_name = tensor_name.replace('.weight', '.scale')
                    if scale_name not in tensor_cache and scale_name in sf.keys():
                        tensor_cache[scale_name] = sf.get_tensor(scale_name)
                preload_ms = (time.time() - t_preload) * 1000
                print(f"  预加载 {len(tensor_cache)} 个张量: {preload_ms:.0f}ms")
                
                # 按形状+量化类型分组
                shape_type_groups = defaultdict(list)
                for layer_id, expert_id, weight_name, tensor_name, shape in shard_experts:
                    weight_type = 'iq2xs' if weight_name in ('w1', 'w3') else 'q2k'
                    shape_type_groups[(shape, weight_type)].append(
                        (layer_id, expert_id, weight_name, tensor_name, shape))
                
                all_batches = []
                # 动态 batch: 限制每批 FP4 解码后 f32 不超过 vram_budget
                vram_budget = 2 * 1024**3  # 2GB per buffer (双缓冲共 4GB, 留空间给中间张量和解码)
                for (shape_key, weight_type), group in shape_type_groups.items():
                    out_dim, packed_in_dim = shape_key
                    in_dim = packed_in_dim * 2
                    # 每个专家解码后 f32 大小
                    bytes_per_expert = out_dim * in_dim * 4
                    adaptive_batch = max(1, min(batch_size, vram_budget // bytes_per_expert))
                    for i in range(0, len(group), adaptive_batch):
                        all_batches.append((shape_key, weight_type, group[i:i+adaptive_batch]))
                
                if not all_batches:
                    continue
                
                # 启动第一个 batch
                shape_key, weight_type, batch = all_batches[0]
                packed_cpu, scale_parts, _ = load_batch_cpu(tensor_cache, batch, weight_type)
                prepare_batch(0, packed_cpu, scale_parts, batch, shape_key, weight_type)
                del packed_cpu
                launch_quantize(0)
                buf_idx = 1
                
                # 预提交下一批
                next_cpu_future = None
                if len(all_batches) > 1:
                    next_shape, next_type, next_batch = all_batches[1]
                    next_cpu_future = cpu_executor.submit(load_batch_cpu, tensor_cache, next_batch, next_type)
                
                # 双缓冲流水线
                for batch_i in range(1, len(all_batches)):
                    shape_key, weight_type, batch = all_batches[batch_i]
                    
                    if next_cpu_future is not None:
                        packed_cpu, scale_parts, _ = next_cpu_future.result()
                    else:
                        packed_cpu, scale_parts, _ = load_batch_cpu(tensor_cache, batch, weight_type)
                    
                    prepare_batch(buf_idx, packed_cpu, scale_parts, batch, shape_key, weight_type)
                    del packed_cpu
                    
                    prev_idx = 1 - buf_idx
                    count = consume_batch(prev_idx)
                    quantized += count
                    
                    launch_quantize(buf_idx)
                    
                    next_batch_i = batch_i + 1
                    if next_batch_i < len(all_batches):
                        next_shape, next_type, next_batch = all_batches[next_batch_i]
                        next_cpu_future = cpu_executor.submit(load_batch_cpu, tensor_cache, next_batch, next_type)
                    else:
                        next_cpu_future = None
                    
                    buf_idx = 1 - buf_idx
                    
                    now = time.time()
                    if now - last_progress_time >= progress_interval:
                        elapsed = now - start_time
                        rate = quantized / elapsed
                        vram_used = torch.cuda.memory_allocated() / 1024**3
                        print(f"\r  [{quantized}/{total_experts}] {quantized/total_experts*100:.1f}% | {rate:.1f} e/s | VRAM {vram_used:.1f}GB", end='', flush=True)
                        last_progress_time = now
                
                last_idx = 1 - buf_idx
                count = consume_batch(last_idx)
                quantized += count
            
            writer.shard_done()
            torch.cuda.empty_cache()
    
    else:
        # ====================================================================
        # 回退路径：CPU 量化
        # ====================================================================
        print("\nCUDA 量化不可用，使用 CPU 路径")
        
        for shard_idx, (shard_path, shard_experts) in enumerate(experts.items()):
            print(f"\n[Shard {shard_idx+1}/{len(experts)}] {os.path.basename(shard_path)}")
            
            with safe_open(shard_path, framework="pt") as sf:
                for layer_id, expert_id, weight_name, tensor_name, shape in shard_experts:
                    tensor = sf.get_tensor(tensor_name)
                    out_dim, packed_in_dim = tensor.shape
                    in_dim = packed_in_dim * 2
                    
                    raw = tensor.numpy().tobytes()
                    f32 = np.frombuffer(raw, dtype=np.uint8)
                    lo = f32 & 0x0F
                    hi = (f32 >> 4) & 0x0F
                    f32 = np.stack([FP4_TABLE_NP[lo], FP4_TABLE_NP[hi]], axis=-1).ravel().astype(np.float32)
                    
                    scale_name = tensor_name.replace('.weight', '.scale')
                    if scale_name in sf.keys():
                        scale_tensor = sf.get_tensor(scale_name)
                        scale_f32 = scale_tensor.view(torch.float8_e8m0fnu).float().numpy()
                        scale_expanded = np.repeat(scale_f32, 32, axis=1).reshape(-1)
                        f32 = f32 * scale_expanded[:f32.size]
                    
                    n_elements = f32.size
                    n_blocks = (n_elements + QK_K - 1) // QK_K
                    if n_elements % QK_K != 0:
                        padded = np.zeros(n_blocks * QK_K, dtype=np.float32)
                        padded[:n_elements] = f32
                        f32 = padded
                    
                    if weight_name in ('w1', 'w3'):
                        from iq2xs_c_wrapper import quantize_iq2_xs_gpu_via_cpu
                        f32_gpu = torch.from_numpy(f32.reshape(1, -1)).cuda()
                        d, qs, scales = quantize_iq2_xs_gpu_via_cpu(f32_gpu)
                        writer.add_iq2xxs_tensor(
                            tensor_name,
                            d.cpu().numpy(), qs.cpu().numpy(), scales.cpu().numpy(),
                            [out_dim, in_dim]
                        )
                    else:
                        d, dmin, scales_arr, qs_arr = quantize_q2k_gpu(
                            torch.from_numpy(f32).cuda(), n_blocks)
                        writer.add_q2k_tensor(
                            tensor_name, d, dmin, scales_arr, qs_arr,
                            [out_dim, in_dim]
                        )
                    
                    quantized += 1
                    if quantized % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = quantized / elapsed
                        print(f"\r  [{quantized}/{total_experts}] {rate:.1f} e/s", end='', flush=True)
    
    print()
    writer.write()
    
    total_time = time.time() - start_time
    print(f"\n完成！总耗时: {total_time:.1f}s, 平均速率: {total_experts/total_time:.1f} e/s")


def main():
    parser = argparse.ArgumentParser(description="混合量化：IQ2_XS + Q2_K")
    parser.add_argument("--ckpt-path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--output", type=str, default="", help="输出 GGUF 文件路径")
    parser.add_argument("--imatrix", type=str, default="", help="importance weights 文件路径")
    parser.add_argument("--batch-size", type=int, default=64, help="批量大小")
    args = parser.parse_args()
    
    output_path = args.output
    if not output_path:
        output_path = "/workspace/gguf/experts_mixed.gguf"
    
    prequant_mixed(
        args.ckpt_path,
        output_path,
        args.imatrix if args.imatrix else None,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
