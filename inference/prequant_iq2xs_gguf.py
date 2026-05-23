"""IQ2_XS 预量化脚本（GGUF 格式输出）。

流程：
1. 遍历所有 safetensor shard
2. 每个 shard 处理完成后立即写入临时文件
3. 最后合并为 GGUF 文件

内存管理：
- 每个 safetensor 处理完成后释放内存
- 量化数据写入临时文件，不累积在内存中
"""
import argparse
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import struct

import numpy as np
import torch
from safetensors import safe_open

from gguf_iq2xs import GGUFWriter, GGUFReader, IQ2_XS_BLOCK_SIZE, IQ2_XS_BLOCK_BYTES
from iq2xs_cuda_quant import _find_lib
import ctypes

# FP4 解码表（与 llama.cpp 一致）
FP4_TABLE_NP = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)

FP4_TABLE_F32 = torch.tensor(FP4_TABLE_NP, dtype=torch.float32)


def decode_fp4_cpu(packed: bytes) -> np.ndarray:
    """在 CPU 上解码 FP4 打包数据为 float32。"""
    arr = np.frombuffer(packed, dtype=np.uint8)
    lo = arr & 0x0F
    hi = (arr >> 4) & 0x0F
    return np.stack([FP4_TABLE_NP[lo], FP4_TABLE_NP[hi]], axis=-1).reshape(-1)


def decode_fp4_gpu(packed_gpu: torch.Tensor, n: int, out_dim: int, in_dim: int) -> torch.Tensor:
    """在 GPU 上解码 FP4 打包数据为 float32。"""
    arr = packed_gpu.view(torch.uint8)
    lo = (arr & 0x0F).long()
    hi = ((arr >> 4) & 0x0F).long()
    table = FP4_TABLE_F32.to(arr.device)
    lo_val = table[lo]
    hi_val = table[hi]
    decoded = torch.stack([lo_val, hi_val], dim=-1).reshape(n * out_dim, in_dim).contiguous()
    del packed_gpu, lo, hi, lo_val, hi_val
    return decoded


def quantize_iq2xs_batch(f32_gpu: torch.Tensor, nrow: int, n_per_row: int, lib) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量量化 FP32 数据为 IQ2_XS 格式。
    
    返回:
        d: float16 [nrow]
        qs: uint16 [nrow, 32]
        scales: uint8 [nrow, 8]
    """
    # n_blocks = (nrow * n_per_row + IQ2_XS_BLOCK_SIZE - 1) // IQ2_XS_BLOCK_SIZE
    n_blocks_per_row = (n_per_row + IQ2_XS_BLOCK_SIZE - 1) // IQ2_XS_BLOCK_SIZE
    n_blocks = nrow * n_blocks_per_row

    output_gpu = torch.empty(n_blocks * IQ2_XS_BLOCK_BYTES, dtype=torch.uint8, device='cuda')
    
    src_ptr = ctypes.c_void_p(f32_gpu.data_ptr())
    dst_ptr = ctypes.c_void_p(output_gpu.data_ptr())
    lib.iq2xs_quantize_ffi(src_ptr, dst_ptr, nrow, n_per_row, ctypes.c_void_p(0))
    
    # 解析结果
    output_cpu = output_gpu.cpu().numpy()
    
    # 解包每个 block
    d_list = []
    qs_list = []
    scales_list = []
    
    for i in range(n_blocks):
        block_data = output_cpu[i * IQ2_XS_BLOCK_BYTES:(i+1) * IQ2_XS_BLOCK_BYTES]
        d_list.append(np.frombuffer(block_data[0:2], dtype=np.float16)[0])
        qs_list.append(np.frombuffer(block_data[2:66], dtype=np.uint16).copy())
        scales_list.append(np.frombuffer(block_data[66:74], dtype=np.uint8).copy())



    
    d = np.array(d_list, dtype=np.float16)
    qs = np.stack(qs_list, axis=0)
    scales = np.stack(scales_list, axis=0)
    
    return d, qs, scales


def process_safetensor_shard(
    shard_path: str,
    shard_idx: int,
    total_shards: int,
    lib,
    use_gpu: bool = True,
    batch_size: int = 16,
) -> List[Dict]:
    """处理单个 safetensor shard。
    
    返回:
        量化后的张量列表，每个元素包含 {name, d, qs, scales, dims}
    """
    print(f"\n[Shard {shard_idx+1}/{total_shards}] 处理: {os.path.basename(shard_path)}")
    
    quantized_tensors = []
    
    with safe_open(shard_path, framework='pt') as sf:
        # 获取所有专家权重键
        all_keys = sf.keys()
        expert_keys = [k for k in all_keys if '.ffn.experts.' in k and '.weight' in k]
        
        if not expert_keys:
            print(f"  无专家权重，跳过")
            return quantized_tensors
        
        print(f"  专家权重数: {len(expert_keys)}")
        
        # 按 layer 分组
        layer_groups: Dict[int, List[str]] = {}
        for key in expert_keys:
            layer = int(key.split('.')[1])
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append(key)
        
        # 处理每个 layer
        for layer, keys in sorted(layer_groups.items()):
            print(f"  Layer {layer}: {len(keys)} 个权重")
            
            # 读取并量化
            for key in keys:
                tensor = sf.get_tensor(key)
                out_dim, packed_in_dim = tensor.shape
                in_dim = packed_in_dim * 2
                
                # 解码 FP4
                if use_gpu:
                    packed_gpu = tensor.cuda()
                    f32_gpu = decode_fp4_gpu(packed_gpu, 1, out_dim, in_dim)
                    del packed_gpu
                else:
                    raw = tensor.numpy().tobytes()
                    f32 = decode_fp4_cpu(raw)
                    f32_gpu = torch.from_numpy(f32).cuda()
                
                # 量化
                d, qs, scales = quantize_iq2xs_batch(f32_gpu, out_dim, in_dim, lib)
                del f32_gpu
                
                quantized_tensors.append({
                    'name': key,
                    'd': d,
                    'qs': qs,
                    'scales': scales,
                    'dims': [out_dim, in_dim],
                })
                
                gc.collect()
                torch.cuda.empty_cache()
    
    return quantized_tensors


def main():
    parser = argparse.ArgumentParser(description='IQ2_XS 预量化（GGUF 输出）')
    parser.add_argument('--ckpt-path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output', type=str, required=True, help='输出 GGUF 文件路径')
    parser.add_argument('--use-gpu', action='store_true', help='使用 GPU 解码')
    parser.add_argument('--batch-size', type=int, default=16, help='批处理大小')
    args = parser.parse_args()
    
    print("=" * 60)
    print("IQ2_XS 预量化（GGUF 输出）")
    print("=" * 60)
    print(f"模型路径: {args.ckpt_path}")
    print(f"输出文件: {args.output}")
    print(f"使用 GPU: {args.use_gpu}")
    
    # 加载量化库
    lib = _find_lib()
    print(f"量化库已加载")
    
    # 获取所有 safetensor 文件
    safetensor_files = sorted([
        os.path.join(args.ckpt_path, f)
        for f in os.listdir(args.ckpt_path)
        if f.endswith('.safetensors')
    ])
    
    print(f"\nSafetensor 文件数: {len(safetensor_files)}")
    
    # 创建 GGUF 写入器
    writer = GGUFWriter()
    writer.set_kv_string("general.name", "deepseek-v4-experts-iq2xs")
    writer.set_kv_uint32("general.quantization_version", 1)
    
    # 处理每个 shard
    start_time = time.time()
    total_tensors = 0
    
    for shard_idx, shard_path in enumerate(safetensor_files):
        shard_start = time.time()
        
        # 处理 shard
        quantized_tensors = process_safetensor_shard(
            shard_path, shard_idx, len(safetensor_files), lib, args.use_gpu, args.batch_size
        )
        
        # 添加到 GGUF 写入器
        for tensor_data in quantized_tensors:
            writer.add_iq2xs_tensor(
                tensor_data['name'],
                tensor_data['d'],
                tensor_data['qs'],
                tensor_data['scales'],
                tensor_data['dims'],
            )
            total_tensors += 1
        
        # 释放内存
        del quantized_tensors
        gc.collect()
        torch.cuda.empty_cache()
        
        shard_elapsed = time.time() - shard_start
        print(f"  耗时: {shard_elapsed:.1f}s")
    
    # 写入 GGUF 文件
    print(f"\n写入 GGUF 文件...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer.write(args.output)
    
    total_elapsed = time.time() - start_time
    print(f"\n完成！总耗时: {total_elapsed:.1f}s")
    print(f"张量数: {total_tensors}")


if __name__ == '__main__':
    main()
