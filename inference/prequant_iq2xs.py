"""IQ2_XS 预量化脚本 — 首次启动时将 FP4 权重量化为 IQ2_XS 归档。

流程：
1. 扫描所有 safetensors shard
2. 提取专家权重（FP4 打包）
3. GPU 加速量化：FP4 → float16 → IQ2_XS（quantize_iq2xs_gpu_optimized）
4. 打包成归档文件 experts.iq2xs
5. 显示进度和统计

量化速度：
  GPU 量化（默认）：~50-100 experts/s（RTX 5060 Ti）
  CPU 回退：~0.3 experts/s（仅 GPU 不可用时使用）

用法：
    python inference/prequant_iq2xs.py --ckpt-path /models --output /models/iq2xs/experts.iq2xs
"""
import os
import sys
import time
import argparse
import struct
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iq2xs_archive import IQ2XSArchiveWriter

# FP4 解码表
FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

def decode_fp4(packed: bytes) -> np.ndarray:
    """解码 FP4 打包数据为 float16。

    每个 uint8 包含 2 个 FP4 值：低 4 位是偶数列，高 4 位是奇数列。
    解码后应交错排列：[lo0, hi0, lo1, hi1, ...]，与原始权重列顺序一致。
    """
    arr = np.frombuffer(packed, dtype=np.uint8)
    lo = arr & 0x0F
    hi = (arr >> 4) & 0x0F
    # 交错排列：每个字节的低4位和高4位交替
    return np.stack([FP4_TABLE[lo], FP4_TABLE[hi]], axis=-1).reshape(-1).astype(np.float16)

def find_expert_weights(ckpt_path: str) -> dict:
    """扫描所有 shard，提取专家权重信息。
    
    使用 safe_open 流式读取，避免全量加载 shard 到内存。
    
    返回: {shard_path: [(layer_id, expert_id, weight_name, tensor_name), ...]}
    """
    experts = {}
    
    shard_files = sorted(Path(ckpt_path).glob("model-*.safetensors"))
    
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt") as f:
            shard_experts = []
            for name in f.keys():
                # 匹配专家权重：layers.{L}.ffn.experts.{E}.{w}.weight
                if '.ffn.experts.' in name and '.weight' in name:
                    parts = name.split('.')
                    # layers.{L}.ffn.experts.{E}.{w}.weight → parts = [layers, L, ffn, experts, E, w, weight]
                    layer_id = int(parts[1])
                    expert_id = int(parts[4])
                    weight_name = parts[5]  # w1, w2, w3
                    shard_experts.append((layer_id, expert_id, weight_name, name))
        
        if shard_experts:
            experts[str(shard_path)] = shard_experts
    
    return experts


def _quantize_expert_gpu(weight_i8: torch.Tensor) -> dict:
    """使用 GPU 加速量化单个专家权重。

    参数:
        weight_i8: int8 FP4 打包权重，形状 [out_dim, in_dim//2]

    返回:
        {"d": float16, "qs": uint16, "scales": uint8, "shape": tuple}
    """
    from quantize_iq2xs_gpu_optimized import quantize_weight_gpu_optimized
    return quantize_weight_gpu_optimized(weight_i8)


def _quantize_expert_cpu(f16: np.ndarray, n_blocks: int) -> dict:
    """CPU 回退量化（GPU 不可用时使用）。

    参数:
        f16: float16 数组，已对齐到 256 的倍数
        n_blocks: block 数量

    返回:
        {"d": float16 numpy, "qs": uint16 numpy, "scales": uint8 numpy, "shape": tuple}
    """
    from iq2xs_c_wrapper import quantize_iq2_xs_gpu_via_cpu, get_qk_k
    QK_K = get_qk_k()

    flat_gpu = torch.from_numpy(f16.reshape(n_blocks, QK_K)).cuda()
    d, qs, scales = quantize_iq2_xs_gpu_via_cpu(flat_gpu)

    return {
        "d": d.cpu().numpy().astype(np.float16),
        "qs": qs.cpu().numpy().astype(np.uint16),
        "scales": scales.cpu().numpy().astype(np.uint8),
    }


def prequant_iq2xs(
    ckpt_path: str,
    output_path: str,
    n_layers: int,
    n_experts: int,
    progress_interval: int = 10,
) -> None:
    """预量化所有专家权重为 IQ2_XS 归档。
    
    优先使用 GPU 加速量化（quantize_iq2xs_gpu_optimized），
    GPU 不可用时回退到 CPU（C 实现）。
    
    参数:
        ckpt_path: 模型 checkpoint 目录
        output_path: 输出归档文件路径
        n_layers: 层数
        n_experts: 每层专家数
        progress_interval: 进度显示间隔（秒）
    """
    # 检测 GPU 可用性，选择量化方式
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        try:
            from quantize_iq2xs_gpu_optimized import quantize_weight_gpu_optimized
            print("[量化] 使用 GPU 加速量化（~50-100 experts/s）")
        except ImportError:
            use_gpu = False
            print("[量化] GPU 量化模块不可用，回退到 CPU（~0.3 experts/s）")
    else:
        print("[量化] GPU 不可用，回退到 CPU（~0.3 experts/s）")

    print("=" * 70)
    print("IQ2_XS 预量化")
    print("=" * 70)
    print(f"输入: {ckpt_path}")
    print(f"输出: {output_path}")
    print(f"量化方式: {'GPU' if use_gpu else 'CPU'}")
    
    # 扫描专家权重
    print("\n[扫描] 查找专家权重...")
    experts = find_expert_weights(ckpt_path)
    
    total_experts = sum(len(v) for v in experts.values())
    print(f"[扫描] 找到 {total_experts} 个专家权重 ({len(experts)} 个 shard)")
    
    # 统计层数和专家数
    all_layer_ids = set()
    all_expert_ids = set()
    for shard_experts in experts.values():
        for layer_id, expert_id, _, _ in shard_experts:
            all_layer_ids.add(layer_id)
            all_expert_ids.add(expert_id)
    
    if n_layers == 0:
        n_layers = max(all_layer_ids) + 1
    if n_experts == 0:
        n_experts = max(all_expert_ids) + 1
    
    print(f"[配置] n_layers={n_layers}, n_experts={n_experts}")
    
    # 创建归档写入器
    writer = IQ2XSArchiveWriter(output_path, n_layers, n_experts, n_weights=3)
    
    # 量化统计
    quantized = 0
    total_bytes = 0
    start_time = time.time()
    last_progress_time = start_time
    
    QK_K = 256
    
    # 逐 shard 处理
    for shard_idx, (shard_path, shard_experts) in enumerate(experts.items()):
        print(f"\n[Shard {shard_idx+1}/{len(experts)}] {os.path.basename(shard_path)}")
        
        # 使用 safe_open 流式读取，避免全量加载 shard 到内存
        with safe_open(shard_path, framework="pt") as sf:
            # 逐专家处理
            for expert_idx, (layer_id, expert_id, weight_name, tensor_name) in enumerate(shard_experts):
                # 读取 FP4 权重（仅读取当前张量，非全量 shard）
                tensor = sf.get_tensor(tensor_name)
                out_dim, packed_in_dim = tensor.shape
                in_dim = packed_in_dim * 2  # FP4 打包，实际维度翻倍

                if use_gpu:
                    # GPU 量化路径：直接传 int8 tensor，GPU 内部解码+量化
                    result = _quantize_expert_gpu(tensor)
                    d_np = result["d"].cpu().numpy().astype(np.float16)
                    qs_np = result["qs"].cpu().numpy().astype(np.uint16)
                    scales_np = result["scales"].cpu().numpy().astype(np.uint8)
                else:
                    # CPU 回退路径：先解码 FP4 → float16，再 C 量化
                    raw = tensor.numpy().tobytes()
                    f16 = decode_fp4(raw)
                    n_elements = f16.size
                    n_blocks = (n_elements + QK_K - 1) // QK_K
                    if n_elements % QK_K != 0:
                        padded = np.zeros(n_blocks * QK_K, dtype=np.float16)
                        padded[:n_elements] = f16
                        f16 = padded
                    else:
                        n_blocks = n_elements // QK_K

                    result = _quantize_expert_cpu(f16, n_blocks)
                    d_np = result["d"]
                    qs_np = result["qs"]
                    scales_np = result["scales"]

                # 计算总 block 数
                n_blocks = d_np.shape[0]
                
                # 添加到归档
                writer.add_expert(layer_id, expert_id, 
                                 {'w1': 0, 'w2': 1, 'w3': 2}[weight_name],
                                 d_np, qs_np, scales_np, out_dim, in_dim)
                
                quantized += 1
                total_bytes += n_blocks * 74
                
                # 进度显示
                now = time.time()
                if now - last_progress_time >= progress_interval or quantized == total_experts:
                    elapsed = now - start_time
                    rate = quantized / elapsed
                    remaining = (total_experts - quantized) / rate if rate > 0 else 0
                    print(f"\r  [{quantized}/{total_experts}] {quantized/total_experts*100:.1f}% "
                          f"| {rate:.1f} experts/s "
                          f"| 已用 {elapsed:.0f}s, 剩余 ~{remaining:.0f}s "
                          f"| {total_bytes/1024**3:.2f} GB", end='', flush=True)
                    last_progress_time = now

                # GPU 量化时及时释放显存
                if use_gpu and quantized % 100 == 0:
                    torch.cuda.empty_cache()
    
    print()
    
    # 写入归档
    print("\n[写入] 打包归档...")
    writer.write()
    
    total_time = time.time() - start_time
    final_size = os.path.getsize(output_path)
    
    print("\n" + "=" * 70)
    print("预量化完成")
    print("=" * 70)
    print(f"  专家数量:   {quantized}")
    print(f"  归档大小:   {final_size/1024**3:.2f} GB")
    print(f"  总耗时:     {total_time:.1f}s")
    print(f"  平均速度:   {quantized/total_time:.1f} experts/s")
    print(f"  量化方式:   {'GPU' if use_gpu else 'CPU'}")
    print(f"  输出文件:   {output_path}")

def main():
    parser = argparse.ArgumentParser(description="IQ2_XS 预量化")
    parser.add_argument("--ckpt-path", type=str, required=True, help="模型 checkpoint 目录")
    parser.add_argument("--output", type=str, default="", help="输出归档文件路径")
    parser.add_argument("--n-layers", type=int, default=0, help="层数（0=自动检测）")
    parser.add_argument("--n-experts", type=int, default=0, help="每层专家数（0=自动检测）")
    args = parser.parse_args()
    
    output_path = args.output
    if not output_path:
        output_path = os.path.join(args.ckpt_path, "iq2xs", "experts.iq2xs")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    prequant_iq2xs(args.ckpt_path, output_path, args.n_layers, args.n_experts)

if __name__ == "__main__":
    main()
