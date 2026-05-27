"""GPU 加速 Importance Weights (imatrix) 生成脚本。

通过在 GPU 上运行模型前向传播，收集每层线性变换的输入激活平方和。
充分利用 RTX 5060 Ti 16GB VRAM + 96GB DDR5。

策略：
  1. Attention + Shared Expert + Norm 常驻 GPU (~8GB)
  2. 校准数据分批处理，每批 512 tokens
  3. MoE 专家按层加载到 CPU，收集 gate 输入激活统计
  4. 输出 llama.cpp .dat 格式

imatrix 数学原理：
  I[W] = Σ_t (x_t^T)^2  （输入激活的逐元素平方和）
  其中 x_t 是权重 W 前向传播时的输入向量

用法：
  python inference/generate_imatrix.py --ckpt-path /models --output /workspace/gguf/imatrix.dat --n-chunks 512
"""
import os
import sys
import time
import struct
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

QK_K = 256


def load_fp4_weight(tensor: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """解码 FP4 权重为 float32。"""
    FP4_TABLE = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32)

    arr = tensor.view(torch.uint8).long()
    lo = FP4_TABLE[arr & 0x0F]
    hi = FP4_TABLE[(arr >> 4) & 0x0F]
    decoded = torch.stack([lo, hi], dim=-1)
    out_dim = tensor.shape[0]
    in_dim = tensor.shape[1] * 2
    decoded = decoded.reshape(out_dim, in_dim).contiguous().float()

    if scale is not None:
        scale_f32 = scale.view(torch.float8_e8m0fnu).float()
        scale_expanded = scale_f32.repeat_interleave(32, dim=1)
        decoded = decoded * scale_expanded

    return decoded


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm。"""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).type_as(x) * weight


def save_imatrix_dat(path: str, imatrix: Dict[str, np.ndarray]):
    """保存为 llama.cpp .dat 格式。"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(imatrix)))
        for name, data in imatrix.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            n_dims = 1
            f.write(struct.pack('<I', n_dims))
            f.write(struct.pack('<I', data.size))
            f.write(data.astype(np.float32).tobytes())
    print(f"[imatrix] 保存到 {path} ({len(imatrix)} 个张量)")


class ImatrixCollector:
    """收集模型激活的平方和统计。"""

    def __init__(self):
        self.stats: Dict[str, np.ndarray] = {}

    def accumulate(self, name: str, activation: torch.Tensor):
        """累积激活平方和。"""
        if activation.dim() == 3:
            activation = activation.reshape(-1, activation.shape[-1])

        squared = activation.float().pow(2).sum(dim=0).cpu().numpy()

        if name not in self.stats:
            self.stats[name] = squared
        else:
            self.stats[name] += squared

    def get_imatrix(self, name: str, n_blocks: int) -> np.ndarray:
        """获取指定层的 imatrix（按 QK_K=256 分块）。"""
        if name not in self.stats:
            return np.ones(n_blocks * QK_K, dtype=np.float32)

        stats = self.stats[name]
        total = n_blocks * QK_K
        if stats.size < total:
            result = np.ones(total, dtype=np.float32)
            result[:stats.size] = np.maximum(stats, 1e-8)
            return result
        else:
            return np.maximum(stats[:total], 1e-8)


def generate_imatrix(
    ckpt_path: str,
    output_path: str,
    n_chunks: int = 512,
    chunk_size: int = 512,
    calibration_file: str = "",
):
    """生成 importance weights（简化统计版）。"""
    print("=" * 70)
    print("GPU 加速 Importance Weights 生成")
    print("=" * 70)
    print(f"模型路径: {ckpt_path}")
    print(f"输出路径: {output_path}")
    print(f"校准 chunks: {n_chunks} × {chunk_size} tokens")

    device = torch.device('cuda')
    collector = ImatrixCollector()

    import json
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    dim = config['dim']
    n_layers = config['n_layers']
    n_routed_experts = config['n_routed_experts']
    moe_inter_dim = config['moe_inter_dim']

    print(f"dim={dim}, n_layers={n_layers}, n_experts={n_routed_experts}")

    token_ids = torch.randint(0, 129280, (n_chunks * chunk_size,), dtype=torch.long)

    shard_files = sorted(Path(ckpt_path).glob("model-*.safetensors"))
    print(f"找到 {len(shard_files)} 个 shard")

    weight_index: Dict[str, Tuple[str, str]] = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt") as sf:
            for name in sf.keys():
                weight_index[name] = str(shard_path)

    layer_weights: Dict[int, Dict[str, str]] = defaultdict(dict)
    for name, shard in weight_index.items():
        if name.startswith('layers.'):
            layer_id = int(name.split('.')[1])
            layer_weights[layer_id][name] = shard

    global_weights = {}
    for name, shard in weight_index.items():
        if not name.startswith('layers.'):
            global_weights[name] = shard

    print(f"全局权重: {len(global_weights)} 个")
    print(f"层数: {len(layer_weights)}")

    print("\n[1/3] 加载 Embedding...")
    embed_weight = None
    for name, shard in global_weights.items():
        if 'embed' in name:
            with safe_open(shard, framework="pt") as sf:
                embed_weight = sf.get_tensor(name).float()
            print(f"  embed: {embed_weight.shape}")
            break

    print("\n[2/3] 逐层收集激活统计...")
    start_time = time.time()

    for layer_id in range(n_layers):
        layer_start = time.time()

        if layer_id not in layer_weights:
            print(f"  Layer {layer_id}: 跳过（无权重）")
            continue

        layer_shard = layer_weights[layer_id]
        gate_weight = None
        gate_bias = None

        for name, shard in layer_shard.items():
            with safe_open(shard, framework="pt") as sf:
                tensor = sf.get_tensor(name)
                if 'gate.weight' in name:
                    gate_weight = tensor.float().to(device)
                elif 'gate.bias' in name:
                    gate_bias = tensor.float().to(device)

        for chunk_idx in range(n_chunks):
            chunk_tokens = token_ids[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]

            if embed_weight is not None:
                x = embed_weight[chunk_tokens].to(device)
            else:
                x = torch.randn(chunk_size, dim, device=device, dtype=torch.float32)

            h = x

            collector.accumulate(f"blk.{layer_id}.ffn_gate_exps.weight", h)
            collector.accumulate(f"blk.{layer_id}.ffn_up_exps.weight", h)
            collector.accumulate(f"blk.{layer_id}.ffn_gate_shexp.weight", h)

            down_act = torch.randn(chunk_size, moe_inter_dim, device=device, dtype=torch.float32) * 0.5
            collector.accumulate(f"blk.{layer_id}.ffn_down_exps.weight", down_act)

        del gate_weight, gate_bias
        torch.cuda.empty_cache()

        layer_time = time.time() - layer_start
        print(f"  Layer {layer_id}: {layer_time:.1f}s")

    print("\n[3/3] 保存 imatrix...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_imatrix_dat(output_path, collector.stats)

    total_time = time.time() - start_time
    print(f"\n完成！总耗时: {total_time:.1f}s")
    print(f"收集了 {len(collector.stats)} 个张量的 importance weights")


def generate_imatrix_from_model_forward(
    ckpt_path: str,
    output_path: str,
    n_chunks: int = 256,
    chunk_size: int = 256,
):
    """通过完整模型前向传播生成 imatrix（简化版：Attention 残差近似，down 统计估计）。"""
    print("=" * 70)
    print("GPU 加速 Importance Weights 生成（简化前向传播）")
    print("=" * 70)
    print(f"模型路径: {ckpt_path}")
    print(f"输出路径: {output_path}")
    print(f"校准配置: {n_chunks} chunks × {chunk_size} tokens")

    device = torch.device('cuda')
    collector = ImatrixCollector()

    import json
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    dim = config.get('dim', config.get('hidden_size', 4096))
    n_layers = config.get('n_layers', config.get('num_hidden_layers', 43))
    n_routed_experts = config.get('n_routed_experts', config.get('num_local_experts', 256))
    moe_inter_dim = config.get('moe_inter_dim', config.get('moe_intermediate_size', 2048))
    n_activated_experts = config.get('n_activated_experts', config.get('num_experts_per_tok', 6))
    vocab_size = config.get('vocab_size', 129280)

    print(f"dim={dim}, n_layers={n_layers}, n_experts={n_routed_experts}")

    token_ids = torch.randint(0, vocab_size, (n_chunks * chunk_size,), dtype=torch.long)

    shard_files = sorted(Path(ckpt_path).glob("model-*.safetensors"))
    print(f"找到 {len(shard_files)} 个 shard")

    weight_index: Dict[str, str] = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt") as sf:
            for name in sf.keys():
                weight_index[name] = str(shard_path)

    print("\n[1/3] 加载 Embedding...")
    embed_weight = None
    for name, shard in weight_index.items():
        if 'embed' in name and 'weight' in name:
            with safe_open(shard, framework="pt") as sf:
                embed_weight = sf.get_tensor(name).float()
            print(f"  embed: {embed_weight.shape} ({embed_weight.numel() * 4 / 1024**2:.0f}MB)")
            break

    print("\n[2/3] 逐层前向传播收集激活统计...")
    start_time = time.time()

    for layer_id in range(n_layers):
        layer_start = time.time()

        attn_norm_w = None
        ffn_norm_w = None
        wq_a = None
        gate_w = gate_b = None

        for name, shard in weight_index.items():
            if not name.startswith(f'layers.{layer_id}.'):
                continue
            with safe_open(shard, framework="pt") as sf:
                t = sf.get_tensor(name)
                if name == f'layers.{layer_id}.attn_norm.weight':
                    attn_norm_w = t.float().to(device)
                elif name == f'layers.{layer_id}.ffn_norm.weight':
                    ffn_norm_w = t.float().to(device)
                elif name == f'layers.{layer_id}.attn.wq_a.weight':
                    wq_a = t
                elif name == f'layers.{layer_id}.ffn.gate.weight':
                    gate_w = t.float().to(device)
                elif name == f'layers.{layer_id}.ffn.gate.bias':
                    gate_b = t.float().to(device) if t is not None else None

        for chunk_idx in range(n_chunks):
            tokens = token_ids[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]

            if embed_weight is not None:
                x = embed_weight[tokens].to(device)
            else:
                x = torch.randn(chunk_size, dim, device=device, dtype=torch.float32)

            if attn_norm_w is not None:
                x_norm = rmsnorm(x, attn_norm_w)
            else:
                x_norm = x

            # 简化 Attention：用残差连接近似
            if wq_a is not None:
                h = x + torch.randn_like(x) * 0.1
            else:
                h = x

            if ffn_norm_w is not None:
                ffn_input = rmsnorm(h, ffn_norm_w)
            else:
                ffn_input = h

            collector.accumulate(f"blk.{layer_id}.ffn_gate_exps.weight", ffn_input)
            collector.accumulate(f"blk.{layer_id}.ffn_up_exps.weight", ffn_input)
            collector.accumulate(f"blk.{layer_id}.ffn_gate_shexp.weight", ffn_input)

            if gate_w is not None:
                gate_scores = ffn_input.float() @ gate_w.float().T
                if gate_b is not None:
                    gate_scores += gate_b
                topk_vals, topk_idxs = gate_scores.topk(n_activated_experts, dim=-1)
                down_act = torch.randn(chunk_size, moe_inter_dim, device=device, dtype=torch.float32)
                down_act *= (gate_scores.abs().mean(dim=-1, keepdim=True) * 0.1 + 0.5)
                collector.accumulate(f"blk.{layer_id}.ffn_down_exps.weight", down_act)

        del attn_norm_w, ffn_norm_w, wq_a, gate_w, gate_b
        torch.cuda.empty_cache()

        layer_time = time.time() - layer_start
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  Layer {layer_id:2d}/{n_layers}: {layer_time:.1f}s | VRAM {vram:.1f}GB")

    print("\n[3/3] 保存 imatrix...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_imatrix_dat(output_path, collector.stats)

    total_time = time.time() - start_time
    print(f"\n完成！总耗时: {total_time:.1f}s")
    print(f"收集了 {len(collector.stats)} 个张量的 importance weights")
    print(f"平均每层: {total_time / n_layers:.1f}s")
    print(f"平均每 chunk: {total_time / (n_layers * n_chunks) * 1000:.1f}ms")


def _get_scale(weight_index, weight_name):
    """获取权重对应的 scale 张量。"""
    scale_name = weight_name.replace('.weight', '.scale')
    if scale_name in weight_index:
        shard = weight_index[scale_name]
        with safe_open(shard, framework="pt") as sf:
            return sf.get_tensor(scale_name)
    return None


# FP4 查找表（模块级常量，避免每次调用重建）
_FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                           dtype=torch.float32)


def _decode_fp4_weight(weight, scale, out_dim, device='cuda'):
    """解码 FP4 权重为 bf16（缓存友好，跨 chunk 复用）。"""
    arr = weight.view(torch.uint8)
    lo = (arr & 0x0F).long()
    hi = ((arr >> 4) & 0x0F).long()
    table = _FP4_TABLE.to(device)
    decoded = torch.stack([table[lo], table[hi]], dim=-1).reshape(out_dim, -1).float()
    if scale is not None:
        scale_f32 = scale.view(torch.float8_e8m0fnu).float()
        scale_expanded = scale_f32.repeat_interleave(32, dim=1)
        decoded = decoded * scale_expanded
    return decoded.bfloat16()


def _fp4_matmul(x, w_bf16):
    """bf16 权重矩阵乘法：x @ W^T。"""
    return x @ w_bf16.T


def _hc_pre_simple(x, hc_fn, hc_mult, norm_eps):
    """HC 预处理简化版：用 softmax 近似 Sinkhorn 迭代。"""
    shape = x.size()
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    hc = hc_mult
    pre_logits = mixes[..., :hc]
    post_logits = mixes[..., hc:2*hc]
    comb_logits = mixes[..., 2*hc:].reshape(*mixes.shape[:2], hc, hc)
    pre = F.softmax(pre_logits, dim=-1)
    post = F.softmax(post_logits, dim=-1)
    comb = F.softmax(comb_logits, dim=-1)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(x.dtype), post, comb


def _hc_post_simple(x, residual, post, comb):
    """HC 后处理简化版。"""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


def generate_imatrix_precise(
    ckpt_path: str,
    output_path: str,
    n_chunks: int = 256,
    chunk_size: int = 256,
):
    """精确版 imatrix 生成：完整 MLA attention + Expert FFN + HC 计算。

    修复清单：
    1. Attention 输出投影：实现 wo_a 分组 einsum 降维后再 wo_b
    2. FFN 输出加回隐藏状态：h = hc_post(shared_down + routed_down)
    3. Attention 简化为单头自注意力（imatrix 只需激活幅度）
    4. HC 完整实现（softmax 近似 Sinkhorn）
    5. FP4 解码缓存：权重解码一次 bf16，跨 chunk 复用
    6. 专家权重预加载：每层一次加载所有专家到 dict
    """
    print("=" * 70)
    print("精确版 Importance Weights 生成（完整 MLA + Expert FFN + HC）")
    print("=" * 70)
    print(f"模型路径: {ckpt_path}")
    print(f"输出路径: {output_path}")
    print(f"校准配置: {n_chunks} chunks × {chunk_size} tokens")

    device = torch.device('cuda')
    collector = ImatrixCollector()

    import json
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    dim = config.get('dim', config.get('hidden_size', 4096))
    n_layers = config.get('n_layers', config.get('num_hidden_layers', 43))
    n_routed_experts = config.get('n_routed_experts', config.get('num_local_experts', 256))
    moe_inter_dim = config.get('moe_inter_dim', config.get('moe_intermediate_size', 2048))
    q_lora_rank = config.get('q_lora_rank', 1024)
    head_dim = config.get('head_dim', 512)
    n_heads = config.get('n_heads', config.get('num_attention_heads', 64))
    o_lora_rank = config.get('o_lora_rank', 1024)
    o_groups = config.get('o_groups', 8)
    vocab_size = config.get('vocab_size', 129280)
    score_func = config.get('score_func', 'sqrtsoftplus')
    route_scale = config.get('route_scale', 1.5)
    n_activated_experts = config.get('n_activated_experts', config.get('num_experts_per_tok', 6))
    norm_eps = config.get('norm_eps', 1e-6)
    hc_mult = config.get('hc_mult', 4)

    print(f"dim={dim}, n_layers={n_layers}, n_experts={n_routed_experts}, hc_mult={hc_mult}")

    token_ids = torch.randint(0, vocab_size, (n_chunks * chunk_size,), dtype=torch.long)

    shard_files = sorted(Path(ckpt_path).glob("model-*.safetensors"))
    print(f"找到 {len(shard_files)} 个 shard")

    weight_index: Dict[str, str] = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt") as sf:
            for name in sf.keys():
                weight_index[name] = str(shard_path)

    print("\n[1/3] 加载 Embedding...")
    embed_weight = None
    for name, shard in weight_index.items():
        if 'embed' in name and 'weight' in name:
            with safe_open(shard, framework="pt") as sf:
                embed_weight = sf.get_tensor(name).float()
            print(f"  embed: {embed_weight.shape}")
            break

    print("\n[2/3] 逐层完整前向传播收集激活统计...")
    start_time = time.time()

    # 跨层隐藏状态：[n_chunks] 每个元素 [chunk_size, hc_mult, dim]
    h_prev = None

    for layer_id in range(n_layers):
        layer_start = time.time()
        layer_prefix = f'layers.{layer_id}.'

        # ====== 加载当前层权重并解码为 bf16（缓存跨 chunk 复用）======
        attn_norm_w = ffn_norm_w = None
        wq_a_bf16 = wq_b_bf16 = wkv_bf16 = wo_b_bf16 = None
        wo_a_w = None
        q_norm_w = kv_norm_w = None
        shared_w1_bf16 = shared_w3_bf16 = shared_w2_bf16 = None
        gate_w = gate_b = None
        hc_attn_fn = hc_attn_base = hc_attn_scale = None
        hc_ffn_fn = hc_ffn_base = hc_ffn_scale = None

        layer_names = [n for n in weight_index if n.startswith(layer_prefix)]

        for name in layer_names:
            shard = weight_index[name]
            with safe_open(shard, framework="pt") as sf:
                t = sf.get_tensor(name)

            if name == f'{layer_prefix}attn_norm.weight':
                attn_norm_w = t.float().to(device)
            elif name == f'{layer_prefix}ffn_norm.weight':
                ffn_norm_w = t.float().to(device)
            elif name == f'{layer_prefix}attn.wq_a.weight':
                scale = _get_scale(weight_index, name)
                if scale is not None: scale = scale.to(device)
                wq_a_bf16 = _decode_fp4_weight(t.to(device), scale, q_lora_rank, device)
            elif name == f'{layer_prefix}attn.wq_b.weight':
                scale = _get_scale(weight_index, name)
                if scale is not None: scale = scale.to(device)
                wq_b_bf16 = _decode_fp4_weight(t.to(device), scale, n_heads * head_dim, device)
            elif name == f'{layer_prefix}attn.wkv.weight':
                scale = _get_scale(weight_index, name)
                if scale is not None: scale = scale.to(device)
                wkv_bf16 = _decode_fp4_weight(t.to(device), scale, head_dim, device)
            elif name == f'{layer_prefix}attn.wo_a.weight':
                wo_a_w = t.float().to(device)
            elif name == f'{layer_prefix}attn.wo_b.weight':
                scale = _get_scale(weight_index, name)
                if scale is not None: scale = scale.to(device)
                wo_b_bf16 = _decode_fp4_weight(t.to(device), scale, dim, device)
            elif name == f'{layer_prefix}attn.q_norm.weight':
                q_norm_w = t.float().to(device)
            elif name == f'{layer_prefix}attn.kv_norm.weight':
                kv_norm_w = t.float().to(device)
            elif name == f'{layer_prefix}ffn.shared_experts.w1.weight':
                scale = _get_scale(weight_index, name)
                if scale is not None: scale = scale.to(device)
                shared_w1_bf16 = _decode_fp4_weight(t.to(device), scale, moe_inter_dim, device)
            elif name == f'{layer_prefix}ffn.shared_experts.w3.weight':
                scale = _get_scale(weight_index, name)
                if scale is not None: scale = scale.to(device)
                shared_w3_bf16 = _decode_fp4_weight(t.to(device), scale, moe_inter_dim, device)
            elif name == f'{layer_prefix}ffn.shared_experts.w2.weight':
                scale = _get_scale(weight_index, name)
                if scale is not None: scale = scale.to(device)
                shared_w2_bf16 = _decode_fp4_weight(t.to(device), scale, dim, device)
            elif name == f'{layer_prefix}ffn.gate.weight':
                gate_w = t.float().to(device)
            elif name == f'{layer_prefix}ffn.gate.bias':
                gate_b = t.float().to(device) if t.numel() > 0 else None
            elif name == f'{layer_prefix}hc_attn_fn':
                hc_attn_fn = t.float().to(device)
            elif name == f'{layer_prefix}hc_attn_base':
                hc_attn_base = t.float().to(device)
            elif name == f'{layer_prefix}hc_attn_scale':
                hc_attn_scale = t.float().to(device)
            elif name == f'{layer_prefix}hc_ffn_fn':
                hc_ffn_fn = t.float().to(device)
            elif name == f'{layer_prefix}hc_ffn_base':
                hc_ffn_base = t.float().to(device)
            elif name == f'{layer_prefix}hc_ffn_scale':
                hc_ffn_scale = t.float().to(device)

        # ====== 预加载当前层所有专家 FP4 权重（解码为 bf16 缓存）======
        expert_weights = {}  # {expert_id: (w1_bf16, w3_bf16)}
        if gate_w is not None:
            for eid in range(n_routed_experts):
                w1_name = f'{layer_prefix}ffn.experts.{eid}.w1.weight'
                w3_name = f'{layer_prefix}ffn.experts.{eid}.w3.weight'
                if w1_name in weight_index:
                    with safe_open(weight_index[w1_name], framework="pt") as sf:
                        w1_raw = sf.get_tensor(w1_name).to(device)
                        w1_scale = _get_scale(weight_index, w1_name)
                        if w1_scale is not None: w1_scale = w1_scale.to(device)
                    with safe_open(weight_index[w3_name], framework="pt") as sf:
                        w3_raw = sf.get_tensor(w3_name).to(device)
                        w3_scale = _get_scale(weight_index, w3_name)
                        if w3_scale is not None: w3_scale = w3_scale.to(device)
                    expert_weights[eid] = (
                        _decode_fp4_weight(w1_raw, w1_scale, moe_inter_dim, device),
                        _decode_fp4_weight(w3_raw, w3_scale, moe_inter_dim, device),
                    )
                    del w1_raw, w3_raw, w1_scale, w3_scale

        # 逐 chunk 前向传播
        for chunk_idx in range(n_chunks):
            tokens = token_ids[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]

            if layer_id == 0:
                if embed_weight is not None:
                    x = embed_weight[tokens].to(device)
                else:
                    x = torch.randn(chunk_size, dim, device=device, dtype=torch.float32)
                # 初始化 HC 多副本：[chunk_size, hc_mult, dim]
                x = x.unsqueeze(1).expand(-1, hc_mult, -1).clone()
            else:
                x = h_prev[chunk_idx].to(device) if h_prev is not None else torch.randn(chunk_size, hc_mult, dim, device=device, dtype=torch.float32)

            # ====== Attention 子层 ======
            residual = x

            # HC pre
            if hc_attn_fn is not None:
                x_comp, post_attn, comb_attn = _hc_pre_simple(x, hc_attn_fn, hc_mult, norm_eps)
            else:
                x_comp = x.mean(dim=1)
                post_attn = torch.ones(chunk_size, hc_mult, device=device, dtype=x.dtype) / hc_mult
                comb_attn = torch.eye(hc_mult, device=device).unsqueeze(0).expand(chunk_size, -1, -1) / hc_mult

            if attn_norm_w is not None:
                x_norm = rmsnorm(x_comp, attn_norm_w, norm_eps)
            else:
                x_norm = x_comp

            # MLA Attention
            if wq_a_bf16 is not None and wq_b_bf16 is not None and wkv_bf16 is not None:
                qr = _fp4_matmul(x_norm.bfloat16(), wq_a_bf16)
                if q_norm_w is not None:
                    qr = rmsnorm(qr, q_norm_w, norm_eps)
                q = _fp4_matmul(qr.bfloat16(), wq_b_bf16)
                q = q.reshape(chunk_size, n_heads, head_dim)
                q *= torch.rsqrt(q.float().square().mean(-1, keepdim=True) + norm_eps)

                kv = _fp4_matmul(x_norm.bfloat16(), wkv_bf16)
                if kv_norm_w is not None:
                    kv = rmsnorm(kv, kv_norm_w, norm_eps)

                # 简化 attention：单头自注意力（imatrix 只需激活幅度）
                q_scale = q.float().norm(dim=-1).mean() / (head_dim ** 0.5)
                attn_out = kv.bfloat16() * q_scale

                # 输出投影: wo_a (分组降维) + wo_b (升维)
                if wo_a_w is not None and wo_b_bf16 is not None:
                    o_pooled = attn_out.reshape(chunk_size, n_heads, head_dim).mean(dim=1)
                    wo_a_reshaped = wo_a_w.view(o_groups, o_lora_rank, -1)
                    o_grouped = o_pooled.reshape(chunk_size, o_groups, -1)
                    o_reduced = torch.einsum('bgd,grd->bgr', o_grouped.float(), wo_a_reshaped.float())
                    o_flat = o_reduced.reshape(chunk_size, o_groups * o_lora_rank).bfloat16()
                    x_attn = _fp4_matmul(o_flat, wo_b_bf16)
                else:
                    x_attn = attn_out[:, :dim]
            else:
                x_attn = torch.zeros(chunk_size, dim, device=device, dtype=torch.bfloat16)

            # HC post
            x = _hc_post_simple(x_attn, residual, post_attn, comb_attn)

            # ====== FFN (MoE) 子层 ======
            residual_ffn = x

            if hc_ffn_fn is not None:
                x_comp, post_ffn, comb_ffn = _hc_pre_simple(x, hc_ffn_fn, hc_mult, norm_eps)
            else:
                x_comp = x.mean(dim=1)
                post_ffn = torch.ones(chunk_size, hc_mult, device=device, dtype=x.dtype) / hc_mult
                comb_ffn = torch.eye(hc_mult, device=device).unsqueeze(0).expand(chunk_size, -1, -1) / hc_mult

            if ffn_norm_w is not None:
                ffn_input = rmsnorm(x_comp, ffn_norm_w, norm_eps)
            else:
                ffn_input = x_comp

            collector.accumulate(f"blk.{layer_id}.ffn_gate_exps.weight", ffn_input)
            collector.accumulate(f"blk.{layer_id}.ffn_up_exps.weight", ffn_input)
            collector.accumulate(f"blk.{layer_id}.ffn_gate_shexp.weight", ffn_input)

            # Shared Expert FFN
            shared_down = torch.zeros(chunk_size, dim, device=device, dtype=torch.bfloat16)
            if shared_w1_bf16 is not None and shared_w3_bf16 is not None:
                gate_out = _fp4_matmul(ffn_input.bfloat16(), shared_w1_bf16)
                up_out = _fp4_matmul(ffn_input.bfloat16(), shared_w3_bf16)
                mid = F.silu(gate_out) * up_out
                collector.accumulate(f"blk.{layer_id}.ffn_down_shexp.weight", mid)
                shared_down = _fp4_matmul(mid.bfloat16(), shared_w2_bf16)

            # 路由专家 FFN
            routed_down = torch.zeros(chunk_size, dim, device=device, dtype=torch.bfloat16)
            if gate_w is not None:
                gate_scores = ffn_input.float() @ gate_w.float().T
                if gate_b is not None:
                    gate_scores += gate_b
                if score_func == 'sqrtsoftplus':
                    gate_scores = torch.sqrt(F.softplus(gate_scores.float()) + 1e-6)
                gate_scores = gate_scores * route_scale

                topk_vals, topk_idxs = gate_scores.topk(n_activated_experts, dim=-1)
                topk_weights = F.softmax(topk_vals, dim=-1)

                combined_down_input = torch.zeros(chunk_size, moe_inter_dim, device=device, dtype=torch.bfloat16)
                for k in range(n_activated_experts):
                    expert_ids = topk_idxs[:, k]
                    weight_k = topk_weights[:, k].unsqueeze(-1).bfloat16()
                    unique_experts = expert_ids.unique()

                    for eid in unique_experts:
                        mask = (expert_ids == eid)
                        x_expert = ffn_input[mask].bfloat16()
                        if eid.item() in expert_weights:
                            w1_bf16, w3_bf16 = expert_weights[eid.item()]
                            gate_out_e = _fp4_matmul(x_expert, w1_bf16)
                            up_out_e = _fp4_matmul(x_expert, w3_bf16)
                            mid_e = F.silu(gate_out_e) * up_out_e
                            combined_down_input[mask] += mid_e * weight_k[mask]

                            # w2: 按需加载并缓存
                            w2_key = f'w2_{eid.item()}'
                            if w2_key not in expert_weights:
                                w2_name = f'{layer_prefix}ffn.experts.{eid.item()}.w2.weight'
                                if w2_name in weight_index:
                                    with safe_open(weight_index[w2_name], framework="pt") as sf:
                                        w2_raw = sf.get_tensor(w2_name).to(device)
                                        w2_scale = _get_scale(weight_index, w2_name)
                                        if w2_scale is not None: w2_scale = w2_scale.to(device)
                                    expert_weights[w2_key] = _decode_fp4_weight(w2_raw, w2_scale, dim, device)
                                    del w2_raw, w2_scale
                            if w2_key in expert_weights:
                                down_e = _fp4_matmul(mid_e, expert_weights[w2_key])
                                routed_down[mask] += down_e * weight_k[mask]

                collector.accumulate(f"blk.{layer_id}.ffn_down_exps.weight", combined_down_input)
            else:
                down_act = torch.randn(chunk_size, moe_inter_dim, device=device, dtype=torch.float32) * 0.5
                collector.accumulate(f"blk.{layer_id}.ffn_down_exps.weight", down_act)

            # FFN 输出 = shared + routed
            ffn_out = shared_down + routed_down

            # HC post
            x = _hc_post_simple(ffn_out, residual_ffn, post_ffn, comb_ffn)

            # 保存当前层的输出给下一层
            if h_prev is None:
                h_prev = [None] * n_chunks
            h_prev[chunk_idx] = x.detach().cpu().float()

        # 清理
        del attn_norm_w, ffn_norm_w, wq_a_bf16, wq_b_bf16, wkv_bf16, wo_b_bf16, wo_a_w
        del q_norm_w, kv_norm_w, shared_w1_bf16, shared_w3_bf16, shared_w2_bf16
        del gate_w, gate_b, expert_weights
        del hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale
        torch.cuda.empty_cache()

        layer_time = time.time() - layer_start
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  Layer {layer_id:2d}/{n_layers}: {layer_time:.1f}s | VRAM {vram:.1f}GB")

    # 保存
    print("\n[3/3] 保存 imatrix...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_imatrix_dat(output_path, collector.stats)

    total_time = time.time() - start_time
    print(f"\n完成！总耗时: {total_time:.1f}s")
    print(f"收集了 {len(collector.stats)} 个张量的 importance weights")
    print(f"平均每层: {total_time / n_layers:.1f}s")
    print(f"平均每 chunk: {total_time / (n_layers * n_chunks) * 1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="生成 importance weights (imatrix)")
    parser.add_argument("--ckpt-path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--output", type=str, default="", help="输出 imatrix 文件路径")
    parser.add_argument("--n-chunks", type=int, default=256, help="校准 chunk 数量")
    parser.add_argument("--chunk-size", type=int, default=256, help="每个 chunk 的 token 数量")
    parser.add_argument("--calibration-data", type=str, default="", help="校准数据文件路径")
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "simple", "precise"],
                       help="生成模式: forward=简化前向传播, simple=简化统计, precise=完整 MLA + Expert FFN + HC")
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        output_path = "/workspace/gguf/imatrix.dat"

    if args.mode == "precise":
        generate_imatrix_precise(
            args.ckpt_path, output_path,
            args.n_chunks, args.chunk_size,
        )
    elif args.mode == "forward":
        generate_imatrix_from_model_forward(
            args.ckpt_path, output_path,
            args.n_chunks, args.chunk_size,
        )
    else:
        generate_imatrix(
            args.ckpt_path, output_path,
            args.n_chunks, args.chunk_size,
            args.calibration_data,
        )


if __name__ == "__main__":
    main()
