import os
import json
import sys
import gc
import time
from collections import OrderedDict, defaultdict
from argparse import ArgumentParser
from typing import List, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors import safe_open

from model import Transformer, ModelArgs

_CONFIG_KEY_MAP = {
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_experts_per_tok": "n_activated_experts",
    "num_nextn_predict_layers": "n_mtp_layers",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "routed_scaling_factor": "route_scale",
    "scoring_func": "score_func",
    "sliding_window": "window_size",
}

current_dir = os.path.dirname(os.path.abspath(__file__))
encoding_dir = os.path.join(current_dir, '../encoding')
sys.path.insert(0, os.path.abspath(encoding_dir))
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


from expert_cache import ExpertCache


def _set_cuda_shared_memory_limit(limit_bytes: int) -> None:
    """
    设置 CUDA 动态共享内存上限。

    RTX 5060 Ti 默认动态共享内存限制为 48KB/block，
    但 TileLang 的 sparse_attn 内核需要约 138KB。

    方法：通过环境变量 CUDA_DEVICE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    告知 CUDA 运行时允许更大的共享内存请求。
    注意：此环境变量必须在 CUDA context 初始化前设置。
    """
    os.environ["CUDA_DEVICE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"] = str(limit_bytes)
    print(f"[CUDA] Set CUDA_DEVICE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN={limit_bytes} ({limit_bytes//1024}KB)")


def _fix_weight_scale_refs(model: Transformer) -> None:
    """
    修复 load_state_dict(assign=True) 导致的 weight.scale 引用丢失。

    Linear.__init__ 中设置了 self.weight.scale = self.scale，
    但 assign=True 会替换 self.weight 和 self.scale 为新的 Parameter 对象，
    导致新 weight 对象上没有 .scale 属性。
    此函数遍历所有 Linear 层，重新绑定 weight.scale = self.scale。
    """
    for module in model.modules():
        if hasattr(module, 'weight') and hasattr(module, 'scale'):
            if module.scale is not None and not hasattr(module.weight, 'scale'):
                module.weight.scale = module.scale


def _dequantize_wo_a_in_state_dict(state_dict: dict) -> None:
    """
    在 state_dict 中反量化 wo_a 权重：FP8 + per-block scale → BF16。

    checkpoint 中 wo_a 以 FP8 格式存储（带 per-128x128 block scale），
    但模型代码中 wo_a 初始化为 BF16（无 scale 参数），且 einsum 不支持 FP8。
    必须在 load_state_dict 前处理，否则 wo_a.scale 键会被跳过。

    反量化公式（与 convert.py 一致）：
      weight[out//128, 128, in//128, 128] * scale[out//128, 1, in//128, 1]
      → flatten → bf16
    """
    names = list(state_dict.keys())
    for name in names:
        if name.endswith("wo_a.weight"):
            weight = state_dict[name]
            if weight.dtype != torch.float8_e4m3fn:
                continue
            scale_name = name.replace("weight", "scale")
            if scale_name in state_dict:
                scale = state_dict.pop(scale_name)
                w = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
                w = w * scale[:, None, :, None].float()
                state_dict[name] = w.flatten(2, 3).flatten(0, 1).bfloat16()
                del scale


def _dequantize_wo_a(model: Transformer) -> None:
    """
    反量化 wo_a 权重：FP8 + per-block scale → BF16。

    checkpoint 中 wo_a 以 FP8 格式存储（带 per-128x128 block scale），
    但模型代码中 wo_a 初始化为 BF16，且 einsum 不支持 FP8 张量。
    因此需要在加载后将 wo_a 反量化为 BF16，并移除对应的 scale 参数。

    反量化公式（与 convert.py 一致）：
      weight[out//128, 128, in//128, 128] * scale[out//128, 1, in//128, 1]
      → flatten → bf16
    """
    for name, module in model.named_modules():
        if not name.endswith("wo_a"):
            continue
        if not hasattr(module, 'weight') or not hasattr(module, 'scale'):
            continue
        if module.scale is None or module.weight.dtype != torch.float8_e4m3fn:
            continue

        weight = module.weight.data
        scale = module.scale.data

        # FP8 per-block 反量化：[out, in] → [out//128, 128, in//128, 128] * scale → [out, in] bf16
        w = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
        w = w * scale[:, None, :, None].float()
        w = w.flatten(2, 3).flatten(0, 1).bfloat16()

        module.weight = nn.Parameter(w)
        module.scale = None
        if hasattr(module.weight, 'scale'):
            del module.weight.scale
        print(f"  [Dequant] {name}: FP8 {weight.shape} → BF16 {w.shape}")


def _move_buffers_to_gpu(model: Transformer) -> None:
    """
    将所有非持久化 buffer（kv_cache、freqs_cis、kv_state、score_state 等）移到 GPU。

    这些 buffer 在模型 CPU 初始化时创建在 CPU 上，
    但推理时需要在 GPU 上访问。
    """
    for name, buf in model.named_buffers():
        if buf is not None and buf.device.type == "cpu":
            buf.data = buf.data.cuda()


def load_weights_from_shards(model: Transformer, ckpt_path: str, rank: int, world_size: int) -> None:
    """
    从多个 safetensors 分片文件流式加载权重到模型。

    针对 150GB 权重 / 90GB 内存 / 16GB 显存的资源约束设计：
      - 逐个分片文件加载，每次只在内存中持有一个分片（~3.4GB）
      - 每个分片加载后立即 load_state_dict 到模型（assign=True 避免拷贝）
      - 加载完一个分片后立即释放，再加载下一个
      - 峰值内存 ≈ 单个分片大小 + 模型参数量 ≈ 3.4GB + 7GB = ~10GB

    参数:
        model: Transformer 模型实例
        ckpt_path: checkpoint 目录路径
        rank: 当前进程 rank
        world_size: 总进程数
    """
    from glob import glob as _glob

    # 检测加载模式：单文件（旧格式）或多分片（新格式）
    single_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if os.path.exists(single_file):
        shard_files = [single_file]
    else:
        shard_files = sorted(_glob(os.path.join(ckpt_path, "*.safetensors")))

    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {ckpt_path}")

    total_size_gb = sum(os.path.getsize(f) for f in shard_files) / (1024**3)
    print(f"[Load] Found {len(shard_files)} shard files, total {total_size_gb:.1f}GB")

    # 懒初始化后 experts 为 None，需先创建空壳 Expert 才能 load_state_dict
    # 此路径仅用于小模型（权重总量 < 内存限制），大模型使用 load_weights_streaming
    for layer in model.layers:
        moe = layer.ffn
        for i in range(moe.n_routed_experts):
            moe._ensure_expert(i)

    # 逐分片加载：每个分片独立 mmap 打开，加载后立即释放
    total_keys = 0
    for shard_idx, shard_path in enumerate(shard_files):
        shard_size_gb = os.path.getsize(shard_path) / (1024**3)
        print(f"[Load] Shard {shard_idx+1}/{len(shard_files)}: {os.path.basename(shard_path)} ({shard_size_gb:.1f}GB)")

        state_dict = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                state_dict[key] = f.get_tensor(key)

        # 反量化 wo_a：checkpoint 中 wo_a 为 FP8 + per-block scale，模型代码使用 BF16 einsum
        _dequantize_wo_a_in_state_dict(state_dict)

        # assign=True: 直接替换参数指针，避免额外内存拷贝
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        total_keys += len(state_dict)

        if missing and shard_idx == 0:
            print(f"[Load] First shard: {len(missing)} keys missing (expected for multi-shard)")
        if unexpected:
            print(f"[WARN] Unexpected keys in shard {shard_idx}: {unexpected[:5]}...")

        del state_dict
        gc.collect()

    # 将常驻权重搬到 GPU，路由专家留在 CPU（mmap 按需加载）
    print("[Load] Moving resident weights to GPU...")
    for name, param in model.named_parameters():
        if "experts" not in name or "shared_experts" in name:
            if param.device.type == "cpu":
                param.data = param.data.cuda()

    # 修复 load_state_dict(assign=True) 导致的 weight.scale 引用丢失
    # Linear.__init__ 中 self.weight.scale = self.scale，但 assign=True 替换参数后引用断开
    _fix_weight_scale_refs(model)

    # 反量化 wo_a：checkpoint 中 wo_a 为 FP8，但模型代码使用 BF16 einsum
    _dequantize_wo_a(model)

    # 将 kv_cache、freqs_cis 等 buffer 移到 GPU
    _move_buffers_to_gpu(model)

    torch.cuda.empty_cache()
    vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"[Load] Done. Loaded {total_keys} keys from {len(shard_files)} shards. GPU VRAM: {vram_mb:.0f}MB")


def load_weights_streaming(model: Transformer, ckpt_path: str, rank: int, world_size: int) -> None:
    """
    流式按需加载权重：路由专家仅加载激活的 top-k 个，其余常驻 GPU。

    针对 150GB 权重 / 90GB 内存 / 16GB 显存的资源约束设计：
      - 非路由专家常驻 GPU（共享专家体积小但高频访问）
      - 路由专家按需加载：Gate 计算后仅加载被激活的 top-k 专家到 GPU
      - 专家计算完毕后立即卸载，释放 GPU 显存
      - 通过 MoE._on_experts_needed / _on_experts_done 回调驱动

    与旧方案（每层加载全部 256 个专家）的区别：
      - 旧方案：每层加载 256 个专家 ≈ 6GB GPU，超出 16GB 限制
      - 新方案：仅加载 top-8 激活专家 ≈ 192MB GPU，完全可行

    参数:
        model: Transformer 模型实例
        ckpt_path: checkpoint 目录路径
        rank: 当前进程 rank
        world_size: 总进程数
    """
    from glob import glob as _glob

    single_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if os.path.exists(single_file):
        shard_files = [single_file]
    else:
        shard_files = sorted(_glob(os.path.join(ckpt_path, "*.safetensors")))

    # 步骤 1：仅解析 safetensors header（不加载张量数据），收集键名并按模块分类
    # 使用 _get_shard_metadata 直接读取 header，避免 safe_open 的 mmap 页缓存膨胀
    # safe_open 会 mmap 整个分片文件（3.4GB/个），46 个分片 × 3.4GB ≈ 156GB 页缓存潜力
    print("[Streaming] Scanning shard headers for key classification (no mmap)...")
    key_to_shard = {}
    attn_keys = []
    ffn_norm_keys = []
    moe_gate_keys = []
    shared_expert_keys = []
    routed_expert_keys = {}
    embed_keys = []
    head_keys = []
    hc_keys = []
    other_keys = []

    iq2xs_dir = os.environ.get("IQ2XS_DIR", "")  # IQ2_XS 归档文件目录
    expert_cache = ExpertCache({}, top_n=6, window_m=5, cpu_cache_size=500,
                               iq2xs_dir=iq2xs_dir)
    for shard_path in shard_files:
        metadata = expert_cache.get_shard_metadata(shard_path)
        for key in metadata.keys():
            key_to_shard[key] = shard_path
            if "embed" in key:
                embed_keys.append(key)
            elif "head" in key:
                head_keys.append(key)
            elif "attn" in key and "norm" not in key:
                attn_keys.append(key)
            elif "ffn_norm" in key:
                ffn_norm_keys.append(key)
            elif "gate" in key and "experts" not in key:
                moe_gate_keys.append(key)
            elif "shared_experts" in key:
                shared_expert_keys.append(key)
            elif "experts" in key and "shared" not in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_id = int(parts[i + 1])
                            if layer_id not in routed_expert_keys:
                                routed_expert_keys[layer_id] = []
                            routed_expert_keys[layer_id].append(key)
                            break
                        except ValueError:
                            pass
            elif "hc" in key:
                hc_keys.append(key)
            else:
                other_keys.append(key)

    # 步骤 2：构建按层、按专家索引的键映射
    # expert_key_map: {layer_id: {expert_id: [(key, shard_path), ...]}}
    expert_key_map = {}
    for layer_id, keys in routed_expert_keys.items():
        expert_key_map[layer_id] = {}
        for key in keys:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "experts" and i + 1 < len(parts):
                    try:
                        expert_id = int(parts[i + 1])
                        if expert_id not in expert_key_map[layer_id]:
                            expert_key_map[layer_id][expert_id] = []
                        expert_key_map[layer_id][expert_id].append((key, key_to_shard[key]))
                        break
                    except ValueError:
                        pass

    # 步骤 3：逐分片加载常驻权重
    resident_keys = set(embed_keys + head_keys + attn_keys + ffn_norm_keys + moe_gate_keys + shared_expert_keys + hc_keys + other_keys)
    print(f"[Streaming] Loading {len(resident_keys)} resident keys from {len(shard_files)} shards...")

    shard_resident_keys = {}
    for key in resident_keys:
        sp = key_to_shard[key]
        if sp not in shard_resident_keys:
            shard_resident_keys[sp] = []
        shard_resident_keys[sp].append(key)

    # 顺序预热页缓存：safe_open 使用 mmap 随机访问，直接访问会导致大量随机 page fault
    # 先用 fadvise WILLNEED 提示 OS 预读，减少随机 I/O 延迟
    import time as _time
    preload_start = _time.time()
    for shard_path in shard_resident_keys:
        try:
            fd = os.open(shard_path, os.O_RDONLY | os.O_DIRECT)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            os.close(fd)
        except (OSError, AttributeError):
            pass
    preload_elapsed = _time.time() - preload_start
    print(f"[Streaming] fadvise WILLNEED for {len(shard_resident_keys)} shards: {preload_elapsed:.1f}s")

    # 常驻权重使用 safe_open 加载（启动时一次性操作）
    load_start = _time.time()
    for shard_idx, (shard_path, keys) in enumerate(shard_resident_keys.items()):
        shard_name = os.path.basename(shard_path)
        t0 = _time.time()
        state_dict = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                state_dict[key] = f.get_tensor(key)

        # 反量化 wo_a：checkpoint 中 wo_a 为 FP8 + per-block scale，模型代码使用 BF16 einsum
        # 必须在 load_state_dict 前处理，因为 wo_a 初始化为 BF16（无 scale 参数）
        # load_state_dict(assign=True) 会跳过 wo_a.scale 键（模型无此参数）
        _dequantize_wo_a_in_state_dict(state_dict)

        model.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict
        gc.collect()
        elapsed = _time.time() - t0
        print(f"  [{shard_idx+1}/{len(shard_resident_keys)}] {shard_name}: {len(keys)} keys, {elapsed:.1f}s")

    total_load_time = _time.time() - load_start
    print(f"[Streaming] Loaded all resident keys in {total_load_time:.1f}s")

    # 将常驻权重搬到 GPU，路由专家留在 CPU（按需加载）
    print("[Streaming] Moving resident weights to GPU...")
    for name, param in model.named_parameters():
        if "experts" not in name or "shared_experts" in name:
            if param.device.type == "cpu":
                param.data = param.data.cuda()

    # 修复 weight.scale 引用
    _fix_weight_scale_refs(model)

    # 反量化 wo_a：checkpoint 中 wo_a 为 FP8，但模型代码使用 BF16 einsum
    _dequantize_wo_a(model)

    # 将 kv_cache、freqs_cis 等 buffer 移到 GPU
    _move_buffers_to_gpu(model)

    torch.cuda.empty_cache()
    vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"[Streaming] GPU VRAM: {vram_mb:.0f}MB")

    # 记录初始 VRAM 空闲量（模型加载后、专家缓存加载前）
    # 使用 mem_get_info() 获取实际可用 VRAM（含 CUDA context 等非 PyTorch 分配）
    vram_free_mb, vram_total_mb = torch.cuda.mem_get_info()
    vram_free_mb = vram_free_mb / (1024**2)
    expert_cache._initial_vram_free_mb = vram_free_mb

    # 步骤 4：更新 expert_cache 的 expert_key_map（步骤 1 中已创建）
    expert_cache.expert_key_map = expert_key_map
    expert_cache.setup(model, len(model.layers),
                       os.environ.get("EXPERT_CACHE_DIR", "/root/.cache/ds4rs"))
    expert_cache.adjust_window_for_vram()
    model._expert_cache = expert_cache

    # 步骤 5：为每层 MoE 设置流式加载回调
    for layer_id, block in enumerate(model.layers):
        moe = block.ffn
        if layer_id in expert_key_map:
            moe._expert_key_map = expert_key_map[layer_id]
            moe._expert_cache = expert_cache
            moe._on_experts_needed = lambda activated, m=moe: _load_activated_experts(m, activated)
            moe._on_experts_done = lambda activated, m=moe: _unload_activated_experts(m, activated)

    total_expert_keys = sum(len(v) for v in routed_expert_keys.values())
    total_layers_with_experts = len(routed_expert_keys)
    print(f"[Streaming] Routed experts: {total_expert_keys} keys across {total_layers_with_experts} layers")

    expert_cache.preload_experts_to_ram()

    print(f"[Streaming] Cache v9: L1 GPU (SLRU prot={expert_cache._gpu_cache.protected_capacity}/prob={expert_cache._gpu_cache.probation_capacity} cap={expert_cache._gpu_cache.capacity}) → L2 CPU (SLRU cap={expert_cache._cpu_cache.capacity}) → L2.5 RAM (FP8 scale ZipNN) → L3 SSD")


def _load_activated_experts(moe, activated_indices: list) -> None:
    """使用 SLRU 缓存加载激活专家的权重到 GPU。

    优化策略（100% pinned pool 模式）：
    1. Gate 计算后立即并行 DMA 所有差集专家（从 pinned pool → GPU）
    2. DMA 与 Expert 空壳创建重叠
    3. 当前层加载完成后，立即预取 L+1 层差集

    缓存查找顺序：L1 GPU SLRU → DMA 预取 → Pinned Pool → SSD
    """
    cache = moe._expert_cache
    layer_id = moe.layer_id
    cache.on_layer_start(layer_id)

    # === 阶段 1：识别差集并并行预取 ===
    # 差集 = 不在 GPU 缓存且不在预取队列中的专家
    diff_keys = []
    gpu_hit_ids = []

    for expert_id in activated_indices:
        key = (layer_id, expert_id)
        if cache._gpu_cache.contains(key):
            gpu_hit_ids.append(expert_id)
        else:
            with cache._prefetch_lock:
                in_prefetch = (layer_id in cache._prefetch_pending and
                               expert_id in cache._prefetch_pending[layer_id])
                in_iq2xs_prefetch = key in cache._iq2xs_prefetch_pending
            if not in_prefetch and not in_iq2xs_prefetch:
                diff_keys.append(key)

    # 并行提交所有差集专家到预取线程池（非阻塞）
    if diff_keys:
        cache.prefetch_experts_batch(diff_keys)

    # === 阶段 2：处理 GPU 缓存命中的专家 ===
    for expert_id in gpu_hit_ids:
        with torch.device('cpu'):
            moe._ensure_expert(expert_id)

    # === 阶段 3：处理差集专家（等待预取完成 + 设置参数） ===
    all_gpu_params = {}
    missed = []

    for expert_id in activated_indices:
        with torch.device('cpu'):
            moe._ensure_expert(expert_id)
        expert = moe.experts[expert_id]
        if expert is None:
            continue

        gpu_params = cache.get_expert_gpu_params(layer_id, expert_id)
        if gpu_params is not None:
            all_gpu_params[expert_id] = gpu_params
        else:
            missed.append(expert_id)

    if missed and layer_id <= 2:
        print(f"[WARN] L{layer_id}: {len(missed)} experts missed GPU params: {missed[:5]}")

    # 未获取到 GPU 参数的专家必须置 None，防止 fp4_gemm 收到 CPU 权重
    for eid in missed:
        if eid < len(moe.experts):
            moe.experts[eid] = None

    for expert_id, gpu_params in all_gpu_params.items():
        expert = moe.experts[expert_id]
        if expert is None:
            continue
        cache._set_expert_params(expert, gpu_params)

    # 验证专家权重在 GPU 上（仅首次推理前3层打印）
    if layer_id <= 2 and not hasattr(moe, '_debug_printed'):
        moe._debug_printed = True

    # === 阶段 4：预取下一层 ===
    cache.prefetch_next_layer(layer_id, activated_indices)


def _unload_activated_experts(moe, activated_indices: list) -> None:
    """LFU 模式下无需显式卸载，缓存自动管理淘汰。"""
    pass


def sample(logits, temperature: float = 1.0):
    """
    基于 Gumbel-max 技巧的采样函数。

    原理：
        对 logits 施加温度缩放后计算 softmax 得到概率分布，
        再为每个概率值采样一个服从指数分布 Exponential(1) 的随机变量，
        取概率除以该随机变量后的最大值对应的索引作为采样结果。

        数学上等价于从多项分布 (multinomial) 中采样，但完全在 GPU 上完成，
        避免了 torch.multinomial 将数据从 GPU 同步回 CPU 的开销，
        因此在批量推理场景下速度更快。

    参数：
        logits: 模型输出的原始 logits，形状通常为 (batch_size, vocab_size)。
        temperature: 温度参数，控制采样随机性。
            - temperature -> 0 时趋近于贪心采样（取概率最大的 token）。
            - temperature = 1 时为标准随机采样。
            - temperature > 1 时分布更平坦，随机性增强；
            - temperature < 1 时分布更尖锐，随机性减弱。
            此处通过 max(temperature, 1e-5) 避免除零错误。

    返回：
        采样得到的 token 索引，形状与 logits 的前缀维度一致。
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    批量文本生成函数，支持 left-pad（左填充）对齐的变长 prompt。

    生成流程分为两个阶段：
        1. Prefill 阶段（预填充）：
           第一个 forward 步一次性处理每个序列中从 min_prompt_len 开始到末尾的所有 prompt tokens，
           同时计算并缓存这些位置的 KV（Key/Value），为后续 decode 做准备。
        2. Decode 阶段（自回归解码）：
           从 min_prompt_len 位置开始，逐个位置生成新 token。
           每一步仅将上一步新生成的 token（或 prompt 中对应位置的 ground-truth token）输入模型，
           利用 KV Cache 避免重复计算历史位置，从而加速生成。

    Ground-truth token 覆盖逻辑：
        对于每个序列，如果当前生成位置 cur_pos 仍然落在原始 prompt 范围内
        （即 prompt_mask[:, cur_pos] 为 True），则直接使用 prompt 中该位置的 ground-truth token，
        而不是使用模型预测出的 token。这保证了模型在 prompt 范围内的行为是确定性的，
        只有超出 prompt 长度的部分才真正进行自回归生成。

    终止条件：
        - 当某个序列生成出 eos_id（结束符）时，该序列被标记为 finished。
        - 当所有序列都 finished，或达到最大生成长度 total_len 时，提前退出循环。

    参数：
        model: 已加载权重的 Transformer 模型实例。
        prompt_tokens: 批量 prompt 的 token 索引列表，每个子列表长度可以不同。
        max_new_tokens: 每个序列最多生成的新 token 数量（不包括 prompt 部分）。
        eos_id: 结束符 token 的索引，遇到该 token 则停止对应序列的生成。
        temperature: 采样温度，传递给 sample() 函数控制随机性。

    返回：
        每个序列的生成结果 token 列表，已截断至 eos_id 之前，并在末尾追加 eos_id。
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    # total_len 为实际需要进行 forward 计算的最大序列长度，
    # 受限于模型支持的最大序列长度 max_seq_len，以及 prompt 长度加上最多新生成的 token 数。
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    # 初始化 tokens 张量，形状为 (batch_size, total_len)，用 -1 填充表示未填充位置。
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    prev_pos = 0
    # finished 标记每个序列是否已经生成了结束符 eos_id。
    finished = torch.tensor([False] * len(prompt_tokens))
    # prompt_mask 标识哪些位置属于原始 prompt（True），哪些位置是需要模型生成的（False）。
    prompt_mask = tokens != -1
    # 从最短 prompt 长度开始迭代：
    # - 对于长度大于 min(prompt_lens) 的序列，第一个 forward 会处理多个 prompt tokens（prefill）。
    # - 所有序列在 cur_pos 达到各自 prompt 长度后进入逐 token decode 阶段。
    for cur_pos in range(min(prompt_lens), total_len):
        # Step 开始：重置 step 级别保护集合
        cache = getattr(model, '_expert_cache', None)
        if cache is not None:
            cache.on_step_start(cur_pos)

        t0 = __import__('time').time()
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        fwd_ms = (__import__('time').time() - t0) * 1000
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        # 如果当前位置属于原始 prompt，则使用 prompt 中的 ground-truth token 覆盖模型预测值。
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        # 仅在非 prompt 位置检查是否生成了 eos_id，更新 finished 状态。
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos

        # 推理进度日志：每步打印 forward 耗时和缓存统计
        # 在 on_step_end 之前打印，这样 step_prot 才能显示当前 step 的保护数
        if cache is not None and cur_pos % 5 == 0:
            s = cache._stats
            total = s['gpu_hits'] + s['cpu_hits'] + s['ssd_hits'] + s['pinned_pool_hits']
            gpu_rate = s['gpu_hits'] / total * 100 if total > 0 else 0
            print(f"[Step {cur_pos}] fwd={fwd_ms:.0f}ms gpu={s['gpu_hits']}({gpu_rate:.0f}%) "
                  f"pinned={s['pinned_pool_hits']} cpu={s['cpu_hits']} ssd={s['ssd_hits']} "
                  f"pf_hit={s['prefetch_hits']} evict={s['gpu_evictions']} "
                  f"step_prot={len(cache._step_protected_keys) if cache._step_protected_keys is not None else -1}", flush=True)

        # Step 结束：清理 step 级别保护
        if cache is not None:
            cache.on_step_end()

        # 每 10 个 token 检查内存压力并回收
        if cur_pos % 10 == 0:
            gc.collect()
            if cache is not None:
                cache.check_memory_pressure()
                cache.finalize_prefetch()
        # 所有序列均完成时提前退出，避免不必要的计算。
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # 截取 prompt 之后、最多 max_new_tokens 个生成 token。
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        # 如果生成结果中包含 eos_id，则截断至 eos_id 之前。
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        # 在结果末尾追加 eos_id，保证输出格式统一。
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    quant_type: str = "fp4",
    iq2xs_dir: str = "",
) -> None:
    """
    文本生成主入口函数，负责分布式环境初始化、模型加载、以及交互式/批处理推理。

    分布式初始化（NCCL）：
        通过读取环境变量 WORLD_SIZE、RANK、LOCAL_RANK 判断当前进程所属的角色：
        - WORLD_SIZE > 1 时，调用 dist.init_process_group("nccl") 初始化 NCCL 进程组，
          支持多 GPU 分布式推理。
        - 非 0 号 rank 的进程将 print 重定义为空操作，避免多进程同时输出造成混乱，
          仅由 rank 0 进程负责与用户交互或输出结果。

    模型加载：
        - 根据 config 路径读取 JSON 配置文件，解析为 ModelArgs。
        - 交互式模式下强制设置 max_batch_size = 1，因为 REPL 每次只处理单条输入。
        - 在 cuda 设备上实例化 Transformer 模型。
        - 使用 AutoTokenizer 从 ckpt_path 加载分词器。
        - 使用 safetensors 的 load_model 加载对应 rank 的模型权重文件
          （格式为 model{rank}-mp{world_size}.safetensors），strict=False 允许部分权重缺失。

    交互式 REPL 逻辑：
        - 维护 messages 列表保存多轮对话历史。
        - 单进程时直接读取用户输入；多进程时由 rank 0 读取并通过 dist.broadcast_object_list
          将输入广播到所有 rank，保证各进程输入一致。
        - 支持 /exit 退出、/clear 清空对话历史。
        - 将用户输入编码为 prompt tokens，调用 generate 生成回复，解码后输出，
          并将模型回复解析后追加到 messages 中，实现多轮对话。

    批处理模式（非交互式）：
        - 从 input_file 读取输入，按双换行符 \n\n 分割为多个 prompt。
        - 对每个 prompt 编码后批量调用 generate，最后逐条输出 prompt 与对应的 completion。

    参数：
        ckpt_path: 模型检查点目录路径，包含 safetensors 权重和 tokenizer 配置。
        config: 模型配置文件（JSON）路径。
        input_file: 批处理模式下的输入文件路径，交互式模式下无需提供。
        interactive: 是否启用交互式 REPL 模式。
        max_new_tokens: 每次生成最多产生的新 token 数量。
        temperature: 采样温度，控制生成随机性。
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # 必须在 CUDA context 初始化前设置共享内存限制
    # 否则 TileLang sparse_attn 内核会因 48KB 默认限制而启动失败
    _set_cuda_shared_memory_limit(200 * 1024)
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        raw_config = json.load(f)

    if "quantization_config" in raw_config:
        qc = raw_config.pop("quantization_config")
        # quantization_config.fmt='e4m3' 对应 ModelArgs.dtype='fp8'
        fmt = qc.get("fmt", "fp8")
        raw_config.setdefault("dtype", "fp8" if fmt in ("e4m3", "fp8") else "bf16")
        raw_config.setdefault("scale_fmt", qc.get("scale_fmt", "ue8m0"))
        raw_config.setdefault("scale_dtype", qc.get("scale_dtype", "fp8"))

    mapped = {}
    valid_fields = {f.name for f in ModelArgs.__dataclass_fields__.values()}
    for k, v in raw_config.items():
        k = _CONFIG_KEY_MAP.get(k, k)
        if k in valid_fields:
            mapped[k] = v

    args = ModelArgs(**mapped)
    if interactive:
        args.max_batch_size = 1
    print(args)
    # 在 CPU 上初始化模型，避免 GPU OOM
    # 256 个路由专家 × 43 层的参数量远超 16GB 显存
    # 改为 CPU 初始化 + 按需搬到 GPU
    with torch.device("cpu"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    # 根据资源约束选择加载策略：
    # - 若权重总量 <= 内存容量，使用多分片流式加载（逐分片加载到 GPU）
    # - 若权重总量 > 内存容量（如 150GB 权重 / 90GB 内存），使用路由专家按层动态加载/卸载
    from glob import glob as _glob
    single_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if os.path.exists(single_file):
        total_weight_gb = os.path.getsize(single_file) / (1024**3)
    else:
        all_shards = _glob(os.path.join(ckpt_path, "*.safetensors"))
        total_weight_gb = sum(os.path.getsize(f) for f in all_shards) / (1024**3)

    print(f"[Quant] Using {quant_type.upper()} quantization for routed experts")

    if quant_type == "iq2xs":
        if not iq2xs_dir:
            candidate_dirs = [
                os.path.join(ckpt_path, "iq2xs"),
                "/models/iq2xs",
                os.path.expanduser("~/.cache/ds4rs/iq2xs"),
            ]
            for d in candidate_dirs:
                if os.path.isdir(d) and os.listdir(d):
                    iq2xs_dir = d
                    break

        # 检查归档文件
        archive_path = None
        if iq2xs_dir and os.path.isdir(iq2xs_dir):
            candidate_archive = os.path.join(iq2xs_dir, "experts.iq2xs")
            if os.path.exists(candidate_archive):
                archive_path = candidate_archive
        
        if archive_path:
            print(f"[IQ2_XS] Found archive: {archive_path}")
            os.environ["IQ2XS_DIR"] = iq2xs_dir
        else:
            # 需要预量化
            print("[IQ2_XS] No archive found, performing pre-quantization...")
            from prequant_iq2xs import prequant_iq2xs

            # 预量化输出目录：优先使用可写路径
            iq2xs_output_dir = os.path.join(ckpt_path, "iq2xs")
            try:
                os.makedirs(iq2xs_output_dir, exist_ok=True)
            except OSError:
                # ckpt_path 可能是只读挂载，使用缓存目录
                iq2xs_output_dir = os.path.expanduser("~/.cache/ds4rs/iq2xs")
                os.makedirs(iq2xs_output_dir, exist_ok=True)
                print(f"[IQ2_XS] ckpt_path is read-only, using cache dir: {iq2xs_output_dir}")
            archive_path = os.path.join(iq2xs_output_dir, "experts.iq2xs")
            
            # 从已解析的 args 获取层数和专家数（避免重复读取 config.json）
            n_layers = args.n_layers if hasattr(args, 'n_layers') else 0
            n_experts = args.n_routed_experts if hasattr(args, 'n_routed_experts') else 0
            
            # 执行预量化
            prequant_iq2xs(ckpt_path, archive_path, n_layers, n_experts)
            
            print(f"[IQ2_XS] Pre-quantization complete, using archive: {archive_path}")
            os.environ["IQ2XS_DIR"] = iq2xs_output_dir
    else:
        os.environ["IQ2XS_DIR"] = ""

    # 动态检测可用内存
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    mem_available_gb = int(line.split()[1]) * 1024 / (1024**3)
                    break
            else:
                mem_available_gb = 90
    except Exception:
        mem_available_gb = 90
    print(f"[INFO] Total weights: {total_weight_gb:.1f}GB, Memory limit: {mem_available_gb}GB")

    if total_weight_gb > mem_available_gb * 0.8:
        print(f"[INFO] Using streaming mode (routed experts load/unload per layer)")
        torch.set_default_device("cuda")
        load_weights_streaming(model, ckpt_path, rank, world_size)
    else:
        print(f"[INFO] Using shard-by-shard load (all weights resident on GPU)")
        load_weights_from_shards(model, ckpt_path, rank, world_size)
        torch.set_default_device("cuda")

    # 清除 lru_cache 中可能在 CPU 设备上下文创建的索引张量
    from model import get_window_topk_idxs, get_compress_topk_idxs
    get_window_topk_idxs.cache_clear()
    get_compress_topk_idxs.cache_clear()

    # 自动 warmup：做一次短推理收集 LFU 统计，然后将热点专家常驻 GPU
    # 必须在 torch.set_default_device("cuda") 之后，因为模型内部创建的索引张量需要在 GPU 上
    expert_cache = getattr(model, '_expert_cache', None)
    if expert_cache is not None:
        # IQ2_XS: 延迟打开归档（常驻权重已加载完，页缓存不再竞争）
        if iq2xs_dir:
            expert_cache._ensure_iq2xs_archive()
            # 归档打开后重新调整 VRAM 容量（IQ2_XS 专家更小，需更多预留）
            expert_cache._vram_adjusted = False
            expert_cache.adjust_window_for_vram()

        cache_loaded = expert_cache.load_cache_state()
        total_freq = expert_cache.total_freq_entries()
        # IQ2_XS 归档加载（跳过 CPU SLRU，归档通过 mmap 按需读取）
        if iq2xs_dir:
            expert_cache.load_iq2xs_to_cpu()
        if cache_loaded and total_freq:
            print("[Warmup] Hot restart: using persisted LFU stats, skipping inference warmup")
            expert_cache.warmup_from_cache(model)
        else:
            print("[Warmup] No cache state found, running inference to collect LFU stats...")
            expert_cache.n_layers = 0
            warmup_prompts = [
                torch.tensor([[0, 1, 2]], dtype=torch.long, device="cuda"),
                torch.tensor([[100, 200, 300]], dtype=torch.long, device="cuda"),
                torch.tensor([[1000, 2000, 3000]], dtype=torch.long, device="cuda"),
            ]
            with torch.inference_mode():
                for wp in warmup_prompts:
                    try:
                        model.forward(wp, 0)
                    except Exception:
                        pass
            expert_cache.clear_gpu_cache()
            gc.collect()
            torch.cuda.empty_cache()
            expert_cache.n_layers = len(model.layers)
            total_freq = expert_cache.total_freq_entries()
            print(f"[Warmup] LFU stats collected: {total_freq} expert entries")
            expert_cache.warmup_from_cache(model)
            expert_cache.save_cache_state()

        # IQ2_XS: 设置 mmap 访问模式
        # 不使用 MADV_RANDOM（会禁止 OS 预读，导致大量 page fault）
        # 使用默认模式，让 OS 根据访问模式自动调整预读
        if iq2xs_dir and expert_cache._iq2xs_archive is not None:
            try:
                import mmap as _mmap
                expert_cache._iq2xs_archive._mmap.madvise(_mmap.MADV_WILLNEED)
                print("[IQ2_XS] mmap 模式: WILLNEED（允许 OS 预读）")
            except (AttributeError, OSError):
                pass

    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    expert_cache = getattr(model, '_expert_cache', None)
    if expert_cache is not None:
        expert_cache.save_cache_state()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--quant-type", type=str, default="fp4", choices=["fp4", "iq2xs"],
                        help="量化类型：fp4（默认）或 iq2xs")
    parser.add_argument("--iq2xs-dir", type=str, default="",
                        help="IQ2_XS 量化文件目录（为空则自动检测）")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, 
         args.max_new_tokens, args.temperature, args.quant_type, args.iq2xs_dir)
