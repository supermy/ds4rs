"""IQ2_XS 预量化脚本 — 首次启动时将 FP4 权重量化为 IQ2_XS 归档。

流程：
1. 扫描所有 safetensors shard
2. 提取专家权重（FP4 打包）
3. 量化：CUDA GPU（C 算法并行版）或 C CPU
4. 打包成归档文件 experts.iq2xs
5. 显示进度和统计

量化速度：
  CUDA 量化（--use-gpu）：~8-15 experts/s（RTX 5060 Ti，批量+流水线）
  C CPU（默认）：~0.3 experts/s

用法：
    python inference/prequant_iq2xs.py --ckpt-path /models --output /models/iq2xs/experts.iq2xs
    python inference/prequant_iq2xs.py --ckpt-path /models --output /models/iq2xs/experts.iq2xs --use-gpu
"""
import os
import sys
import time
import argparse
import ctypes
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iq2xs_archive import IQ2XSArchiveWriter

# FP4 解码表 (float32, 用于 GPU 解码)
FP4_TABLE_F32 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                               0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                              dtype=torch.float32)

# FP4 解码表 (float32, 用于 CPU 解码)
FP4_TABLE_NP = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

QK_K = 256  # IQ2_XS super-block 大小


def decode_fp4_gpu(packed_tensor: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    """在 GPU 上解码 FP4 打包数据为 float32。

    每个 uint8 包含 2 个 FP4 值：低 4 位是偶数列，高 4 位是奇数列。
    """
    arr = packed_tensor.view(torch.uint8)
    lo = (arr & 0x0F).long()
    hi = ((arr >> 4) & 0x0F).long()
    table = FP4_TABLE_F32.to(arr.device)
    lo_val = table[lo]
    hi_val = table[hi]
    return torch.stack([lo_val, hi_val], dim=-1).reshape(out_dim, in_dim)


def decode_fp4_cpu(packed: bytes) -> np.ndarray:
    """在 CPU 上解码 FP4 打包数据为 float32（IQ2_XS 量化需要 FP32 精度）。"""
    arr = np.frombuffer(packed, dtype=np.uint8)
    lo = arr & 0x0F
    hi = (arr >> 4) & 0x0F
    return np.stack([FP4_TABLE_NP[lo], FP4_TABLE_NP[hi]], axis=-1).reshape(-1)


def find_expert_weights(ckpt_path: str) -> dict:
    """扫描所有 shard，提取专家权重信息。

    使用 get_slice().get_shape() 获取形状，不加载张量数据。

    返回: {shard_path: [(layer_id, expert_id, weight_name, tensor_name, shape), ...]}
    """
    experts = {}
    shard_files = sorted(Path(ckpt_path).glob("model-*.safetensors"))

    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt") as f:
            shard_experts = []
            for name in f.keys():
                if '.ffn.experts.' in name and '.weight' in name:
                    parts = name.split('.')
                    layer_id = int(parts[1])
                    expert_id = int(parts[4])
                    weight_name = parts[5]
                    shape = tuple(f.get_slice(name).get_shape())
                    shard_experts.append((layer_id, expert_id, weight_name, name, shape))
        if shard_experts:
            experts[str(shard_path)] = shard_experts

    return experts


def _quantize_expert_cpu(f32: np.ndarray, n_blocks: int) -> dict:
    """CPU 回退量化（GPU 不可用时使用）。

    参数:
        f32: float32 权重数组，形状 [n_blocks * QK_K]
        n_blocks: IQ2_XS block 数量
    """
    from iq2xs_c_wrapper import quantize_iq2_xs_gpu_via_cpu, get_qk_k
    qk_k = get_qk_k()
    flat_gpu = torch.from_numpy(f32.reshape(n_blocks, qk_k)).cuda()
    d, qs, scales = quantize_iq2_xs_gpu_via_cpu(flat_gpu)
    return {
        "d": d.cpu().numpy().astype(np.float16),
        "qs": qs.cpu().numpy().astype(np.uint16),
        "scales": scales.cpu().numpy().astype(np.uint8),
    }


def _parse_iq2xs_batch(output_cpu: np.ndarray, n_blocks: int,
                        blocks_per_expert: int) -> list:
    """向量化解析批量 block_iq2xs 输出。

    block_iq2_xs 布局: d(2) + qs[32](64) + scales[8](8) = 74 bytes

    返回: [(d_np, qs_np, scales_np), ...] 每个专家一组
    """
    n_experts = n_blocks // blocks_per_expert

    # 向量化解析: 一次性提取所有 d, qs, scales
    # 使用 numpy 的 stride trick 避免 Python 循环
    # 将 1D 字节数组视为 (n_blocks, 74) 的结构化视图
    raw = output_cpu.reshape(n_blocks, 74)

    # d: 每块前 2 字节, float16
    d_all = raw[:, :2].copy().view(np.float16).reshape(n_blocks)

    # qs: 每块 2~66 字节, 32 个 uint16
    qs_all = raw[:, 2:66].copy().view(np.uint16).reshape(n_blocks, 32)

    # scales: 每块 66~74 字节, 8 个 uint8
    scales_all = raw[:, 66:74].copy().reshape(n_blocks, 8)

    # 按专家切分
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


def prequant_iq2xs(
    ckpt_path: str,
    output_path: str,
    n_layers: int,
    n_experts: int,
    progress_interval: int = 10,
    use_gpu_flag: bool = False,
    batch_size: int = 0,
) -> None:
    """预量化所有专家权重为 IQ2_XS 归档。"""
    # 量化方式优先级: CUDA > C CPU
    quant_mode = "c"
    if use_gpu_flag:
        try:
            from iq2xs_cuda_quant import is_cuda_quantize_available, build_cuda_quantize
            if is_cuda_quantize_available():
                quant_mode = "cuda"
            elif torch.cuda.is_available():
                try:
                    build_cuda_quantize()
                    quant_mode = "cuda"
                except Exception as e:
                    print(f"[量化] CUDA 量化编译失败: {e}，使用 C 量化")
        except ImportError:
            print("[量化] CUDA 量化模块不可用，使用 C 量化")
    else:
        print("[量化] 使用 C 量化（与 llama.cpp 完全一致）")

    # 自动计算批量大小: 显存可用 ~14GB, 每个专家 ~32MB(float32), 双缓冲 2x
    # 实测 bs=32 时 GPU 量化速度 24.6 e/s (已饱和), bs=16 只有 7.8 e/s
    if quant_mode == "cuda":
        if batch_size == 0:
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            # 每个专家 ~32MB float32 输入 + ~8MB 输出, 双缓冲需要 2x
            # 保留 2GB 给 CUDA 上下文和查找表
            batch_size = max(16, min(64, int((vram_mb - 2048) / (40 * 2))))
        print(f"[量化] 使用 CUDA 批量量化 (batch_size={batch_size})")

    print("=" * 70)
    print("IQ2_XS 预量化")
    print("=" * 70)
    print(f"输入: {ckpt_path}")
    print(f"输出: {output_path}")
    print(f"量化方式: {quant_mode}")

    # 扫描专家权重
    print("\n[扫描] 查找专家权重...")
    experts = find_expert_weights(ckpt_path)

    total_experts = sum(len(v) for v in experts.values())
    print(f"[扫描] 找到 {total_experts} 个专家权重 ({len(experts)} 个 shard)")

    # 统计层数和专家数
    all_layer_ids = set()
    all_expert_ids = set()
    for shard_experts in experts.values():
        for layer_id, expert_id, _, _, _ in shard_experts:
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

    if quant_mode == "cuda":
        # ====================================================================
        # CUDA 批量 + 双缓冲流水线
        # ====================================================================
        # 核心优化:
        #   1. 批量: 一次 kernel 调用处理 N 个同形状专家，压满 GPU
        #   2. 双缓冲: GPU 量化 batch A 时，CPU 准备 batch B
        #   3. 按形状分组: w1/w3 形状相同可合并，w2 单独处理
        #   4. GPU FP4 解码: 避免 CPU→GPU 传输瓶颈
        # ====================================================================
        from iq2xs_cuda_quant import _find_lib
        from concurrent.futures import ThreadPoolExecutor
        lib = _find_lib()

        stream_compute = torch.cuda.Stream()
        stream_xfer = torch.cuda.Stream()

        # 用 event 追踪 GPU 量化完成, 避免 synchronize 阻塞 CPU
        compute_done_event = torch.cuda.Event(enable_timing=False)

        # 线程池: 在后台线程准备下一批 CPU 数据, 与 GPU 量化并行
        cpu_executor = ThreadPoolExecutor(max_workers=1)

        # 批量缓冲
        class BatchBuffer:
            """批量缓冲: 存放一组同形状专家的输入/输出"""
            def __init__(self):
                self.f32_gpu = None       # [N*rows, cols] float32 GPU
                self.output_gpu = None    # [total_blocks*74] uint8 GPU
                self.meta_list = []       # [(layer_id, expert_id, weight_idx, out_dim, in_dim), ...]
                self.n_experts = 0
                self.blocks_per_expert = 0
                self.total_blocks = 0
                self.ready = False

            def clear(self):
                if self.f32_gpu is not None:
                    del self.f32_gpu
                    self.f32_gpu = None
                if self.output_gpu is not None:
                    del self.output_gpu
                    self.output_gpu = None
                self.meta_list = []
                self.n_experts = 0
                self.ready = False

        buf = [BatchBuffer(), BatchBuffer()]

        def load_batch_cpu(tensor_cache, expert_list):
            """CPU 端: 从缓存中获取并拼接 FP4 数据和 scale"""
            packed_parts = [tensor_cache[name] for _, _, _, name in expert_list]
            # 同时加载对应的 scale 张量
            scale_parts = []
            for _, _, wn, name in expert_list:
                scale_name = name.replace('.weight', '.scale')
                if scale_name in tensor_cache:
                    scale_parts.append(tensor_cache[scale_name])
                else:
                    scale_parts.append(None)
            return torch.stack(packed_parts, dim=0), scale_parts

        def prepare_batch(buf_idx, packed_cpu, scale_parts, expert_list, shape_key):
            """GPU 端: 上传 + 解码 + 应用 scale + 分配输出"""
            b = buf[buf_idx]
            b.clear()

            out_dim, packed_in_dim = shape_key
            in_dim = packed_in_dim * 2
            blocks_per_expert = out_dim * in_dim // QK_K
            n = len(expert_list)

            # GPU 端: 上传 + 解码 + 应用 scale (在传输 stream 上异步执行)
            with torch.cuda.stream(stream_xfer):
                packed_gpu = packed_cpu.cuda(non_blocking=True)
                del packed_cpu

                # GPU 上批量解码 FP4
                arr = packed_gpu.view(torch.uint8)
                lo = (arr & 0x0F).long()
                hi = ((arr >> 4) & 0x0F).long()
                table = FP4_TABLE_F32.to(arr.device)
                lo_val = table[lo]
                hi_val = table[hi]
                decoded = torch.stack([lo_val, hi_val], dim=-1).reshape(n * out_dim, in_dim).contiguous().float()
                del packed_gpu, lo, hi, lo_val, hi_val

                # 应用 FP4 scale (float8_e8m0fnu)
                # scale shape: [out_dim, in_dim/32], 每 32 个元素共享一个 scale
                if scale_parts[0] is not None:
                    scale_scales = torch.stack(scale_parts, dim=0)  # [n, out_dim, in_dim/32]
                    scale_gpu = scale_scales.cuda(non_blocking=True).view(torch.float8_e8m0fnu).float()
                    # repeat_interleave 使 scale 与 decoded 对齐
                    scale_expanded = scale_gpu.reshape(n * out_dim, -1).repeat_interleave(32, dim=1)
                    decoded = decoded * scale_expanded
                    del scale_scales, scale_gpu, scale_expanded

                b.f32_gpu = decoded

                # 分配输出缓冲
                b.total_blocks = n * blocks_per_expert
                b.output_gpu = torch.empty(b.total_blocks * 74, dtype=torch.uint8, device='cuda')

            b.meta_list = [
                (lid, eid, {'w1': 0, 'w2': 1, 'w3': 2}[wn], out_dim, in_dim)
                for lid, eid, wn, _ in expert_list
            ]
            b.n_experts = n
            b.blocks_per_expert = blocks_per_expert
            b.ready = False

        def launch_quantize(buf_idx):
            """在计算 stream 上启动 GPU 量化 (异步)"""
            b = buf[buf_idx]
            with torch.cuda.stream(stream_compute):
                stream_compute.wait_stream(stream_xfer)
                # decoded 形状: [n * out_dim, in_dim]
                # 正确参数: nrow = n * out_dim, n_per_row = in_dim
                nrow = b.f32_gpu.shape[0]  # n * out_dim
                n_per_row = b.f32_gpu.shape[1]  # in_dim
                src_ptr = ctypes.c_void_p(b.f32_gpu.data_ptr())
                dst_ptr = ctypes.c_void_p(b.output_gpu.data_ptr())
                lib.iq2xs_quantize_ffi(src_ptr, dst_ptr, nrow, n_per_row, ctypes.c_void_p(0))
                # 记录 event: GPU 量化完成时触发
                compute_done_event.record(stream_compute)
            b.ready = True

        def consume_batch(buf_idx):
            """等待 GPU 完成, 解析结果, 写入归档"""
            b = buf[buf_idx]
            if not b.ready:
                return 0

            # 用 event 等待 GPU 量化完成 (比 synchronize 更精确)
            compute_done_event.synchronize()

            # 解析输出
            output_cpu = b.output_gpu.cpu().numpy()
            results = _parse_iq2xs_batch(output_cpu, b.total_blocks, b.blocks_per_expert)

            # 写入归档
            count = 0
            for i, (lid, eid, widx, od, idim) in enumerate(b.meta_list):
                d_np, qs_np, scales_np = results[i]
                writer.add_expert(lid, eid, widx, d_np, qs_np, scales_np, od, idim)
                count += 1

            b.clear()
            return count

        # 逐 shard 处理
        for shard_idx, (shard_path, shard_experts) in enumerate(experts.items()):
            print(f"\n[Shard {shard_idx+1}/{len(experts)}] {os.path.basename(shard_path)}")

            with safe_open(shard_path, framework="pt") as sf:
                # 预加载所有专家张量到 CPU 内存 (避免重复磁盘 I/O)
                t_preload = time.time()
                tensor_cache = {}
                for layer_id, expert_id, weight_name, tensor_name, shape in shard_experts:
                    tensor_cache[tensor_name] = sf.get_tensor(tensor_name)
                    # 同时加载对应的 scale 张量
                    scale_name = tensor_name.replace('.weight', '.scale')
                    if scale_name not in tensor_cache and scale_name in sf.keys():
                        tensor_cache[scale_name] = sf.get_tensor(scale_name)
                preload_ms = (time.time() - t_preload) * 1000
                print(f"  预加载 {len(tensor_cache)} 个张量: {preload_ms:.0f}ms")

                # 按形状分组
                shape_groups = defaultdict(list)
                for layer_id, expert_id, weight_name, tensor_name, shape in shard_experts:
                    shape_groups[shape].append(
                        (layer_id, expert_id, weight_name, tensor_name))

                # 对每个形状组，按 batch_size 分批
                all_batches = []
                for shape_key, group in shape_groups.items():
                    for i in range(0, len(group), batch_size):
                        all_batches.append((shape_key, group[i:i+batch_size]))

                # 双缓冲流水线处理 (CPU 后台预加载 + GPU 量化并行)
                if not all_batches:
                    continue

                # 预加载第一个 batch
                shape_key, batch = all_batches[0]
                t_prep = time.time()
                packed_cpu, scale_parts = load_batch_cpu(tensor_cache, batch)
                prepare_batch(0, packed_cpu, scale_parts, batch, shape_key)
                del packed_cpu
                t_launch = time.time()
                launch_quantize(0)
                buf_idx = 1

                # 预提交下一批 CPU 加载到后台线程
                next_cpu_future = None
                if len(all_batches) > 1:
                    next_shape, next_batch = all_batches[1]
                    next_cpu_future = cpu_executor.submit(load_batch_cpu, tensor_cache, next_batch)

                for batch_i in range(1, len(all_batches)):
                    shape_key, batch = all_batches[batch_i]

                    # 1. 等待后台 CPU 加载完成, 上传到 GPU
                    t_wait = time.time()
                    if next_cpu_future is not None:
                        packed_cpu, scale_parts = next_cpu_future.result()
                    else:
                        packed_cpu, scale_parts = load_batch_cpu(tensor_cache, batch)

                    t_upload = time.time()
                    prepare_batch(buf_idx, packed_cpu, scale_parts, batch, shape_key)
                    del packed_cpu

                    # 2. 消费上一个 batch (等 GPU 完成 + 解析 + 写归档)
                    t_consume = time.time()
                    prev_idx = 1 - buf_idx
                    count = consume_batch(prev_idx)
                    quantized += count
                    t_consumed = time.time()

                    # 3. 启动当前 batch 的 GPU 量化
                    launch_quantize(buf_idx)

                    # 4. 提交下一批 CPU 加载到后台线程 (与 GPU 量化并行)
                    next_batch_i = batch_i + 1
                    if next_batch_i < len(all_batches):
                        next_shape, next_batch = all_batches[next_batch_i]
                        next_cpu_future = cpu_executor.submit(load_batch_cpu, tensor_cache, next_batch)
                    else:
                        next_cpu_future = None

                    # 5. 交替 buffer
                    buf_idx = 1 - buf_idx

                    # 进度显示 (含详细计时)
                    now = time.time()
                    if now - last_progress_time >= progress_interval:
                        elapsed = now - start_time
                        rate = quantized / elapsed
                        remaining = (total_experts - quantized) / rate if rate > 0 else 0
                        vram_used = torch.cuda.memory_allocated() / 1024**3
                        # 显示各阶段耗时
                        prep_ms = (t_upload - t_wait) * 1000
                        consume_ms = (t_consumed - t_consume) * 1000
                        print(f"\r  [{quantized}/{total_experts}] {quantized/total_experts*100:.1f}% "
                              f"| {rate:.1f} e/s "
                              f"| prep {prep_ms:.0f}ms consume {consume_ms:.0f}ms "
                              f"| VRAM {vram_used:.1f}GB", end='', flush=True)
                        last_progress_time = now

                # 消费最后一个 batch
                last_idx = 1 - buf_idx
                count = consume_batch(last_idx)
                quantized += count

            # shard 间释放显存
            torch.cuda.empty_cache()

    else:
        # ====================================================================
        # C CPU 回退路径
        # ====================================================================
        for shard_idx, (shard_path, shard_experts) in enumerate(experts.items()):
            print(f"\n[Shard {shard_idx+1}/{len(experts)}] {os.path.basename(shard_path)}")

            with safe_open(shard_path, framework="pt") as sf:
                for layer_id, expert_id, weight_name, tensor_name, shape in shard_experts:
                    tensor = sf.get_tensor(tensor_name)
                    out_dim, packed_in_dim = tensor.shape
                    in_dim = packed_in_dim * 2

                    raw = tensor.numpy().tobytes()
                    f32 = decode_fp4_cpu(raw)

                    # 应用 FP4 scale (float8_e8m0fnu)
                    scale_name = tensor_name.replace('.weight', '.scale')
                    if scale_name in sf.keys():
                        scale_tensor = sf.get_tensor(scale_name)
                        scale_f32 = scale_tensor.view(torch.float8_e8m0fnu).float().numpy()
                        # scale shape: [out_dim, in_dim/32], 每 32 个元素共享一个 scale
                        scale_expanded = np.repeat(scale_f32, 32, axis=1).reshape(-1)
                        f32 = f32 * scale_expanded[:f32.size]

                    n_elements = f32.size
                    n_blocks = (n_elements + QK_K - 1) // QK_K
                    if n_elements % QK_K != 0:
                        padded = np.zeros(n_blocks * QK_K, dtype=np.float32)
                        padded[:n_elements] = f32
                        f32 = padded

                    result = _quantize_expert_cpu(f32, n_blocks)
                    d_np = result["d"]
                    qs_np = result["qs"]
                    scales_np = result["scales"]

                    n_blocks = d_np.shape[0]
                    writer.add_expert(layer_id, expert_id,
                                     {'w1': 0, 'w2': 1, 'w3': 2}[weight_name],
                                     d_np, qs_np, scales_np, out_dim, in_dim)

                    quantized += 1
                    total_bytes += n_blocks * 74

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
    print(f"  量化方式:   {quant_mode}")
    print(f"  输出文件:   {output_path}")


def main():
    parser = argparse.ArgumentParser(description="IQ2_XS 预量化")
    parser.add_argument("--ckpt-path", type=str, required=True, help="模型 checkpoint 目录")
    parser.add_argument("--output", type=str, default="", help="输出归档文件路径")
    parser.add_argument("--n-layers", type=int, default=0, help="层数（0=自动检测）")
    parser.add_argument("--n-experts", type=int, default=0, help="每层专家数（0=自动检测）")
    parser.add_argument("--use-gpu", action="store_true", help="使用 GPU 加速量化")
    parser.add_argument("--batch-size", type=int, default=0, help="批量大小（0=自动）")
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        output_path = os.path.join(args.ckpt_path, "iq2xs", "experts.iq2xs")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prequant_iq2xs(args.ckpt_path, output_path, args.n_layers, args.n_experts,
                   use_gpu_flag=args.use_gpu, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
