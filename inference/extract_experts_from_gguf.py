"""从 llama.cpp GGUF 文件提取路由专家权重，保存为 GGUF 格式。

源 GGUF 格式：
  - 256 专家拼接为单个 tensor: blk.{L}.ffn_{gate|up|down}_exps.weight
  - 支持多种量化类型：IQ2_XXS, IQ2_XS, Q2_K, Q4_K

输出 GGUF 格式：
  - 每个专家权重为独立 tensor: layers.{L}.experts.{E}.{w1|w2|w3}
  - 保留原始量化格式（零损失拷贝，不做反量化/重量化）
  - 附带 sidecar JSON 索引

用法：
  python inference/extract_experts_from_gguf.py \
    --input /data/ai/models/gguf/DeepSeek-V4-Flash-Q2_K.gguf \
    --output /data/ai/ds4rs/gguf/experts_q2k.gguf

  python inference/extract_experts_from_gguf.py \
    --input /mnt/shared_data/.../DeepSeek-V4-Flash-IQ2XXS-w2Q2K-....gguf \
    --output /data/ai/ds4rs/gguf/experts_mixed.gguf
"""
import os
import sys
import struct
import json
import argparse
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

QK_K = 256

# GGML 类型 ID
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 28

# GGUF 类型名 -> GGML 类型 ID
GGUF_TYPE_TO_GGML = {
    'Q2_K': GGML_TYPE_Q2_K,
    'Q3_K': GGML_TYPE_Q3_K,
    'Q4_K': GGML_TYPE_Q4_K,
    'IQ2_XXS': GGML_TYPE_IQ2_XXS,
    'IQ2_XS': GGML_TYPE_IQ2_XS,
}

# GGML 类型 ID -> block 字节数
GGML_BLOCK_BYTES = {
    GGML_TYPE_Q2_K: 84,    # d:2 + dmin:2 + scales:16 + qs:64
    GGML_TYPE_Q3_K: 110,   # d:2 + dmin:2 + scales:12 + qs:64 + hmask:2 + qs_h:28
    GGML_TYPE_Q4_K: 144,   # d:2 + dmin:2 + scales:12 + qs:128
    GGML_TYPE_IQ2_XXS: 66, # d:2 + qs:64
    GGML_TYPE_IQ2_XS: 74,  # d:2 + qs:64 + scales:8
}

# GGUF 格式常量
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF KV 类型
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT32 = 4


def _write_string(f, s: str):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _pad_to_alignment(f, alignment=GGUF_ALIGNMENT):
    pos = f.tell()
    padded = (pos + alignment - 1) // alignment * alignment
    if padded > pos:
        f.write(b"\x00" * (padded - pos))


def extract_experts_to_gguf(input_path: str, output_path: str):
    """从源 GGUF 提取路由专家权重，拆分为独立 tensor 写入新 GGUF。"""
    from gguf import GGUFReader

    print(f"[Extract] 读取 GGUF: {input_path}")
    reader = GGUFReader(input_path)

    # 收集专家 tensor
    expert_tensors = {}
    for t in reader.tensors:
        if 'exps' in t.name and 'shared' not in t.name and 'shexp' not in t.name:
            expert_tensors[t.name] = t

    # 确定层数
    layer_ids = set()
    for name in expert_tensors:
        parts = name.split('.')
        layer_ids.add(int(parts[1]))
    n_layers = max(layer_ids) + 1
    n_experts = 256

    print(f"[Extract] 层数: {n_layers}, 专家数: {n_experts}")
    print(f"[Extract] 专家 tensor 数: {len(expert_tensors)}")

    # GGUF weight_key -> 归档 weight_name
    weight_key_map = {
        'ffn_gate_exps': 'w1',
        'ffn_up_exps': 'w3',
        'ffn_down_exps': 'w2',
    }

    # 第一遍：收集所有 tensor 信息（名称、维度、类型、数据偏移）
    tensor_infos = []  # [(name, dims, ggml_type, data_bytes)]
    data_tmp = output_path + ".data.tmp"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    start_time = time.time()
    current_data_offset = 0

    with open(data_tmp, 'wb') as df:
        for layer_id in range(n_layers):
            layer_start = time.time()

            for weight_key, weight_name in weight_key_map.items():
                tensor_name = f"blk.{layer_id}.{weight_key}.weight"
                if tensor_name not in expert_tensors:
                    print(f"  [WARN] 缺少 {tensor_name}")
                    continue

                t = expert_tensors[tensor_name]
                data = t.data  # numpy memmap

                # 自动检测量化类型
                type_name = t.tensor_type.name
                if type_name not in GGUF_TYPE_TO_GGML:
                    print(f"  [WARN] 不支持的量化类型: {type_name} ({tensor_name})")
                    continue
                ggml_type = GGUF_TYPE_TO_GGML[type_name]
                block_bytes = GGML_BLOCK_BYTES[ggml_type]

                ne0, ne1, ne2 = [int(x) for x in t.shape]
                out_dim = ne0
                in_dim = ne1

                for eid in range(n_experts):
                    expert_data = data[eid].ravel().tobytes()
                    n_blocks = out_dim * in_dim // QK_K
                    assert len(expert_data) == n_blocks * block_bytes, \
                        f"Size mismatch: {len(expert_data)} != {n_blocks * block_bytes}"

                    tensor_name_out = f"layers.{layer_id}.experts.{eid}.{weight_name}"
                    tensor_infos.append((tensor_name_out, [out_dim, in_dim], ggml_type, len(expert_data)))

                    df.write(expert_data)
                    # 对齐填充
                    padding = (GGUF_ALIGNMENT - (len(expert_data) % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
                    if padding:
                        df.write(b'\x00' * padding)

            layer_time = time.time() - layer_start
            written_gb = os.path.getsize(data_tmp) / 1024**3
            if (layer_id + 1) % 5 == 0 or layer_id == 0:
                print(f"  Layer {layer_id:2d}/{n_layers}: {layer_time:.1f}s ({written_gb:.2f} GB written)")

    # 第二遍：组装 GGUF 文件
    print(f"[Extract] 组装 GGUF 文件...")
    n_tensors = len(tensor_infos)

    # 计算元数据大小
    # KV 对
    kv_pairs = [
        ("general.architecture", GGUF_TYPE_STRING, "deepseek-v4"),
        ("general.name", GGUF_TYPE_STRING, "DeepSeek-V4-Flash-experts"),
        ("expert.n_layers", GGUF_TYPE_UINT32, n_layers),
        ("expert.n_experts", GGUF_TYPE_UINT32, n_experts),
        ("expert.n_weights", GGUF_TYPE_UINT32, 3),
    ]

    meta_size = 4 + 4 + 8 + 8  # magic + version + n_tensors + n_kv
    for key, kv_type, value in kv_pairs:
        meta_size += 8 + len(key.encode("utf-8")) + 4
        if kv_type == GGUF_TYPE_STRING:
            meta_size += 8 + len(value.encode("utf-8"))
        elif kv_type == GGUF_TYPE_UINT32:
            meta_size += 4

    for name, dims, ggml_type, _ in tensor_infos:
        meta_size += 8 + len(name.encode("utf-8"))
        meta_size += 4  # n_dims
        meta_size += 8 * len(dims)
        meta_size += 4  # type
        meta_size += 8  # offset

    data_offset = (meta_size + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT

    with open(output_path, 'wb') as f:
        # 文件头
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<q", n_tensors))
        f.write(struct.pack("<q", len(kv_pairs)))

        # KV 对
        for key, kv_type, value in kv_pairs:
            _write_string(f, key)
            f.write(struct.pack("<i", kv_type))
            if kv_type == GGUF_TYPE_STRING:
                _write_string(f, value)
            elif kv_type == GGUF_TYPE_UINT32:
                f.write(struct.pack("<I", value))

        # Tensor 信息
        current_offset = 0
        for name, dims, ggml_type, data_len in tensor_infos:
            _write_string(f, name)
            f.write(struct.pack("<I", len(dims)))
            for dim in dims:
                f.write(struct.pack("<q", dim))
            f.write(struct.pack("<i", ggml_type))
            f.write(struct.pack("<Q", current_offset))
            current_offset += data_len
            padding = (GGUF_ALIGNMENT - (data_len % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
            current_offset += padding

        # 填充到数据区
        _pad_to_alignment(f, GGUF_ALIGNMENT)

        # 数据区：从临时文件拷贝
        with open(data_tmp, 'rb') as df:
            while True:
                chunk = df.read(64 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    os.remove(data_tmp)

    # 写 sidecar JSON 索引
    idx_path = output_path + ".idx.json"
    idx = {
        "n_layers": n_layers,
        "n_experts": n_experts,
        "n_weights": 3,
        "quant_types": {},
        "tensors": {},
    }
    for name, dims, ggml_type, _ in tensor_infos:
        idx["tensors"][name] = {
            "dims": dims,
            "ggml_type": ggml_type,
        }
    # 记录每层的量化类型
    for layer_id in range(n_layers):
        for weight_key, weight_name in weight_key_map.items():
            tensor_name = f"blk.{layer_id}.{weight_key}.weight"
            if tensor_name in expert_tensors:
                type_name = expert_tensors[tensor_name].tensor_type.name
                key = f"layer{layer_id}.{weight_name}"
                idx["quant_types"][key] = type_name

    with open(idx_path, 'w') as f:
        json.dump(idx, f, indent=2)

    total_size = os.path.getsize(output_path)
    total_time = time.time() - start_time
    print(f"\n[Extract] 完成！")
    print(f"  输出: {output_path}")
    print(f"  索引: {idx_path}")
    print(f"  大小: {total_size / 1024**3:.2f} GB")
    print(f"  张量数: {n_tensors}")
    print(f"  耗时: {total_time:.1f}s")


def verify_gguf(output_path: str, input_path: str, n_check: int = 5):
    """验证输出 GGUF 与源 GGUF 的一致性。"""
    from gguf import GGUFReader

    print(f"\n[Verify] 验证输出 GGUF...")
    reader_src = GGUFReader(input_path)
    reader_out = GGUFReader(output_path)

    # 构建输出 GGUF 的 tensor 索引
    out_tensors = {t.name: t for t in reader_out.tensors}

    weight_key_map = {'w1': 'ffn_gate_exps', 'w2': 'ffn_down_exps', 'w3': 'ffn_up_exps'}

    import random
    n_layers = 43
    n_experts = 256
    ok = 0
    fail = 0

    for _ in range(n_check):
        lid = random.randint(0, n_layers - 1)
        eid = random.randint(0, n_experts - 1)
        wn = random.choice(['w1', 'w2', 'w3'])

        out_name = f"layers.{lid}.experts.{eid}.{wn}"
        if out_name not in out_tensors:
            print(f"  [FAIL] 缺少 {out_name}")
            fail += 1
            continue

        out_data = out_tensors[out_name].data.ravel().tobytes()

        # 从源 GGUF 读取
        src_key = f'blk.{lid}.{weight_key_map[wn]}.weight'
        src_tensor = None
        for t in reader_src.tensors:
            if t.name == src_key:
                src_tensor = t
                break
        if src_tensor is None:
            print(f"  [FAIL] 源 GGUF 缺少 {src_key}")
            fail += 1
            continue

        src_data = src_tensor.data[eid].ravel().tobytes()

        if out_data == src_data:
            print(f"  [OK] {out_name}: {len(out_data)} bytes")
            ok += 1
        else:
            print(f"  [FAIL] {out_name}: 数据不匹配")
            fail += 1

    print(f"[Verify] {ok} OK, {fail} FAIL")


def main():
    parser = argparse.ArgumentParser(description="从 GGUF 提取路由专家权重到 GGUF")
    parser.add_argument("--input", type=str, required=True, help="源 GGUF 文件路径")
    parser.add_argument("--output", type=str, default="", help="输出 GGUF 文件路径")
    parser.add_argument("--verify", action="store_true", help="提取后验证")
    parser.add_argument("--n-verify", type=int, default=5, help="验证检查数量")
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf')
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(output_dir, f"{base}_experts.gguf")

    extract_experts_to_gguf(args.input, output_path)

    if args.verify:
        verify_gguf(output_path, args.input, args.n_verify)


if __name__ == "__main__":
    main()
