"""将自定义 IQ2XS 归档转换为 GGUF 格式。

用法:
    python inference/convert_iq2xs_to_gguf.py --input /models/iq2xs/experts.iq2xs --output /models/iq2xs/experts.gguf

输出:
    - experts.gguf: GGUF 格式的专家权重
    - experts.idx.json: 侧载索引文件
"""
import os
import sys
import json
import struct
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iq2xs_archive import IQ2XSArchiveReader, MAGIC as IQ2XS_MAGIC, VERSION as IQ2XS_VERSION

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGML_TYPE_IQ2_XS = 17
GGML_TYPE_Q2_K = 10


def write_gguf_string(f, s: str):
    data = s.encode('utf-8')
    f.write(struct.pack('<Q', len(data)))
    f.write(data)


def write_gguf_metadata_uint32(f, key: str, value: int):
    write_gguf_string(f, key)
    f.write(struct.pack('<I', 4))
    f.write(struct.pack('<I', value))


def write_gguf_metadata_string(f, key: str, value: str):
    write_gguf_string(f, key)
    f.write(struct.pack('<I', 8))
    write_gguf_string(f, value)


def convert_iq2xs_to_gguf(input_path: str, output_path: str):
    print(f"[Convert] {input_path} -> {output_path}")
    
    reader = IQ2XSArchiveReader(input_path)
    
    n_layers = reader.n_layers
    n_experts = reader.n_experts
    n_weights = reader.n_weights
    
    print(f"[Convert] n_layers={n_layers}, n_experts={n_experts}, n_weights={n_weights}")
    
    tensor_infos = []
    data_buffers = []
    current_offset = 0
    
    for layer_id in range(n_layers):
        for expert_id in range(n_experts):
            for weight_type in range(n_weights):
                weight_name = ['w1', 'w2', 'w3'][weight_type]
                tensor_name = f"layers.{layer_id}.experts.{expert_id}.{weight_name}"
                
                result = reader.get_expert(layer_id, expert_id, weight_type)
                if result is None:
                    continue
                
                d, qs, scales, shape = result
                out_dim, in_dim = shape
                n_blocks = d.shape[0] * d.shape[1]
                
                d_data = d.astype('<f2').tobytes()
                qs_data = qs.astype('<u2').tobytes()
                scales_data = scales.astype('<u1').tobytes()
                tensor_data = d_data + qs_data + scales_data
                
                tensor_infos.append({
                    'name': tensor_name,
                    'dims': [out_dim, in_dim],
                    'ggml_type': GGML_TYPE_IQ2_XS,
                    'offset': current_offset,
                    'size': len(tensor_data),
                })
                
                data_buffers.append(tensor_data)
                current_offset += len(tensor_data)
    
    reader.close()
    
    print(f"[Convert] {len(tensor_infos)} tensors, {current_offset / 1024**3:.2f} GB data")
    
    with open(output_path, 'wb') as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensor_infos)))
        f.write(struct.pack('<Q', 4))
        
        write_gguf_metadata_string(f, "general.name", "DeepSeek-V4-Flash")
        write_gguf_metadata_string(f, "general.architecture", "deepseek")
        write_gguf_metadata_uint32(f, "deepseek.n_layers", n_layers)
        write_gguf_metadata_uint32(f, "deepseek.n_experts", n_experts)
        
        for info in tensor_infos:
            write_gguf_string(f, info['name'])
            f.write(struct.pack('<I', len(info['dims'])))
            for dim in info['dims']:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', info['ggml_type']))
            f.write(struct.pack('<Q', info['offset']))
        
        for data in data_buffers:
            f.write(data)
    
    total_size = os.path.getsize(output_path)
    print(f"[Convert] GGUF written: {output_path} ({total_size / 1024**3:.2f} GB)")
    
    index_path = output_path.replace('.gguf', '.idx.json')
    index = {
        'n_layers': n_layers,
        'n_experts': n_experts,
        'n_weights': n_weights,
        'quant_type': 'iq2_xs',
        'tensors': {info['name']: {
            'offset': info['offset'],
            'size': info['size'],
            'dims': info['dims'],
            'ggml_type': info['ggml_type'],
        } for info in tensor_infos}
    }
    
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"[Convert] Index written: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert IQ2XS archive to GGUF")
    parser.add_argument("--input", type=str, required=True, help="Input IQ2XS archive")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF file")
    args = parser.parse_args()
    
    convert_iq2xs_to_gguf(args.input, args.output)


if __name__ == "__main__":
    main()
