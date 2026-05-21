# ds4.rs

DeepSeek V4 Flash 专用单机推理引擎，基于 96G DDR5 + RTX 5060 Ti 16G 定制开发。

## 性能指标

| 指标 | 值 |
|------|-----|
| 推理速度 | 1.0-1.25 t/s (decode) |
| GPU VRAM | 12.6 GB (常驻 + 200专家缓存) |
| 系统内存 | ~6 GB (稳定，无泄漏) |
| 模型权重 | 149 GB (FP4) |
| 缓存命中率 | L1 GPU 0% (warmup后提升) |

## 架构概览

```
┌─────────────────────────────────────────────┐
│  GPU VRAM (16GB)                            │
│  ┌─────────────────────────────────────┐    │
│  │ 常驻: 非路由专家 + KV Cache +       │    │
│  │       热点路由专家 (LFU Top-200)     │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ 空闲槽位: 非缓存专家计算 (~3GB)      │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                │ PCIe 5.0 x8 (~16 GB/s)
                │ Direct I/O (非 mmap)
┌──────────────────┴──────────────────────────┐
│  Host RAM (96GB DDR5)                       │
│  ┌─────────────────────────────────────┐    │
│  │ 模型常驻权重 + OS (~40GB)           │    │
│  │ L2 CPU 缓存: 禁用 (避免内存膨胀)     │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                │ safetensors 直接读取
┌──────────────────┴──────────────────────────┐
│  SSD: 全量路由专家权重 (149GB)              │
└─────────────────────────────────────────────┘
```

## 快速开始

### Python 推理 (inference/)

```bash
# 进入容器
docker exec -it ds4rs-dev bash

# 运行推理
cd /workspace/inference
python3 generate.py --ckpt-path /models --config /models/config.json \
    --interactive --max-new-tokens 100 --temperature 0.6

# 或使用 Python API
python3 -c "
from model import Transformer, ModelArgs
from generate import load_weights_streaming, generate
from transformers import AutoTokenizer
import json, torch

torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_device(0)

with open('/models/config.json') as f:
    raw = json.load(f)
# ... 字段映射见 generate.py main() ...

with torch.device('cpu'):
    model = Transformer(args)

load_weights_streaming(model, '/models', 0, 1)
torch.set_default_device('cuda')

tokenizer = AutoTokenizer.from_pretrained('/models')
tokens = tokenizer.encode('你好')
out = generate(model, [tokens], 50, tokenizer.eos_token_id, temperature=0.6)
print(tokenizer.decode(out[0]))
"
```

### Rust 推理 (待实现)

```bash
make build
./target/release/ds4_cli --model /models --prompt "Hello"
```

## 推理全流程

```
Input Tokens
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Transformer Layer (×43)                              │
│  ┌──────────┐   ┌───────────────────┐   ┌─────────┐ │
│  │ RMSNorm  │──▶│ Hybrid Attention  │──▶│ Residual│ │
│  └──────────┘   │ (SWA+CSA+HCA)     │   │ (HC)    │ │
│                 └───────────────────┘   └─────────┘ │
│  ┌──────────┐   ┌───────────────────┐   ┌─────────┐ │
│  │ RMSNorm  │──▶│ MoE FFN           │──▶│ Residual│ │
│  └──────────┘   │ (Shared+Routed)   │   │ (HC)    │ │
│                 └───────────────────┘   └─────────┘ │
└──────────────────────────────────────────────────────┘
    │
    ▼
  Output Logits
```

### 关键数据流

1. **Prefill**: Token序列 → RMSNorm → QKV投影 → Hybrid Attention(SWA滑动窗口 + CSA压缩注意力 + HCA层级注意力) → MoE(共享专家 + 路由专家)
2. **Decode**: 单Token → 同上路径，KV Cache环形缓冲区管理

### MoE 专家加载流程

```
MoE.forward(x, input_ids)
    │
    ├─▶ Gate 计算: weights, indices = gate(x, input_ids)
    │
    ├─▶ 统计激活专家: activated = torch.unique(indices).tolist()
    │
    ├─▶ _on_experts_needed(activated)  ← 三级缓存加载
    │       │
    │       ├─▶ L1 GPU 缓存命中? → 设置参数到 Expert (0ms)
    │       │
    │       └─▶ L3 SSD 读取? → 直接 I/O + H2D (~30ms)
    │
    ├─▶ 专家计算: for i in activated: y += expert(x)
    │
    └─▶ _on_experts_done(activated)  ← 删除非 GPU 缓存 Expert 对象
```

## 技术架构

### 模块结构

| 模块 | 文件 | 职责 |
|------|------|------|
| 配置 | `config.rs` | 模型超参数、硬件配置 |
| 权重 | `weight.rs` | SafeTensors权重加载 |
| 张量 | `tensor.rs` | CPU/GPU张量操作、D2D传输 |
| 量化 | `quant.rs` | FP4/FP8量化、Hadamard变换 |
| 位置编码 | `rope.rs` | RoPE预计算 |
| 层 | `layer.rs` | Transformer层前向传播 |
| 注意力 | `kv_cache.rs` | KV Cache环形缓冲区、热启动 |
| 压缩 | `compressor.rs` | KV压缩(GPU后处理管道) |
| 索引 | `indexer.rs` | 压缩KV索引(GPU因果掩码) |
| 门控 | `gate.rs` | MoE路由、GPU TopK |
| 专家 | `expert.rs` | 专家调度、三级缓存 |
| 模型 | `model.rs` | 整体前向逻辑 |
| 内核 | `tvm_ffi.rs` | TileLang内核加载与调用 |
| cuBLAS | `cublas.rs` | GEMM矩阵运算封装 |
| DLPack | `dlpack.rs` | 跨框架张量协议 |

### Python 推理模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 模型 | `inference/model.py` | Transformer、MoE、Attention 定义 |
| 生成 | `inference/generate.py` | 流式加载、两级缓存、生成循环 |
| 内核 | `inference/kernel.py` | TileLang GPU 内核封装 |
| 转换 | `inference/convert.py` | 权重格式转换 (FP4/FP8) |

### TileLang 内核

| 内核 | 功能 |
|------|------|
| `rmsnorm_*` | RMSNorm归一化 |
| `rmsnorm_f32_weighted_*` | 加权RMSNorm(FP32) |
| `compressor_pool_*` | Compressor池化 |
| `compressor_rope_f32_*` | Compressor专用RoPE |
| `fp4_qdq_f32_*` | FP4量化/反量化 |
| `cast_*` | BF16↔FP32类型转换 |
| `indexer_score_*` | Indexer评分+TopK |
| `indexer_causal_adjust_*` | 因果掩码+偏移调整 |
| `compressor_group_*` | Compressor分组(gather+slice+ape) |
| `scale_f32_*` | 元素级缩放 |
| `fp4_gemm_*` | FP4 GEMM (激活FP8 × 权重FP4) |
| `fp8_gemm_*` | FP8 GEMM |

### GPU化路径

- **Compressor**: GEMM(FP32) → Group(GPU: gather+slice+ape) → Pool(GPU) → RMSNorm(GPU) → RoPE(GPU) → Hadamard(cuBLAS GEMM) → FP4-QDQ(GPU) → BF16(GPU)
- **Indexer**: GEMM(BF16) → q_proj后处理(GPU: RoPE+Hadamard+FP4-QDQ) → Score+TopK(GPU) → Causal Adjust(GPU) → D2D Scatter写入
- **KV Cache**: D2D行提取、环形缓冲区管理、checkpoint热启动
- **MoE Gate**: TopK路由(GPU)

## 关键技术

### 混合稀疏注意力 (Hybrid Attention)

- **SWA**: 滑动窗口注意力，处理近期Token
- **CSA**: 压缩稀疏注意力，通过Compressor压缩历史KV
- **HCA**: 层级压缩注意力，跨层聚合信息

### Hyper-Connection (HC)

替代传统残差连接，通过可学习门控实现动态特征融合。

### MoE 两级缓存

路由专家权重按需加载，通过GPU/SSD两级缓存隐藏PCIe传输延迟：

| 缓存层 | 容量 | 策略 | 延迟 |
|--------|------|------|------|
| L1 GPU | 200 专家 (~2.5GB) | LFU | 0ms |
| L3 SSD | 全量 11008 专家 | 直接 I/O | ~30ms |

**关键设计决策**：
- **L2 CPU 缓存禁用**: pinned memory 会锁定物理页，96GB 内存中模型常驻权重 + OS 已占 ~40GB，剩余空间不足以缓存大量专家。且旧方案将 GPU 参数移到 CPU 保留导致内存泄漏。
- **Expert 对象卸载策略**: 非缓存专家计算完后直接删除 Expert 对象（设为 None），而非移到 CPU。下次需要时由 `_ensure_expert` 重建空壳，缓存重新加载权重。
- **直接 I/O 而非 mmap**: safetensors mmap 会导致 OS 页缓存膨胀（46 个分片 × 3.4GB ≈ 156GB 潜力），改用 `seek/read` 直接读取所需字节。

### FP4 量化

路由专家权重以FP4 e2m1fn格式存储(int8打包)，按行每32列一组FP8 e8m0fnu缩放因子。

**存储格式**：
- 数据类型：FP4 e2m1fn (1符号位 + 2指数位 + 1尾数位)
- 打包密度：2个FP4值 → 1个int8字节
- 缩放因子：FP8 e8m0fnu，按行、每32列一组

### 内存安全机制

- **内存水位监控**: 每 10 个 token 检查 `/proc/meminfo`，可用内存 < 8GB 时触发紧急清理
- **紧急清理**: 释放 safetensors header 缓存、强制 GC、清空 CUDA 缓存
- **自动 warmup**: 启动时执行短推理收集 LFU 统计，将 top-200 热点专家常驻 GPU

## 构建与测试

```bash
# 开发环境 (Docker容器)
docker exec -it ds4rs-dev bash

# 编译检查
make check

# 构建
make build

# 运行测试
make test

# 详细测试输出
make dev-test

# Python 推理测试
cd /workspace/inference
python3 generate.py --ckpt-path /models --config /models/config.json \
    --interactive --max-new-tokens 50 --temperature 0.6
```

## 依赖

- Rust 2021 Edition
- CUDA 12.9+
- cuBLAS
- TileLang >= 0.1.9 (内核编译)
- cudarc 0.17 (CUDA驱动)
- safetensors 0.7 (权重加载)
- PyTorch >= 2.5 (Python 推理)
- transformers (分词器)
