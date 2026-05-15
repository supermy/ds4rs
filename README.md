# ds4.rs

DeepSeek V4 Flash 专用单机推理引擎，基于 96G DDR5 + RTX 5060 Ti 16G 定制开发。

## 架构概览

```
┌─────────────────────────────────────────────┐
│  GPU VRAM (16GB)                            │
│  常驻: 非路由专家 + KV Cache + 热点路由专家  │
│  空闲槽位: 预取目标                          │
└──────────────────┬──────────────────────────┘
                │ PCIe 5.0 x8 (~16 GB/s)
┌──────────────────┴──────────────────────────┐
│  Host RAM (96GB DDR5)                       │
│  Pinned Memory Pool (SLRU Cache)            │
└──────────────────┬──────────────────────────┘
                │ SSD MMAP 预取
┌──────────────────┴──────────────────────────┐
│  SSD: 全量路由专家权重                       │
└─────────────────────────────────────────────┘
```

## 推理全流程

```
Input Tokens
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Transformer Layer (×61)                              │
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

### Compressor/Indexer 数据流

```
KV/Scores → Pool(softmax weighted) → RMSNorm(weighted) → RoPE → Hadamard → FP4-QDQ → BF16
                                                              ↓
                                                        Compressed KV Cache
```

```
Query → Q投影 → RoPE → Hadamard → FP4-QDQ
                                    ↓
Indexer: Score(Q, CompressedKV) × Weights → TopK → Causal Adjust → Index
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

### GPU化路径

- **Compressor**: Pool(GPU) → RMSNorm(GPU) → RoPE(GPU) → Hadamard(cuBLAS GEMM) → FP4-QDQ(GPU) → BF16(GPU)
- **Indexer**: Score+TopK(GPU) → Causal Adjust(GPU) → D2D Scatter写入
- **KV Cache**: D2D行提取、环形缓冲区管理、checkpoint热启动
- **MoE Gate**: TopK路由(GPU)

## 关键技术

### 混合稀疏注意力 (Hybrid Attention)

- **SWA**: 滑动窗口注意力，处理近期Token
- **CSA**: 压缩稀疏注意力，通过Compressor压缩历史KV
- **HCA**: 层级压缩注意力，跨层聚合信息

### Hyper-Connection (HC)

替代传统残差连接，通过可学习门控实现动态特征融合。

### MoE 三级缓存

路由专家权重按需加载，通过GPU/内存/SSD三级缓存隐藏PCIe传输延迟：
- GPU: LFU热点专家常驻
- 内存: Pinned Memory SLRU缓存
- SSD: MMAP全量专家预取

### FP4 量化

路由专家权重以FP4 e2m1fn格式存储(int8打包)，按行每32列一组FP8 e8m0fnu缩放因子。

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
```

## 依赖

- Rust 2021 Edition
- CUDA 12.9+
- cuBLAS
- TileLang >= 0.1.9 (内核编译)
- cudarc 0.17 (CUDA驱动)
- safetensors 0.7 (权重加载)
