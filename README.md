# ds4.rs

DeepSeek V4 Flash 专用单机推理引擎，基于 96G DDR5 + RTX 5060 Ti 16G 定制开发。

## 性能指标

| 指标 | FP4 | IQ2_XS | 混合 (IQ2_XXS+Q2_K) | 混合 + GPU FFN + P1层内重叠 |
|------|-----|--------|---------------------|--------------------------|
| 推理速度 | 1.0-1.25 t/s (decode) | ~0.8-1.0 t/s (decode) | ~0.76 t/s (decode) | **~0.7 t/s** (imatrix GGUF) |
| GPU VRAM | 12.6 GB (常驻 + 200专家缓存) | ~10 GB (常驻 + SLRU缓存) | ~10 GB (常驻 + SLRU缓存) | ~14 GB (常驻 + 535专家缓存) |
| 系统内存 | ~6 GB (稳定) | ~80 GB (全量CPU专家) | ~84 GB (全量CPU专家) | ~84 GB (全量CPU专家) |
| 模型权重 | 149 GB (FP4) | ~76 GB (IQ2_XS) | ~80 GB (混合) | ~80 GB (混合) |
| GPU 缓存策略 | LFU | SLRU (90% protected / 10% probation) | SLRU (90%/10%) | LFU+LRU 二次触发准入 + P1层内重叠 |
| GPU 命中率 | ~6% | ~78% (16GB卡) / ~100% (24GB卡) | ~78% (16GB卡) | ~37-47% (535专家, prefetch=4L) |
| CPU 命中率 | N/A | 100% (全量pinned pool) | 100% (全量pinned pool) | 100% (全量pinned pool) |
| CPU FFN 延迟 | ~3.9ms/专家 (AVX-512 amd7600) | ~2.7ms/专家 (AVX-512) | ~4.6ms/专家 (AVX-512 amd7600) | ~0.49ms/专家 (异步，与GPU计算重叠) |
| GPU FFN 延迟 | — | — | — | ~0.49ms/专家 (hit, M=1) |
| 每专家上传 | — | — | — | ~7MB (量化格式直传) |

## 架构概览

```
┌─────────────────────────────────────────────┐
│  GPU VRAM (16GB)                            │
│  ┌─────────────────────────────────────┐    │
│  │ 常驻: 非路由专家 + KV Cache +       │    │
│  │       热点路由专家 (SLRU 90%/10%)    │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ 空闲槽位: 非缓存专家计算 (~3GB)      │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                │ PCIe 5.0 x8 (~16 GB/s)
                │ 异步预取流水线
┌──────────────────┴──────────────────────────┐
│  Host RAM (96GB DDR5)                       │
│  ┌─────────────────────────────────────┐    │
│  │ 模型常驻权重 + OS (~40GB)           │    │
│  │ FP4: CPU缓存禁用(避免内存膨胀)       │    │
│  │ IQ2_XS: 全量pinned pool (~74GB)      │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                │ mmap + OS页缓存 (IQ2_XS)
                │ seek/read 直接I/O (FP4)
┌──────────────────┴──────────────────────────┐
│  SSD: 全量路由专家权重                       │
│  FP4: 149GB safetensors                     │
│  IQ2_XS: ~80GB .iq2xs 归档 (mmap)           │
└─────────────────────────────────────────────┘
```

## 快速开始

### Python 推理 (inference/)

```bash
# 进入容器
docker exec -it ds4rs-dev bash

# FP4 推理 (默认) — 官方参数: temp=1.0, top_p=1.0
cd /workspace/inference
python3 generate.py --ckpt-path /models --config /models/config.json \
    --interactive --max-new-tokens 100 --temperature 1.0 --top-p 1.0

# IQ2_XS 推理 (首次启动自动预量化) — 量化补偿参数: temp=0.6, top_p=0.95, min_p=0.01
python3 generate.py --ckpt-path /models --config /models/config.json \
    --quant-type iq2xs --interactive --max-new-tokens 100 \
    --temperature 0.6 --top-p 0.95 --min-p 0.01

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
out = generate(model, [tokens], 50, tokenizer.eos_token_id, temperature=1.0)  # FP4用1.0, IQ2_XS用0.6
print(tokenizer.decode(out[0]))
"
```

### Rust 推理 (待实现)

```bash
make build
./target/release/ds4_cli --model /models --prompt "Hello"
```

## ⚠️ 推理参数（重要）

**不同量化类型必须使用不同的推理参数，否则输出质量会严重下降。**

量化会引入精度损失，低比特量化的 logits 分布更尖锐（方差缩小），需要通过温度和采样策略补偿：

| 量化类型 | temperature | top_p | min_p | 说明 |
|---------|-------------|-------|-------|------|
| **FP4** | **1.0** | **1.0** | — | 精度最高，保持原始 logits 分布，无需采样补偿 |
| **IQ2_XS** | **0.6** | **0.95** | **0.01** | 2.3bit 量化精度损失大，需降温+截断抑制低概率 token |
| **混合 (IQ2_XS+Q2_K)** | **0.6** | **0.95** | **0.01** | 同 IQ2_XS，down 权重进一步量化，采样策略一致 |

**为什么 IQ2_XS 需要降温？** IQ2_XS 量化误差使 logits 峰值偏高（"过度自信"），temperature=0.6 重新展平分布，top_p=0.95 + min_p=0.01 双重截断过滤尾部噪声 token。FP4 精度足够，保持 temperature=1.0 即可还原模型原始采样行为。

```bash
# FP4 — 官方推荐参数
python3 generate.py --ckpt-path /models --config /models/config.json \
    --interactive --temperature 1.0 --top-p 1.0

# IQ2_XS — 量化补偿参数
python3 generate.py --ckpt-path /models --config /models/config.json \
    --quant-type iq2xs --interactive --temperature 0.6 --top-p 0.95 --min-p 0.01

# 混合量化 — 同 IQ2_XS 参数
python3 generate.py --ckpt-path /models --config /models/config.json \
    --quant-type mixed --interactive --temperature 0.6 --top-p 0.95 --min-p 0.01

# 混合量化 + GPU FFN — 量化格式直传GPU（M=1 decode 无加速，适用于 batch≥4）
python3 generate.py --ckpt-path /models --config /models/config.json \
    --quant-type mixed --gpu-ffn --interactive --temperature 0.6 --top-p 0.95 --min-p 0.01 \
    --prefetch-count 50 --prefetch-layers 4
```

## 量化类型

系统支持三种量化格式，通过 `--quant-type` 参数选择：

| 特性 | FP4 | IQ2_XS | 混合 (IQ2_XXS+Q2_K) |
|------|-----|--------|---------------------|
| 比特率 | 4 bit/weight | 2.3125 bit/weight | gate/up 2.0625 + down 2.5625 bit |
| 模型大小 | ~149 GB | ~76 GB | ~80 GB |
| 精度 | 较高 | 较低 (MSE ~0.47) | 中等 (down 精度略优) |
| CPU FFN 延迟 | ~3.9ms/专家 | — | ~4.6ms/专家 |
| CPU FFN 带宽利用率 | ~70% | — | ~50% |
| GPU缓存策略 | LFU | SLRU (90%/10%) | SLRU (90%/10%) |
| GPU命中率 | ~6% | ~78% (16GB) | ~78% (16GB) |
| CPU缓存 | 部分pinned pool | 全量pinned pool (100%) | 全量pinned pool (100%) |
| SSD加载 | seek/read 直接I/O | mmap + OS页缓存 | mmap + OS页缓存 |
| 首次启动 | 直接推理 | 自动预量化(需等待) | 需预量化(见下方) |
| 推荐场景 | 显存充足、追求精度 | 内存充足、节省磁盘 | 内存充足、down层精度优先 |

### 推理流程对比

```
┌─────────────────────────────────────────────────────────────┐
│                      FP4 推理流程                            │
├─────────────────────────────────────────────────────────────┤
│  x (BF16) → act_quant → x_fp8 + scale                       │
│           ↓                                                  │
│  fp4_gemm(x_fp8, scale, weight_fp4, weight_scale)           │
│           ↓                                                  │
│  输出 (BF16)                                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    IQ2_XS 推理流程                            │
├─────────────────────────────────────────────────────────────┤
│  x (BF16)                                                    │
│           ↓                                                  │
│  iq2xs_gemm(x, qs, scales, d)  ← TileLang融合算子            │
│           ↓                                                  │
│  输出 (BF16)                                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│          混合量化 + GPU FFN + P1层内重叠 推理流程             │
├─────────────────────────────────────────────────────────────┤
│  x (BF16)                                                    │
│           ↓                                                  │
│  Gate 计算 → 分离 GPU hit / CPU miss                         │
│           ↓                                                  │
│  ┌─ GPU hit → iq2xxs_gemm(x, qs, d) [gate/up]               │
│  │           + q2k_gemm(mid, qs, scales, d, dmin) [down]     │
│  │           + SwiGLU融合 → 输出 (BF16)                      │
│  │                                                           │
│  └─ CPU miss → compute_ffn_async (线程池提交，不阻塞)        │
│                                                              │
│  P1 层内重叠时间线：                                          │
│    1. 提交所有 CPU miss 到异步线程池                          │
│    2. 计算所有 GPU hit（与 CPU 并行）                         │
│    3. 等待 CPU 异步结果 (future.result())                     │
│    4. 合并到 y，返回完整 FFN 输出 → hc_post                  │
│                                                              │
│  辅助优化：                                                   │
│    + 二次触发准入：第2次命中才异步上传到GPU                    │
│    + 预测性预取：未来4层高频专家异步上传（prefetch=4L）        │
│    + 频率持久化（热启动直接加载）                              │
│    + imatrix 校准量化（down 层精度提升）                      │
│                                                              │
│  注：跨层专家延迟(P0)在 HC 架构下不可行（Sinkhorn非线性）     │
└─────────────────────────────────────────────────────────────┘
```

### IQ2_XS 预量化流程

首次使用 `--quant-type iq2xs` 时，系统自动执行预量化：

```
FP4 权重 → GPU FP4解码(float32) → CUDA批量量化(IQ2_XS) → .iq2xs归档文件
```

- 默认使用 CUDA 批量量化（C 算法 GPU 并行版，~24 e/s）
- GPU 不可用时回退到 CPU（C 实现通过 ctypes 调用，~0.3 e/s）
- 量化完成后打包为 `.iq2xs` 归档文件，后续启动直接 mmap 加载
- 量化期间显示进度、速度、VRAM 使用和预计剩余时间
- 支持断点续传：归档文件存在时跳过预量化

### 混合量化流程 (IQ2_XS + Q2_K)

混合量化策略：gate/up 权重使用 IQ2_XS（2.3125 bit），down 权重使用 Q2_K（~2.7 bit），通过 imatrix 校准提升 down 层量化精度：

```bash
# 混合量化（需 imatrix 校准数据）
python inference/prequant_mixed_iq2xxs_q2k.py \
    --ckpt-path /models \
    --imatrix /models/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat \
    --output /workspace/gguf/experts_mixed.gguf \
    --batch-size 256
```

- 输出目录：`/workspace/gguf/`
- imatrix 校准数据：`/models/imatrix/`
- 混合量化总大小：~80GB（vs IQ2_XS ~76GB，down 层精度提升）
- 量化速率：~30 e/s (CUDA kernel)

预量化性能（RTX 5060 Ti 16G，33792 个专家权重）：

| 阶段 | 速度 | 说明 |
|------|------|------|
| C CPU 量化 | 0.3 e/s | 基线 |
| CUDA 量化 (1 thread/block) | 1.1 e/s | kernel 并行度低 |
| CUDA 量化 (16 threads/block) | 2.1 e/s | 子块并行 |
| 双缓冲流水线 | 8.8 e/s | CPU/GPU 重叠 |
| CUDA 批量量化 | 23.7 e/s | 批量+向量化解析 |
| **CUDA 混合量化 (最终版)** | **30 e/s** | IQ2_XS + Q2_K 混合，imatrix校准 |

详细技术文档见 [docs/cuda_quantize.md](docs/cuda_quantize.md)

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
    │       ├─▶ L2 CPU pinned pool命中? → H2D DMA传输 (~5ms)
    │       │
    │       └─▶ L3 SSD 读取? → mmap + H2D (~30ms)
    │
    ├─▶ 路由预测预取: RoutePredictor → 预取L+1层差集到GPU
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
| 模型 | `inference/model.py` | Transformer、MoE、Attention 定义，量化类型路由 |
| 生成 | `inference/generate.py` | 流式加载、两级缓存、生成循环、`--quant-type` 参数 |
| 内核 | `inference/kernel.py` | TileLang GPU 内核封装 |
| 转换 | `inference/convert.py` | 权重格式转换 (FP4/FP8) |
| 缓存 | `inference/expert_cache.py` | 三级缓存(GPU/CPU/SSD)、SLRU策略(90%/10%)、step保护、路由预测预取、W-TinyLFU(可选) |
| IQ2_XS GEMM | `inference/iq2xs_gemm_tilelang.py` | TileLang IQ2_XS 融合GEMM算子 |
| IQ2_XS 归档 | `inference/iq2xs_archive.py` | .iq2xs 归档读写、mmap 加载 |
| IQ2_XS 预量化 | `inference/prequant_iq2xs.py` | FP4→IQ2_XS 预量化流水线 |
| IQ2_XS CUDA量化 | `inference/iq2xs_cuda_quant.py` | CUDA 批量量化封装（C 算法 GPU 并行版） |
| IQ2_XS C封装 | `inference/iq2xs_c_wrapper.py` | Python ctypes 调用 C 量化实现 |
| 混合预量化 | `inference/prequant_mixed_iq2xxs_q2k.py` | IQ2_XS + Q2_K 混合预量化（gate/up IQ2_XS, down Q2_K, imatrix校准） |
| 混合专家池 | `inference/rust_cpu_expert.py` | MixedQuantExpertPool: CPU FFN + GPU FFN、二次触发准入SLRU、预测性预取、量化格式直传、频率持久化 |
| 混合GEMM | `tilelang/mixed_quant_gemm.py` | TileLang IQ2_XXS + Q2_K 融合GEMM算子（反量化+矩阵乘法单kernel） |

### C/FFI 层

| 文件 | 职责 |
|------|------|
| `csrc/iq2_xs.h` | IQ2_XS 量化/反量化完整 C 实现 (llama.cpp) |
| `csrc/iq2_xs_bridge.c` | Rust FFI 桥接层，FP16 转换和 FFI 接口 |
| `csrc/iq2_xs.cu` | IQ2_XS CUDA GPU 量化（16 threads/block、常量内存、批量启动） |

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
| `iq2xs_gemm_*` | IQ2_XS 融合GEMM (反量化+矩阵乘法) |
| `iq2xxs_gemm_*` | IQ2_XXS 融合GEMM (256-entry grid, aux32解码) |
| `q2k_gemm_*` | Q2_K 融合GEMM (4-bit packed scales, 2-halves qs) |
| `mixed_quant_ffn` | 混合量化FFN (gate/up IQ2_XXS + down Q2_K + SwiGLU) |

### GPU化路径

- **Compressor**: GEMM(FP32) → Group(GPU: gather+slice+ape) → Pool(GPU) → RMSNorm(GPU) → RoPE(GPU) → Hadamard(cuBLAS GEMM) → FP4-QDQ(GPU) → BF16(GPU)
- **Indexer**: GEMM(BF16) → q_proj后处理(GPU: RoPE+Hadamard+FP4-QDQ) → Score+TopK(GPU) → Causal Adjust(GPU) → D2D Scatter写入
- **KV Cache**: D2D行提取、环形缓冲区管理、checkpoint热启动
- **MoE Gate**: TopK路由(GPU)
- **IQ2_XS GEMM**: TileLang 融合算子，反量化+矩阵乘法单 kernel 完成
- **混合量化 GPU FFN**: 量化格式直传 GPU（~7MB/专家），TileLang IQ2_XXS + Q2_K 融合 GEMM，二次触发准入 SLRU，预测性预取（prefetch=4L），频率持久化

## 优化策略汇总

所有已探索的优化策略，按状态分类：

### ✅ 已实现

| # | 优化策略 | 版本 | 效果 | 说明 |
|---|---------|------|------|------|
| 1 | **P1 层内重叠** | v0.9.13 | CPU-GPU 并行 | CPU miss 异步提交 → GPU hit 计算 → 等 CPU 结果，避免跨层 HC 校正 |
| 2 | **SLRU 缓存 (90%/10%)** | v0.9.0 | GPU 命中率 78% | Protected 段保护热点专家，probation 段接纳新专家，step+layer 双重保护 |
| 3 | **二次触发准入** | v0.9.10 | 减少缓存颠簸 | 第 1 次 Gate 选中仅记录频率，第 2 次命中才异步上传到 GPU |
| 4 | **预测性预取 (prefetch=4L)** | v0.9.10 | 命中率 47.4% | 未来 4 层高频专家异步上传，L3-L4 是甜蜜点 |
| 5 | **GPU FFN 混合量化** | v0.9.9 | 量化格式直传 | IQ2_XXS + Q2_K 融合 GEMM，~7MB/专家直传 GPU（vs BF16 ~24MB） |
| 6 | **imatrix GGUF 校准** | v0.9.12 | 输出连贯性↑ | imatrix 校准量化 down 层，推理速度 0.7 t/s，GPU 命中率 36.9% |
| 7 | **CPU FFN AVX-512 优化** | v0.9.7 | 单专家 4.59ms | 256-bit maddubs 管线、双 ib32 融合、memcpy 批量读取、scale 融入 madd |
| 8 | **FP4 CPU FFN** | v0.9.8 | 单专家 3.93ms | x_split + permutex2var + E8M0 scale，比 IQ2_XXS 快 15% |
| 9 | **IQ2_XS 量化流水线** | v0.9.0 | 模型 76GB | CUDA 批量量化 23.7 e/s，归档格式 v2 零拷贝读取 |
| 10 | **混合量化 (IQ2_XXS+Q2_K)** | v0.9.6 | 模型 80GB | gate/up IQ2_XXS 抢速度，down Q2_K 保质量，CUDA 混合量化 30 e/s |
| 11 | **三级缓存架构** | v0.7.0 | GPU/CPU/SSD | L1 GPU SLRU + L2 CPU pinned pool + L3 SSD mmap |
| 12 | **路由预测预取** | v0.9.0 | 隐藏 PCIe 延迟 | RoutePredictor 统计层间专家共现，差集预取减少带宽浪费 |
| 13 | **频率持久化** | v0.9.10 | 热启动秒级 | freq JSON 保存/加载，跳过 warmup 直接进入高频缓存 |
| 14 | **GPU 化后处理管道** | v0.2.0-v0.3.0 | 消除 D2H 热点 | Compressor/Indexer/KV Cache 全 GPU 路径，消除 ~48MB D2H/H2D 往返 |
| 15 | **内存安全机制** | v0.8.0 | 防止 OOM | 内存水位监控、紧急清理、Expert 对象卸载策略 |
| 16 | **归档格式 v2 零拷贝** | v0.9.3 | 消除 3 次 copy | d/qs/scales 分离连续存储，mmap → pinned memory 直通 |
| 17 | **route_weight 修复** | v0.9.12 | 修复输出重复 | CPU FFN 内部已乘 route_weight，外部不再重复乘 |

### ❌ 不可行

| # | 优化策略 | 原因 | 说明 |
|---|---------|------|------|
| 1 | **P0 跨层专家延迟** | HC Sinkhorn 非线性 | `hc_pre` 的 Sinkhorn 归一化非线性放大误差，`hc_pre(x+δ) ≠ hc_pre(x)+hc_pre(δ)`，延迟合并导致输出重复。尝试了直接合并、post 缩放校正、comb 一阶校正均失败 |
| 2 | **热/冷双路 mul_mat_id** | 架构不匹配 | llama.cpp 的 mul_mat_id 要求所有专家同格式，GPU 热专家(BF16)与 CPU 冷专家(量化)格式不同，无法统一调用 |
| 3 | **W-TinyLFU 缓存** | 命中率更低 | 实测 71% < SLRU 78%，MoE 访问模式稳定不需要频率衰减，Window 段浪费 5% 容量 |
| 4 | **6 专家并行 (CPU)** | DDR5 带宽瓶颈 | 2 路并行反而更慢（59.9ms vs 串行 28.4ms），多线程争抢带宽降低 DDR burst 效率 |
| 5 | **GPU FFN (M=1 decode)** | kernel 启动开销 | TileLang kernel 启动开销占主导（3 次 kernel），CPU FFN AVX-512 对 M=1 已足够快，GPU 优势仅在 batch≥4 |

### ⏳ 待实现

| # | 优化策略 | 来源 | 预期效果 | 说明 |
|---|---------|------|---------|------|
| 1 | **P2 CPU-GPU 异步流水线** | 内部设计 | 隐藏 CPU 等待 | CPU FFN 与 GPU Attention 跨层重叠，需解决 HC 校正问题 |
| 2 | **P4 融合 MoE 算子** | 内部设计 | 减少 kernel 启动 | 将 gate+up+SwiGLU+down 融合为单个 TileLang kernel |
| 3 | **P5 Cache-Friendly 分块** | 内部设计 | 提高 L1/L2 命中 | 按管线顺序排列权重（gate+up 连续，down 尾随），L3 预热 d 数组触发硬件预取 |
| 4 | **Dynamic Expert Update** | KTransformers | 动态缓存更新 | 运行时动态替换 GPU 缓存中的专家，基于实时路由统计 |
| 5 | **Token-wise Prefetch** | SMOE (IPDPS 2026) | decode +20.9% | 按 token 级别预取专家，比层级别更精细 |
| 6 | **PreSched 跨层调度** | LayerScope (ICS 2026) | 吞吐 +141% | LLaPor 预测器 + 跨层调度 + AsyncIO |
| 7 | **FP4 CPU 路由** | 内部设计 | FP4 全 CPU 路径 | FP4 CPU FFN 已实现(3.93ms)，但 GPU 缓存策略和预取尚未适配 |
| 8 | **NUMA-Aware** | KTransformers | 多 socket 优化 | NUMA 感知的内存分配和线程绑定，单 socket 暂不需要 |
| 9 | **Layerwise Prefill** | KTransformers | prefill 加速 | prefill 阶段按层分配不同策略（GPU 密集层 vs CPU 密集层） |
| 10 | **全 IQ2_XXS 量化** | 内部分析 | down 投影 -26% | gate/up/down 均用 IQ2_XXS，down 从 Q2_K 1.95ms 降至 ~1.45ms |
| 11 | **IQ3_XXS down 投影** | AGENTS.md 推荐 | 精度敏感场景 | "查表精度型" 0.8ms，精度高于 IQ2_XXS，速度适中 |

### 优化技术来源

| 来源 | 关键技术 | 适用性 |
|------|---------|--------|
| **KTransformers (SOSP 2025)** | Expert Deferral、Dynamic Expert Update、NUMA-Aware、Layerwise Prefill、Native Precision Backend | Expert Deferral 在 HC 架构下不可行；其余待评估 |
| **SMOE (IPDPS 2026)** | Token-wise Prefetch | 待实现，需适配 HC 架构 |
| **LayerScope (ICS 2026)** | LLaPor 预测器、PreSched 跨层调度、AsyncIO | 待评估，跨层调度需解决 HC 校正 |
| **llama.cpp** | IQ2_XS/IQ2_XXS/Q2_K 量化格式、mul_mat_id、热/冷双路 | 量化格式已采用；mul_mat_id 双路不适用 |
| **内部优化** | P1 层内重叠、二次触发准入、SLRU 90%/10%、预测性预取、AVX-512 深度优化 | 已实现 |

## 关键技术

### 混合稀疏注意力 (Hybrid Attention)

- **SWA**: 滑动窗口注意力，处理近期Token
- **CSA**: 压缩稀疏注意力，通过Compressor压缩历史KV
- **HCA**: 层级压缩注意力，跨层聚合信息

### Hyper-Connection (HC)

替代传统残差连接，通过可学习门控实现动态特征融合。

### MoE 三级缓存

路由专家权重按需加载，通过GPU/CPU/SSD三级缓存隐藏PCIe传输延迟：

#### FP4 缓存策略

| 缓存层 | 容量 | 策略 | 延迟 |
|--------|------|------|------|
| L1 GPU | 200 专家 (~2.5GB) | LFU | 0ms |
| L2 CPU | 禁用 | - | - |
| L3 SSD | 全量 11008 专家 | 直接 I/O | ~30ms |

#### IQ2_XS 缓存策略

| 缓存层 | 容量 | 策略 | 延迟 |
|--------|------|------|------|
| L1 GPU | ~642 专家 (~4.5GB, 16GB卡) | SLRU (90% protected / 10% probation) | 0ms |
| L2 CPU | 全量 11008 专家 (~74GB) | 全量 pinned pool | ~5ms (H2D) |
| L3 SSD | 全量专家 | mmap + OS页缓存 | ~30ms |

**GPU 命中率**：16GB 卡 ~78%，24GB 卡可达 ~100%（受 VRAM 容量限制）

**关键设计决策**：
- **FP4 L2 CPU 缓存禁用**: pinned memory 会锁定物理页，96GB 内存中模型常驻权重 + OS 已占 ~40GB，剩余空间不足以缓存大量专家
- **IQ2_XS CPU 全量 pinned pool**: IQ2_XS 每专家仅 ~7MB，全量 11008 专家约 74GB，96GB 内存可容纳，CPU 命中率 100%
- **SLRU 90% protected**: MoE 推理访问模式稳定，90% 容量保护热点专家，仅 10% 留给新专家
- **Step 级别专家保护**: 当前 step 已访问的专家不被淘汰（缓存容量 ≥ 2× 工作集时自动启用）
- **SLRU > W-TinyLFU**: 实测 W-TinyLFU 命中率 71% < SLRU 78%，MoE 访问模式稳定，准入策略无意义
- **Expert 对象卸载策略**: 非缓存专家计算完后直接删除 Expert 对象（设为 None），而非移到 CPU
- **FP4 直接 I/O 而非 mmap**: safetensors mmap 会导致 OS 页缓存膨胀（46 个分片 × 3.4GB ≈ 156GB 潜力）
- **IQ2_XS 使用 mmap**: 归档文件按专家索引组织，mmap 零拷贝加载，由 OS 页缓存管理

### SLRU 缓存策略

Segmented LRU 将缓存分为 protected 段和 probation 段：

- **Protected 段 (90%)**: 被多次访问的热点专家，不会被轻易淘汰
- **Probation 段 (10%)**: 新进入缓存的专家，再次访问后晋升到 protected 段
- **Step 保护**: 当前 step 已访问的专家不被淘汰（缓存容量 ≥ 2× 工作集时启用）
- **层保护**: 当前层专家不被淘汰
- **淘汰顺序**: 先淘汰 probation 段最久未访问的（跳过受保护的），再淘汰 protected 段

相比 LFU，SLRU 对访问模式变化更敏感。相比 W-TinyLFU，SLRU 在 MoE 场景下命中率更高（78% vs 71%），因为 MoE 访问模式稳定，不需要准入策略过滤。

### 路由预测预取

利用 MoE 推理中相邻层 topk 专家的相关性，预测并预取下一层可能需要的专家：

```
GPU 计算 L 层 → CPU 预取 L+1 层差集到 GPU → GPU DMA 异步传输
```

- **RoutePredictor**: 基于历史路由结果统计层间专家共现频率
- **异步预取**: 预取失败不影响主流程
- **差集预取**: 仅预取不在 GPU 缓存中的专家，减少 PCIe 带宽浪费

### FP4 量化

路由专家权重以FP4 e2m1fn格式存储(int8打包)，按行每32列一组FP8 e8m0fnu缩放因子。

**存储格式**：
- 数据类型：FP4 e2m1fn (1符号位 + 2指数位 + 1尾数位)
- 打包密度：2个FP4值 → 1个int8字节
- 缩放因子：FP8 e8m0fnu，按行、每32列一组

### IQ2_XS 量化

2.3125 bit/weight 超低比特量化，源自 llama.cpp GGUF 格式。

**存储格式**：
- 数据结构：`block_iq2_xs`（74 字节 / 256 元素）
- 编码：grid 查找 + 符号位 + scale，每个 super-block 独立量化
- 查找表：`iq2xs_grid[512]`、`ksigns_iq2xs[128]`、`kmask_iq2xs[8]`
- 归档格式：自定义二进制格式，含文件头、偏移量表、mmap 零拷贝加载

### 内存安全机制

- **内存水位监控**: 每 10 个 token 检查 `/proc/meminfo`，可用内存 < 8GB 时触发紧急清理
- **紧急清理**: 释放 safetensors header 缓存、强制 GC、清空 CUDA 缓存
- **自动 warmup**: 启动时执行短推理收集统计，将热点专家常驻 GPU
- **硬件检测**: 启动时检测 GPU 显存和系统内存，推荐合适的量化类型

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

# Python FP4 推理测试 — 官方参数: temp=1.0, top_p=1.0
cd /workspace/inference
python3 generate.py --ckpt-path /models --config /models/config.json \
    --interactive --max-new-tokens 50 --temperature 1.0 --top-p 1.0

# Python IQ2_XS 推理测试 — 量化补偿参数: temp=0.6, top_p=0.95, min_p=0.01
python3 generate.py --ckpt-path /models --config /models/config.json \
    --quant-type iq2xs --interactive --max-new-tokens 50 \
    --temperature 0.6 --top-p 0.95 --min-p 0.01

# Python 混合量化 + GPU FFN — 量化格式直传GPU推理
python3 generate.py --ckpt-path /models --config /models/config.json \
    --quant-type mixed --gpu-ffn --interactive --max-new-tokens 50 \
    --temperature 0.6 --top-p 0.95 --min-p 0.01
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
