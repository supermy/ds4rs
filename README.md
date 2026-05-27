# ds4.rs

DeepSeek V4 Flash 专用单机推理引擎，基于 96G DDR5 + RTX 5060 Ti 16G 定制开发。

## 性能指标

| 指标 | FP4 | IQ2_XS | 混合 (IQ2_XS+Q2_K) |
|------|-----|--------|---------------------|
| 推理速度 | 1.0-1.25 t/s (decode) | ~0.8-1.0 t/s (decode) | ~0.76 t/s (decode) |
| GPU VRAM | 12.6 GB (常驻 + 200专家缓存) | ~10 GB (常驻 + SLRU缓存) | ~10 GB (常驻 + SLRU缓存) |
| 系统内存 | ~6 GB (稳定) | ~80 GB (全量CPU专家) | ~84 GB (全量CPU专家) |
| 模型权重 | 149 GB (FP4) | ~76 GB (IQ2_XS) | ~80 GB (混合) |
| GPU 缓存策略 | LFU | SLRU (90% protected / 10% probation) | SLRU (90%/10%) |
| GPU 命中率 | ~6% | ~78% (16GB卡) / ~100% (24GB卡) | ~78% (16GB卡) |
| CPU 命中率 | N/A | 100% (全量pinned pool) | 100% (全量pinned pool) |
| CPU FFN 延迟 | N/A | ~2.7ms/专家 (AVX-512) | ~4.6ms/专家 (AVX-512 amd7600) |

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
```

## 量化类型

系统支持三种量化格式，通过 `--quant-type` 参数选择：

| 特性 | FP4 | IQ2_XS | 混合 (IQ2_XS+Q2_K) |
|------|-----|--------|---------------------|
| 比特率 | 4 bit/weight | 2.3125 bit/weight | gate/up 2.3125 + down ~2.7 bit |
| 模型大小 | ~149 GB | ~76 GB | ~80 GB |
| 精度 | 较高 | 较低 (MSE ~0.47) | 中等 (down 精度略优) |
| GPU缓存策略 | LFU | SLRU (90%/10%) | SLRU (90%/10%) |
| GPU命中率 | ~6% | ~78% (16GB) | ~78% (16GB) |
| CPU缓存 | 禁用 | 全量pinned pool (100%) | 全量pinned pool (100%) |
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

### GPU化路径

- **Compressor**: GEMM(FP32) → Group(GPU: gather+slice+ape) → Pool(GPU) → RMSNorm(GPU) → RoPE(GPU) → Hadamard(cuBLAS GEMM) → FP4-QDQ(GPU) → BF16(GPU)
- **Indexer**: GEMM(BF16) → q_proj后处理(GPU: RoPE+Hadamard+FP4-QDQ) → Score+TopK(GPU) → Causal Adjust(GPU) → D2D Scatter写入
- **KV Cache**: D2D行提取、环形缓冲区管理、checkpoint热启动
- **MoE Gate**: TopK路由(GPU)
- **IQ2_XS GEMM**: TileLang 融合算子，反量化+矩阵乘法单 kernel 完成

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
