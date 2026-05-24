# Agent 备忘录

`ds4.rs` 是 DeepSeek V4 Flash 的专用单机推理引擎，基于 96G DDR5 + RTX5060Ti16G 定制开发。
目标：构建小巧、可读、高性能的 Rust 代码库，实现本地推理 Agent 自由。

## 目标

### 核心约束：资源受限

单机 16GB VRAM vs 11008 专家 × 109MB = 1.2TB 权重 → **显存只能装 0.4% 专家**。
这是所有设计决策的根因：缓存策略不是优化，而是生存。

### 设计哲学：二八法则 × 马太效应

MoE 专家激活呈幂律分布：20% 专家承载 80% 激活，且越热的专家越容易被再次激活（马太效应）。
三级缓存机制贯彻这一规律：

| 层级 | 容量 | 覆盖 | 命中率 | 延迟 | 策略 |
|------|------|------|--------|------|------|
| L1 GPU VRAM | ~642 专家 (4.5GB) | 头部 6% | ~78% | 0.2ms | SLRU 锁定热专家，马太效应保证命中率 |
| L2 CPU RAM | 11008 专家 (74GB) | 全量 100% | ~22% | 2.7ms | Pinned Pool 全量常驻，零 miss |
| L3 SSD | 全量归档 | 兜底 | 0% | 30ms | mmap，仅启动时加载 |

**二八法则**：GPU 只需缓存 6% 专家即可覆盖 78% 激活 → 有限 VRAM 的最优分配
**马太效应**：热专家越用越热 → SLRU protected 段自动锁定高频专家 → 命中率随推理递增
**延迟倒挂**：CPU FFN (2.7ms) < DMA (5.0ms) → GPU miss 走 CPU 而非 DMA，反直觉但正确

### MoE 特点

MoE 权重呈"低秩共享底座 + 正交特化专家"结构，decode 阶段算术强度极低，属内存受限负载。
延迟 ∝ 加载的 unique 专家数，缓存策略本质是用空间换带宽。

### 硬件资源

| 组件 | 规格 | 用途 |
|------|------|------|
| GPU | RTX 5060 Ti 16GB | Attention + Shared Expert + 热专家 FFN |
| CPU | Ryzen 5 7600 6C/12T, AVX-512 | 冷专家 FFN (AVX-512 SIMD) |
| RAM | DDR5-4800 48GB×2 = 96GB | 全量专家 pinned pool (~74GB) |
| SSD | NVMe | 专家归档 (mmap, 0% 命中率) |

### 量化配置

| 量化类型 | 专家总大小 | CPU FFN 延迟 | GPU FFN 延迟 |
|---------|-----------|-------------|-------------|
| FP4 (e2m1 + E8M0 scale) | 150GB | 4.1ms | ~1.15ms |
| IQ2_XS | 76GB | 2.7ms | 1.24ms |
| Q2K+IQ2_XSS | 80GB | 1.92ms | — |

- 默认 IQ2_XS，支持FP4 （`--quant-type=[iq2xs|fp4]`）
- 启动时检测硬件配置，提示推荐量化类型

## 缓存策略

核心：GPU hot + CPU cold 双路 MoE，Gate 选中即计数，统一频率驱动预取。
GPU miss 走 CPU FFN (2.7ms) 而非 DMA (5.0ms)。

### 内存布局

```
GPU 16GB 常驻：
├─ Attention + Shared Expert + Norm    (~8GB)
├─ KV Cache                            (~2-4GB)
├─ 热门专家池 SLRU                     (~4.5GB, ~642专家, ~78% 命中)
└─ 应急缓冲                            (~1-2GB)

CPU 96GB：
├─ Pinned Pool: 全量 11008 专家 IQ2_XS (~74GB, 100% 命中)
├─ Rust SLRU (IQ2_XS 2.7ms, FP4 4.1ms)
└─ 系统 + 框架                          (~14GB)

SSD: IQ2_XS Archive (mmap) / FP4 Safetensors, 0% 命中率, ~30ms
```

### 分层预加载

```
Step N 开始
├── 阶段 0: record_access() — Gate 选中即计数（唯一 freq 入口）
├── 阶段 1: GPU SLRU 命中 → GPU GEMM; miss → CPU FFN
├── 阶段 2: 预取下一层按频率分层
│     Top-N → GPU DMA
│     TopN+1~M → CPU SLRU + L3 预热 (warmup_layer_targeted, 20MB 预算)
│     M 之后 → CPU SLRU (cold)
└── 加权延迟: 0.78×0.2 + 0.15×2.0 + 0.07×2.7 = 3.87ms (6专家)
```

### 两套 SLRU + 频率统计

| | GPU SLRU (Python) | CPU SLRU (Rust) |
|--|---|---|
| 管理 | GPU VRAM | CPU RAM (Iq2XsWeight) |
| 淘汰 | LRU | LFU(freq) + LRU(last_access) |
| 频率 | 无 | record_access() 统一计数 |
| 保护 | step + layer | step + layer |
| 持久化 | 无 | save_freq/load_freq (JSON) |

freq 只在 `record_access()` 递增（Gate 选中时）；add_expert/touch_key/compute_expert 只更新 last_access。
- freq → 驱动预取排序（Top-N / TopN+1~M 分层）
- last_access → 驱动 LRU 淘汰（protected → probation → 驱逐）

## GPU+CPU 异构并行推理

```
                MoE.forward()
                     │
        ┌────────────┼────────────┐
        ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│  Shared Expert  │      │  Routed Experts  │
│  始终在 GPU     │      │  GPU hot + CPU cold│
│  BF16 GEMM      │      │                   │
└─────────────────┘      └────────┬──────────┘
                                 │
                      ┌──────────┼──────────┐
                      ▼                      ▼
               GPU 缓存命中             GPU 缓存未命中
               expert != None          expert == None
                      │                      │
                      ▼                      ▼
              ┌──────────────┐      ┌──────────────────┐
              │  GPU FFN     │      │  CPU FFN          │
              │  Tensor Core │      │  AVX-512 SIMD     │
              └──────────────┘      └──────────────────┘
```

### CPU FFN 算子

| 算子 | 延迟 | 权重/专家 | 关键优化 |
|------|------|----------|---------|
| IQ2_XS FFN | 2.7ms | 109MB | AVX-512 VNNI maddubs |
| FP4 FFN | 4.1ms | 201MB | AVX-512 FMA + E8M0 scale + x_split |

FP4 优化历程：标量 210ms → AVX-512 FMA 5.0ms → E8M0 scale 4.5ms → x_split 4.1ms

### GPU 算子

TileLang 编写，编译为 .so 由 Rust tvm-ffi 加载。JIT 测试，AOT 生产。

- fp4_gemm, fp8_gemm, iq2xs_gemm, iq2k_iq2xs_gemm
- fused_shared_ffn, HybridAttention(SWA+CSA+HCA)
- KV Cache 压缩解压 (CSA/HCA)

## 规则与约束

**NEVER**
- 修改官方 TileLang 代码中数据精度
- 保留存在未解释注意力、KV 缓存或 logits 漂移的更快路径

**ALWAYS**
- 正确性优先于速度
- Rust 代码权重精度参见官方 `inference/` 目录下 python 代码
- 生产路径为全 GPU 推理（效率优先可例外，如 CPU FFN 兜底）
- 官方推理逻辑无需验证，根据本地硬件配置调整

**代码规范**

- 测试驱动：单元测试 + 集成测试
- 重要推理代码添加注释（缓存生命周期、内存策略、形状约束）
- 注释在实现旁边，保持简洁，重构改善既有代码的设计
- 公共 API 窄小，CLI/服务端不了解张量内部结构
- 不引入 C/C++ 代码；python 推理代码优先，rust 参考 python
- Rust 全 GPU 路径 panic 时直接退出进程

**其他**
- KV Cache 完全在 GPU，GPU 化 Compressor/Indexer，缓存落盘支持热启动
- MTP 收益不高，不建议使用

## 安全

- 不要并发运行多个巨型模型进程
- 优先使用简短的 TileLang 冒烟测试进行构建验证
- 内存按 90GB 使用，预留 6GB 系统使用

## 工程

### 项目布局

- `ds4.rs`：模型加载、分词器、图调度、会话、磁盘缓存序列化
- `ds4_cli.rs`：命令行 REPL
- `ds4_server.rs`：OpenAI/Anthropic 兼容 HTTP API、流式传输、工具调用
- `tilelang/*.rs`：计算内核
- `tests/`：单元和集成测试
- `misc/`：备忘录、实验材料

### 开发环境

容器 `ds4rs-dev`，测试命令：`docker exec ds4rs-dev bash -c "..."`

```bash
# 创建容器
docker run --gpus all -it --name ds4rs-dev --shm-size=8g \
    -v /data/ai/ds4rs:/workspace -v /data/ai/models/dsv4:/models:ro \
    -v /data/cache:/root/.cache -w /workspace nvcr.io/nvidia/pytorch:25.05-py3

# 安装依赖（Rust 镜像 + tilelang + cargo 镜像）
docker exec ds4rs-dev bash -c "
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install tilelang>=0.1.9
    export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
    curl --proto '=https' --tlsv1.2 -sSf https://mirrors.ustc.edu.cn/rust-static/rustup/rustup-init.sh | sh
    mkdir -p ~/.cargo && cat > ~/.cargo/config.toml << 'EOF'
[source.crates-io]
replace-with = 'aliyun'
[source.aliyun]
registry = \"sparse+https://mirrors.aliyun.com/crates.io-index/\"
EOF
"
```

### 测试与集成

- `make` 构建，日志输出模型参数/内存/显存
- 发布：`Rust binary + kernel.so → CUDA 驱动加载 → 零 Python 依赖`
- CI/CD：GitHub Actions，多平台 + 多 GPU

### 模型资源（容器内路径）

- 论文：`/workspace/DeepSeek_V4.pdf`
- 推理代码：`/workspace/inference`
- 服务代码：`/workspace/encoding/encoding_dsv4.py`
- 模型权重：`/models/`

### 文档

- `README.md`：使用手册 + 技术架构
- `CHANGELOG.md`：变更日志
- `src/` + `tilelang/`：源码中文注释
- tilelang 核心算子接口文档
