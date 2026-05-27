# Agent 备忘录

`ds4.rs` 是 DeepSeek V4 Flash 的专用单机推理引擎，基于 96G DDR5 + RTX5060Ti16G 定制开发。
目标：构建小巧、可读、高性能的 Rust 代码库，实现本地推理 Agent 自由。

## 目标

### 核心约束：资源受限

单机 16GB VRAM vs 11008 专家 → **显存只能装 <1% 专家**。
这是所有设计决策的根因：缓存策略不是优化，而是生存。

### 设计哲学：二八法则 × 马太效应

MoE 专家激活呈幂律分布：20% 专家承载 80% 激活，且越热的专家越容易被再次激活（马太效应）。
三级缓存机制贯彻这一规律，缓存容量根据量化类型和模型层数动态计算：

| 层级 | 容量 | 命中率 | 延迟 | 策略 |
|------|------|--------|------|------|
| L1 GPU VRAM | 动态（IQ2_XS ~642专家/4.5GB, FP4 ~350专家/4.5GB） | ~78% | 0.2ms | SLRU 锁定热专家，马太效应保证命中率 |
| L2 CPU RAM | 动态（IQ2_XS 全量74GB, FP4 部分约60GB） | IQ2_XS 100% / FP4 ~90% | 2.7~4.1ms | Pinned Pool 常驻，容量不足时 SLRU 淘汰 |
| L3 SSD | 全量归档 | 兜底 | 30ms | mmap，仅启动时加载 |

**二八法则**：GPU 尽量确保每层 Top-20% 专家常驻 → 256×20%=51专家/层 × 43层 = 2193 专家（分层滚动预取，非同时驻留；同帧 VRAM ~642 专家）
**马太效应**：热专家越用越热 → SLRU protected 段自动锁定高频专家 → 命中率随推理递增
**延迟倒挂**：CPU FFN (2.7~4.1ms) < DMA (5.0ms) → GPU miss 走 CPU 而非 DMA，反直觉但正确
**异步隐藏**：DMA 预取与 GPU 计算重叠 → L+1 层专家在 L 层计算期间异步传输 → PCIe 延迟对推理不可见

### MoE 特点

MoE 权重呈"低秩共享底座 + 正交特化专家"结构，decode 阶段算术强度极低，属内存受限负载。
延迟 ∝ 加载的 unique 专家数，缓存策略本质是用空间换带宽。

### MoE 架构与参数规模
| 指标          | V4-Flash          |
| ----------- | ----------------- |
| **总参数量**    | 284B              |
| **激活参数量**   | 13B（每 token）      |
| **层数**      | 43 层              |
| **路由专家数**   | 256 个             |
| **共享专家数**   | 1 个               |
| **每次激活专家数** | 6 个路由专家 + 1 个共享专家 |
| **激活比**     | ~4.6%             |

官方fp4推理参数：temperature = 1.0, top_p = 1.0

量化IQ2推理参数：
    --temp 0.6 
    --top-p 0.95 
    --min-p 0.01

### 硬件资源

| 组件 | 规格 | 用途 |
|------|------|------|
| GPU | RTX 5060 Ti 16GB | Attention + Shared Expert + 热专家 FFN |
| CPU | Ryzen 5 7600 6C/12T, AVX-512 | 冷专家 FFN (AVX-512 SIMD) |
| RAM | DDR5-4800 48GB×2 = 96GB | 全量专家 pinned pool |
| SSD | NVMe | 专家归档 (mmap, 0% 命中率) |

### 量化配置

IQ2_XXS 是"查表极速型"（0.52ms），IQ3_XXS 是"查表精度型"（0.8ms），IQ2_S 是"混合夹生型"（1.2ms），Q2_K 是"计算基线型"（1.5ms）。

最实用的组合是：gate/up 用 IQ2_XXS 抢速度，down 用 IQ3_XXS 保质量，共享层和 Router 不动。

gate_proj / up_proj: IQ2_XXS（极致速度）
down_proj: Q2_K 或 Q4_K_M（精度敏感，保留稍高精度）

| 量化类型 | 专家总大小 | 每专家 GPU | 每专家 CPU | CPU FFN | GPU FFN |
|---------|-----------|----------|----------|---------|---------|
| IQ2_XS | 76GB | ~7MB | ~109MB | 2.7ms | 1.24ms |
| FP4 (e2m1+E8M0) | 150GB | ~7MB | ~201MB | 4.1ms | ~1.15ms |
| Q2K+IQ2_XSS | 80GB | — | — | 1.92ms | — |

- 默认 IQ2_XS，支持 FP4（`--quant-type=[iq2xs|fp4]`）
- 启动时检测硬件配置，提示推荐量化类型

## 缓存策略

核心：GPU hot + CPU cold 双路 MoE，Gate 选中即计数，统一频率驱动预取。
GPU miss 走 CPU FFN (2.7ms) 而非 DMA (5.0ms)。

### 内存布局

```
GPU 16GB 常驻：
├─ Attention + Shared Expert + Norm    (~8GB)
├─ KV Cache                            (~2-4GB)
├─ 热门专家池 SLRU                     (动态容量，按量化类型计算)
└─ 应急缓冲                            (~1-2GB)

CPU 96GB：
├─ Pinned Pool (IQ2_XS 全量74GB / FP4 部分约60GB)
├─ Rust SLRU (IQ2_XS 2.7ms, FP4 4.1ms)
└─ 系统 + 框架                          (~14GB)

SSD: IQ2_XS Archive (mmap) / FP4 Safetensors, 0% 命中率, ~30ms
```

### 分层预加载

```
Step N 开始
├── 阶段 0: record_access() — Gate 选中即计数（唯一 freq 入口）
├── 阶段 1: GPU SLRU 命中 → GPU GEMM; miss → CPU FFN
├── 阶段 2: 异步预取 L+1 层（与 L 层 GPU 计算重叠，隐藏 PCIe 延迟）
│     Top-N → 异步 DMA (pinned_pool → GPU, ~5ms 但被计算覆盖)
│     TopN+1~M → CPU SLRU + L3 预热 (warmup_layer_targeted, 以 L3 容量为上限)
│     M 之后 → CPU SLRU (cold)
└── 有效延迟: 0.78×0.2 + 0.22×2.7 = 0.76ms (DMA 被异步隐藏)
```

异步 DMA 时间线：
```
L 层:  [GPU Attention][GPU GEMM hit][CPU FFN miss]  [GPU Attention L+1]
L+1:  [──异步DMA预取──][──DMA完成──]                 [GPU GEMM hit]
                      ↑ PCIe延迟被L层计算完全覆盖
```

### 两套 SLRU + 频率统计

| | GPU SLRU (Python) | CPU SLRU (Rust) |
|--|---|---|
| 管理 | GPU VRAM | CPU RAM (Iq2XsWeight; FP4 待支持) |
| 淘汰 | LRU | LFU(freq) + LRU(last_access) |
| 频率 | _layer_freq (ExpertCache) | record_access() 统一计数 |
| 保护 | step + layer | step + layer |
| 持久化 | 无 | save_freq/load_freq (JSON) |

freq 只在 `record_access()` 递增（Gate 选中时）；add_expert/touch_key/compute_expert 只更新 last_access。
- freq → 驱动预取排序（Top-N / TopN+1~M 分层）
- last_access → 驱动 LRU 淘汰（protected → probation → 驱逐）

## GPU+CPU 异构并行推理

**原因**：GPU VRAM 不够装全量专家 → 二八法则 → GPU 只缓存 Top-20% 热专家
**方法**：预加载确保 GPU Top-20% + 异步隐藏 PCIe 延迟 + CPU L3 预热提高命中率

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

### GPU：异步预加载隐藏 PCIe 延迟

GPU 计算当前层 L 时，异步 DMA 预取 L+1（或 L+2）层差集专家到 GPU SLRU。
PCIe 延迟 (~5ms) 被 GPU Attention + GEMM 计算时间覆盖，对推理不可见。

```
L 层:   [GPU Attention][GPU GEMM hit][CPU FFN miss]  [GPU Attention L+1]
L+1 层: [──异步DMA预取──][──DMA完成──]                 [GPU GEMM hit]
                        ↑ PCIe延迟被L层计算完全覆盖
```

### CPU：L3 预热提高命中率

CPU L3 = 32MB，预热当前层 CPU 负责的 80% 专家中的 20%（即总量的 16%），以 L3 容量为上限。
`warmup_layer_targeted()` 只预热当前层 TopN+1~TopN+M 范围的专家（跳过 GPU 已有的），
使 CPU FFN 从 cold RAM (2.7ms) 加速到 L3 warm (~2.0ms)。

### 专家 FFN 管线设计

每个被选中的专家内部执行: gate → up → SwiGLU → down

```
x ──┬─ gate_weight ──→ gate ──→ sigmoid ──→ × ──→ mid ──→ down_weight ──→ y
    └─ up_weight  ───→ up  ────────────────┘
         ↑ 共享 x        ↑ SwiGLU 融合              ↑ 依赖 mid
```

**管线特点**：gate 和 up 共享输入 x（并行计算），down 依赖 SwiGLU 输出（时序后置）。

**缓存/预加载**：
- L3 预热按管线顺序：先 gate+up 的 d 数组（触发硬件预取 qs/scales），计算期间预取 down 的 d 数组
- 权重存储: gate+up 连续排列（共享 x 时访问模式一致），down 尾随
- Pinned Pool 紧凑格式 (~7MB/专家) → SLRU 展开为计算格式 (~109MB/专家)

**算子优化**：
- IQ2_XS: Q8 预量化 x，gate 和 up 共享同一份 Q8 激活（省 1 次量化）
- FP4: x_split 预拆分 x_even/x_odd，gate 和 up 共享拆分结果（省 2 次 permutex2var/block）
- SwiGLU 融合: sigmoid(gate) × gate × up 在 gate+up 输出上原地计算

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
- 注释在实现旁边，保持简洁，
- 关键代码生成需要tdd，代码修改要tdd;
- 代码臃肿，需要重构改善既有代码的设计，review;
- 公共 API 窄小，CLI/服务端不了解张量内部结构
- 不引入 C/C++ 代码；
- python 推理代码优先，rust 参考 python
- Rust 全 GPU 路径 panic 时直接退出进程

**安全**
- 不并发运行多个巨型模型进程
- 优先使用简短的 TileLang 冒烟测试进行构建验证
- 内存按 90GB 使用，预留 6GB 系统使用

**其他**
- KV Cache 完全在 GPU，GPU 化 Compressor/Indexer，缓存落盘支持热启动
- MTP 收益不高，不建议使用

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
    -v /data/ai/ds4rs:/workspace -v /data/ai/models/dsv4:/models:rw \
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
- GGUF模型：`/workspace/gguf/`


### 文档

- `README.md`：使用手册 + 技术架构
- `CHANGELOG.md`：变更日志
- `src/` + `tilelang/`：源码中文注释
- tilelang 核心算子接口文档
