# Agent 备忘录

`ds4.rs` 是 DeepSeek V4 Flash 的专用单机推理引擎，基于 96G DDR5 + RTX5060Ti16G 定制开发。目标是构建一个小巧、可读、高性能的 Rust 代码库，复刻 inference 目录下官方推理，inference 目录下官方 TileLang（Python）将算子编译成共享库（.so），然后在 Rust 加载执行。内核存放在 `tilelang/` 目录下。

## 目标

    FP4 专家：150GB
    IQ2_XS 专家：80GB
    内存：90GB
    显存：16GB
    系统两种量化都支持
### 推理流程

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
    │                    IQ2_XS 推理流程 (优化后)                   │
    ├─────────────────────────────────────────────────────────────┤
    │  x (BF16)                                                    │
    │           ↓                                                  │
    │  iq2xs_gemm_optimized(x, qs, scales, d)                     │
    │           ↓                                                  │
    │  输出 (BF16)                                                 │
    └─────────────────────────────────────────────────────────────┘

### 推理路径

- 保持生产路径为全 GPU 推理，效率优先可例外
- 官方推理逻辑无需验证，根据本地硬件配置调整
- 正确性优先于速度。不要保留存在未解释注意力、KV 缓存或 logits 漂移的更快路径

### 专家缓存

- GPU/CPU M 层 N 专家编号 + LFU 频率到磁盘，支持热启动
- 多级异步预取流水线：GPU 计算 L 层 → CPU 预取 L+2→GPU → SSD 预取 L+5→CPU
- 专家在 FP4 scale 支持 Delta 分解 + ZipNN 熵编码压缩；只缓存 fp4 weight，scale 从 RAM 读取

### 量化配置
- 量化类型：默认 FP4，支持 IQ2_XS（启动参数 `--quant-type=[fp4|iq2xs]`）；如果配置iq2xs,检测本地是否有 IQ2_XS 文件，否则使用预量化保存到本地；
- 缓存容量：运行时根据量化类型动态计算，不硬编码
- 启动检测：推理启动时检测硬件配置，提示推荐量化类型

#### FP4缓存策略
- 非路由专家常驻 GPU
- 路由专家三级缓存（GPU/CPU/SSD）：
    - GPU 显存：采用 LFU
    - CPU 内存：采用 SLRU
    - SSD5 硬盘：全量路由专家
- TODO: CPU/GPU: 模型分层滑窗 + 热点；SLRU/FIFO

#### IQ2_XS缓存策略
- 非路由专家常驻 GPU
- 路由专家三级缓存（GPU/CPU/SSD）：
    - GPU 显存：采用 SLRU
    - CPU 内存：全量专家全部加载
    - SSD5 硬盘：全量专家mmap 加载，OS 页缓存管理 

- 专家IQ2_XS 量化/反量化路径：解码 FP4 →TileLang(float32 → IQ2_XS 量化 )→ 缓存 → 传 GPU → TileLang IQ2_XS GEMM kernel


### 其他

- KV Cache 完全在 GPU 上实现，GPU 化的 Compressor 和 Indexer，缓存落盘支持热启动
- 支持插件机制，方便扩展功能
- MTP 收益不高，不建议使用

## 判断边界

**NEVER**
- 不要试图修改官方 TileLang 代码中数据精度
- 路由专家计算禁止 CPU fallback，仅在 GPU 上执行

**ALWAYS**
- Rust 代码权重精度参见官方 inference 目录下 python 代码

## Rust 实现

- Rust 复刻 inference 目录下推理逻辑及 TileLang 算子
- TileLang 编写融合内核 fused_shared_ffn、Shared Expert FFN 融合算子
- 核心算子单元测试集成测试，生成核心算子 API 文档

## 算子融合

TileLang Mega-kernel 能力多个算子融合进一个 persistent kernel，极致压榨 GPU 利用率。

- TileLang 编写融合内核：HybridAttention(SWA+CSA+HCA)
- DeepSeek TileKernels(MoE/Quant): MoE 路由 + FP8/FP4 GEMM
- TileLang KV Cache 实现压缩解压(CSA/HCA)
- 完整版用来集成完整的 V4 Flash Attention + MoE 逻辑
- Rust tvm-ffi crate 加载 TileLang 算子（.so）
- 测试时候采用 JIT 编译，验证算子融合是否生效。生产时候 AOT 编译

## 算子接口文档

- 文件：`docs/tilelang_kernel_api.md`
- 定期更新
- 调用查询接口规范

## 质量规则

- 在重要推理代码处添加注释，当模型机制、缓存生命周期、内存策略或 API 编排无法从局部代码明显看出时
- 优先选择实现旁边的注释，而非单独的设计文档
- 保持注释具有指导性和简洁性：解释为何存在某种形状、排序、缓存边界或内存选择
- 保持公共 API 窄小。CLI/服务端代码不应了解张量内部结构
- 不要在标志后添加永久语义变体。诊断开关在验证单一发布路径时是可接受的
- 不要引入 C++ 代码
- python 推理代码优先，rust 代码参考 python 代码
- Rust 全 GPU 推理路径，禁止 CPU fallback。panic 时，直接退出进程

## 安全

- 不要并发运行多个巨型模型进程
- 优先使用简短的 TileLang 冒烟测试进行构建验证
- 内存按 90GB 使用，预留 6GB 系统使用

## 布局

- `ds4.rs`：模型加载、分词器、CPU 参考代码、图调度、会话、磁盘缓存负载序列化
- `ds4_cli.rs`：命令行、linenoise REPL、交互式转录处理
- `ds4_server.rs`：OpenAI/Anthropic 兼容 HTTP API、工作队列、流式传输、工具调用映射、磁盘 KV 缓存策略。参考官方代码 encoding/encoding_dsv4.py。引入 nng crate 提供端口服务
- `tilelang/*.rs`：计算内核
- `tests/`：单元和实时集成测试
- `misc/`：被忽略的备忘录、实验和旧规划材料

## 测试

- 测试驱动开发：单元测试 + 集成测试
- 使用 `make` 进行构建验证。当模型和 TileLang 可用时，使用 `make test` 进行单元/回归测试
- 仅在有意测试 API 表面时使用实时服务端测试
- 打印日志：模型参数以及向量名称与大小，内存与显存消耗情况以及剩余
- 推理时候显示模型加载日志，向量名称与大小；Transformer 每一层加载向量名称及大小；内存与显存消耗情况以及剩余

### 测试环境

在容器环境测试：

```bash
docker exec -it ds4rs-dev bash -c "..."
```

## 集成

发布阶段（用户机器）：

```
Rust binary + kernel.so → 直接 CUDA 驱动加载 → 零 Python 依赖
```

## 开发环境

### 配置容器 ds4rs-dev

#### 初始化

```bash
docker run --gpus all -it \
    --name ds4rs-dev \
    --shm-size=8g \
    -v /data/ai/ds4rs:/workspace \
    -v /data/ai/models/dsv4:/models:ro \
    -v /data/cache:/root/.cache \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.05-py3
```

#### 安装依赖

```bash
docker exec -it ds4rs-dev bash -c "
    apt install htop glances

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

    pip install tilelang>=0.1.9

    # 临时使用（当前终端生效）
    export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
    export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static

    # 然后执行安装
    curl --proto '=https' --tlsv1.2 -sSf https://mirrors.ustc.edu.cn/rust-static/rustup/rustup-init.sh | sh

    # 国内镜像 cargo
    mkdir -p ~/.cargo
    cat > ~/.cargo/config.toml << 'EOF'
    [source.crates-io]
    replace-with = 'aliyun'

    [source.aliyun]
    registry = \"sparse+https://mirrors.aliyun.com/crates.io-index/\"
    EOF

        rm -f ~/.cargo/.package-cache
    "
    ```

## 模型相关

### 官方模型代码及文档

- 模型论文：容器目录 `/workspace/DeepSeek_V4.pdf`
- 模型推理代码：容器目录 `/workspace/inference`
- 模型服务代码：容器目录 `/workspace/encoding/encoding_dsv4.py`
- 模型：容器目录 `/models/`
- 容器代码路径：`:/workspace`

## 文档

- `README.md`：项目介绍，使用手册，推理全流程，技术架构，关键技术
- `CHANGELOG.md`：项目变更日志
- 目录 `src` 与 `tilelang` 所有源代码增加中文注释；如何人工审查代码说明文档
- MoE: 专家激活热力图 参数加载/卸载时间线
- 内存/显存时间线图
- tilelang 核心算子接口文档

## CI/CD

- GitHub Actions: 自动构建和测试
- 多平台支持：Windows, macOS, Linux；依赖 Rust 与 TileLang 多平台支持
- 多 GPU 支持
