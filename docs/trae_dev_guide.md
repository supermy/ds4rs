# Trae 开发环境配置与最佳实践

本文档面向 ds4.rs 项目，描述在 Trae IDE 中进行本地开发和服务器开发的完整环境配置、工作流与最佳实践。

## 1. 本地 Trae Solo 模式

### 1.1 安装与启动

**Trae IDE 安装**：

从 [Trae 官网](https://trae.ai) 下载 Linux 版本安装包，或通过命令行安装：

```bash
# 下载并安装 Trae IDE
wget -O trae.deb <下载链接>
sudo dpkg -i trae.deb
```

**Solo 模式启动**：

1. 启动 Trae IDE，选择 **Solo 模式**（本地 AI Agent，无需远程服务）
2. 打开项目目录：`File → Open Folder → /data/ai/ds4rs`
3. Trae 自动识别项目根目录下的 `AGENTS.md`，加载项目上下文

**工作区配置**：

- 项目根目录：`/data/ai/ds4rs`
- Trae 自动索引项目文件，Agent 可通过语义搜索定位代码
- 建议关闭不必要的文件监听（如 `.venv/`、`gguf/`），减少索引开销

### 1.2 项目规则配置

**AGENTS.md 的作用**：

`AGENTS.md` 位于项目根目录 `/data/ai/ds4rs/AGENTS.md`，是 AI Agent 的项目上下文备忘录。Agent 在每次会话启动时自动加载，获取：

- 项目核心约束（资源受限、缓存策略、硬件配置）
- NEVER/ALWAYS 规则（正确性优先、不修改精度、不引入 C/C++ 代码等）
- 开发环境命令（Docker 容器创建、依赖安装）
- 模型资源和文档路径

配置方法：直接编辑 `AGENTS.md`，使用 Markdown 格式。内容变更后 Agent 在下次会话自动生效，无需重启 IDE。保持内容精炼，Agent 上下文窗口有限。

**.trae/rules/project_rules.md 的配置方法**：

`.trae/rules/project_rules.md` 补充 AGENTS.md 中未覆盖的 IDE 行为配置，如 lint/typecheck 命令、测试命令等。创建文件：

```bash
mkdir -p /data/ai/ds4rs/.trae/rules
```

示例内容：

```markdown
# Project Rules

## Build & Test
- Build: `make build` (cargo build --release)
- Test: `make test` (cargo test -- --test-threads=1)
- Lint: `make clippy` (cargo clippy -- -D warnings)
- Check: `make check` (cargo check)

## Code Style
- Rust 代码遵循项目既有风格，参考 src/ 目录
- 推理代码添加中文注释，保持简洁
- 公共 API 窄小，内部实现不暴露给 CLI/服务端
```

**规则编写最佳实践**：

- **精简**：每条规则一句话，避免冗长解释
- **结构化**：用标题和列表组织，便于 Agent 快速定位
- **包含具体命令**：直接给出可执行的命令，而非描述性文字（如 `make test` 而非"运行测试"）
- **优先级**：NEVER/ALWAYS 规则放在最前面，Agent 优先遵守
- **路径用容器内路径**：所有路径使用 `/workspace` 而非宿主机路径

### 1.3 Solo 模式工作流

**查看日志**：

- **Trae 输出面板**：Agent 的对话和操作日志显示在 IDE 底部输出面板
- **终端日志**：Agent 执行的命令输出在对应终端中查看；使用 `CheckCommandStatus` 工具可获取非阻塞命令的最新输出

**编辑文件**：

- **Agent 自动编辑**：Agent 通过 Edit/Write 工具直接修改文件，修改前会先 Read 文件内容
- **手动编辑**：用户可在 IDE 中直接编辑文件，Agent 会感知文件变更
- **协作模式**：建议复杂逻辑先让 Agent 生成代码，人工审查后确认

**配置规则**：

让 Agent 遵循项目规范的关键：
1. `AGENTS.md` 中写明 NEVER/ALWAYS 规则
2. `.trae/rules/project_rules.md` 中写明 lint/test 命令
3. 在对话中明确引用规则（如"按照 AGENTS.md 中的规范"）

**配置对话流**：

引导 Agent 完成复杂任务的技巧：
1. **拆分任务**：将大任务拆为多个小步骤，逐步完成
2. **先分析后修改**：让 Agent 先阅读相关代码，理解上下文，再动手修改
3. **验证驱动**：每步修改后要求 Agent 运行测试验证
4. **指定范围**：明确告诉 Agent 只修改哪些文件，避免过度修改

**Solo 沙箱运行**：

全自动 coding 的配置与安全边界：
- Agent 可自动执行安全命令（文件读取、搜索、构建）
- 危险操作（git push、删除文件、修改 AGENTS.md）需用户确认
- Rust 全 GPU 路径 panic 时直接退出进程，Agent 不会尝试恢复

**终端管理**：

Trae IDE 最多支持 **5 个终端**。本项目使用 Docker 容器，所有命令通过 `docker exec` 执行。

| 终端用途 | 命令模板 |
|---------|---------|
| Rust 构建/测试 | `docker exec ds4rs-dev bash -c "cd /workspace && cargo build --release"` |
| Python 推理脚本 | `docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python inference/xxx.py"` |
| TileLang kernel 编译 | `docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python tilelang/compile_kernels.py"` |
| 量化流程 | `docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python inference/prequant_mixed_iq2xxs_q2k.py ..."` |
| GPU 监控 | `docker exec ds4rs-dev nvidia-smi` |

注意事项：
- 在非空闲终端执行命令会终止该终端正在运行的命令
- 长时间运行的命令（如量化流程）使用独立终端，避免阻塞其他操作
- `docker exec` 命令中路径使用容器内路径 `/workspace`，而非宿主机路径 `/data/ai/ds4rs`

### 1.4 权限模式

**自动允许 vs 手动确认**：

- **自动执行模式**：Agent 自动执行安全命令（文件读取、搜索、构建），危险操作需确认
- **确认模式**：所有命令执行前需用户确认

推荐开发时使用自动执行模式，生产部署时切换为确认模式。

**文件读写权限**：

- 文件读取：自动允许（Agent 需要阅读代码才能工作）
- 文件写入/编辑：自动允许（Agent 的核心功能）
- 删除文件：需确认

**终端命令执行权限**：

以下操作始终需要确认：
- `git push`、`git reset --hard` 等破坏性 Git 操作
- 删除文件
- 修改 `AGENTS.md` 或项目配置

安全命令自动执行：
- `cargo build`、`cargo test`、`cargo clippy`
- `docker exec` 中的构建和测试命令
- 文件搜索和代码分析

## 2. 服务器开发环境

### 2.1 SSH 连接服务器

**SSH 配置（~/.ssh/config）**：

```bash
# 编辑 SSH 配置
vim ~/.ssh/config

# 添加服务器配置
Host gpu-server
    HostName <服务器IP>
    User <用户名>
    Port 22
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**SSH key 配置与免密登录**：

```bash
# 生成 SSH key（如果还没有）
ssh-keygen -t ed25519 -C "your@email.com"

# 复制公钥到服务器
ssh-copy-id gpu-server

# 测试免密登录
ssh gpu-server
```

**VSCode/Trae Remote SSH 插件配置**：

1. 安装 Remote SSH 插件（Trae IDE 内置或从扩展市场安装）
2. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → 选择 `gpu-server`
3. 连接后打开远程项目目录

**端口转发配置（GPU 服务器访问）**：

```bash
# 转发 ds4_server 端口（本地 8080 → 服务器 8080）
ssh -L 8080:localhost:8080 gpu-server

# 转发 TensorBoard 端口
ssh -L 6006:localhost:6006 gpu-server

# 在 ~/.ssh/config 中配置持久转发
Host gpu-server
    HostName <服务器IP>
    User <用户名>
    LocalForward 8080 localhost:8080
```

### 2.2 Docker 容器开发

**容器创建命令**：

```bash
docker run --gpus all -it --name ds4rs-dev --shm-size=8g \
    -v /data/ai/ds4rs:/workspace \
    -v /data/ai/models/dsv4:/models:rw \
    -v /data/cache:/root/.cache \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.05-py3
```

参数说明：
- `--gpus all`：挂载所有 GPU，推理和 TileLang 编译必需
- `--shm-size=8g`：共享内存 8GB，PyTorch DataLoader 和多进程通信需要
- `-v /data/ai/ds4rs:/workspace`：项目代码挂载到容器内 `/workspace`
- `-v /data/ai/models/dsv4:/models:rw`：模型权重挂载到 `/models`，读写权限
- `-v /data/cache:/root/.cache`：缓存目录（pip、HuggingFace 等），避免重复下载
- `-w /workspace`：容器工作目录

**容器内环境配置**：

```bash
docker exec ds4rs-dev bash -c "
    # pip 镜像
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

    # 安装 TileLang
    pip install tilelang>=0.1.9

    # Rust 镜像安装
    export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
    curl --proto '=https' --tlsv1.2 -sSf \
        https://mirrors.ustc.edu.cn/rust-static/rustup/rustup-init.sh | sh

    # Cargo 镜像
    mkdir -p ~/.cargo && cat > ~/.cargo/config.toml << 'EOF'
[source.crates-io]
replace-with = 'aliyun'
[source.aliyun]
registry = \"sparse+https://mirrors.aliyun.com/crates.io-index/\"
EOF
"
```

**GPU 访问验证**：

```bash
# 检查 GPU 状态
docker exec ds4rs-dev nvidia-smi

# 检查 CUDA 版本
docker exec ds4rs-dev bash -c "python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'"

# 检查 GPU 显存
docker exec ds4rs-dev bash -c "python -c 'import torch; print(torch.cuda.get_device_properties(0))'"
```

**数据卷挂载**：

| 宿主机路径 | 容器路径 | 用途 | 权限 |
|-----------|---------|------|------|
| `/data/ai/ds4rs` | `/workspace` | 项目代码 | rw |
| `/data/ai/models/dsv4` | `/models` | 模型权重 | rw |
| `/data/cache` | `/root/.cache` | pip/HF 缓存 | rw |

容器内关键路径：
- 论文：`/workspace/DeepSeek_V4.pdf`
- 推理代码：`/workspace/inference`
- 服务代码：`/workspace/encoding/encoding_dsv4.py`
- 模型权重：`/models/`
- GGUF 模型：`/workspace/gguf/`

### 2.3 终端监控

**nvtop 查看 GPU 占用**：

```bash
# 安装（如果未安装）
sudo apt install nvtop

# 启动监控
nvtop
```

常用快捷键：
- `q`：退出
- `h`：帮助
- `F9`：终止选中进程
- `F6`：排序方式切换

监控指标：
- **VRAM**：显存使用量/总量（本项目常驻 ~14GB/16GB）
- **GPU 利用率**：计算单元使用率
- **进程**：GPU 上的进程列表及显存占用

**htop 查看 CPU/内存占用**：

```bash
# 安装
sudo apt install htop

# 启动监控
htop
```

常用快捷键：
- `F5`：树形/列表视图切换
- `F6`：排序
- `F9`：终止进程
- `F10`：退出

监控指标：
- **CPU 核心**：每个核心的使用率（6C/12T Ryzen 5 7600）
- **内存**：RAM 使用量（96GB DDR5，Pinned Pool 常驻 ~74GB）
- **SWAP**：交换空间使用量

**glances 综合监控**：

```bash
# 安装
pip install glances

# 启动
glances

# Web 模式（远程访问）
glances -w
```

特色功能：
- **容器视图**：按 `c` 切换容器视图，查看 Docker 容器资源占用
- **进程排序**：按 CPU/内存/IO 自动排序
- **告警**：资源超阈值自动高亮

**实际监控示例：量化任务运行时的资源占用分析**：

IQ2_XS 量化任务运行时典型资源占用：

| 资源 | 占用 | 说明 |
|------|------|------|
| GPU VRAM | ~12-14GB | 模型权重加载 + imatrix 计算 |
| GPU 利用率 | 60-80% | GEMM 计算密集 |
| CPU | 30-50% | 数据预处理 + Python 开销 |
| RAM | ~80GB | 模型权重 + Pinned Pool |
| 磁盘 IO | 高 | 读取模型权重、写入 GGUF |

FP4 量化任务更重：VRAM ~15GB，RAM ~90GB，磁盘 IO 更高（权重 150GB）。

## 3. 多语言项目开发最佳实践

### 3.1 Rust 代码

**构建与测试**：

```bash
# 开发构建
docker exec ds4rs-dev bash -c "cd /workspace && cargo build"

# 发布构建
docker exec ds4rs-dev bash -c "cd /workspace && make build"

# 运行测试（单线程避免 GPU 竞争）
docker exec ds4rs-dev bash -c "cd /workspace && make test"

# 详细测试输出
docker exec ds4rs-dev bash -c "cd /workspace && make dev-test"

# Lint 检查
docker exec ds4rs-dev bash -c "cd /workspace && make clippy"
```

**关键约束**：

- 全 GPU 路径 panic 时直接退出进程（`std::process::exit`），不使用 unwrap 后恢复
- 权重精度参考官方 `inference/` 目录下 Python 代码
- 不引入 C/C++ 代码；C 互操作通过 `cc` crate 编译的 `.c` 文件除外（如 `csrc/` 下的桥接代码）
- 公共 API 窄小，CLI/服务端不了解张量内部结构

**项目结构**：

- `src/`：核心库代码（模型加载、缓存、推理管线）
- `src/bin/`：CLI 和工具入口（`ds4_cli.rs`）
- `src/cpu_expert/`：CPU FFN 实现（AVX-512 SIMD）
- `tests/`：集成测试

**中文注释规范**：

重要推理代码添加注释，内容包括：
- 缓存生命周期（何时分配、何时释放、谁持有所有权）
- 内存策略（Pinned Pool → SLRU → GPU 的流转路径）
- 形状约束（张量维度、batch size 限制）

注释放在实现旁边，保持简洁。示例：

```rust
// Pinned Pool 紧凑格式 (~7MB/专家) → SLRU 展开为计算格式 (~109MB/专家)
// 展开后 gate+up 连续排列（共享 x 时访问模式一致），down 尾随
fn expand_expert(packed: &[u8], layer: usize, expert: usize) -> Iq2XsWeight {
```

### 3.2 Python 推理代码

Python 代码位于 `inference/` 目录，是推理逻辑的参考实现和原型验证环境。

**关键文件**：

| 文件 | 用途 |
|------|------|
| `model.py` | 模型定义和前向传播 |
| `cpu_expert.py` | CPU 专家 FFN 实现 |
| `expert_cache.py` | 专家缓存管理（GPU SLRU） |
| `generate.py` | 文本生成入口 |
| `prequant_mixed_iq2xxs_q2k.py` | 混合量化脚本 |
| `gguf_iq2xs.py` | GGUF 格式序列化 |

**虚拟环境管理**：

```bash
# 激活虚拟环境
docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate"

# 安装依赖
docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && pip install -r inference/requirements.txt"
```

**开发原则**：

- Python 推理代码优先，Rust 参考 Python 实现
- 官方推理逻辑无需验证，根据本地硬件配置调整
- 使用 `.venv` 虚拟环境隔离依赖

### 3.3 TileLang Kernel

TileLang kernel 位于 `tilelang/` 目录，编译为 `.so` 由 Rust `tvm-ffi` 加载。

**JIT 测试 vs AOT 生产**：

- **JIT 测试**：开发阶段使用 TileLang JIT 编译即时测试，快速迭代
  ```bash
  docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python tilelang/test_fp8_gemm.py"
  ```
- **AOT 生产**：生产环境使用 `compile_kernels.py` 预编译为 `.so`，避免运行时编译开销
  ```bash
  docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python tilelang/compile_kernels.py"
  ```

**核心算子列表**：

| 算子 | 用途 |
|------|------|
| `fp4_gemm` | FP4 精度 GEMM |
| `fp8_gemm` | FP8 精度 GEMM |
| `iq2xs_gemm` | IQ2_XS 精度 GEMM |
| `iq2k_iq2xs_gemm` | 混合 Q2K+IQ2_XSS GEMM |
| `fused_shared_ffn` | 融合共享专家 FFN |
| `HybridAttention` | 混合注意力（SWA + CSA + HCA） |
| KV Cache 压缩解压 | CSA/HCA 压缩格式 |

**约束**：不修改官方 TileLang 代码中数据精度；优先使用简短的 TileLang 冒烟测试进行构建验证。

### 3.4 代码规范

| 规范 | 说明 |
|------|------|
| 测试驱动 | 单元测试 + 集成测试，先写测试再实现 |
| 中文注释 | 重要推理代码添加注释（缓存生命周期、内存策略、形状约束） |
| 注释位置 | 在实现旁边，保持简洁 |
| 公共 API | 窄小，CLI/服务端不了解张量内部结构 |
| 重构原则 | 重构改善既有代码的设计，不增加无关功能 |
| 正确性优先 | 不保留存在未解释注意力、KV 缓存或 logits 漂移的更快路径 |

## 4. 常见工作流

### 4.1 量化流程

量化流程将 FP4/BF16 模型权重转换为 IQ2_XS 或混合量化格式，输出 GGUF 文件供 Rust 引擎加载。

```
imatrix 生成 → 混合量化 → GGUF 输出
```

**步骤 1：生成 imatrix**（重要性矩阵，用于量化校准）：

```bash
docker exec ds4rs-dev bash -c "
cd /workspace && source .venv/bin/activate
python inference/generate_imatrix.py \
    --ckpt-path /models \
    --output /models/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat
"
```

**步骤 2：混合量化**（IQ2_XXS gate/up + Q2_K down）：

```bash
docker exec ds4rs-dev bash -c "
cd /workspace && source .venv/bin/activate
mkdir -p /workspace/gguf
python inference/prequant_mixed_iq2xxs_q2k.py \
    --ckpt-path /models \
    --imatrix /models/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat \
    --output /workspace/gguf/experts_mixed.gguf \
    --batch-size 256
"
```

**步骤 3：验证 GGUF**（可选）：

```bash
docker exec ds4rs-dev bash -c "
cd /workspace && gcc -o verify_gguf inference/verify_gguf_iq2xs.c -lm && ./verify_gguf /workspace/gguf/experts_mixed.gguf
"
```

**量化参数选择**：

| 量化类型 | 专家总大小 | CPU FFN 延迟 | 说明 |
|---------|-----------|-------------|------|
| IQ2_XS | 76GB | 2.7ms | 默认，CPU 可全量常驻 |
| FP4 | 150GB | 4.1ms | `--quant-type=fp4`，CPU 部分常驻 |
| Q2K+IQ2_XSS | 80GB | 1.92ms | 混合量化，CPU 最快 |

### 4.2 TileLang Kernel 开发

```
编写 kernel → JIT 测试 → AOT 编译
```

1. **编写 kernel**：在 `tilelang/` 目录下创建 Python 脚本
2. **JIT 测试**：使用 TileLang JIT 编译即时验证正确性
   ```bash
   docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python tilelang/test_fp8_gemm.py"
   ```
3. **AOT 编译**：通过 `compile_kernels.py` 预编译为 `.so`
   ```bash
   docker exec ds4rs-dev bash -c "cd /workspace && source .venv/bin/activate && python tilelang/compile_kernels.py"
   ```

### 4.3 Rust 集成

```
Python 验证 → Rust 实现 → 集成测试
```

1. **Python 验证**：在 `inference/` 下用 Python 实现并验证推理逻辑
2. **Rust 实现**：参考 Python 代码在 `src/` 下实现 Rust 版本
3. **集成测试**：在 `tests/` 下编写测试，对比 Python 和 Rust 输出

```bash
# 运行特定集成测试
docker exec ds4rs-dev bash -c "cd /workspace && cargo test test_inference -- --test-threads=1 --nocapture"
```

## 5. 调试技巧

### 5.1 Docker 容器内调试

**进入容器交互式 shell**：

```bash
docker exec -it ds4rs-dev bash
```

**查看容器资源使用**：

```bash
docker stats ds4rs-dev
```

**查看容器日志**：

```bash
docker logs ds4rs-dev --tail 100
```

**常见问题**：

| 问题 | 解决方案 |
|------|---------|
| 容器未启动 | `docker start ds4rs-dev` |
| GPU 不可用 | 检查 `--gpus all` 参数和 NVIDIA 驱动 |
| 共享内存不足 | 确认 `--shm-size=8g`，PyTorch 多进程需要足够共享内存 |
| Rust 编译失败 | 检查 `~/.cargo/config.toml` 镜像配置是否正确 |

### 5.2 GPU 内存监控

**nvidia-smi 实时监控**：

```bash
# 持续监控（每 1 秒刷新）
docker exec ds4rs-dev bash -c "watch -n 1 nvidia-smi"

# 单次查询
docker exec ds4rs-dev nvidia-smi

# 查询显存使用详情
docker exec ds4rs-dev bash -c "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv"
```

**PyTorch 显存追踪**：

```python
import torch

# 当前已分配显存
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 峰值显存
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# 缓存显存
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 重置峰值统计
torch.cuda.reset_peak_memory_stats()
```

**显存预算参考**（RTX 5060 Ti 16GB）：

| 组件 | 占用 |
|------|------|
| Attention + Shared Expert + Norm | ~8GB |
| KV Cache | ~2-4GB |
| 热门专家池 SLRU | 动态（IQ2_XS ~4.5GB / FP4 ~4.5GB） |
| 应急缓冲 | ~1-2GB |

超出预算会导致 OOM，需调整缓存容量参数。

### 5.3 量化精度验证

**反量化对比**：

逐层对比 Python vs Rust 输出，定位误差来源：

```bash
# Python 端输出中间结果
docker exec ds4rs-dev bash -c "
cd /workspace && source .venv/bin/activate
python inference/debug_layer_compare.py --layer 0
"

# Rust 端输出中间结果
docker exec ds4rs-dev bash -c "
cd /workspace && cargo run -- check-st /workspace/gguf/experts_mixed.gguf
"
```

**余弦相似度**：

```python
import torch

def cosine_similarity(a, b):
    """计算两个张量的余弦相似度"""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

# 使用示例
sim = cosine_similarity(python_output, rust_output)
print(f"余弦相似度: {sim:.6f}")  # 期望 > 0.999
```

**常见精度问题**：

| 问题 | 排查方向 |
|------|---------|
| IQ2_XS 解量化偏差 | 检查 kmap 查找表是否正确加载（`kmap_iq2xs.npy`） |
| FP4 scale 异常 | 确认 E8M0 scale 因子正确应用 |
| SwiGLU 融合误差 | 验证 sigmoid(gate) × gate × up 的计算顺序 |
| 累积误差放大 | 逐层对比，定位误差放大的起始层 |

**验证工具**：

| 工具 | 用途 |
|------|------|
| `inference/test_cpu_expert.py` | CPU 专家 FFN 精度测试 |
| `inference/test_p0_avx512.py` | AVX-512 算子精度测试 |
| `inference/test_p0_ffn.py` | FFN 管线精度测试 |
| `tilelang/validate_datapath.py` | 数据通路端到端验证 |
| `tilelang/debug_compare.py` | 逐算子对比调试 |
