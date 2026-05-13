# Agent 备忘录

`ds4.rs` 是 DeepSeek V4 Flash 的专用单机推理引擎，基于96G DDR5+RTX5060Ti16G定制开发。目标是构建一个小巧、可读、高性能的 Rust 代码库，复刻inference目录下官方推理，inference目录下官方TileLang（Python）将算子编译成共享库（.so），然后在Rust加载执行。内核存放在 `tilelang/` 目录下。

## 目标

- 保持生产路径为全GPU推理；
- 尽可能复刻inference目录下python转为TileLang，构建常用算子；
- 非路由专家常驻GPU；
- 路由专家三级缓存：权重流式经过L2,shared memory 仅用于高频访问的activations与临时buffers。
    GPU缓存热点路由专家：LFU+Top Segment缓存+DMA异步读取Bottom Segment CPU缓存 ；    
    跨层预取：L层计算预测L+1层可能专家集合，cuda stream+pinned memory异步从gpu ram拷贝到gpu空闲槽位；隐藏PCIe传输延迟；
    内存缓存路由专家：内存划分SLRU缓存与L1-N层路由专家SSD MMAP预取；推理效率优先动态划分；
    SSD5 PCIe5x8 独立缓存，15G/s:全量路由专家；路由专家SSD MMAP预取；
- 正确性优先于速度。不要保留存在未解释注意力、KV 缓存或 logits 漂移的更快路径。
- 通过实时 KV 复用和磁盘 KV 检查点，路由专家缓存落盘支持热启动，使长时本地 Agent 会话变得实用。
- 支持插件机制，方便扩展功能。

### MoE 专家权重流式传输最佳实践

    当前模型文件中的路由专家权重仍然是 FP4 (int8 打包格式) ，没有经过 --expert-dtype fp8 转换！

    基于 cudarc 的能力和 RTX 5060 Ti 硬件特性，推荐以下架构：

    三级缓存 + 异步预取架构
 
    ┌─────────────────────────────────────────────┐
    │  GPU VRAM (16GB)                            │
    │  ┌─────────────────────────────────────┐    │
    │  │ 常驻: 非路由专家 + KV Cache +       │    │
    │  │       热点路由专家 (LFU Top-K)       │    │
    │  └─────────────────────────────────────┘    │
    │  ┌─────────────────────────────────────┐    │
    │  │ 空闲槽位: 预取目标                   │    │
    │  └─────────────────────────────────────┘    │
    └──────────────────┬──────────────────────────┘
                    │ PCIe 5.0 x8 (~16 GB/s)
                    │ DMA Async (Pinned Memory)
    ┌──────────────────┴──────────────────────────┐
    │  Host RAM (96GB DDR5)                       │
    │  ┌─────────────────────────────────────┐    │
    │  │ Pinned Memory Pool (SLRU Cache)     │    │
    │  │ - 热层: 常驻 pinned buffer          │    │
    │  │ - 冷层: 可分页 + cudaHostRegister   │    │
    │  └─────────────────────────────────────┘    │
    └──────────────────┬──────────────────────────┘
                    │ SSD MMAP 预取 [PCIe 5.0 x8 (~16 GB/s)]
    ┌──────────────────┴──────────────────────────┐
    │  SSD: 全量路由专家权重                       │
    └─────────────────────────────────────────────┘

![alt text](assets/三级缓存可行性评估.png)

## 算子融合

    TileLang Mega-kernel能力多个算子融合进一个persistent kernel，极致压榨GPU利用率。

- TileLang编写融合内核：HybridAttention(SWA+CSA+HCA) 
- DeepSeeK TileKernels(MoE/Quant):MoE路由+FP8/FP4 GEMM
- TileLang KV Cache实现压缩解压(CSA/HCA)
- 简化版用来验证Rust-->TileLang数据通路是否正确：融合Attention+FFN，无MoE
- 完整版用来集成完整的V4 Flash Attention+MoE逻辑:MoE整层Transformer，Mega-kernel(RMSNorm->QKV->Hybrid Attention->Proj->Residual->FFN)
- 先跑通简化版+层间expert预取，验证Rust-->TileLang通路；再扩展为完整版Moe Mege-Kernel，达到最高性能。
- Rust tvm-ffi crate 加载TileLang算子（.so）
- 数据交换 DLPack协议
- 测试时候采用JIT编译，验证算子融合是否生效。生产时候AOT编译。

## 质量规则

- 在重要推理代码处添加注释，当模型机制、缓存生命周期、内存策略或 API 编排无法从局部代码明显看出时。
- 优先选择实现旁边的注释，而非单独的设计文档。
- 保持注释具有指导性和简洁性：解释为何存在某种形状、排序、缓存边界或内存选择。
- 保持公共 API 窄。CLI/服务端代码不应了解张量内部结构。
- 不要在标志后添加永久语义变体。诊断开关在验证单一发布路径时是可接受的。
- 不要引入 C 或 C++ 代码。

## 安全

- 不要并发运行多个巨型模型进程。
- 优先使用简短的 TileLang 冒烟测试进行构建验证。

## 布局

- `ds4.rs`：模型加载、分词器、CPU 参考代码、 图调度、会话、磁盘缓存负载序列化。
- `ds4_cli.rs`：命令行、linenoise REPL、交互式转录处理。
- `ds4_server.rs`：OpenAI/Anthropic 兼容 HTTP API、工作队列、流式传输、工具调用映射、磁盘 KV 缓存策略。参考官方代码encoding/encoding_dsv4.py。
- `tilelang/*.rs`：计算内核。
- `tests/`：单元和实时集成测试。
- `misc/`：被忽略的备忘录、实验和旧规划材料。

## 测试

    使用 `make` 进行构建验证。当模型和 TileLang 可用时，使用 `make test` 进行单元/回归测试。仅在有意测试 API 表面时使用实时服务端测试。

## 开发环境
### 配置容器 ds4rs-dev 

#### 初始化

    docker run --gpus all -it \
    --name ds4rs-dev \
    --shm-size=8g \
    -v /data/ai/ds4rs:/workspace \
    -v /data/ai/models/dsv4:/models:ro \
    -v /data/cache:/root/.cache \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.05-py3

#### 安装依赖

    docker exec -it ds4rs-dev bash -c "
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    pip install tilelang>=0.1.9 "

### 测试环境

测试命令： docker exec -it ds4rs-dev bash

## 模型相关

### 官方模型代码及文档
 
    模型论文 容器目录/models/DeepSeek_V4.pdf 
    模型推理代码 容器目录/models/inference
    模型服务代码 容器目录/models/encoding/encoding_dsv4.py。
    模型 容器目录/models/
    容器 代码路径 :/workspace

