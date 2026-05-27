# Changelog

All notable changes to this project will be documented in this file.

## [0.9.7] - 2025-05-27

### CPU FFN AVX-512 内核深度优化（AMD Ryzen 5 7600 专用）

#### IQ2_XXS AVX-512 内核优化

- **256-bit maddubs 管线**: grid_u64 预计算表（8KB L1 常驻）+ `_mm256_set_epi64x` 打包，替代标量逐字节 grid 查表
- **KSIGNS_IQ2XS_U64 预计算表**: 128 项 × 8 bytes = 1KB L1 常驻，将 i8 符号掩码重解释为 u64，供 `_mm256_set_epi64x` 直接使用
- **双 ib32 融合**: 一次处理 2 个 ib32（64 个 q8 值），2 次 256-bit maddubs 结果累加后 1 次 hsum，减少 hsum 次数 50%
- **去掉 grid 符号吸收**: 发现 IQ2_XXS grid 值全部在 [8, 43] 范围（unsigned byte），sign 只需应用到 q8，grid 直接传给 maddubs
- **memcpy 批量读取**: `std::ptr::read_unaligned` 一次读 16 字节（8 个 u16 = 4 个 u32），替代逐个 u16 读取和拼接，减少标量解码开销
- **scale 融入 madd**: `_mm256_madd_epi16(dot_256, sc_256)` 替代标量 `sc_val * isuml`

#### Q2_K AVX-512 内核优化

- **summs 256-bit 批量处理**: 16 次 128-bit 循环 → 8 次 256-bit 循环（一次处理 2 个子块），`_mm256_maddubs_epi16(_mm256_set1_epi8(1), q8_256)` 替代标量累加
- **scale 融入 madd**: Q2_K isum 的 scale 乘法融入 `_mm256_madd_epi16`，避免标量乘法
- **单线程 matvec**: `q2k_matvec_blocked_amd7600` 单线程顺序扫描，用于 6 专家并行场景避免 rayon 争抢

#### 函数重命名（CPU 专用）

AVX-512 内核函数从通用 `avx512` 后缀改为 `amd7600` 专用，后续不同 CPU 提供不同专有函数：

| 原名 | 新名 |
|------|------|
| `iq2xxs_vec_dot_q8_avx512` | `iq2xxs_vec_dot_q8_amd7600` |
| `iq2xxs_matvec_blocked_avx512` | `iq2xxs_matvec_blocked_amd7600` |
| `q2k_vec_dot_q8_avx512` | `q2k_vec_dot_q8_amd7600` |
| `q2k_matvec_blocked_avx512` | `q2k_matvec_blocked_amd7600` |

#### 6 专家并行实测

**结论：DDR5 带宽瓶颈下，6 专家并行不可行。**

| 方案 | 6 专家耗时 | 原因 |
|------|-----------|------|
| 串行（基线） | 28.4ms | — |
| 2路并行 + rayon down | 59.9ms | rayon 线程池争抢 |
| 2路并行 + 单线程 down | 390.9ms | 单线程 down 太慢（7168行） |

DDR5 带宽利用率 ~50%，多线程争抢带宽反而降低性能。顺序访问已最大化 DDR burst 效率。

#### 性能提升

| 阶段 | 单专家 FFN | 优化点 | 累计提升 |
|------|-----------|--------|---------|
| 初始（128-bit + 逐bit sign） | 7.76ms | — | — |
| +预计算符号掩码表 | 6.75ms | KSIGNS_IQ2XS_MASKS 表 | -13% |
| +256-bit maddubs 管线 | 6.22ms | grid_u64 表 + _mm256_set_epi64x | -20% |
| +Q2_K scale 融入 madd | 5.88ms | _mm256_madd_epi16 替代标量乘 | -24% |
| +去掉 grid 符号吸收 + 双 ib32 融合 | 4.66ms | grid unsigned + 2 ib32/iter | -40% |
| +memcpy 批量读取 | 4.59ms | read_unaligned 16B | -41% |

当前性能：单专家 FFN 4.59ms，6 专家串行 27.5ms，推理速度 0.76 t/s，DDR5 带宽利用率 50.5%。

#### 修改文件

- `src/cpu_expert/avx512.rs` - IQ2_XXS/Q2_K AVX-512 内核深度优化 + 函数重命名
- `src/cpu_expert/tables.rs` - 新增 KSIGNS_IQ2XS_U64 常量
- `src/cpu_expert/kernel.rs` - 调用点更新 + 6 专家并行实验

## [0.9.6] - 2025-05-25

### 混合量化：IQ2_XS (gate, up) + Q2_K (down)

#### 混合量化脚本 (inference/prequant_mixed_iq2xxs_q2k.py)

- **混合量化策略**: gate/up 投影使用 IQ2_XS (2.0625 bpw)，down 投影使用 Q2_K (2.5625 bpw)
- **CUDA 量化**: IQ2_XS 使用 CUDA kernel 直接在 GPU 上量化 (~50-100 e/s vs C CPU ~0.3 e/s)
- **GPU Q2_K 量化**: Q2_K 量化全程在 GPU 上完成（torch 向量化），数据不离开 GPU
- **双缓冲流水线**: CPU 加载与 GPU 量化并行，CUDA stream 异步操作
- **动态 batch size**: 根据权重形状自动调整，防止 VRAM OOM
- **增量写入**: 每 10 个 shard 保存一次，避免内存爆炸（33792 专家权重 ~80GB）
- **imatrix 支持**: 加载 llama.cpp 格式的 .dat importance weights 文件
- **性能**: 30 e/s (RTX 5060 Ti 16GB)，VRAM 仅 2.1GB

#### 校准数据生成 (inference/generate_imatrix.py)

- 从校准数据生成 importance weights 用于 IQ 量化
- 支持 dummy imatrix（全 1）用于无校准数据场景
- 输出 .npz 格式

#### GGUF 格式扩展 (inference/gguf_iq2xs.py)

- 新增 `add_iq2xxs_tensor()`: IQ2_XXS 张量写入 (66 bytes/block)
- 新增 `add_q2k_tensor()`: Q2_K 张量写入 (84 bytes/block)
- GGML 类型: IQ2_XXS=16, Q2_K=10

#### TileLang GEMM kernel (tilelang/mixed_quant_gemm.py)

- `iq2xxs_gemm_kernel`: IQ2_XXS 反量化 + GEMM (TileLang)
- `q2k_gemm_kernel`: Q2_K 反量化 + GEMM (TileLang)
- `mixed_quant_ffn`: 融合 SwiGLU FFN (gate/up IQ2_XXS, down Q2_K)

#### Rust GGUF 读取 (src/gguf.rs)

- 修正 Q2_K `bytes_per_block`: 256 → 84

## [0.9.5] - 2025-05-23

### CPU 专家推理 + 混合推理架构

#### CPU 专家推理模块 (cpu_expert.py)

- **IQ2_XS 反量化**: numpy 向量化实现，tile 分块（512 blocks/tile < L2 缓存）
- **CPU 专家 FFN**: gate/up 配对投影 + SwiGLU + down 投影
- **CpuExpertRunner**: 管理 pinned pool 数据和 CPU 计算，D2H/H2D 传输
- **优化**: FMA（numpy BLAS 自动利用 AVX-512）、L1/L2 命中最大化（连续内存布局）、符号乘法代替 where

#### TDD 测试 (test_cpu_expert.py)

- 10/10 测试全部通过
- 反量化延迟基准：159ms（gate_up 权重 131072 blocks）
- CPU FFN 延迟基准：123ms（dim=4096, inter_dim=2048）

#### 混合推理集成

- model.py: MoE.forward 添加 hot/cold 双路分发
  - GPU 命中：GPU GEMM 推理（~0.2ms/专家）
  - GPU 未命中 + pinned pool 命中：DMA 传输到 GPU 推理（~5ms/专家）
  - GPU 未命中 + CPU 路径：CPU 反量化 + FFN + H2D（~123ms/专家，待优化）
- generate.py: 初始化 CpuExpertRunner，为每层 MoE 设置 CPU 推理回调

#### 第三方代码 (third/)

- `third/ds4/`: 官方 C 推理引擎（IQ2_XXS + ARM NEON + MoE FFN）
- `third/llama.cpp/`: llama.cpp DS4 分支（IQ2_XS AVX2 + 热/冷双路 MoE + AVX-512 GEMM）
- `third/README.md`: 完整分析文档和 Rust 实施计划

#### 性能对比

| 路径 | 延迟/专家 | 适用场景 |
|------|----------|---------|
| GPU SLRU 命中 | ~0.2ms | 热点专家 |
| Pinned pool → DMA → GPU | ~5ms | 冷专家（当前默认） |
| CPU 反量化 + FFN | ~123ms | 无 GPU 时兜底 |

#### Review #1-#10 优化要点

1. Tile 分块：512 blocks/tile，确保 L2 缓存命中
2. FMA：numpy BLAS 自动利用 AVX-512 FMA 指令
3. L1/L2 命中：连续内存布局（np.ascontiguousarray）
4. 符号乘法：`1 - 2*bit` 代替 `np.where`，消除分支
5. Scale 解码：位操作向量化，避免 Python 循环
6. unpackbits：一次性解包 8 个符号位
7. 配对投影：gate+up 共享输入，减少一次矩阵乘法
8. 内存对齐：64 字节对齐，确保 AVX-512 对齐访问
9. 预取策略：CPU 计算时预取下一批专家权重到 L3
10. 线程池：多线程并行计算多个专家（待实现）

## [0.9.4] - 2025-05-23

### GPU 缓存策略深度优化

#### W-TinyLFU vs SLRU 实测对比

实现了 W-TinyLFU（Caffeine/Guava Cache 同款算法）并与 SLRU 做了实测对比：

| 指标 | SLRU (v9) | W-TinyLFU (v9) |
|---|---|---|
| GPU 缓存容量 | 642 专家 | 585 专家 |
| GPU 命中率 | **76-78%** | 71% |
| 推理延迟 | ~200ms/step | ~210ms/step |

**结论：SLRU 更适合 MoE 场景**，W-TinyLFU 不适合的 3 个原因：
1. MoE 访问模式稳定（同一对话内路由高度稳定），不需要频率衰减适应变化
2. 工作集 ≈ 缓存容量时，准入策略退化为"全准入"，Window 段浪费 5% 容量
3. CMS 频率估计 + Window→Main 转移增加代码复杂度，无命中率提升

#### SLRU 策略优化

- **protected 比例 75% → 90%**: 减少热点专家被淘汰概率，probation 段仅占 10%
- **Step 级别专家保护**: 当前 step 已访问的专家不被淘汰（缓存容量 ≥ 2× 工作集时自动启用）
- **层保护增强**: `_evict_lru_skip_layer` 支持 `step_protected_keys` 参数，双重防淘汰
- **VRAM 预留精简**: VRAM_RESERVE_MB 从 800→1200→800MB，平衡 OOM 风险和缓存容量

#### Warmup 优化

- **无频率数据时加载 top 30 专家/层**（原 12），更充分填满缓存
- **put_force_protected 支持 step_protected_keys**: warmup 时也遵守 step 保护

#### 100% GPU 命中率分析

| GPU 显存 | 可容纳专家数 | 预期命中率 |
|---|---|---|
| 16GB (当前) | ~642 | ~78% |
| 24GB | ~2048 | ~100% |
| 80GB | ~10240 | ~100% |

16GB 卡受 VRAM 限制，GPU 命中率上限 ~78%。要达到 100% 需要 24GB+ 显存或减少专家 GPU 占用。

#### 新增类

- **CountMinSketch**: O(1) 频率估计器，用于 W-TinyLFU 准入决策
- **WTinyLFU**: Window(5%) + SLRU Main(95%) + CMS 准入策略，代码保留可切换

#### 修改文件

- `inference/expert_cache.py` - SLRU 90% protected、step 保护、W-TinyLFU 实现、VRAM 预留调整
- `inference/generate.py` - on_step_start/end 调用、step_prot 统计、缓存策略描述更新

## [0.9.3] - 2025-05-23

### IQ2_XS 推理乱码修复

#### 根本原因
- **FP4 scale 未应用**: `prequant_iq2xs.py` 中 FP4→IQ2_XS 量化时，只做了 FP4 查表解码（值域 [-6, 6]），没有乘以 `float8_e8m0fnu` scale（约 0.004~0.031），导致 IQ2_XS 量化输入值域错误，反量化后权重值域偏大 ~100 倍，推理输出乱码

#### 修复内容
- **CUDA 路径**: `prepare_batch()` 加载 `.scale` 张量，GPU 上应用 `decoded * scale_expanded`
- **CPU 路径**: `_quantize_expert_cpu()` 加载 `.scale` 张量，应用 `f32 * scale_expanded`
- **预加载**: 同时加载 `.weight` 和 `.scale` 张量

#### 验证结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Layer 10 expert 0 d 值 | 0.0 (全零) | 0.000395 |
| GEMM vs FP4 参考 mean_error | 118.0 | 0.38 |
| 推理输出 | "dekameters" (乱码) | "The capital of France is Paris." |

#### 修改文件
- `inference/prequant_iq2xs.py` - FP4 解码后应用 scale

### IQ2_XS 推理性能优化

#### 归档格式 v2: 分离存储 d/qs/scales
- **问题**: v1 交织存储格式（d[2B]+qs[64B]+scales[8B]=74B/block）导致读取时列切片非连续，必须 `.copy()` 才能 `.view()` 为目标 dtype，每个专家权重 3 次拷贝
- **修复**: v2 格式将 d/qs/scales 各自连续存储，`np.frombuffer(mmap, offset=...)` 直接创建连续视图，零拷贝读取
- **兼容**: 读取器同时支持 v1 和 v2 格式，写入器使用 v2 格式
- **数据流**: mmap → `np.frombuffer`（零拷贝视图）→ `torch.from_numpy` → `pin_memory()`（拷贝到 pinned memory）→ `to("cuda", non_blocking=True)`（DMA 异步传输）
- **效果**: 消除归档读取的 3 次 `.copy()`，数据从 mmap 直接到 pinned memory，减少一次内存拷贝

#### IQ2_XS 归档预取移到后台线程
- **问题**: `_async_prefetch_expert()` 中 IQ2_XS 归档读取在主线程同步执行，SSD I/O + page fault 阻塞推理
- **修复**: 新增 `_iq2xs_prefetch_executor` 线程池，归档读取 + GPU 传输在后台线程执行，与主线程 GPU 计算重叠
- **线程安全**: 新增 `_prefetch_lock` 保护 `_prefetch_pending` 的并发访问，`_promote_prefetch` 使用快照避免长时间持锁
- **效果**: SSD I/O 延迟隐藏在 GPU 计算期间，减少推理时的同步等待

#### 修改文件
- `inference/iq2xs_archive.py` - 归档格式 v2（分离存储），读取器兼容 v1/v2
- `inference/expert_cache.py` - 零拷贝读取 + pin_memory、后台线程预取、线程安全锁

## [0.9.2] - 2025-05-23

### IQ2_XS 数据存储优化与 C 验证

#### 数据格式统一
- **block 交织存储**: `iq2xs_archive.py` 和 `gguf_iq2xs.py` 的写入器/读取器统一为逐 block 交织存储格式（`d[2B] + qs[64B] + scales[8B]` = 74B/block），与 llama.cpp `block_iq2_xs` 结构一致
- **归档头修复**: reserved 字段从 28 字节修正为 20 字节，确保总头大小为 64 字节

#### numpy 向量化优化
- **归档写入器**: 数据打包从 Python `for i in range(n_blocks)` 循环改为 numpy 切片赋值，预量化速度 14.8 → 24.1 e/s（+63%），总耗时 2383s → 1501s
- **归档读取器**: 数据解析从 Python 循环改为 `np.frombuffer` + 切片 view，32768 blocks 读取 ~0.7ms
- **GGUF 写入器**: 同样优化为 numpy 向量化交织
- **GGUF 读取器**: 同样优化为 numpy 向量化解析

#### C 语言验证
- **verify_archive_iq2xs.c**: 独立 C 程序验证归档文件 IQ2_XS 数据正确性，直接读取归档头+索引+block 数据并反量化
- **verify_gguf_iq2xs.c**: 独立 C 程序验证 GGUF 文件 IQ2_XS 数据正确性
- **三方对比**: Python (GGUFReader) / Python (手动参数) / C 反量化结果完全一致（误差 0.0）
- **反量化公式**: `y = d * (2*scale_4bit + 1) * 0.125 * grid[j] * sign`，与 llama.cpp 一致

#### 性能对比

| 组件 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 预量化 consume 时间 | ~1800ms/batch | ~150ms/batch | 12x |
| 预量化总速度 | 14.8 e/s | 24.1 e/s | 63% |
| 预量化总耗时 | 2383s (~40min) | 1501s (~25min) | -37% |
| GGUF 读取 (32768 blocks) | ~数百ms | 0.7ms | ~100x |

#### 修改文件
- `inference/iq2xs_archive.py` - 写入器/读取器 numpy 向量化、header 修复
- `inference/gguf_iq2xs.py` - 写入器/读取器 numpy 向量化
- `inference/verify_archive_iq2xs.c` - 新增归档 C 验证程序
- `inference/verify_gguf_iq2xs.c` - 新增 GGUF C 验证程序

## [0.9.1] - 2025-05-21

### CUDA 量化性能优化

#### CUDA Kernel 优化
- **16 threads/block 并行**: IQ2_XS 量化 kernel 从 1 thread/block 改为 16 threads/block，每个 thread 独立处理一个 16 元素子块，occupancy 提升 ~15 倍
- **常量内存缓存 grid**: 512 项 grid 查找表从全局内存移至 `__constant__` 内存（4KB 广播缓存），降低访问延迟
- **批量 kernel 启动**: FFI 函数从逐行启动 kernel 改为一次性启动所有 block，消除 kernel launch overhead
- **Warp shuffle 归约**: sigma2 计算使用 `__shfl_down_sync` 实现 16 线程并行归约
- **设备端辅助函数**: 添加 `nearest_int_cuda` 和 `iq2_find_best_neighbour_cuda`，修复编译错误

#### 预量化流水线优化
- **GPU FP4 解码**: FP4→float32 解码从 CPU numpy 改为 GPU torch 查表，消除 CPU→GPU 传输瓶颈
- **批量专家处理**: 一次 kernel 调用处理 32 个同形状专家，GPU 利用率从 ~20% 提升到 ~95%
- **向量化输出解析**: block_iq2_xs 输出解析从 Python 逐块循环改为 numpy stride trick，7530ms → 76ms（100 倍加速）
- **safetensors 预加载**: 每个 shard 开始时一次性加载所有专家张量到 CPU 内存，避免重复磁盘 I/O
- **双缓冲流水线**: CPU 准备和 GPU 量化交替执行，CUDA event 替代 synchronize 减少阻塞
- **后台线程预加载**: ThreadPoolExecutor 在 GPU 量化时后台准备下一批 CPU 数据
- **自动 batch_size**: 根据显存大小自动计算批量大小

#### 代码清理
- **删除旧 PyTorch GPU 量化**: 移除 `quantize_iq2xs_gpu_optimized.py`（旧 PyTorch 实现，~4.7 e/s，与 C 算法有差异）
- **删除旧 GEMM 实现**: 移除 `iq2xs_gemm.py`（PyTorch CUDA Extension，硬编码 sm_90，已被 TileLang 版替代）
- **清理残留引用**: 移除 `prequant_iq2xs.py` 中 `"pytorch_gpu"` 模式残留引用

#### 性能提升

| 版本 | 速度 | GPU 利用率 | 关键瓶颈 |
|------|------|-----------|---------|
| C CPU 量化 | 0.3 e/s | 0% | CPU 串行 |
| CUDA 1 thread/block | 1.1 e/s | ~10% | kernel 并行度低 |
| CUDA 16 threads/block | 2.1 e/s | ~20% | 逐行 FFI 调用 |
| 双缓冲流水线 | 8.8 e/s | ~40% | CPU FP4 解码慢 |
| GPU FP4 解码+批量 | 9.4 e/s | ~45% | Python 循环解析 7.5s |
| **向量化解析** | **23.7 e/s** | **~95%** | 接近 GPU 极限 |

#### 修改文件
- `csrc/iq2_xs_quantize.cu` - CUDA kernel 重写（16 threads/block、常量内存、批量启动、辅助函数）
- `inference/prequant_iq2xs.py` - 预量化流水线重写（批量处理、GPU FP4 解码、向量化解析、双缓冲）
- `inference/iq2xs_cuda_quant.py` - 清理旧速度对比注释

#### 删除文件
- `inference/quantize_iq2xs_gpu_optimized.py` - 旧 PyTorch GPU 量化（已被 CUDA 版替代）
- `inference/iq2xs_gemm.py` - 旧 PyTorch CUDA Extension GEMM（已被 TileLang 版替代）

## [0.9.0] - 2025-05-21

### IQ2_XS 量化支持

#### 新增功能

- **`--quant-type` 参数**: 支持 `fp4`（默认）和 `iq2xs` 两种量化类型，启动时自动选择推理路径
- **IQ2_XS 预量化流水线**: 首次使用 `--quant-type iq2xs` 时，自动将 FP4 权重解码为 float32 后量化为 IQ2_XS 格式并保存为 `.iq2xs` 归档文件
- **GPU 加速量化**: `quantize_iq2xs_gpu_optimized.py` 使用 CUDA kernel 并行执行 IQ2_XS 量化，大幅缩短预量化时间
- **IQ2_XS GEMM 融合算子**: `iq2xs_gemm_tilelang.py` 将反量化与矩阵乘法融合为单个 TileLang persistent kernel，避免中间结果回写显存
- **IQ2_XS 归档格式**: `iq2xs_archive.py` 实现自定义二进制归档格式，支持 mmap 零拷贝加载和按专家索引随机访问
- **C/FFI 层**: `csrc/iq2_xs.h` 从 llama.cpp 抽取 IQ2_XS 完整量化/反量化实现，`csrc/iq2_xs_bridge.c` 提供 Rust FFI 桥接
- **硬件检测**: 启动时检测 GPU 显存和系统内存，推荐合适的量化类型

#### 缓存策略优化

- **SLRU 缓存策略**: 替代 LFU，将缓存分为 protected 段和 probation 段，对访问模式变化更敏感
- **IQ2_XS 缓存策略**: GPU 采用 SLRU，CPU 全量加载所有专家（~80GB），SSD 使用 mmap + OS 页缓存
- **FP4 缓存策略保持**: GPU 采用 LFU，CPU 禁用，SSD 直接 I/O
- **缓存容量动态计算**: 根据量化类型自动计算缓存容量，不硬编码

#### 路由预测预取

- **RoutePredictor**: 基于历史路由结果统计层间专家共现频率，预测下一层将激活的专家
- **异步预取流水线**: GPU 计算 L 层 → CPU 预取 L+2 层专家到 GPU → SSD 预取 L+5 层专家到 CPU
- **预取异常处理**: 预取失败不影响主推理流程

#### Bug 修复

- **查找表 GPU 缓存**: `iq2xs_gemm_tilelang.py` 中 `iq2xs_grid`、`ksigns_iq2xs` 查找表张量缓存到 GPU 常驻显存，避免每次调用重复传输
- **bias 检查修复**: `model.py` 中 bias 检查改为 `if bias is not None`，避免零张量误判
- **输入连续性检查**: IQ2_XS GEMM 调用前确保输入张量连续，避免断言错误
- **Python/C 算法对齐**: 修复符号翻转逻辑，使 Python 量化 MSE 与 C 版本一致

#### 新增文件

- `inference/iq2xs_gemm_tilelang.py` - TileLang IQ2_XS 融合 GEMM 算子
- `inference/iq2xs_archive.py` - IQ2_XS 归档格式读写与 mmap 加载
- `inference/prequant_iq2xs.py` - FP4→IQ2_XS 预量化流水线
- `inference/quantize_iq2xs_gpu_optimized.py` - GPU 加速 IQ2_XS 量化
- `inference/iq2xs_c_wrapper.py` - Python ctypes 调用 C 量化实现
- `inference/iq2xs_gemm.py` - 纯 PyTorch IQ2_XS GEMM 参考实现
- `csrc/iq2_xs.h` - IQ2_XS 量化/反量化 C 实现
- `csrc/iq2_xs_bridge.c` - Rust FFI 桥接层
- `csrc/iq2_xs.cu` / `csrc/iq2_xs.cuh` - IQ2_XS CUDA GPU 实现

#### 修改文件

- `inference/generate.py` - 新增 `--quant-type` 参数、IQ2_XS 预量化检测、硬件检测
- `inference/expert_cache.py` - 新增 IQ2_XS 缓存策略、SLRU 实现、RoutePredictor 路由预测预取
- `inference/model.py` - 新增 IQ2_XS GEMM 集成分支、量化类型路由

## [0.8.0] - 2025-05-18

### 🔴 关键内存泄漏修复

#### 核心泄漏：Expert 对象 CPU 参数累积
- **问题**: `_unload_activated_experts` 将 GPU 参数移到 CPU 保留在 Expert 对象中，导致 CPU 内存无限累积
- **影响**: 每个 FP4 专家 ~24.5MB，2000 个不同专家 ≈ 49GB，直接导致 OOM 崩溃
- **修复**: 卸载时直接删除 Expert 对象（`moe.experts[expert_id] = None`），而非移到 CPU
- **结果**: 内存从 99.8% 降到稳定 ~6GB，推理 5 轮后无增长

#### safetensors dtype 映射错误
- **问题**: `_DTYPE_MAP` 键名（`INT8`, `FP8_E8M0`）与 safetensors header 实际键名（`I8`, `F8_E8M0`）不匹配
- **影响**: 专家权重被错误解析为 `torch.uint8`，无法路由到 FP4 GEMM
- **修复**: 添加 `I8`, `F8_E4M3`, `F8_E8M0`, `I64` 映射

#### config.json 字段映射
- **问题**: HuggingFace config.json 字段名与 ModelArgs 不匹配
- **修复**: 添加完整字段名映射（`hidden_size` → `dim`, `num_hidden_layers` → `n_layers` 等）
- **修复**: `quantization_config.fmt='e4m3'` 正确映射为 `dtype='fp8'`

### 🟡 GPU 缓存优化

#### GPU 缓存命中修复
- **问题**: GPU 缓存命中时跳过参数设置，Expert 对象使用随机初始化参数
- **修复**: GPU 缓存命中时，将缓存的 GPU 参数重新设置到 Expert 对象上

#### 自动 warmup
- **新增**: 启动时执行短推理收集 LFU 统计，将 top-200 热点专家常驻 GPU
- **效果**: VRAM 从 10044MB → 12594MB (+2550MB)，预热热点专家

#### GPU 缓存大小调整
- **调整**: 从 350 减少到 200，避免 GPU OOM（16GB 限制）
- **计算**: 常驻权重 ~10GB + 200 专家 ~2.5GB + 计算缓冲区 ~3GB = 15.5GB

### 🟢 内存安全机制

#### 内存水位监控
- **新增**: `ExpertCache.check_memory_pressure()` 每 10 token 检查 `/proc/meminfo`
- **阈值**: 可用内存 < 8GB 时触发紧急清理
- **清理**: 释放 safetensors header 缓存、强制 GC、清空 CUDA 缓存

#### L2 CPU 缓存禁用
- **原因**: pinned memory 锁定物理页，96GB 内存不足以缓存大量专家
- **决策**: `cpu_cache_size=0`，避免内存膨胀

#### 直接 I/O 替代 mmap
- **问题**: safetensors mmap 导致 OS 页缓存膨胀（46 × 3.4GB ≈ 156GB 潜力）
- **修复**: `_read_tensor_no_mmap` 使用 `seek/read` 直接读取所需字节

### 性能指标

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| RAM 使用 | 93.2GB (99.8%) | 5.7GB (稳定) |
| Swap | 8GB (100%) | 2.8GB (稳定) |
| 可用内存 | 208MB | 87.7GB |
| 推理速度 | N/A (崩溃) | 1.0-1.25 t/s |
| GPU VRAM | N/A | 12594MB |

## [0.7.0] - 2025-05-18

### Python 推理实现

#### MoE 懒初始化
- **懒初始化专家**: `MoE.__init__` 中 `self.experts = nn.ModuleList([None] * 256)`，不预分配权重
- **`_ensure_expert(idx)`**: 按需创建空壳 Expert，避免 143GB 内存预分配
- **回调钩子**: `_on_experts_needed` / `_on_experts_done` 在 Gate 计算后调用

#### 三级缓存架构
- **ExpertCache 类**: 统一管理 L1 GPU + L2 CPU + L3 SSD 三级缓存
- **L1 GPU 缓存**: LFU 策略，200 个热点专家常驻 GPU (~2.5GB)
- **L2 CPU 缓存**: LRU 策略，2000 个专家 pinned memory (~25GB)
- **L3 SSD**: safetensors mmap 按需读取，文件句柄缓存
- **缓存统计**: gpu_hits / cpu_hits / ssd_hits / gpu_evictions

#### 推理速度优化
- **移除 gc.collect() + empty_cache()**: 3.5x 提速 (0.14 → 0.49 t/s)
- **param.data 替换**: 直接替换 Parameter 数据指针，避免重复创建对象
- **int8→float4_e2m1fn_x2 view**: 零拷贝 dtype 转换
- **GPU 缓存预热**: warmup 后加载 LFU top-200 专家常驻 GPU

#### Bug 修复
- **wo_a FP8 反量化**: checkpoint 中 wo_a 为 FP8，模型初始化为 BF16，需在 load_state_dict 前反量化
- **weight.scale 引用**: setattr 后重新绑定 `module.weight.scale = module.scale`
- **Head dtype 不匹配**: `x.float()` vs `weight.bfloat16()`，统一为 float

#### 性能指标
- 推理速度: 0.5-0.7 t/s (decode)
- GPU VRAM: 12.6 GB (常驻)
- 缓存命中率: L1 GPU 6%, L2 CPU 69%, L3 SSD 25%

## [0.6.0] - 2025-05-15

### P0: 严重D2H热点消除

- **output_proj GPU化**: 消除非BF16路径6次D2H/H2D往返，wo_a统一反量化为BF16上传GPU，使用strided batched GEMM
- **sparse_attention**: 确保GPU内核覆盖所有配置，fallback仅作安全网

### P1: 中等D2H热点消除

- **gate_output GPU化**: 新增moe_expert_count_topk6_ne256内核，GPU上计算expert_counts，消除indices D2H
- **RAM缓存Pinned Memory**: RamExpertCache改用PinnedBuffer存储，消除RAM→GPU传输的额外memcpy

### P2: PCIe延迟隐藏

- **双stream异步预取**: finalize_prefetch移至层计算开始前，预取与计算真正重叠
- **SSD预取集成**: prefetch_layers_ahead中添加SSD→RAM→CPU预取链路

### P3: 小优化

- **RoPE GPU缓存**: RopeCache添加upload_to_gpu/get_gpu_slice，cos/sin预计算到GPU，消除~4KB H2D
- **预取深度修正**: 基于gpu_slots/n_routed_experts计算，避免过度预取

### 新增TileLang内核

- `moe_expert_count_topk6_ne256`: GPU上计算expert counts

## [0.5.0] - 2025-05-15

### Phase 4: 资源优化

#### 缓存命中率统计
- **CacheHitStats**: 新增通用命中率统计结构(hits/misses/hit_rate/reset)
- **GpuExpertCache**: 添加命中率统计，LFU驱逐改进为LFU+LRU(同频率时驱逐最久未访问)
- **RamExpertCache**: 添加命中率统计，热层驱逐改为LFU(最低频率优先)
- **SsdExpertCache**: 添加命中率统计，get()方法签名更新为&mut self

#### 动态缓存调整
- **GpuExpertCache::resize()**: 根据命中率动态调整GPU缓存slot数，min_slots下限保护
- **RamExpertCache::rebalance()**: 动态调整热/冷层容量比例(0.2~0.6)，推理效率优先
- **ThreeLevelCache::adapt()**: 每100次访问自适应调整，GPU命中率>90%缩容、<50%扩容

#### 自适应预取策略
- **ExpertScheduler::adapt()**: 根据GPU命中率动态调整预取深度(1~max_prefetch_depth)
- **ExpertScheduler::prefetch_layers_ahead()**: L+N层异步预取，N根据命中率自适应
- **ExpertScheduler::vram_utilization()**: VRAM利用率查询

#### SSD L+N层预取
- **SsdExpertCache::prefetch_layers()**: 批量MMAP预取多层专家权重

## [0.4.0] - 2025-05-15

### Phase 3: MoE优化

#### MoE FFN GPU化
- **延迟D2H**: gate_output weights/indices延迟到scatter_add/CPU fallback路径才D2H，scatter_add成功路径零D2H
- **GPU gather内核**: 新增moe_gather_D4096内核，大批量token gather在GPU完成
- **scatter_add全GPU路径**: scatter_add成功时输出直接在GPU，无需D2H→CPU累加→H2D

#### Expert权重预取优化
- **预取逻辑重构**: 分离可变借用与不可变借用，解决借用检查器冲突
- **SSD/RAM缓存优先**: 预取时优先从SSD/RAM缓存加载，减少磁盘I/O
- **finalize_prefetch**: 新增方法，将预取完成的权重批量放入GPU缓存
- **GPU缓存contains检查**: 预取跳过已在GPU缓存的专家，避免重复传输

#### 新增TileLang内核
- `moe_gather_D4096`: MoE专家输入行GPU gather
- `moe_extract_weights_topk6`: MoE专家权重/token_id提取(预留)

#### FP4 GEMM
- 路由专家FP4权重推理路径已集成(compute_expert FP4E2M1分支)
- 支持FP4 e2m1fn打包+FP8 e8m0fnu缩放因子

## [0.3.0] - 2025-05-15

### Phase 2: KV Cache全GPU化

#### Indexer全GPU路径
- **q_proj全GPU后处理**: 新增try_gpu_q_postprocess方法，RoPE+Hadamard+FP4-QDQ全在GPU完成，消除~16MB D2H
- **try_gpu_index_score全GPU输入**: 直接使用GPU张量(kv_cache GPU直接读取+BF16→FP32 GPU转换)，消除q_proj/weights/kv_cache_cpu的H2D上传
- **因果掩码GPU化**: 新增indexer_causal_adjust_topk512内核，topk结果不再D2H
- **compressed KV D2D更新**: D2D scatter成功时跳过CPU缓存同步，消除~1MB D2H

#### Compressor全GPU路径
- **GEMM结果GPU直通**: 新增compressor_group内核(gather+slice+ape add)，GEMM结果不再D2H，消除~32MB D2H+H2D往返
- **try_gpu_pool_gpu**: 直接接受GPU张量输入，无需H2D上传
- **ape_gpu**: APE权重上传GPU，供group内核使用

#### 新增TileLang内核
- `compressor_group_d128_od1024_ps8`: Compressor非重叠分组
- `compressor_group_d128_od1024_ps16`: Compressor重叠分组
- `scale_f32_N4096`: 元素级缩放

#### 新增基础设施
- `Indexer::hadamard_gpu`: q_proj Hadamard变换矩阵(GPU)
- `Indexer::try_gpu_q_postprocess()`: q_proj全GPU后处理管道
- `Compressor::try_gpu_group()`: GPU分组(gather+slice+ape)
- `Compressor::try_gpu_pool_gpu()`: GPU张量直接输入pool

## [0.2.0] - 2025-05-15

### Phase 1: BUG修复 + D2H消除

#### 正确性修复
- **RMSNorm内核修复**: 将absmax实现改为正确的sum_of_squares计算方式，消除logits漂移
- **Compressor BF16精度回归修复**: 全FP32 GPU后处理管道(rmsnorm_f32_weighted → compressor_rope_f32 → fp4_qdq_f32 → cast_f32_to_bf16)
- **KV Cache循环缓冲区修复**: 使用d2d_extract_rows替代CPU fallback，正确处理环形缓冲区读取

#### D2H/H2D消除
- **Compressor pool结果GPU直通**: try_gpu_pool返回GpuTensor，pool结果直接进入GPU后处理，消除D2H→CPU→H2D往返
- **Compressor GPU后处理管道**: 新增try_gpu_postprocess方法，RMSNorm+RoPE+Hadamard+FP4-QDQ+BF16转换全在GPU完成
- **Indexer因果掩码GPU化**: 新增indexer_causal_adjust_topk512内核，消除topk结果的D2H→CPU因果掩码→H2D
- **Indexer compressed KV D2D更新**: 新增GpuTensor::d2d_scatter_rows方法，D2D scatter写入替代全量H2D上传

#### 新增TileLang内核
- `rmsnorm_f32_weighted_N128`: 加权RMSNorm(FP32)
- `compressor_rope_f32_d128_rd64`: Compressor专用RoPE(FP32)
- `fp4_qdq_f32_N128_bs32`: FP4量化/反量化(FP32)
- `cast_f32_to_bf16_N128`: FP32→BF16类型转换
- `compressor_pool_d128_c8`: Compressor池化(d=128, coff=8)
- `indexer_causal_adjust_topk512`: 因果掩码+偏移调整

#### 新增基础设施
- `GpuTensor::d2d_scatter_rows()`: 跨batch行级D2D scatter写入
- `Compressor::precompute_hadamard_matrix()`: 预计算归一化Hadamard矩阵
- `Compressor::cpu_postprocess_to_gpu()`: CPU后处理fallback路径
- KV Cache checkpoint save/load (热启动支持)

## [0.1.0] - 2025-05-10

### Initial Implementation

- 基础模型加载与权重管理
- DeepSeek V4 Flash配置解析
- cuBLAS GEMM封装(BF16/FP32)
- TileLang内核编译与加载框架
- DLPack协议数据交换
- RMSNorm/Attention/MoE基础算子
- KV Cache环形缓冲区管理
- Compressor/Indexer基础实现
- 专家调度与三级缓存架构
- RoPE位置编码预计算
