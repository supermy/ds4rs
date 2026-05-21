# Changelog

All notable changes to this project will be documented in this file.

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
