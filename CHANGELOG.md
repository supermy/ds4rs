# Changelog

All notable changes to this project will be documented in this file.

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
