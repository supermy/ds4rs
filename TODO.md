容器模型目录调整为读写；

参照python混合量化推理完善rust推理，内核尽可能使用tilelang；
计算图？

热/冷双路分发，需要将 MoE forward 中的逐专家循环改为 batch 化的 hot/cold 双路分发。
将 CPU FFN 计算延迟到下一层 attention 期间执行，GPU attention 和 CPU expert 并行。这是 ds4.rs 当前最大的优化空间。

### 混合推理时间线瓶颈分析
以 prefetch_layers=4（最优配置）为例：
Timeline (23 steps, total=46.3s, 2012ms/step)
├─ attn:    591ms (29.4%)
└─ ffn:    1422ms (70.6%)
   ├─ gate:      14ms (1.0%)
   ├─ expert:  1335ms (93.9%)  ← 瓶颈
   ├─ shared:    45ms (3.2%)
   └─ hc_over:   28ms (1.9%)

优化方向 ：
1. expert FFN 是绝对瓶颈（66%） ：提高 GPU 命中率 → 减少 CPU FFN 调用。当前 47% 命中率意味着 53% 走 CPU（0.48ms vs GPU 0.14ms）
2. attn 占 29% ：hc_mult=4 导致 4 倍注意力计算，可考虑动态 hc_mult
3. CPU FFN/GPU FFN 并行化 ：当前逐专家串行计算，可用多线程并行（6 专家可 2-3 路并行）
4. GPU FFN 延迟 0.14ms vs CPU 0.48ms ：GPU 仅快 3.4x，对 M=1 来说优势有限（kernel 启动开销占比大）

docker exec ds4rs-dev bash -c "cd /workspace/inference && python generate.py --ckpt-path /models --config /models/config.json --input-file /workspace/test_prompt.txt --quant-type iq2xxs_q2k --max-new-tokens 50 --temperature 0.6 --top-p 0.95 --min-p 0.01 "

✅ 从下面imatrix文件里面提取专家到gguf目录，然后进行混合精度测试：_load_raw() 支持打包格式，imatrix GGUF 推理正确（0.7 t/s，36.9% GPU 命中率）
/mnt/shared_data/models/antirez-ds4-macos/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf


热/冷双路分发最先进 p1
优先级 优化方法 预期收益 实现难度 适用条件
P0 专家延迟机制 吞吐 +45% 中 残差延迟容忍性验证（⚠️ 需修复 hc_post 集成，当前禁用）
P1 动态层内负载均衡 延迟 -20% 中 多专家 miss 时
P2 CPU-GPU 异步流水线 延迟 -30% 高 需重构 MoE forward
P3 延迟感知预取 命中率 +5% 低 替换频率排序 ✅ 已实现
P4 融合 MoE 算子 batch≥4 有收益 中 prefill 路径
P5 Cache-Friendly 分块 CPU FFN -10% 低 Rust 侧优化

dynamic-expert-update 运行时动态调整 推荐 ，prefill 时重分配

性能 （4×RTX 4090, Qwen3-Next-80B）：dynamic-expert-update 在 10% GPU 占比时 70.2 t/s vs uniform 56.6 t/s（+24%）

### DeepSeek-V4-Flash 支持（2026.05 新增）
- MXFP4 MoE + SGLang 混合推理
- 单 RTX 5090： 20+ tok/s
- --kt-method MXFP4 ：CPU 侧 MXFP4 格式
- --kt-num-gpu-experts 10 ：GPU 放 10 个专家
- MTP 投机解码：26.5 → 32.74 tok/s（+23%）

技术 ds4.rs 现状 可借鉴 
Dynamic Expert Update SLRU 频率驱动 ✅ prefill 时动态重分配 GPU 专家 
Token-wise Prefetch 层级预取（freq×latency） ✅ 
用 gate 输出预测下一 token 专家 NUMA-Aware 无（单 NUMA） ❌ 硬件不支持 
MXFP4 CPU Backend IQ2_XS/Q2_K ✅ 可考虑 FP4 CPU 路径 
MTP 投机解码 已评估，收益不高 ❌ 
Expert Deferral HC 架构下不可行 ❌ 
PreSched 跨层调度 逐层贪心 ✅ 可考虑跨层全局预取优化


传统串行：  [GPU Attn L0][CPU Expert L0][GPU Attn L1][CPU Expert L1]...
专家延迟：  [GPU Attn L0][GPU Attn L1]...
                         [CPU Expert L0][CPU Expert L1]...
                         ↑ CPU 计算与 GPU Attention 重叠

## 待办

mixed_iq2xxs_q2k
用下面的校准数据重新生成量化文件
/workspace/gguf/imatrix.dat
优化校准数据生成脚本，根据现有的硬件环境，在cuda 16GB 内存80GB 生成,充分使用显存。重新生成校准数据集所需时间。

专家延迟：使用fp4路由测试；如果是fp4路由注意内存溢出，mmap专家缓存；
跨层专家延迟在 HC 架构下 结构性不可行 ，不是 float32/bf16 精度问题。P1 层内重叠是当前正确的方案——CPU miss 异步提交、GPU hit 并行计算、层内等待合并，保证 FFN 输出完整后再进入 hc_post 。

如果未来要做跨层延迟，需要 跳过下一层的 hc_pre Sinkhorn ，直接将校正加到 hc_pre 的输出（即 norm 之前），但这需要更深的架构改动。

cpu-gpu 混合推理达到 10t/s

 docker exec ds4rs-dev bash -c "cd /workspace/inference && python generate.py --ckpt-path /models --config /models/config.json --input-file /workspace/test_prompt.txt --quant-type iq2xxs_q2k --max-new-tokens 50 --temperature 0.6 --top-p 0.95 --min-p 0.01 "

 docker exec ds4rs-dev bash -c "echo 'Hello' | DS4RS_MAX_SEQLEN=256 timeout 180 /workspace/target/release/ds4_cli"
