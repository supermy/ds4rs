容器模型目录调整为读写；

结论 ：当前硬件配置（Ryzen 5 7600 + DDR5-4800）下，CPU FFN 已优化至 4.59ms/专家，接近 DDR5 带宽极限（50.5% 利用率，计算开销 49%）。

- 计算瓶颈 + 内存瓶颈并存：带宽利用率 50.5%，CPU 计算速度跟不上 DDR5 带宽
- 6 专家并行不可行：DDR5 双通道无法从多线程读取中获益，实测 2 路并行比串行慢 3-14 倍
- IQ2_XXS 标量解码是剩余瓶颈：aux32→aux8→grid 查表→sign 查表→_mm256_set_epi64x 数据依赖链无法被 SIMD 加速
- CPU FFN 函数已重命名为 amd7600 专用，后续不同 CPU 提供不同专有函数

### 单层时间线
每层 30.5ms
├── Attention (GPU)          2.0ms   6.6%
├── 6× CPU FFN (串行)       27.5ms  90.2%  ← 瓶颈
│   ├─ 6× gate (IQ2_XXS)   8.4ms  27.5%
│   ├─ 6× up   (IQ2_XXS)   7.8ms  25.6%
│   ├─ 6× down (Q2_K)     11.7ms  38.4%
│   └─ 6× SwiGLU+Q8        0.5ms   1.6%
└── Shared Expert (GPU)     1.0ms   3.3%

单专家 FFN 4.59ms (min)
├── gate (IQ2_XXS)    1.45ms  30.3%  ← 标量解码瓶颈
├── up   (IQ2_XXS)    1.26ms  26.2%  ← 标量解码瓶颈
├── down (Q2_K)       1.95ms  40.6%  ← 最大瓶颈
├── Q8量化(x)         0.02ms   0.4%
├── SwiGLU            0.06ms   1.3%
└── Q8量化(mid)       0.06ms   1.3%

优化历程：7.76ms → 4.59ms（-41%）
├── 预计算符号掩码表          6.75ms  -13%
├── 256-bit maddubs 管线     6.22ms  -20%
├── Q2_K scale 融入 madd     5.88ms  -24%
├── 去掉 grid 符号吸收 + 双 ib32 融合  4.66ms  -40%
└── memcpy 批量读取          4.59ms  -41%

docker exec ds4rs-dev bash -c "cd /workspace/inference && python generate.py --ckpt-path /models --config /models/config.json --input-file /workspace/test_prompt.txt --quant-type iq2xxs_q2k --max-new-tokens 50 --temperature 0.6 --top-p 0.95 --min-p 0.01 "


kvcache 支持？

mixed_iq2xxs_q2k
用下面的校准数据重新生成量化文件
/workspace/gguf/imatrix.dat
优化校准数据生成脚本，根据现有的硬件环境，在cuda 16GB 内存80GB 生成,充分使用显存。重新生成校准数据集所需时间。

纯rust推理；

cpu-gpu 混合推理达到 10t/s
