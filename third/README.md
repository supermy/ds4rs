# CPU 路由专家推理：分析与实施计划

## 1. 策略概述

**热点专家 GPU 推理 + 非热点专家 CPU 推理**

```
Token N:
  GPU: [Attention][Shared Expert][Hot Expert 1-6 (SLRU命中)]
  CPU: [等待hidden][Cold Expert 1-2 (pinned pool)][FFN AVX-512][回传]
```

混合推理流程：
1. GPU 计算 Attention + Shared Expert → hidden_state (BF16)
2. Gate 计算 TopK → 分离 hot/cold 专家
3. Hot 专家：GPU SLRU 命中 → GPU GEMM
4. Cold 专家：hidden_state D2H → CPU IQ2_XS 反量化 + AVX-512 GEMM → 结果 H2D
5. GPU 合并 hot + cold 结果 → 残差连接

## 2. 代码源分析

### 2.1 /data/ai/ds4 (ds4.c)

**量化格式**：IQ2_XXS（2-bit，非 IQ2_XS）
**SIMD 优化**：仅 ARM NEON，x86 为纯标量回退
**关键函数**：

| 函数 | 行号 | 说明 |
|------|------|------|
| `block_iq2_xxs` | 166-175 | IQ2_XXS 量化块：f16 d + uint16 qs[32] = 66B/256元素 |
| `iq2xxs_grid[256]` | 250-315 | 256 项查找表（uint64_t） |
| `iq2xxs_signed_grid` | 317-339 | 预计算带符号网格，pthread_once 延迟初始化 |
| `dot_iq2_pair_16()` | 341-358 | 16 元素 IQ2 点积核心（NEON + 标量） |
| `ds4_vec_dot_iq2_xxs_q8_K()` | 1715-1802 | IQ2_XXS × Q8_K 完整向量点积 |
| `ds4_vec_dot_iq2_xxs_pair_q8_K()` | 1804-1883 | 配对点积（gate+up 共享激活） |
| `swiglu()` | 4968-4972 | SwiGLU 激活 |
| `matvec_iq2_xxs_expert_pair_prequant()` | 3662-3701 | 单专家 gate+up 投影 |
| `matvec_iq2_xxs_experts_mid_prequant()` | 3719-3788 | 多专家 gate/up + SwiGLU |
| `matvec_q2_k_experts_accum_prequant()` | 3798-3899 | Q2_K down 投影累加 |
| `layer_routed_moe_one()` | 5244-5340 | 完整 MoE CPU 推理流程 |

**MoE CPU 数据流**：
```
x (float32) → Q8_K 量化 → TopK 选6专家 →
  对每个专家: IQ2_XXS gate/up 配对点积 → clamp → silu(gate)*up*weight →
  中间结果 Q8_K 量化 → Q2_K down 投影累加 → 输出 (float32)
```

### 2.2 llama.cpp (nisparks/deepseek-v4-support)

**量化格式**：IQ2_XS（2.3125-bit）
**SIMD 优化**：AVX2 + AVX1 + 标量，AVX-512 仅用于 F32/BF16 GEMM
**关键文件**：

| 文件 | 说明 |
|------|------|
| `ggml-common.h` | block_iq2_xs 结构体 + iq2xs_grid[512] 查找表 |
| `quants_x86.c` | AVX2 优化 IQ2_XS × Q8_K 点积 |
| `quants_generic.c` | 标量 fallback |
| `sgemm.cpp` | tinyBLAS 框架：AVX-512 F32/BF16/VNNI GEMM |
| `deepseek4.cpp` | DS4 模型图构建：热/冷双路 MoE 分发 |
| `llama-deepseek4-hot.h` | 热门专家管理器：GPU 固定 + 冷门 CPU 计算 |

**IQ2_XS AVX2 核心优化**（quants_x86.c:2654-2770）：
- `_mm256_and_si256(q2_data, m511)` 一次提取 16 个 9-bit grid index
- `_mm256_sign_epi8` 批量应用符号
- `_mm256_maddubs_epi16` + `_mm256_madd_epi16` 双层乘加
- 每次迭代处理 4×32=128 元素

**热/冷双路分发**（deepseek4.cpp:534-697）：
- `hot_remap_table`: 热门专家 → [0, K)，冷门 → K（哨兵）
- `cold_remap_table`: 冷门专家 → 原始 ID，热门 → 0
- GPU 热门路径：冷门 pick 命中哑专家（零权重），输出自然为 0
- CPU 冷门路径：热门 pick 被 mask 清零
- 最终 `ggml_add(hot_out, cold_out)` 合并

**AVX-512 GEMM**（sgemm.cpp）：
- F32: `tinyBLAS<16, __m512>` 4×6 微内核
- BF16: `_mm512_dpbf16_ps` 融合乘加（32 对 BF16 → 16 F32）
- VNNI: `_mm256_dpbusd_epi32` 无符号×有符号 8-bit 乘加

## 3. AMD Ryzen 5 7600 硬件特性

| 特性 | 参数 | 优化方向 |
|------|------|---------|
| 架构 | Zen 4 (6C/12T) | 多线程 + NUMA 感知 |
| L3 缓存 | 32MB 共享 | 专家权重分块 ≤ 32MB |
| AVX-512 | 支持 VNNI, BF16 | 量化 GEMM + BF16 推理 |
| DDR5 | 双通道 96GB (~60GB/s) | 带宽充足，延迟 ~80ns |
| PCIe 5.0 | x8 (~16GB/s) | D2H/H2D 传输 |

**关键优化点**：
1. **L3 缓存分块**：单个专家 ~7MB，4 个专家 ≈ 28MB < 32MB L3，可完全缓存
2. **AVX-512 VNNI**：`_mm512_dpbusd_epi32` 一次处理 64 个 int8 乘加
3. **AVX-512 BF16**：`_mm512_dpbf16_ps` 一次处理 32 对 BF16 乘加
4. **DDR5 双通道**：~60GB/s 带宽，7MB 专家加载 ~0.12ms

## 4. Rust 实施计划

### Phase 1: CPU IQ2_XS 反量化 + AVX-512 GEMM（2 周）

#### 1.1 IQ2_XS 反量化内核
- 移植 `quants_x86.c` 中 `ggml_vec_dot_iq2_xs_q8_K` AVX2 路径
- 添加 AVX-512 优化：`_mm512_and_si256` → `_mm512_and_si512`，32 个 ZMM 寄存器
- Rust 实现：`std::arch::x86_64` intrinsics 或 `core::simd`

```rust
// 目标 API
fn vec_dot_iq2_xs_q8_k_avx512(
    n: i32,
    x: &block_iq2_xs,  // IQ2_XS 权重
    y: &block_q8_k,     // Q8_K 激活
) -> f32;
```

#### 1.2 CPU 专家 FFN
- 移植 `ds4.c` 中 `matvec_iq2_xxs_expert_pair_prequant` 逻辑
- 适配 IQ2_XS 格式（block_iq2_xs vs block_iq2_xxs）
- 实现 gate/up 配对投影 + SwiGLU + down 投影

```rust
// 目标 API
fn expert_ffn_cpu(
    x: &[f32],           // 输入 hidden_state (dim=4096)
    gate_up: &Iq2XsWeight, // gate+up 权重 (IQ2_XS)
    down: &Iq2XsWeight,    // down 权重 (IQ2_XS 或 Q2_K)
    route_weight: f32,     // 路由权重
) -> Vec<f32>;            // 输出 (dim=4096)
```

### Phase 2: 混合推理集成（1 周）

#### 2.1 Python 原型
- 修改 `expert_cache.py`：非热点专家走 CPU 推理路径
- 添加 `ExpertCache.get_expert_cpu_params()` 方法
- 修改 `model.py`：CPU 专家 FFN 计算

#### 2.2 Rust 集成
- `expert.rs`：添加 `compute_expert_cpu()` 方法
- `moe.rs`：hot/cold 双路分发
- `cache.rs`：GPU SLRU 命中走 GPU，未命中走 CPU

### Phase 3: 性能优化（1 周）

#### 3.1 AVX-512 优化
- IQ2_XS 点积：AVX2 → AVX-512（ZMM 寄存器，一次处理 32 元素）
- GEMM 分块：L3 缓存感知（4 专家/块 ≈ 28MB）
- BF16 推理：hidden_state BF16 → `_mm512_dpbf16_ps`

#### 3.2 延迟优化
- D2H/H2D 异步：CPU 计算与 GPU 计算重叠
- 预取：CPU 计算冷专家时，GPU 预取下一批热专家
- 线程池：6 核 12 线程，每核处理 1 个专家

### Phase 4: 集成测试（1 周）

- 正确性验证：CPU 专家输出 vs GPU 专家输出（误差 < 1%）
- 性能基准：单专家延迟、吞吐量、GPU 命中率
- 端到端测试：混合推理输出质量

## 5. 性能预估

### CPU 单专家延迟

| 操作 | 计算 | 延迟 |
|------|------|------|
| D2H hidden_state | 4096 × 2B = 8KB | ~1μs (PCIe) |
| Q8_K 量化 | 4096 × 4B → 4096B | ~2μs |
| IQ2_XS gate/up 点积 | 2 × 2048×4096 × 2.3bit | ~0.8ms (AVX-512) |
| SwiGLU | 2048 × 4B | ~2μs |
| Q8_K 量化中间结果 | 2048 × 4B → 2048B | ~1μs |
| Q2_K down 点积 | 4096×2048 × 2bit | ~0.4ms (AVX-512) |
| H2D 结果 | 4096 × 2B = 8KB | ~1μs (PCIe) |
| **总计** | | **~1.2ms/专家** |

### GPU vs CPU 混合推理

| 场景 | GPU 命中 | CPU 计算 | 总延迟 |
|------|---------|---------|--------|
| 6 专家全 GPU 命中 | 6×0.2ms | 0 | ~1.2ms |
| 4 GPU + 2 CPU | 4×0.2ms | 2×1.2ms (并行) | ~2.6ms |
| 0 GPU + 6 CPU | 0 | 6×1.2ms (3并行) | ~2.4ms |

### 预期提升

| 指标 | 当前 (全 GPU) | 混合推理 | 提升 |
|------|-------------|---------|------|
| GPU 命中率 | ~78% | ~100% (CPU 兜底) | +22% |
| 推理速度 | ~0.8 t/s | ~1.2 t/s | +50% |
| GPU VRAM | ~15GB | ~12GB (减少缓存) | -3GB |

## 6. 文件结构

```
third/
├── ds4/
│   ├── ds4.c              # 原始 C 推理引擎（IQ2_XXS + ARM NEON）
│   ├── ds4.h              # 公共 API
│   ├── ds4_bf16.h         # BF16 辅助
│   ├── Makefile            # 构建系统
│   └── model.py            # Python 参考实现
├── llama.cpp/
│   ├── ggml-common.h       # block_iq2_xs + iq2xs_grid[512]
│   ├── quants_x86.c        # AVX2 IQ2_XS 点积
│   ├── quants_generic.c    # 标量 fallback
│   ├── sgemm.cpp           # tinyBLAS AVX-512 GEMM
│   ├── deepseek4.cpp       # DS4 模型 + 热/冷双路 MoE
│   ├── llama-deepseek4-hot.h   # 热门专家管理器
│   └── llama-deepseek4-hot.cpp # 热门专家实现
```

## 7. 优先级排序

1. **P0**: IQ2_XS AVX-512 反量化内核（Rust）— 核心计算
2. **P0**: CPU 专家 FFN（gate/up + SwiGLU + down）— 完整推理路径
3. **P1**: Python 原型验证 — 快速验证正确性
4. **P1**: 混合推理集成 — GPU hot + CPU cold
5. **P2**: AVX-512 VNNI/BF16 优化 — 性能提升
6. **P2**: L3 缓存分块 + 线程池 — 延迟优化
