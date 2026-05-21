# TileLang 算子 API 文档

> ds4rs 推理引擎 TileLang 内核算子参考手册
> 生成日期: 2026-05-17
> 内核总数: 64 (已编译 .so) | 融合算子: 1 | 参数校验: 全覆盖

---

## 目录

1. [数据类型约定](#1-数据类型约定)
2. [调用约定](#2-调用约定)
3. [量化算子 (Quantization)](#3-量化算子)
4. [GEMM 算子 (矩阵乘法)](#4-gemm-算子)
5. [注意力算子 (Attention)](#5-注意力算子)
6. [归一化算子 (Normalization)](#6-归一化算子)
7. [激活函数算子 (Activation)](#7-激活函数算子)
8. [位置编码算子 (RoPE)](#8-位置编码算子)
9. [类型转换算子 (Cast)](#9-类型转换算子)
10. [Hyper-Connection 算子](#10-hyper-connection-算子)
11. [MoE 算子 (混合专家)](#11-moe-算子)
12. [KV Cache 压缩算子 (Compressor)](#12-kv-cache-压缩算子)
13. [KV Cache 索引算子 (Indexer)](#13-kv-cache-索引算子)
14. [融合算子 (Fused)](#14-融合算子)
15. [Rust 侧调用接口](#15-rust-侧调用接口)
16. [参数校验规范](#16-参数校验规范)
17. [内核实例清单](#17-内核实例清单)

---

## 1. 数据类型约定

| TileLang 类型 | Rust DType | DLPack (code, bits, lanes) | 字节数 | 说明 |
|---|---|---|---|---|
| `bfloat16` | `BF16` | (4, 16, 1) | 2 | BF16 浮点 |
| `float32` | `FP32` | (2, 32, 1) | 4 | FP32 浮点 |
| `float8_e4m3fn` | `FP8E4M3` | (10, 8, 1) | 1 | FP8 E4M3 (范围 ±448) |
| `float4_e2m1fn` | `FP4E2M1` | (17, 4, 2) | 1 | FP4 E2M1 打包 (2值/字节) |
| `float8_e8m0fnu` | `FP8E8M0` | (14, 8, 1) | 1 | FP8 E8M0 纯指数缩放因子 |
| `int32` | `INT32` | (0, 32, 1) | 4 | 32位整数 |
| `uint8` | `UINT8` | (1, 8, 1) | 1 | 8位无符号整数 |

**缩放因子说明:**
- `FP8E8M0` (e8m0): 纯指数格式，无尾数，仅表示 2^exp。用于 block-wise 量化缩放因子。
  - 转换公式: `f32_val = 2^((bits as u32) << 23)` (当 bits != 0)
  - bits == 0 时值为 0.0

**量化 block 大小约定:**
- FP8 激活量化: block_size = 128 (每128列一个scale)
- FP4 权重量化: block_size = 32 (每32列一个scale)
- FP4 激活 QDQ: block_size = 32

---

## 2. 调用约定

### 2.1 加载机制

内核通过 TVM FFI C API 加载:
1. `libtvm_ffi.so` 提供 TVM 运行时
2. 每个 `.so` 文件包含一个 TileLang 编译的 prim_func
3. 通过 `ffi.ModuleLoadFromFile` 加载模块
4. 通过 `ffi.ModuleGetFunction` 获取函数句柄
5. 通过 `TVMFFITensorFromDLPack` 将 GPU 张量转为 TVM Tensor
6. 通过 `TVMFFIFunctionCall` 执行内核

### 2.2 张量传递

所有张量通过 DLPack 格式传递:
- GPU 张量: `DLDevice.type = kDLCUDA, device_id = 0`
- 形状以 `Vec<i64>` 传递
- 步长为 `null` (紧凑布局)
- `byte_offset = 0`

### 2.3 内核命名规则

```
{算子类别}_{参数编码}
```

例: `fp8_gemm_N4096_K8192` → FP8 GEMM, N=4096, K=8192

### 2.4 函数名映射

每个 `.so` 文件内的函数名由 `kernels.json` 中的 `func` 字段指定，与 TileLang `@T.prim_func` 定义的函数名一致。

---

## 3. 量化算子

### 3.1 act_quant_kernel — 激活 FP8 量化

将 BF16 激活量化为 FP8 E4M3，输出缩放因子。

**内核签名:**
```python
act_quant_kernel_(
    X: Tensor[(M, N), bfloat16],       # 输入激活
    Y: Tensor[(M, N), float8_e4m3fn],  # 量化后激活 (inplace模式为bfloat16)
    S: Tensor[(M, ceil(N/group_size)), scale_dtype],  # 缩放因子
)
```

**参数说明:**
- `M`: 符号化维度 (token数)，运行时确定
- `N`: 量化维度 (特征维度)，编译时常量
- `group_size`: 量化块大小，默认 128
- `scale_dtype`: 缩放因子类型，默认 `float8_e8m0fnu`
- `round_scale=True`: 使用 2 的幂次取整缩放 (与权重 scale 格式对齐)
- `inplace=True`: 量化后反量化回 BF16 (QDQ 模式，用于 KV nope 部分)

**量化公式:**
```
amax[i] = max(|X[i, :]|)  per row
scale[i] = round_pow2(amax[i] / 448.0)  (round_scale模式)
Y[i, j] = clamp(X[i, j] / scale[i], -448, 448)
S[i, block] = scale[i]
```

**已编译实例:**

| 内核名 | N | block_size | scale_dtype | round_scale | inplace |
|---|---|---|---|---|---|
| `act_quant_N4096_bs128` | 4096 | 128 | e8m0 | ✓ | ✗ |
| `act_quant_N8192_bs128` | 8192 | 128 | e8m0 | ✓ | ✗ |
| `act_quant_N2048_bs128` | 2048 | 128 | e8m0 | ✓ | ✗ |
| `act_quant_N1024_bs128` | 1024 | 128 | e8m0 | ✓ | ✗ |
| `act_quant_N448_bs64_inplace` | 448 | 64 | fp32 | ✓ | ✓ |

**Rust 调用:**
```rust
kernels.call("act_quant_N4096_bs128", &[&x_2d, &y, &s])?;
```

---

### 3.2 fp4_quant_kernel — FP4 权重量化

将 BF16 权重量化为 FP4 E2M1，输出 e8m0 缩放因子。

**内核签名:**
```python
fp4_quant_kernel_(
    X: Tensor[(M, N), bfloat16],           # 输入
    Y: Tensor[(M, N), float4_e2m1fn],      # 量化后 (inplace模式为bfloat16)
    S: Tensor[(M, ceil(N/group_size)), float8_e8m0fnu],  # 缩放因子
)
```

**参数说明:**
- `group_size = 32`: FP4 权重按 32 列分组
- `round_scale=True`: 使用 2 的幂次取整缩放
- FP4 最大值: 6.0

**已编译实例:**

| 内核名 | N | block_size | inplace |
|---|---|---|---|
| `fp4_quant_N448_bs32_inplace` | 448 | 32 | ✓ |

---

### 3.3 fp4_qdq_f32_kernel — FP4 QDQ (FP32)

FP32 输入的 FP4 量化-反量化，保持 FP32 精度。用于 Compressor 流程中的激活预处理。

**内核签名:**
```python
fp4_qdq_f32_kernel_(
    X: Tensor[(M, N), float32],   # 输入
    Y: Tensor[(M, N), float32],   # QDQ 输出
)
```

**已编译实例:**

| 内核名 | N | block_size |
|---|---|---|
| `fp4_qdq_f32_N128_bs32` | 128 | 32 |

---

## 4. GEMM 算子

### 4.1 fp8_gemm_kernel — FP8 激活 × FP8 权重

计算 `C[M,N] = A[M,K] @ B[N,K]^T`，两个输入均为 FP8 E4M3，输出 BF16。
支持 per-128-block 缩放。

**内核签名:**
```python
fp8_gemm_kernel_(
    A: Tensor[(M, K), float8_e4m3fn],                              # 激活 (量化后)
    B: Tensor[(N, K), float8_e4m3fn],                              # 权重 (存储为 [N,K])
    C: Tensor[(M, N), bfloat16],                                   # 输出
    scales_a: Tensor[(M, ceil(K/128)), scale_dtype],               # 激活缩放
    scales_b: Tensor[(ceil(N/128), ceil(K/128)), scale_dtype],     # 权重缩放
)
```

**计算逻辑:**
```
对每个 K 块:
    C_local += A_shared @ B_shared^T
    C_accum += C_local * (scale_a * scale_b)
```

**关键约束:**
- B 矩阵存储为 `[N, K]`，GEMM 时 `transpose_B=True`
- `scales_a` shape: `[M, ceil(K/128)]` — 每行每128列一个scale
- `scales_b` shape: `[ceil(N/128), ceil(K/128)]` — 每128行每128列一个scale
- 使用双累加器 (C_local + C_accum) 提高精度

**已编译实例:**

| 内核名 | N | K | scale_dtype | 用途 |
|---|---|---|---|---|
| `fp8_gemm_N32768_K1024` | 32768 | 1024 | e8m0 | wq_b (Q LoRA-B) |
| `fp8_gemm_N512_K4096` | 512 | 4096 | e8m0 | wkv (KV 投影) |
| `fp8_gemm_N1024_K4096` | 1024 | 4096 | e8m0 | wq_a (Q LoRA-A) |
| `fp8_gemm_N4096_K8192` | 4096 | 8192 | e8m0 | wo_b (O 投影-B) |
| `fp8_gemm_N2048_K4096` | 2048 | 4096 | e8m0 | shared_w1/w3 (共享专家) |
| `fp8_gemm_N4096_K2048` | 4096 | 2048 | e8m0 | shared_w2 (共享专家下投影) |
| `fp8_gemm_N8192_K1024` | 8192 | 1024 | e8m0 | gate (MoE 门控) |

**Rust 调用:**
```rust
kernels.call("fp8_gemm_N4096_K8192", &[&x_fp8, &w_fp8, &c_bf16, &x_scale, &w_scale])?;
```

---

### 4.2 fp4_gemm_kernel — FP8 激活 × FP4 权重

计算 `C[M,N] = A[M,K] @ B[N,K]^T`，激活为 FP8，权重为 FP4 E2M1，输出 BF16。
用于路由专家的 FP4 权重 GEMM。

**内核签名:**
```python
fp4_gemm_kernel_(
    A: Tensor[(M, K), float8_e4m3fn],                              # FP8 激活
    B: Tensor[(N, K), float4_e2m1fn],                              # FP4 权重 (存储为 [N, K/2])
    C: Tensor[(M, N), bfloat16],                                   # 输出
    scales_a: Tensor[(M, ceil(K/128)), scale_dtype],               # 激活缩放 (per-128)
    scales_b: Tensor[(N, ceil(K/32)), scale_dtype],                # 权重缩放 (per-32!)
)
```

**关键约束:**
- FP4 权重存储: `[N, K/2]` (2个FP4值打包为1字节)
- 激活缩放: per-128 block
- **权重缩放: per-32 block** (与 FP8 GEMM 的 per-128 不同!)
- 内核内部先将 FP4 解包为 FP8 再计算

**已编译实例:**

| 内核名 | N | K | scale_dtype | 用途 |
|---|---|---|---|---|
| `fp4_gemm_N2048_K4096` | 2048 | 4096 | e8m0 | 专家 w1/w3 (上投影) |
| `fp4_gemm_N4096_K2048` | 4096 | 2048 | e8m0 | 专家 w2 (下投影) |

**Rust 调用:**
```rust
kernels.call("fp4_gemm_N2048_K4096", &[&x_fp8, &w_fp4, &c_bf16, &x_scale, &w_scale])?;
```

---

## 5. 注意力算子

### 5.1 sparse_attn_kernel — 稀疏滑动窗口注意力

实现 DeepSeek V4 的稀疏注意力: 每个头独立选择 topk 位置进行注意力计算，
支持滑动窗口 (SWA) 和 attn_sink 机制。

**内核签名:**
```python
sparse_attn_kernel_(
    q: Tensor[(b, m, h, d), bfloat16],        # Query [batch, seq, heads, head_dim]
    kv: Tensor[(b, n, d), bfloat16],           # KV cache [batch, cache_len, head_dim]
    o: Tensor[(b, m, h, d), bfloat16],         # 输出 [batch, seq, heads, head_dim]
    attn_sink: Tensor[(h,), float32],           # 每个头的 sink 偏置
    topk_idxs: Tensor[(b, m, topk), int32],    # 每个位置选择的 topk 索引
)
```

**计算逻辑:**
```
对每个 head_group (hg=16 个头一组):
    对每个 topk 块:
        scores = Q @ KV^T * scale
        online_softmax(scores, max, sum)
        acc_o = acc_o * scale_factor + softmax_weights @ KV
    acc_o /= (sum_exp + exp(attn_sink - max))
```

**关键参数:**
- `h=64`: 注意力头数
- `d=512`: 头维度
- `head_group_size=16`: 每组16个头共享 KV 加载
- `scale = 1/sqrt(512) ≈ 0.0442`
- `block=64`: KV 按64个位置分块

**已编译实例:**

| 内核名 | h | d | head_group_size |
|---|---|---|---|
| `sparse_attn_h64_d512` | 64 | 512 | 16 |

**Rust 调用:**
```rust
kernels.call("sparse_attn_h64_d512", &[&q, &kv, &o, &sink_f32, &topk_idxs])?;
```

---

## 6. 归一化算子

### 6.1 rmsnorm_kernel — BF16 RMSNorm (带/不带权重)

**内核签名 (带权重):**
```python
rmsnorm_kernel_(
    X: Tensor[(M, N), bfloat16],    # 输入
    W: Tensor[(N,), float32],       # 权重 (has_weight=True) 或占位 (has_weight=False)
    Y: Tensor[(M, N), bfloat16],    # 输出
)
```

**计算公式:**
```
rsqrt = 1 / sqrt(sum(X^2) / N + 1e-6)
Y = X * rsqrt * W   (带权重)
Y = X * rsqrt        (不带权重)
```

**已编译实例:**

| 内核名 | N | has_weight | 用途 |
|---|---|---|---|
| `rmsnorm_N4096` | 4096 | ✓ | attn_norm, ffn_norm |
| `rmsnorm_N1024` | 1024 | ✓ | q_norm |
| `rmsnorm_N512` | 512 | ✓ | kv_norm |
| `rmsnorm_no_weight_N1024` | 1024 | ✗ | Q head_dim 归一化 (q_lora_rank=1024) |
| `rmsnorm_no_weight_N512` | 512 | ✗ | Q head 维度归一化 (head_dim=512) |

---

### 6.2 rmsnorm_f32_kernel — FP32 RMSNorm (无权重)

**内核签名:**
```python
rmsnorm_f32_kernel_(
    X: Tensor[(M, N), float32],     # 输入
    Y: Tensor[(M, N), float32],     # 输出
)
```

**已编译实例:**

| 内核名 | N | 用途 |
|---|---|---|
| `rmsnorm_f32_N4096` | 4096 | HC pre 归一化 |
| `rmsnorm_f32_N7168` | 7168 | HC pre 归一化 (hc*dim) |
| `rmsnorm_f32_N16384` | 16384 | HC pre 归一化 (hc*dim) |

---

### 6.3 rmsnorm_f32_weighted_kernel — FP32 RMSNorm (带权重)

**内核签名:**
```python
rmsnorm_f32_weighted_kernel_(
    X: Tensor[(M, N), float32],     # 输入
    W: Tensor[(N,), float32],       # 权重
    Y: Tensor[(M, N), float32],     # 输出
)
```

**已编译实例:**

| 内核名 | N | 用途 |
|---|---|---|
| `rmsnorm_f32_weighted_N128` | 128 | Compressor KV 归一化 |
| `rmsnorm_f32_weighted_N512` | 512 | Compressor KV 归一化 |

---

## 7. 激活函数算子

### 7.1 swiglu_kernel — SwiGLU 激活

**内核签名:**
```python
swiglu_kernel_(
    Gate: Tensor[(M, N), bfloat16],   # gate 分支
    Up: Tensor[(M, N), bfloat16],     # up 分支
    Y: Tensor[(M, N), bfloat16],      # 输出
)
```

**计算公式:**
```
silu(g) = g / (1 + exp(-g))
Y = silu(clamp(g, -limit, limit)) * clamp(u, -limit, limit)
```
- `swiglu_limit = 10.0`: 防止 FP16 溢出

**已编译实例:**

| 内核名 | N | limit |
|---|---|---|
| `swiglu_N2048` | 2048 | 10.0 |

---

## 8. 位置编码算子

### 8.1 rope_interleaved_kernel — 交错布局 RoPE

对张量最后 D 维应用交错布局的旋转位置编码。

**内核签名:**
```python
rope_interleaved_kernel_(
    X_rope: Tensor[(M, H, D), bfloat16],     # RoPE 部分
    Cos: Tensor[(M, D//2), float32],          # cos 频率
    Sin: Tensor[(M, D//2), float32],          # sin 频率
    Y_rope: Tensor[(M, H, D), bfloat16],      # 输出
)
```

**交错布局:** 旋转对为 `(2k, 2k+1)`，即:
```
y[2k]   = x[2k] * cos - sign * x[2k+1] * sin
y[2k+1] = sign * x[2k] * sin + x[2k+1] * cos
```
- `sign = -1.0` (inverse=True) 用于逆 RoPE

**已编译实例:**

| 内核名 | D | inverse | 用途 |
|---|---|---|---|
| `rope_interleaved_fwd_D64` | 64 | ✗ | Q/KV 正向 RoPE |
| `rope_interleaved_inv_D64` | 64 | ✓ | O 逆 RoPE |

**Rust 调用 (Q RoPE):**
```rust
// 先切分 nope/rope 列，对 rope 部分应用 RoPE，再拼接
kernels.call("rope_interleaved_fwd_D64", &[&rope_gpu, &cos_gpu, &sin_gpu, &y_rope])?;
```

---

### 8.2 rotary_emb_kernel — 分离布局 RoPE (旧版)

对 nope 和 rope 分离存储的张量应用 RoPE。

**内核签名:**
```python
rotary_emb_kernel_(
    X_nope: Tensor[(M, H, Total_D-D), bfloat16],   # nope 部分 (直通)
    X_rope: Tensor[(M, H, D), bfloat16],            # rope 部分
    Freqs: Tensor[(M, D//2), float32],              # 频率 (cos only, sin=√(1-cos²))
    Y_nope: Tensor[(M, H, Total_D-D), bfloat16],    # nope 输出
    Y_rope: Tensor[(M, H, D), bfloat16],             # rope 输出
)
```

**已编译实例:**

| 内核名 | D | Total_D | inverse |
|---|---|---|---|
| `rope_forward_D64_TD512` | 64 | 512 | ✗ |
| `rope_inverse_D64_TD512` | 64 | 512 | ✓ |

---

## 9. 类型转换算子

### 9.1 cast_bf16_f32_kernel — BF16 ↔ FP32 类型转换

**内核签名:**
```python
cast_bf16_f32_kernel_(
    X: Tensor[(M, N), src_dtype],    # 输入
    Y: Tensor[(M, N), dst_dtype],    # 输出
)
```

**已编译实例:**

| 内核名 | src → dst | N |
|---|---|---|
| `cast_bf16_to_f32_N128` | BF16 → FP32 | 128 |
| `cast_bf16_to_f32_N512` | BF16 → FP32 | 512 |
| `cast_bf16_to_f32_N1024` | BF16 → FP32 | 1024 |
| `cast_bf16_to_f32_N4096` | BF16 → FP32 | 4096 |
| `cast_bf16_to_f32_N8192` | BF16 → FP32 | 8192 |
| `cast_bf16_to_f32_N16384` | BF16 → FP32 | 16384 |
| `cast_f32_to_bf16_N128` | FP32 → BF16 | 128 |
| `cast_f32_to_bf16_N512` | FP32 → BF16 | 512 |
| `cast_f32_to_bf16_N1024` | FP32 → BF16 | 1024 |
| `cast_f32_to_bf16_N4096` | FP32 → BF16 | 4096 |
| `cast_f32_to_bf16_N16384` | FP32 → BF16 | 16384 |

---

## 10. Hyper-Connection 算子

### 10.1 hc_split_sinkhorn_kernel — HC Sinkhorn 分解

DeepSeek V4 的 Hyper-Connection 机制: 将混合系数分解为 pre/post/comb 三部分，
comb 矩阵通过 Sinkhorn 归一化变为双随机矩阵。

**内核签名:**
```python
hc_split_sinkhorn_kernel_(
    mixes: Tensor[(n, mix_hc), float32],      # 混合系数 (n, (2+hc)*hc)
    hc_scale: Tensor[(3,), float32],           # 三个缩放因子 [pre_scale, post_scale, comb_scale]
    hc_base: Tensor[(mix_hc,), float32],       # 偏置项
    pre: Tensor[(n, hc), float32],             # 输出: pre 系数
    post: Tensor[(n, hc), float32],            # 输出: post 系数
    comb: Tensor[(n, hc, hc), float32],        # 输出: comb 双随机矩阵
)
```

**计算逻辑:**
```
pre[j]  = sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + eps
post[j] = 2 * sigmoid(mixes[j+hc] * hc_scale[1] + hc_base[j+hc])
comb[j,k] = mixes[j*hc+k+2*hc] * hc_scale[2] + hc_base[j*hc+k+2*hc]

# Sinkhorn 归一化 (iters 轮):
for iter:
    comb = comb / row_sum(comb)   # 行归一化
    comb = comb / col_sum(comb)   # 列归一化
```

**已编译实例:**

| 内核名 | hc | sinkhorn_iters | eps |
|---|---|---|---|
| `hc_sinkhorn_hc4_it20` | 4 | 20 | 1e-6 |

**Rust 调用:**
```rust
kernels.call("hc_sinkhorn_hc4_it20", &[&mixes_2d, &hc_scale_f32, &hc_base_f32, &pre_out, &post_out, &comb_out])?;
```

---

### 10.2 sigmoid_kernel — 标量 sigmoid

**内核签名:**
```python
sigmoid_kernel_(
    In: Tensor[(N,), float32],       # 输入
    Scale: Tensor[(1,), float32],    # 缩放因子
    Base: Tensor[(N,), float32],     # 偏置
    Out: Tensor[(N,), float32],      # 输出
)
```

**计算:** `Out[i] = 1 / (1 + exp(-(In[i] * Scale[0] + Base[i])))`

**已编译实例:** `sigmoid_N4`, `sigmoid_dynamic`

---

### 10.3 hc_sigmoid_kernel — HC 头级 sigmoid (广播)

**内核签名:**
```python
hc_sigmoid_kernel_(
    In: Tensor[(N, HC), float32],     # 输入
    Scale: Tensor[(1,), float32],     # 缩放因子
    Base: Tensor[(HC,), float32],     # 偏置 (广播到每行)
    Out: Tensor[(N, HC), bfloat16],   # 输出 (BF16)
)
```

**计算:** `Out[i,j] = BF16(sigmoid(In[i,j] * Scale[0] + Base[j]) + eps)`

**已编译实例:** `hc_sigmoid_hc4`

---

## 11. MoE 算子

### 11.1 moe_route_kernel — MoE 门控路由

计算专家选择分数 + topk 选择 + 归一化。

**内核签名:**
```python
moe_route_kernel_(
    Scores: Tensor[(M, N), float32],          # GEMM 后的原始分数
    Bias: Tensor[(N,), float32],               # 专家偏置
    TopkWeights: Tensor[(M, topk), float32],   # 输出: topk 权重
    TopkIndices: Tensor[(M, topk), int32],      # 输出: topk 索引
)
```

**计算逻辑:**
1. 激活函数: `sqrt_softplus(v) = sqrt(log(1 + exp(v)))`
2. 选择分数: `select[i] = activated[i] + Bias[i]`
3. Topk: 串行扫描选 topk
4. 归一化: `w[k] /= sum(w) * route_scale`

**已编译实例:**

| 内核名 | N | topk | score_func | has_bias | route_scale |
|---|---|---|---|---|---|
| `moe_route_sqrtsp_N256_topk6` | 256 | 6 | sqrt_softplus | ✓ | 1.5 |

---

### 11.2 scatter_add_kernel — MoE 专家输出加权合并

**内核签名:**
```python
scatter_add_kernel_(
    Src: Tensor[(N, D), bfloat16],       # 专家输出
    Weights: Tensor[(N,), float32],       # 路由权重
    TokenIds: Tensor[(N,), int32],        # 目标 token 位置
    Dst: Tensor[(M, D), bfloat16],        # 累加目标
)
```

**计算:** `Dst[token_ids[i], d] += weights[i] * Src[i, d]`

**已编译实例:**

| 内核名 | D |
|---|---|
| `scatter_add_D4096` | 4096 |
| `scatter_add_D7168` | 7168 |

---

### 11.3 moe_gather_kernel — 按 GPU 索引收集行

**内核签名:**
```python
moe_gather_kernel_(
    Src: Tensor[(M, D), bfloat16],       # 源张量
    Indices: Tensor[(N,), int32],         # 行索引
    Dst: Tensor[(N, D), bfloat16],        # 输出
)
```

**计算:** `Dst[i, d] = Src[indices[i], d]`

**已编译实例:** `moe_gather_D4096`

---

### 11.4 moe_extract_weights_kernel — 提取专家权重和 token ID

**内核签名:**
```python
moe_extract_weights_kernel_(
    GateIndices: Tensor[(Total, topk), int32],   # 门控索引
    GateWeights: Tensor[(Total, topk), float32],  # 门控权重
    ExpertId: Tensor[(1,), int32],                # 目标专家 ID
    OutWeights: Tensor[(Total,), float32],         # 输出权重
    OutTokenIds: Tensor[(Total,), int32],           # 输出 token ID
    OutCount: Tensor[(1,), int32],                  # 匹配数量
)
```

**已编译实例:** `moe_extract_weights_topk6`

---

### 11.5 moe_expert_count_kernel — 统计每专家 token 数

**内核签名:**
```python
moe_expert_count_kernel_(
    Indices: Tensor[(Total, topk), int32],   # 门控索引
    Counts: Tensor[(n_experts,), int32],      # 输出: 每专家计数
)
```

**已编译实例:** `moe_expert_count_topk6_ne256`

---

## 12. KV Cache 压缩算子

### 12.1 compressor_pool_kernel — 压缩器 softmax 门控池化

**内核签名:**
```python
compressor_pool_kernel_(
    KV: Tensor[(M, coff, head_dim), float32],     # KV 投影
    Gate: Tensor[(M, coff, head_dim), float32],    # 门控分数
    Out: Tensor[(M, head_dim), float32],            # 压缩后输出
)
```

**计算:** 对每个 token，沿 coff 维度做 softmax 加权求和:
```
Out[tok, d] = sum_c(softmax(Gate[tok, :, d])[c] * KV[tok, c, d])
```

**已编译实例:**

| 内核名 | head_dim | coff |
|---|---|---|
| `compressor_pool_d128_c8` | 128 | 8 |
| `compressor_pool_d512_c8` | 512 | 8 |

---

### 12.2 compressor_rope_f32_kernel — 压缩器 RoPE (FP32)

**内核签名:**
```python
compressor_rope_f32_kernel_(
    X: Tensor[(M, d), float32],              # 输入
    Cos: Tensor[(M, rd//2), float32],        # cos 频率
    Sin: Tensor[(M, rd//2), float32],        # sin 频率
    Y: Tensor[(M, d), float32],              # 输出
)
```

前 `(d-rd)` 维直通，后 `rd` 维应用交错 RoPE。

**已编译实例:**

| 内核名 | d | rd |
|---|---|---|
| `compressor_rope_f32_d128_rd64` | 128 | 64 |
| `compressor_rope_f32_d512_rd64` | 512 | 64 |

---

### 12.3 compressor_group_kernel — 压缩器分组收集

**内核签名:**
```python
compressor_group_kernel_(
    Src: Tensor[(N, out_dim), float32],          # 源 GEMM 结果
    Ape: Tensor[(ratio, out_dim), float32],       # APE 偏置
    RowIdx: Tensor[(M, pool_size), int32],        # 行索引
    ColOff: Tensor[(pool_size,), int32],           # 列偏移
    IsScore: Tensor[(1,), int32],                  # 是否为 score (1=score, 0=kv)
    Dst: Tensor[(M, pool_size, d), float32],       # 输出
)
```

**已编译实例:**

| 内核名 | d | out_dim | pool_size |
|---|---|---|---|
| `compressor_group_d512_od1024_ps8` | 512 | 1024 | 8 |
| `compressor_group_d128_od256_ps8` | 128 | 256 | 8 |

---

## 13. KV Cache 索引算子

### 13.1 indexer_score_kernel — 索引器评分 + topk

**内核签名:**
```python
indexer_score_kernel_(
    Q: Tensor[(M, n_heads, head_dim), float32],          # Query
    KV: Tensor[(1, N, head_dim), float32],                # 压缩 KV 缓存
    Weights: Tensor[(M, n_heads), float32],                # 头权重
    TopkIndices: Tensor[(M, index_topk), int32],           # 输出 topk 索引
)
```

**计算:** 对每个 token，计算 Q 与压缩 KV 的注意力分数 (ReLU 加权)，选 topk。

**已编译实例:** `indexer_score_h64_d128_topk512`

---

### 13.2 indexer_causal_adjust_kernel — 因果掩码调整

**内核签名:**
```python
indexer_causal_adjust_kernel_(
    Indices: Tensor[(M, topk), int32],       # 原始索引
    CausalLimit: Tensor[(M,), int32],         # 因果限制位置
    Offset: Tensor[(1,), int32],              # 偏移量
    Out: Tensor[(M, topk), int32],            # 调整后索引
)
```

**计算:** `Out[i,k] = if Indices[i,k] >= CausalLimit[i] then -1 else Indices[i,k] + Offset[0]`

**已编译实例:** `indexer_causal_adjust_topk512`

---

## 14. 融合算子

### 14.1 fused_shared_ffn_kernel — Shared Expert FFN 融合

融合 SwiGLU + act_quant + FP8 GEMM(w2) 为单个内核，消除中间张量的全局内存读写。

**原始调用链:**
```
gate [M, Inter] BF16 ─┐
                       ├→ swiglu → [M, Inter] BF16 → act_quant → [M, Inter] FP8 → fp8_gemm(w2) → [M, Dim] BF16
up   [M, Inter] BF16 ─┘
```
4 次内核启动 + 2 次全局内存中间张量

**融合后:**
```
gate [M, Inter] BF16 ─┐
                       ├→ fused_shared_ffn ──→ [M, Dim] BF16
up   [M, Inter] BF16 ─┘
```
1 次内核启动 + 0 次全局内存中间张量

**内核签名:**
```python
fused_shared_ffn_kernel_(
    Gate: Tensor[(M, Inter), bfloat16],       # gate 分支输出 (来自 fp8_gemm(w1))
    Up: Tensor[(M, Inter), bfloat16],          # up 分支输出 (来自 fp8_gemm(w3))
    W2: Tensor[(Dim, Inter), float8_e4m3fn],  # down 投影权重
    W2_S: Tensor[(Dim/128, Inter/128), float8_e8m0fnu],  # down 投影缩放
    Y: Tensor[(M, Dim), bfloat16],            # 输出
)
```

**内部流程:**
```
对每个 K 块 (block_K=128):
  1. 从 Gate/Up 读取 block_K 列 → SwiGLU 计算 (fragment, 不写回全局内存)
  2. act_quant: amax → round_scale → 量化为 FP8 (shared memory)
  3. 读取 W2 权重 (shared memory)
  4. 计算 scale_a * scale_b
  5. GEMM: A_shared @ B_shared^T → 累加到 C_local_accum
```

**已编译实例:**

| 内核名 | Dim | Inter | scale_dtype |
|---|---|---|---|
| `fused_shared_ffn_D4096_I2048` | 4096 | 2048 | e8m0 |

**Rust 调用:**
```rust
kernels.call("fused_shared_ffn_D4096_I2048", &[&gate_2d, &up_2d, &w2_2d, &w2_s_2d, &y])?;
```

**参数校验 (Rust 侧):**
- gate.dtype == BF16, up.dtype == BF16
- gate.shape == [total, Inter], up.shape == gate.shape
- w2.dtype == FP8E4M3, w2.shape == [Dim, Inter]
- w2_s.shape == [Dim/128, Inter/128]

---

## 15. Rust 侧调用接口

### 15.1 KernelRegistry

```rust
pub struct KernelRegistry { ... }

impl KernelRegistry {
    /// 从目录加载所有内核 (优先使用 kernels.json 清单)
    pub fn load_dir(&self, dir: &str) -> Result<usize>

    /// 调用指定内核
    pub fn call(&self, name: &str, tensors: &[&GpuTensor]) -> Result<()>
}
```

### 15.2 TlKernel

```rust
pub struct TlKernel { ... }

impl TlKernel {
    /// 加载单个 .so 内核
    pub fn load(runtime: &TvmRuntime, so_path: &str, func_name: &str) -> Result<Self>

    /// 执行内核，传入 GPU 张量列表
    pub fn call(&self, tensors: &[&GpuTensor]) -> Result<()>
}
```

### 15.3 TvmRuntime

```rust
pub struct TvmRuntime { ... }

impl TvmRuntime {
    /// 初始化 TVM FFI 运行时 (自动查找 libtvm_ffi.so)
    pub fn new() -> Result<Self>

    /// 指定路径初始化
    pub fn with_lib_path(path: &str) -> Result<Self>
}
```

### 15.4 调用流程

```
1. TvmRuntime::new()                     → 加载 TVM 运行时
2. KernelRegistry::new(runtime)           → 创建内核注册表
3. registry.load_dir("tilelang/build/")   → 加载所有 .so 内核
4. registry.call("kernel_name", &[...])   → 执行内核
```

### 15.5 layer.rs 中的内核调用映射

| Rust 函数 | 内核名 | 参数 |
|---|---|---|
| `fp8_gemm_act_quant` | `fp8_gemm_N{N}_K{K}` | `[&x_fp8, &w_fp8, &c, &x_scale, &w_scale]` |
| `fp4_gemm_act_quant` | `fp4_gemm_N{N}_K{K}` | `[&x_fp8, &w_fp4, &c, &x_scale, &w_scale]` |
| `act_quant_gpu` | `act_quant_N{N}_bs{bs}` | `[&x_2d, &y, &s]` |
| `rmsnorm` | `rmsnorm_N{N}` / `rmsnorm_no_weight_N{N}` | `[&x_2d, &w_f32, &y]` |
| `rmsnorm_f32` | `rmsnorm_f32_N{N}` | `[&x_2d, &y_2d]` |
| `sparse_attention` | `sparse_attn_h64_d512` | `[&q, &kv, &o, &sink, &topk_idxs]` |
| `hc_pre` | `hc_sinkhorn_hc4_it20` | `[&mixes, &hc_scale, &hc_base, &pre, &post, &comb]` |
| `apply_rope_q/kv` | `rope_interleaved_fwd_D64` | `[&rope, &cos, &sin, &y_rope]` |
| `apply_inverse_rope` | `rope_interleaved_inv_D64` | `[&rope, &cos, &sin, &y_rope]` |
| `swiglu` | `swiglu_N2048` | `[&gate, &up, &y]` |
| `cast_to_f32` | `cast_bf16_to_f32_N{N}` | `[&x_2d, &y_2d]` |
| `cast_to_bf16` | `cast_f32_to_bf16_N{N}` | `[&x_2d, &y_2d]` |
| `act_quant_inplace_nope` | `act_quant_N448_bs64_inplace` | `[&nope_2d, &y_nope, &s]` |
| `output_proj` | (wo_a: cublas BF16, wo_b: fp8_gemm) | — |

---

## 16. 参数校验规范

### 16.1 三层校验架构

| 层级 | 位置 | 校验内容 |
|---|---|---|
| L1: `TlKernel::call()` | tvm_ffi.rs | 参数数量校验 (通过 `infer_nargs` 映射) |
| L2: `KernelRegistry::call()` | tvm_ffi.rs | 内核名存在性校验 + 错误信息包含 shape |
| L3: `layer.rs` 封装函数 | layer.rs | dtype、shape、维度一致性校验 |

### 16.2 L1 参数数量映射

| 内核函数前缀 | 期望参数数 |
|---|---|
| `fp8_gemm_kernel_` / `fp4_gemm_kernel_` | 5 |
| `act_quant_kernel_` / `fp4_quant_kernel_` / `fp4_qdq_f32_kernel_` | 3 |
| `sparse_attn_kernel_` | 5 |
| `hc_split_sinkhorn_kernel_` | 6 |
| `rmsnorm_kernel_` / `rmsnorm_f32_weighted_kernel_` | 3 |
| `rmsnorm_f32_kernel_` / `cast_bf16_f32_kernel_` | 2 |
| `swiglu_kernel_` | 3 |
| `rope_interleaved_kernel_` / `compressor_rope_f32_kernel_` | 4 |
| `rotary_emb_kernel_` | 5 |
| `scatter_add_kernel_` / `sigmoid_kernel_` / `hc_sigmoid_kernel_` | 4 |
| `moe_route_kernel_` / `indexer_score_kernel_` / `indexer_causal_adjust_kernel_` | 4 |
| `compressor_pool_kernel_` / `moe_gather_kernel_` | 3 |
| `moe_expert_count_kernel_` | 2 |
| `compressor_group_kernel_` / `moe_extract_weights_kernel_` | 6 |
| `fused_shared_ffn_kernel_` | 5 |
| `scale_f32_kernel_` | 3 |

### 16.3 L3 算子级校验清单

| 算子 | 校验项 |
|---|---|
| `fp8_gemm_act_quant` | x.dtype==BF16, weight.dtype==FP8E4M3, weight.shape==[N,K], scale.shape==[ceil(N/128),ceil(K/128)], scale.dtype==FP8E8M0 |
| `fp4_gemm_act_quant` | x.dtype==BF16, weight.dtype==FP4E2M1, weight.shape==[N,K/2], scale.shape==[N,ceil(K/32)], scale.dtype==FP8E8M0 |
| `act_quant_gpu` | x.dtype==BF16, K % block_size == 0 |
| `rmsnorm` | x.dtype==BF16, weight.dtype∈{BF16,FP32}, weight.length==last_dim |
| `rmsnorm_f32` | x.dtype==FP32 |
| `sparse_attention` | q.shape==[bsz,seqlen,h,d], kv.shape[2]==head_dim, attn_sink==FP32[h], topk_idxs==INT32 |
| `hc_pre` | x.shape[2]==hc, hc_fn.shape==[mix_hc, hc*dim], hc_scale.shape==[3], hc_base.shape==[mix_hc] |
| `swiglu` | gate.shape==up.shape, gate/up.dtype==BF16 |
| `fused_shared_ffn` | gate/up.dtype==BF16, gate.shape==[total,Inter], up.shape==gate.shape, w2.dtype==FP8E4M3, w2.shape==[Dim,Inter], w2_s.shape==[Dim/128,Inter/128] |

### 16.4 dlopen Fallback

当 `ModuleLoadFromFile` 失败时 (type_index=0)，`TlKernel::load` 自动回退到 `load_via_dlopen`:
1. `dlopen` 加载 .so 文件
2. 调用 `__tvm_ffi_main` 注册全局函数
3. 通过 `get_global_func` 查找目标函数

此 fallback 机制确保不同版本的 TileLang 编译的 .so 文件都能被正确加载。

---

## 17. 内核实例清单

共 64 个已编译内核:

| # | 内核名 | 函数名 | .so |
|---|---|---|---|
| 1 | act_quant_N1024_bs128 | act_quant_kernel_ | ✓ |
| 2 | act_quant_N2048_bs128 | act_quant_kernel_ | ✓ |
| 3 | act_quant_N4096_bs128 | act_quant_kernel_ | ✓ |
| 4 | act_quant_N448_bs64_inplace | act_quant_kernel_ | ✓ |
| 5 | act_quant_N8192_bs128 | act_quant_kernel_ | ✓ |
| 6 | cast_bf16_to_f32_N1024 | cast_bf16_f32_kernel_ | ✓ |
| 7 | cast_bf16_to_f32_N128 | cast_bf16_f32_kernel_ | ✓ |
| 8 | cast_bf16_to_f32_N16384 | cast_bf16_f32_kernel_ | ✓ |
| 9 | cast_bf16_to_f32_N4096 | cast_bf16_f32_kernel_ | ✓ |
| 10 | cast_bf16_to_f32_N512 | cast_bf16_f32_kernel_ | ✓ |
| 11 | cast_bf16_to_f32_N8192 | cast_bf16_f32_kernel_ | ✓ |
| 12 | cast_f32_to_bf16_N1024 | cast_bf16_f32_kernel_ | ✓ |
| 13 | cast_f32_to_bf16_N128 | cast_bf16_f32_kernel_ | ✓ |
| 14 | cast_f32_to_bf16_N16384 | cast_bf16_f32_kernel_ | ✓ |
| 15 | cast_f32_to_bf16_N4096 | cast_bf16_f32_kernel_ | ✓ |
| 16 | cast_f32_to_bf16_N512 | cast_bf16_f32_kernel_ | ✓ |
| 17 | compressor_group_d128_od256_ps8 | compressor_group_kernel_ | ✓ |
| 18 | compressor_group_d512_od1024_ps8 | compressor_group_kernel_ | ✓ |
| 19 | compressor_pool_d128_c8 | compressor_pool_kernel_ | ✓ |
| 20 | compressor_pool_d512_c8 | compressor_pool_kernel_ | ✓ |
| 21 | compressor_rope_f32_d128_rd64 | compressor_rope_f32_kernel_ | ✓ |
| 22 | compressor_rope_f32_d512_rd64 | compressor_rope_f32_kernel_ | ✓ |
| 23 | fp4_gemm_N2048_K4096 | fp4_gemm_kernel_ | ✓ |
| 24 | fp4_gemm_N4096_K2048 | fp4_gemm_kernel_ | ✓ |
| 25 | fp4_qdq_f32_N128_bs32 | fp4_qdq_f32_kernel_ | ✓ |
| 26 | fp8_gemm_N1024_K4096 | fp8_gemm_kernel_ | ✓ |
| 27 | fp8_gemm_N2048_K4096 | fp8_gemm_kernel_ | ✓ |
| 28 | fp8_gemm_N32768_K1024 | fp8_gemm_kernel_ | ✓ |
| 29 | fp8_gemm_N4096_K2048 | fp8_gemm_kernel_ | ✓ |
| 30 | fp8_gemm_N4096_K8192 | fp8_gemm_kernel_ | ✓ |
| 31 | fp8_gemm_N512_K4096 | fp8_gemm_kernel_ | ✓ |
| 32 | fp8_gemm_N8192_K1024 | fp8_gemm_kernel_ | ✓ |
| 33 | hc_sigmoid_hc4 | hc_sigmoid_kernel_ | ✓ |
| 34 | hc_sinkhorn_hc4_it20 | hc_split_sinkhorn_kernel_ | ✓ |
| 35 | indexer_causal_adjust_topk512 | indexer_causal_adjust_kernel_ | ✓ |
| 36 | indexer_score_h64_d128_topk512 | indexer_score_kernel_ | ✓ |
| 37 | moe_expert_count_topk6_ne256 | moe_expert_count_kernel_ | ✓ |
| 38 | moe_extract_weights_topk6 | moe_extract_weights_kernel_ | ✓ |
| 39 | moe_gather_D4096 | moe_gather_kernel_ | ✓ |
| 40 | moe_route_sqrtsp_N256_topk6 | moe_route_kernel_ | ✓ |
| 41 | rmsnorm_N1024 | rmsnorm_kernel_ | ✓ |
| 42 | rmsnorm_N4096 | rmsnorm_kernel_ | ✓ |
| 43 | rmsnorm_N512 | rmsnorm_kernel_ | ✓ |
| 44 | rmsnorm_f32_N16384 | rmsnorm_f32_kernel_ | ✓ |
| 45 | rmsnorm_f32_N4096 | rmsnorm_f32_kernel_ | ✓ |
| 46 | rmsnorm_f32_N7168 | rmsnorm_f32_kernel_ | ✓ |
| 47 | rmsnorm_f32_weighted_N128 | rmsnorm_f32_weighted_kernel_ | ✓ |
| 48 | rmsnorm_f32_weighted_N512 | rmsnorm_f32_weighted_kernel_ | ✓ |
| 49 | rmsnorm_no_weight_N1024 | rmsnorm_kernel_ | ✓ |
| 50 | rmsnorm_no_weight_N512 | rmsnorm_kernel_ | ✓ |
| 51 | rope_forward_D64_TD512 | rotary_emb_kernel_ | ✓ |
| 52 | rope_interleaved_fwd_D64 | rope_interleaved_kernel_ | ✓ |
| 53 | rope_interleaved_inv_D64 | rope_interleaved_kernel_ | ✓ |
| 54 | rope_inverse_D64_TD512 | rotary_emb_kernel_ | ✓ |
| 55 | scatter_add_D4096 | scatter_add_kernel_ | ✓ |
| 56 | scatter_add_D7168 | scatter_add_kernel_ | ✓ |
| 57 | sigmoid_N4 | sigmoid_kernel_ | ✓ |
| 58 | sigmoid_dynamic | sigmoid_kernel_ | ✓ |
| 59 | sparse_attn_h64_d512 | sparse_attn_kernel_ | ✓ |
| 60 | swiglu_N2048 | swiglu_kernel_ | ✓ |
| 61 | scale_f32_N4096 | scale_f32_kernel_ | ✓ |
| 62 | scale_f32_N64 | scale_f32_kernel_ | ✓ |
| 63 | fp4_quant_N448_bs32_inplace | fp4_quant_kernel_ | ✓ |
| 64 | fused_shared_ffn_D4096_I2048 | fused_shared_ffn_kernel_ | ✓ |

---

## 附录 A: 编译命令

```bash
# 在容器内编译所有内核
docker exec ds4rs-dev python /workspace/tilelang/compile_kernels.py --batch all

# 按批次编译
docker exec ds4rs-dev python /workspace/tilelang/compile_kernels.py --batch A1  # 量化+GEMM+注意力+HC
docker exec ds4rs-dev python /workspace/tilelang/compile_kernels.py --batch A2  # 归一化+SwiGLU+RoPE
docker exec ds4rs-dev python /workspace/tilelang/compile_kernels.py --batch B1  # FP4 GEMM
```

## 附录 B: 内核编译参数

| 编译参数 | 值 | 说明 |
|---|---|---|
| `pass_configs` | `TL_DISABLE_WARP_SPECIALIZED=True, TL_DISABLE_TMA_LOWER=True` | 禁用 warp 特化和 TMA 降低 (兼容性) |
| `execution_backend` | `tvm_ffi` | 使用 TVM FFI 后端 |
| `tilelang.set_log_level` | `WARNING` | 仅显示警告及以上日志 |
