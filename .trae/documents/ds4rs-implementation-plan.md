# ds4.rs — DeepSeek V4 Flash 推理引擎实施计划

## 项目概述

基于 AGENTS.md 的目标，构建一个小巧、可读、高性能的 Rust 推理引擎，复刻 `/data/ai/models/dsv4/inference/` 下官方 Python 推理代码。核心策略：TileLang 编译算子为 .so 共享库，Rust 通过 `tvm-ffi` crate 加载执行，DLPack 协议实现 GPU 张量零拷贝。

## 模型架构关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| hidden_size (dim) | 4096 | 隐藏维度 |
| n_layers | 43 | Transformer 层数 |
| n_heads | 64 | 注意力头数 |
| head_dim | 512 | 每头维度（含 RoPE 64 + 非 RoPE 448） |
| n_kv_heads | 1 | KV 头数（MLA 单头压缩） |
| q_lora_rank | 1024 | Q 投影 LoRA 秩 |
| o_lora_rank | 1024 | O 投影 LoRA 秩 |
| o_groups | 8 | O 投影分组数 |
| qk_rope_head_dim | 64 | RoPE 维度 |
| n_routed_experts | 256 | 路由专家总数 |
| n_shared_experts | 1 | 共享专家数 |
| n_activated_experts | 6 | 每 token 激活专家数 |
| moe_inter_dim | 2048 | MoE FFN 中间维度 |
| hc_mult | 4 | Hyper-Connection 副本数 |
| hc_sinkhorn_iters | 20 | Sinkhorn 迭代次数 |
| vocab_size | 129280 | 词表大小 |
| sliding_window | 128 | 滑动窗口大小 |
| index_n_heads | 64 | 索引器头数 |
| index_head_dim | 128 | 索引器头维度 |
| index_topk | 512 | 索引器 Top-K |
| scoring_func | sqrtsoftplus | 专家评分函数 |
| route_scale | 1.5 | 路由专家缩放因子 |
| swiglu_limit | 10.0 | SwiGLU 截断值 |
| rope_theta | 10000 | RoPE 基础频率 |
| compress_rope_theta | 160000 | 压缩 KV RoPE 频率 |
| yarn_factor | 16 | YaRN 缩放因子 |
| compress_ratios | [0,0,4,128,4,128,...,4,0] | 每层压缩比（43 值，对应 43 层；层0-1纯SWA, 层2-41交替 ratio=4(有Indexer)/128(无Indexer), 层42纯SWA；MTP 层 ratio=0 不在此数组中） |
| n_hash_layers | 3 | 哈希路由层数（前 3 层） |
| n_mtp_layers | 1 | MTP 层数 |
| expert_dtype | FP4 (e2m1fn_x2 packed as I8) + E8M0 scale, per-row group_size=32 | 专家权重精度（safetensors 中存储为 I8 打包格式，shape [out, in//2]；可用 convert.py --expert-dtype fp8 转为 FP8） |
| non_expert_dtype | FP8 (e4m3fn, ue8m0 scale, block 128×128) | 非专家权重精度 |
| 模型总大小 | ~159.6 GB (decimal) / ~148.6 GiB | 46 个 safetensors 分片；其中专家权重 ~140.3 GB (88.3%)，非专家权重 ~8.4 GB (5.3%) |

---

## TileLang 内核全景分析

对 `inference/model.py` (827行) 和 `inference/kernel.py` (536行) 逐行审查，识别所有可转化为 TileLang 内核的 GPU 计算操作。

### A. 已有 TileLang 内核（kernel.py 中 6 个）

| # | 内核名 | model.py 调用位置 | 功能 | 编译时参数 | 运行时参数 |
|---|---|---|---|---|---|
| K1 | `act_quant_kernel` | L114, L117, L372, L416, L506 | 块级 FP8 量化 | N, block_size | M |
| K2 | `fp4_quant_kernel` | L370, L416 | 块级 FP4 量化 | N, block_size=32 | M |
| K3 | `fp8_gemm_kernel` | L118 | FP8×FP8 矩阵乘 | N, K | M |
| K4 | `fp4_gemm_kernel` | L115 | FP8×FP4 矩阵乘 | N, K | M |
| K5 | `sparse_attn_kernel` | L528, L533 | 稀疏注意力 | h=64, d=512 | b, m, n, topk |
| K6 | `hc_split_sinkhorn_kernel` | L679 | HC Sinkhorn 分解 | hc=4, iters=20, eps=1e-6 | n |

### B. 需新增的 TileLang 内核（model.py 中识别）

以下按优先级排列，分析每个操作的来源代码、计算模式、融合收益。

---

#### P0 — 关键路径，高频调用，融合收益大

**K7: `rmsnorm_kernel` — 融合 RMSNorm**

- **来源**: model.py L183-196 (`RMSNorm.forward`)、L498 (Q norm: `rsqrt(q.square().mean(-1) + eps)`)
- **调用频率**: 每层 2 次 (attn_norm + ffn_norm) + 每层 1 次 (kv_norm) + 每层 1 次 (q_norm) + Compressor 内 1 次 ≈ **~172 次/前向传播**
- **当前 Python 实现**:
  ```python
  x = x.float()
  var = x.square().mean(-1, keepdim=True)
  x = x * torch.rsqrt(var + self.eps)
  return (self.weight * x).to(dtype)
  ```
- **TileLang 内核设计**:
  - 输入: `X[M, N]` (BF16), `W[N]` (FP32)
  - 输出: `Y[M, N]` (BF16)
  - 计算: `Y[i,j] = (X[i,j] * rsqrt(mean(X[i,:]^2) + eps) * W[j])` — 全融合
  - 分块: `blk_m=32`, `threads=128`
  - **融合收益**: 消除 3 次 kernel launch (square→mean→rsqrt→mul) + 2 次 dtype 转换，单次 kernel 完成
- **变体**: `rmsnorm_no_weight` — 无 weight 的 RMSNorm（用于 Q norm L498）

**K8: `rotary_emb_kernel` — 融合旋转位置编码**

- **来源**: model.py L232-244 (`apply_rotary_emb`)
- **调用频率**: 每层 Q RoPE + KV RoPE + O 逆 RoPE + Compressor KV RoPE ≈ **~130 次/前向传播**
- **当前 Python 实现**:
  ```python
  x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
  if inverse: freqs_cis = freqs_cis.conj()
  x = torch.view_as_real(x * freqs_cis).flatten(-2)
  y.copy_(x)
  ```
- **TileLang 内核设计**:
  - 输入: `X[M, H, D_rope]` (BF16), `freqs_cis[seq_len, D_rope//2]` (complex FP32)
  - 输出: `Y[M, H, D_rope]` (BF16, in-place 可选)
  - 计算: 对每对 (x[2i], x[2i+1]) 执行 2D 旋转 `(x[2i]*cos - x[2i+1]*sin, x[2i]*sin + x[2i+1]*cos)`
  - 支持 `inverse=True`（取共轭）
  - 分块: `blk_m=32`, `threads=128`
  - **融合收益**: 消除 unflatten→complex_view→mul→real_view→flatten 的 5 次 kernel launch + 中间 tensor 分配

**K9: `swiglu_kernel` — 融合 SwiGLU 激活 + 截断**

- **来源**: model.py L596-606 (`Expert.forward`)
- **调用频率**: 每层最多 6 路由专家 + 1 共享专家 ≈ **~301 次/前向传播** (43层 × 7专家)；实际路由专家数取决于 Gate 选择，非哈希层每 token 选 6 个
- **当前 Python 实现**:
  ```python
  gate = self.w1(x).float()
  up = self.w3(x).float()
  if self.swiglu_limit > 0:
      up = torch.clamp(up, min=-10.0, max=10.0)
      gate = torch.clamp(gate, max=10.0)
  x = F.silu(gate) * up
  ```
- **TileLang 内核设计**:
  - 输入: `gate[M, inter_dim]` (FP32/BF16), `up[M, inter_dim]` (FP32/BF16), `limit` (FP32)
  - 输出: `Y[M, inter_dim]` (BF16)
  - 计算: `Y = silu(clamp(gate, max=limit)) * clamp(up, min=-limit, max=limit)`
  - 注意: model.py L600-602 中 up 的 clamp 下限是 `-swiglu_limit` 而非 `-10.0`（swiglu_limit=10.0 为配置值）
  - 分块: `blk_m=32`, `blk_n=128`, `threads=128`
  - **融合收益**: 消除 2×clamp + silu + mul 的 4 次 kernel launch；与 GEMM 输出零拷贝衔接

**K10: `gate_topk_kernel` — 融合 MoE 门控评分 + Top-K 选择**

- **来源**: model.py L564-584 (`Gate.forward`)
- **调用频率**: 每层 1 次 ≈ **43 次/前向传播**
- **当前 Python 实现**:
  ```python
  scores = linear(x.float(), self.weight.float())
  scores = F.softplus(scores).sqrt()
  original_scores = scores
  scores = scores + self.bias
  indices = scores.topk(self.topk, dim=-1)[1]
  weights = original_scores.gather(1, indices)
  weights /= weights.sum(dim=-1, keepdim=True)
  weights *= self.route_scale
  ```
- **TileLang 内核设计**:
  - 输入: `scores[M, 256]` (FP32, GEMM 输出), `bias[256]` (FP32), `topk=6`, `route_scale=1.5`
  - 输出: `weights[M, 6]` (FP32), `indices[M, 6]` (INT32)
  - 计算: sqrtsoftplus → +bias → topk → gather → normalize → scale — 全融合
  - 分块: `blk_m=32`, `threads=256`
  - **融合收益**: 消除 5 次 kernel launch；topk 是关键瓶颈，融合后避免中间大 tensor 写回显存
- **变体**: 哈希路由层 (L0-2) 不需要此内核，直接查表 `tid2eid[input_ids]`（shape [129280, 6], dtype I64，5.92 MB/层）

**K11: `indexer_score_kernel` — 融合 Indexer 评分计算**

- **来源**: model.py L420-427 (`Indexer.forward`)
- **调用频率**: ratio=4 的层（约 20 层）≈ **20 次/前向传播**
- **当前 Python 实现**:
  ```python
  index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
  index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
  topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
  ```
- **TileLang 内核设计**:
  - 输入: `q[bsz, seq, 64, 128]` (BF16), `kv_cache[bsz, n_compress, 128]` (BF16), `weights[bsz, seq, 64]` (BF16)
  - 输出: `topk_idxs[bsz, seq, 512]` (INT32)
  - 计算: batched matmul (64头并行) → ReLU → weighted sum → causal mask → top-512
  - 分块: 利用 TileLang T.gemm 做批量 matmul，然后融合后续操作
  - **融合收益**: 融合 ReLU+sum+topk 避免中间 `[bsz, seq, 64, n_compress]` 大 tensor

---

#### P1 — 重要优化，中等融合收益

**K12: `hadamard_kernel` — Hadamard 旋转**

- **来源**: model.py L247-251 (`rotate_activation`)
- **调用频率**: Indexer 内 Q 和 KV 各 1 次（ratio=4 层）≈ **40 次/前向传播**
- **当前 Python 实现**:
  ```python
  from fast_hadamard_transform import hadamard_transform
  return hadamard_transform(x, scale=x.size(-1) ** -0.5)
  ```
- **TileLang 内核设计**:
  - 输入: `X[M, N]` (BF16), N 必须是 2 的幂
  - 输出: `Y[M, N]` (BF16)
  - 计算: 快速 Walsh-Hadamard 变换 + scale = N^(-0.5)
  - 分块: `blk_m=32`, `blk_n=N`, `threads=128`
  - **融合收益**: 消除对 `fast_hadamard_transform` 外部库的依赖；可与 FP4 量化融合

**K13: `compressor_kernel` — 融合 KV 压缩门控池化**

- **来源**: model.py L316-377 (`Compressor.forward`)
- **调用频率**: 有压缩的层（41 层）≈ **41 次/前向传播**
- **核心计算** (decode 阶段, L343-359):
  ```python
  kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
  score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
  kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
  ```
- **TileLang 内核设计**:
  - 输入: `kv_state[bsz, 2*ratio, 2*d]` (FP32), `score_state[bsz, 2*ratio, 2*d]` (FP32), `ape[ratio, 2*d]` (FP32)
  - 输出: `compressed_kv[bsz, 1, d]` (BF16)
  - 计算: 重叠窗口重组 → softmax → 加权求和 — 全融合
  - 分块: `threads=128`
  - **融合收益**: 消除 cat→softmax→mul→sum 的 4 次 kernel launch + 中间 tensor

**K14: `hc_pre_kernel` — 融合 HC 前置混合**

- **来源**: model.py L673-681 (`Block.hc_pre`)
- **调用频率**: 每层 2 次 (attn + ffn) ≈ **86 次/前向传播**
- **当前 Python 实现**:
  ```python
  x = x.flatten(2).float()
  rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
  mixes = F.linear(x, hc_fn) * rsqrt
  pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)
  y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
  ```
- **TileLang 内核设计**:
  - 输入: `x[bsz, seq, hc, dim]` (BF16), `hc_fn[mix_hc, hc*dim]` (FP32), `hc_scale[3]` (FP32), `hc_base[mix_hc]` (FP32)
  - 输出: `y[bsz, seq, dim]` (BF16), `post[bsz, seq, hc]` (FP32), `comb[bsz, seq, hc, hc]` (FP32)
  - 计算: flatten→rmsnorm→linear→*rsqrt→sinkhorn→pre*sum — 与 K6 融合
  - **融合收益**: 将 RMSNorm + GEMM + Sinkhorn + 加权求和 融合为 2 个 kernel (GEMM 单独 + 融合后处理)

**K15: `hc_post_kernel` — 融合 HC 后置混合**

- **来源**: model.py L683-686 (`Block.hc_post`)
- **调用频率**: 每层 2 次 ≈ **86 次/前向传播**
- **当前 Python 实现**:
  ```python
  y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
  ```
- **TileLang 内核设计**:
  - 输入: `x[bsz, seq, dim]` (BF16), `residual[bsz, seq, hc, dim]` (BF16), `post[bsz, seq, hc]` (FP32), `comb[bsz, seq, hc, hc]` (FP32)
  - 输出: `y[bsz, seq, hc, dim]` (BF16)
  - 计算: `y[i,h,:] = post[i,h]*x[i,:] + sum_h'(comb[i,h,h']*residual[i,h',:])` — 矩阵-向量乘 + 广播加法
  - 分块: `blk_m=32`, `blk_n=128`, `threads=128`
  - **融合收益**: 消除 unsqueeze→broadcast→mul→sum 的多次 kernel launch

**K16: `hc_head_kernel` — 融合 HC Head 混合**

- **来源**: model.py L728-735 (`ParallelHead.hc_head`)
- **调用频率**: 每次生成 1 次
- **当前 Python 实现**:
  ```python
  x = x.flatten(2).float()
  rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
  mixes = F.linear(x, hc_fn) * rsqrt
  pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
  y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
  ```
- **TileLang 内核设计**:
  - 输入: `x[bsz, seq, hc, dim]` (BF16), `hc_fn[hc, hc*dim]` (FP32), `hc_scale[1]` (FP32), `hc_base[hc]` (FP32)
  - 输出: `y[bsz, seq, dim]` (BF16)
  - 计算: flatten→rmsnorm→linear→sigmoid→加权求和 — 与 K14 类似但用 sigmoid 替代 sinkhorn
  - **融合收益**: 与 K14 相同模式，消除 5 次 kernel launch

---

#### P2 — 可选优化，低融合收益或复杂度高

**K17: `wo_a_einsum_kernel` — 分组 O 投影 einsum**

- **来源**: model.py L538-541
  ```python
  wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
  o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
  ```
- **分析**: batched matmul，`g=8` 组，可用 cuBLAS batched GEMM 实现
- **建议**: 使用 cudarc cuBLAS batched GEMM，不单独写 TileLang 内核

**K18: `moe_dispatch_kernel` — MoE 专家分发/归约**

- **来源**: model.py L629-644 (`MoE.forward`)
- **分析**: 包含 bincount、where 索引、逐专家计算、加权累加。模式不规则
- **建议**: 初版用 Rust CPU 端调度 + 逐专家 GEMM；后续可考虑 TileLang grouped GEMM

**K19: `logits_kernel` — 融合 logits 计算**

- **来源**: model.py L715-716
- **分析**: `[1, dim] × [vocab, dim]` 的大 GEMM
- **建议**: 使用 cudarc cuBLAS

**K20: `embedding_kernel` — 融合词嵌入查表**

- **来源**: model.py L96-105
- **分析**: 简单的 GPU 查表操作
- **建议**: 使用 cudarc 手动实现 GPU embedding 查表

---

### C. 不适合转化为 TileLang 的操作

| 操作 | 来源 | 原因 |
|---|---|---|
| `precompute_freqs_cis` | L199-229 | 一次性预计算，CPU 端执行即可 |
| `get_window_topk_idxs` | L254-265 | 索引计算，CPU 端小 tensor |
| `get_compress_topk_idxs` | L268-276 | 同上 |
| `Compressor.overlap_transform` | L307-314 | tensor 重排，可用 cudarc memcpy |
| KV cache 索引/拼接 | L518-533 | 内存操作，非计算密集 |
| `convert.py` 全部 | - | 离线权重转换工具，非推理路径 |
| `generate.py` 采样逻辑 | L19-69 | CPU 端控制流 |

### D. TileLang 内核总览与实例化参数

每个内核需要按模型参数实例化编译。以下列出 DS-V4 模型下各内核的具体编译参数：

| 内核 | 编译实例 | 固定参数 | 符号维度 | 用途 |
|---|---|---|---|---|
| K1 `act_quant` | act_quant_N4096_bs128 | N=4096, bs=128, round_scale=True, scale=E8M0 | M | 主线性层激活量化 (wq_a, wkv, w1, w3) |
| K1 `act_quant` | act_quant_N8192_bs128 | N=8192, bs=128, round_scale=True, scale=E8M0 | M | wo_b 激活量化 |
| K1 `act_quant` | act_quant_N2048_bs128 | N=2048, bs=128, round_scale=True, scale=E8M0 | M | expert w2 激活量化 |
| K1 `act_quant` | act_quant_N1024_bs128 | N=1024, bs=128, round_scale=True, scale=E8M0 | M | wq_b, Indexer wq_b 激活量化 |
| K1 `act_quant` | act_quant_N448_bs64 | N=448, bs=64, round_scale=True, scale=E8M0, inplace=True | M | KV 非 RoPE 维度 FP8 模拟量化 (L372, L506, Compressor/Indexer) |
| K2 `fp4_quant` | fp4_quant_N512_bs32 | N=512, bs=32 | M | Compressor KV / Indexer Q FP4 模拟 |
| K3 `fp8_gemm` | fp8_gemm_N32768_K1024 | N=32768, K=1024, scale=E8M0 | M | wq_b |
| K3 `fp8_gemm` | fp8_gemm_N512_K4096 | N=512, K=4096, scale=E8M0 | M | wkv |
| K3 `fp8_gemm` | fp8_gemm_N1024_K4096 | N=1024, K=4096, scale=E8M0 | M | wq_a |
| K3 `fp8_gemm` | fp8_gemm_N8192_K4096 | N=8192, K=4096, scale=E8M0 | M | wo_a (仅当未转换时使用; convert.py 转换后走 BF16 einsum) |
| K3 `fp8_gemm` | fp8_gemm_N4096_K8192 | N=4096, K=8192, scale=E8M0 | M | wo_b |
| K3 `fp8_gemm` | fp8_gemm_N2048_K4096 | N=2048, K=4096, scale=E8M0 | M | shared_expert w1/w3 |
| K3 `fp8_gemm` | fp8_gemm_N4096_K2048 | N=4096, K=2048, scale=E8M0 | M | shared_expert w2 |
| K3 `fp8_gemm` | fp8_gemm_N8192_K1024 | N=8192, K=1024, scale=E8M0 | M | Indexer wq_b |
| K4 `fp4_gemm` | fp4_gemm_N2048_K4096 | N=2048, K=4096, scale=E8M0 | M | routed_expert w1/w3 |
| K4 `fp4_gemm` | fp4_gemm_N4096_K2048 | N=4096, K=2048, scale=E8M0 | M | routed_expert w2 |
| K5 `sparse_attn` | sparse_attn_h64_d512 | h=64, d=512 | b, m, n, topk | 稀疏注意力 |
| K6 `hc_sinkhorn` | hc_sinkhorn_hc4_it20 | hc=4, iters=20, eps=1e-6 | n | HC Sinkhorn |
| K7 `rmsnorm` | rmsnorm_N4096 | N=4096 | M | attn_norm, ffn_norm |
| K7 `rmsnorm` | rmsnorm_N1024 | N=1024 | M | q_norm |
| K7 `rmsnorm` | rmsnorm_N512 | N=512 | M | kv_norm |
| K7 `rmsnorm` | rmsnorm_no_weight_N1024 | N=1024 | M | Q norm (L498) |
| K8 `rotary_emb` | rotary_emb_D64 | D_rope=64 | M, H | Q/KV/O RoPE |
| K9 `swiglu` | swiglu_N2048 | inter_dim=2048 | M | Expert SwiGLU |
| K10 `gate_topk` | gate_topk_E256_K6 | E=256, topk=6 | M | MoE 门控 |
| K11 `indexer_score` | indexer_score_h64_d128 | h=64, d=128 | bsz, seq, n_compress | Indexer 评分 |
| K12 `hadamard` | hadamard_N512 | N=512 | M | Indexer Q/KV Hadamard 旋转 (head_dim=512) |
| K13 `compressor` | compressor_r4_d512 | ratio=4, d=512 | bsz | ratio=4 层压缩 |
| K13 `compressor` | compressor_r128_d512 | ratio=128, d=512 | bsz | ratio=128 层压缩 |
| K14 `hc_pre` | hc_pre_hc4_dim4096 | hc=4, dim=4096 | bsz, seq | HC 前置混合 |
| K15 `hc_post` | hc_post_hc4_dim4096 | hc=4, dim=4096 | bsz, seq | HC 后置混合 |
| K16 `hc_head` | hc_head_hc4_dim4096 | hc=4, dim=4096 | bsz, seq | HC Head 混合 |

**注意**: K3 `fp8_gemm` 和 K4 `fp4_gemm` 需要为每对 (N, K) 编译独立实例。K1 `act_quant` 的 N 对应 GEMM 的 K 维度（激活的列维度），block_size 按调用处参数确定。

### E. model.py 中线性层 (N, K) 参数完整清单

| 层/组件 | 权重名 | 形状 | N | K | dtype |
|---|---|---|---|---|---|
| Attention | wq_a | [1024, 4096] | 1024 | 4096 | FP8 |
| Attention | wq_b | [32768, 1024] | 32768 | 1024 | FP8 |
| Attention | wkv | [512, 4096] | 512 | 4096 | FP8 |
| Attention | wo_a | [8192, 4096] FP8 或 [8, 1024, 4096] BF16 | 8192 | 4096 | FP8 (原始) / BF16 (convert.py 反量化后) |
| Attention | wo_b | [4096, 8192] | 4096 | 8192 | FP8 |
| Compressor | wkv | [1024, 4096] | 1024 | 4096 | BF16 (运行时 FP32) |
| Compressor | wgate | [1024, 4096] | 1024 | 4096 | BF16 (运行时 FP32) |
| Indexer | wq_b | [8192, 1024] | 8192 | 1024 | FP8 |
| Indexer | weights_proj | [64, 4096] | 64 | 4096 | BF16 |
| Shared Expert | w1 | [2048, 4096] | 2048 | 4096 | FP8 |
| Shared Expert | w2 | [4096, 2048] | 4096 | 2048 | FP8 |
| Shared Expert | w3 | [2048, 4096] | 2048 | 4096 | FP8 |
| Routed Expert | w1 | [2048, 2048] (I8/FP4 packed) | 2048 | 4096 (逻辑) | FP4 (e2m1fn_x2 packed as I8, scale [2048,128] F8_E8M0, per-row group_size=32) |
| Routed Expert | w2 | [4096, 1024] (I8/FP4 packed) | 4096 | 2048 (逻辑) | FP4 (e2m1fn_x2 packed as I8, scale [4096,64] F8_E8M0, per-row group_size=32) |
| Routed Expert | w3 | [2048, 2048] (I8/FP4 packed) | 2048 | 4096 (逻辑) | FP4 (e2m1fn_x2 packed as I8, scale [2048,128] F8_E8M0, per-row group_size=32) |
| Gate | weight | [256, 4096] | 256 | 4096 | BF16 |
| Head | weight | [129280, 4096] | 129280 | 4096 | BF16 |
| Embed | weight | [129280, 4096] | 129280 | 4096 | BF16 |

去重后的 GEMM 实例：

| 类型 | (N, K) 对 |
|---|---|
| FP8 GEMM | (1024,4096), (32768,1024), (512,4096), (4096,8192), (8192,1024), (2048,4096), (4096,2048) — 注意 (8192,4096) wo_a 仅当模型未转换时走 FP8 GEMM；convert.py 转换后走 BF16 einsum |
| FP4 GEMM | (2048,4096), (4096,2048) — 路由专家 (FP4 模型; I8 packed shape [2048,2048] 和 [4096,1024]，逻辑 K=4096/2048; scale per-row group_size=32) |
| BF16 GEMM | (256,4096), (64,4096), (129280,4096), (8192,4096 batched) — Gate, Indexer weights_proj, Head, wo_a einsum; 用 cuBLAS |
| FP32 GEMM | (1024,4096) — Compressor wkv/wgate, BF16 存储→FP32 计算, 用 cuBLAS; (24,16384) — hc_attn_fn, FP32; (4,16384) — hc_head_fn, FP32 |

---

## Mega-Kernel 融合方案可行性评估

对 AGENTS.md 提出的融合方案逐项评估。

### 方案 0：简化版 Mega-Kernel（融合 Attention+FFN，无 MoE）— 验证通路

**可行性：✅ 高度可行，应优先实现**

AGENTS.md 明确要求先跑通简化版验证 Rust→TileLang 数据通路，再扩展为完整版。

**简化版融合内核 `simple_block_kernel` (MK0)**：

将单层 Transformer Block 的 Attention 侧 + 简化 FFN（用共享专家替代 MoE）融合为一个 persistent kernel：

```
┌─────────────────────────────────────────────────────────────┐
│ MK0: Simple Block Mega-Kernel (验证 Rust→TileLang 通路)      │
│                                                               │
│  Phase 1: Attention Side                                     │
│    RMSNorm → act_quant → QKV GEMM → RoPE                    │
│    → sparse_attn(SWA only, 无CSA) → O Proj → HC_post        │
│                                                               │
│  Phase 2: FFN Side (简化为共享专家，无路由专家)                 │
│    RMSNorm → HC_pre → act_quant → FP8 GEMM (w1,w3)          │
│    → SwiGLU → FP8 GEMM (w2) → HC_post                       │
│                                                               │
│  简化点:                                                      │
│    - 无 MoE 路由 (Gate TopK)                                 │
│    - 无路由专家 (FP4 GEMM)                                    │
│    - 无 KV 压缩 (Compressor/Indexer)                         │
│    - 仅滑动窗口注意力 (SWA)，无 CSA                           │
│    - 仍保留 Hyper-Connection (HC)                            │
│    - 仍保留 FP8 量化路径                                      │
└─────────────────────────────────────────────────────────────┘
```

**验证目标**：
1. Rust 通过 tvm-ffi 加载 .so 并调用 TileLang 内核 — **数据通路正确性**
2. GpuTensor → DLPack → TileLang kernel → DLPack → GpuTensor — **零拷贝正确性**
3. CUDA stream 管理 — **异步执行正确性**
4. 融合内核的数值结果与分步内核一致 — **计算正确性**
5. 层间专家预取机制 — **PCIe 传输正确性**

**简化版与完整版的差异**：

| 特性 | 简化版 MK0 | 完整版 MK4 |
|---|---|---|
| Attention | SWA only (window=128) | SWA + CSA (HybridAttention) |
| FFN | 共享专家 (1×FP8 GEMM) | MoE (6×FP4 GEMM + Gate) |
| KV Cache | 滑动窗口环形缓冲 | 滑动窗口 + 压缩 KV + Indexer |
| 专家预取 | 无 (无路由专家) | 三级缓存 + 跨层预取 |
| HC | ✅ 保留 | ✅ 保留 |
| FP8 量化 | ✅ 保留 | ✅ 保留 |
| RoPE | ✅ 保留 | ✅ 保留 |
| 内核复杂度 | 低 (2 phase, 无动态调度) | 高 (多 phase, MoE 动态路由) |

**简化版使用的内核子集**：K3 (fp8_gemm), K5 (sparse_attn), K6 (hc_sinkhorn), K7 (rmsnorm), K8 (rotary_emb), K9 (swiglu), K1 (act_quant), K14 (hc_pre), K15 (hc_post), K16 (hc_head)

**简化版不需要的内核**：K2 (fp4_quant), K4 (fp4_gemm), K10 (gate_topk), K11 (indexer_score), K12 (hadamard), K13 (compressor)

**两阶段策略**：

```
阶段 A: 简化版验证通路
  1. 实现 MK0 简化版融合内核 (TileLang)
  2. 实现 Rust tvm-ffi 加载 + DLPack 零拷贝
  3. 实现 GpuTensor 管理 + CUDA stream
  4. 实现权重加载 (仅非专家权重 + 共享专家)
  5. 实现简化版单层前向传播
  6. 与官方 Python 结果逐层对比验证
  7. 实现层间专家预取框架 (无路由专家时验证框架正确性)
  ✓ 验证 Rust→TileLang 数据通路完全正确

阶段 B: 完整版 MoE Mega-Kernel
  1. 添加 K2, K4, K10-K13 内核
  2. 实现 MK1 HybridAttention (SWA+CSA+HCA)
  3. 实现 MK2 MoE-GEMM (路由+FP4+SwiGLU)
  4. 实现 MK3 KV-Compress (压缩+Norm+RoPE+Quant)
  5. 实现路由专家三级缓存 + 跨层预取
  6. 实现 MK4 全层 Mega-Kernel
  7. 与官方 Python 结果逐层对比验证
  ✓ 达到最高性能
```

**JIT vs AOT 编译策略**（AGENTS.md 要求）：
- **测试阶段**：使用 TileLang JIT 编译，即时验证算子融合是否生效
- **生产阶段**：使用 AOT 预编译 .so，避免运行时编译延迟

### 方案 1：TileLang 融合内核 HybridAttention(SWA+CSA+HCA)

**可行性：✅ 可行**

当前官方 `sparse_attn_kernel` (K5) 已实现 SWA+CSA 混合注意力。融合方案在此基础上进一步整合 Indexer 评分：

**融合内核 `hybrid_attn_kernel` (MK1)**：
- 将 Indexer 评分 (K11) + 索引拼接 + sparse_attn (K5) 融合为 1 个 persistent kernel
- 输入: `q[bsz, seq, h, d]`, `kv_cache[bsz, n, d]`, `compressed_kv[bsz, n_c, d]`, `attn_sink[h]`, 窗口参数, 压缩参数
- 输出: `o[bsz, seq, h, d]`
- **收益**: 消除 Indexer 评分 + 索引拼接 + sparse_attn 之间的多次 kernel launch 和中间 tensor 写回全局内存
- **技术依据**: TileLang FTG 支持单 kernel 内融合多步计算；DeepSeek-V4 技术报告已验证 Compressor+RMSNorm+RoPE 融合

**风险**: 内核复杂度高；需仔细处理 SWA/CSA 索引计算的并行策略

### 方案 2：DeepSeek TileKernels(MoE/Quant): MoE路由+FP8/FP4 GEMM

**可行性：✅ 可行**

**融合内核 `moe_gemm_kernel` (MK2)**：
- 基于 TileLang `T.Persistent` 原语，启动 `sm_num` 个持久化 thread block
- 每个 SM 自主从全局任务队列领取 (token, expert) 对
- 计算流程：Gate 评分 + Top-K → act_quant → FP4 GEMM (w1,w3) → SwiGLU → FP4 GEMM (w2) → 加权归约
- **收益**: 消除 MoE 层内 ~20 次 kernel launch；利用 persistent kernel 减少 launch 开销
- **技术依据**: TileLang 已有 W4A8 GEMM CV fusion 先例；DeepSeek-V4 技术报告提到 Lightning TopK 和 mHC kernel

**风险**: MoE 路由的动态性使任务调度复杂；需仔细设计 persistent block 的工作分配策略

### 方案 3：TileLang KV Cache 压缩解压(CSA/HCA)

**可行性：✅ 可行**

**融合内核 `kv_compress_kernel` (MK3)**：
- Prefill: `wkv/wgate GEMM → overlap_transform → softmax → weighted_sum → RMSNorm → RoPE → FP8/FP4 quant` — 全融合
- Decode: `kv_state更新 → softmax → weighted_sum → RMSNorm → RoPE → FP8/FP4 quant` — 全融合
- **收益**: 消除 Compressor 内 5-6 次 kernel launch；压缩后 KV 直接写入 cache
- **技术依据**: DeepSeek-V4 技术报告明确提到 "Compressor+RMSNorm+RoPE+cache insertion" 融合已有先例

**风险**: prefill/decode 两个路径的分支逻辑需在 kernel 内处理

### 方案 4：Mega-kernel(RMSNorm→QKV→Attention→Proj→Residual→FFN) 全层融合

**可行性：⚠️ 有限可行，需分阶段推进**

这是将整个 Transformer Block 融合为单个 persistent kernel 的方案。调研发现：

**硬件约束**：

| 资源 | RTX 5060Ti (Blackwell) | 全层融合需求 | 差距 |
|---|---|---|---|
| Shared Memory / SM | ~228 KB | ~250 KB+ (QKV+Attn+O_proj+FFN) | **超出容量** |
| 寄存器 / SM | 65536 × 32-bit | 多算子累加器指数增长 | 严重溢出 |
| SM 数量 | 34 | grid.sync() 开销占比大 | 同步瓶颈 |
| CUDA Stream | 单 stream | 跨算子依赖需全局同步 | 同步瓶颈 |

**关键发现**（来自 MegaQwen 实测数据）：
- 全层 Cooperative Megakernel 在 batch=1 下带宽利用率仅 **5%**
- 140+ 次 `grid.sync()` × 0.7µs = ~100µs 纯同步等待，比内存加载还慢
- Cooperative Groups 比 CUDA Graphs 仅快 19.7µs，但需额外 ~2.7 MB 中间缓冲区

**学术进展**：
- **MPK (Mirage Persistent Kernel)**: 编译器自动将 LLM 编译为单 mega-kernel，Qwen3-8B 上比 vLLM 快 1.16×；多 GPU 场景最高 6.7×
- **ETC (Event Tensor Compiler)**: MLSys 2026，支持 MoE 动态路由的 mega-kernel，TTFT 降低 18-32%
- **Stanford Megakernel**: Llama-1B 在 H100 上达 78% 带宽利用率，比 vLLM 快 1.5×

**TileLang 能力边界**：

| 融合层次 | TileLang 支持 | 说明 |
|---|---|---|
| 单算子内 | ✅ 成熟 | GEMM+pipelining, FlashAttention |
| 相邻算子 | ✅ 支持 | RMSNorm+Residual, GEMM+SiLU |
| Attention 整体 | ✅ 支持 | Q·K^T+Softmax+·V |
| **整层融合** | ⚠️ 有限 | 需 shared memory 精细复用，无内置支持 |
| **跨层融合** | ❌ 不支持 | 需编译器级 task graph，超出 FTG 范围 |

**推荐策略：分阶段递进**

| 阶段 | 方案 | 可行性 | 收益 | 依赖 |
|---|---|---|---|---|
| 近期 | 算子级融合（K7-K16） | ✅ 高 | 减少 kernel launch 30-50% | 无 |
| 中期 | MK1 HybridAttention + MK2 MoE-GEMM + MK3 KV-Compress | ✅ 高 | Attention/MoE/Compressor 各 1 kernel | K1-K16 验证正确 |
| 远期 | 半层融合：Attn 侧 (RMSNorm→QKV→Attn→Proj→HC_post) | ⚠️ 中 | Attn 侧 1 kernel | MK1 + shared memory 精细复用 |
| 远期 | 半层融合：FFN 侧 (RMSNorm→Gate→MoE→HC_post) | ⚠️ 中 | FFN 侧 1 kernel | MK2 + 动态调度 |
| 远期 | 全层融合 (RMSNorm→QKV→Attn→Proj→Residual→FFN) | ⚠️ 低-中 | 整层 1 kernel | 自研 task graph 调度器或等待 ETC 开源 |
| 远期 | 跨层持久化 + 预取 | ❌ 极低 | 超出 TileLang 能力 | 需 MPK/ETC 级编译器 |

**全层 Mega-Kernel 设计 (MK4)**：

如果未来实现，架构如下：
```
┌─────────────────────────────────────────────────────────────┐
│ MK4: Block Mega-Kernel (persistent, sm_num thread blocks)    │
│                                                               │
│  Phase 1: Attn Side                                          │
│    Worker blocks: RMSNorm → act_quant → QKV GEMM → RoPE     │
│    → HybridAttention(SWA+CSA) → O Proj → HC_post            │
│                                                               │
│  Phase 2: FFN Side                                           │
│    Worker blocks: RMSNorm → HC_pre → Gate TopK               │
│    → MoE persistent GEMM → HC_post                           │
│                                                               │
│  Scheduling:                                                  │
│    - Attn 阶段空闲 SM 预取 FFN 权重到 L2 cache              │
│    - Event-driven 依赖调度（借鉴 ETC）                        │
│    - MoE 路由结果驱动动态任务分配                              │
└─────────────────────────────────────────────────────────────┘
```

**前提条件**：
1. MK1-MK3 验证正确且性能达标
2. TileLang 支持跨算子 shared memory 复用或自研调度层
3. RTX 5060Ti 的 SM 数量（34）足够支撑 persistent 调度
4. 或等待 ETC 开源后集成其动态 mega-kernel 编译能力

### 方案 5：Rust tvm-ffi crate 加载 TileLang 算子（.so）

**可行性：✅ 可行，推荐采用**

Apache TVM 官方已发布 `tvm-ffi` Rust crate (v0.1.0-alpha.0)，由 TVM 创始人 Tianqi Chen 维护。

| 能力 | 支持情况 |
|---|---|
| 加载 .so 共享库 | ✅ `Module::load_from_file()` |
| 调用导出函数 | ✅ `Module::get_function()` + `into_typed_fn!` 宏 |
| DLPack 零拷贝 | ✅ Tensor 底层就是 DLTensor |
| CUDA stream 管理 | ✅ `current_stream()`, `with_stream()` |
| GPU Tensor | ⚠️ 需自定义 NDAllocator 或通过 DLPack 桥接 |
| 维护状态 | ✅ 活跃（Apache 基金会项目，2026-05-04 发布） |

**注意事项**：
- `tvm-ffi` 仍处于 alpha 阶段，API 可能变化
- 依赖 `libtvm_ffi` C 共享库（需 `pip install apache-tvm-ffi`）
- 构建时需要 `tvm-ffi-config` 工具定位链接路径

**回退方案**：如果 `tvm-ffi` 不稳定，可回退到 `libloading` + 手动符号解析

### 方案 6：DLPack 协议 GPU 零拷贝

**可行性：✅ 可行（不使用 Candle）**

| 组件 | DLPack 支持 | 说明 |
|---|---|---|
| Candle (huggingface) | ❌ 不支持 | 源码中无任何 DLPack 代码，不公开 GPU 设备指针 |
| cudarc | ❌ 不直接支持 | 但提供 `DevicePtr` trait 可提取 `CUdeviceptr` |
| tvm-ffi crate | ✅ 原生支持 | Tensor 底层就是 DLTensor |

**方案**：不使用 Candle，改用 cudarc + tvm-ffi 自建轻量张量层

```
┌──────────────────────────────────────────────────────┐
│                    ds4.rs 推理引擎                      │
│                                                        │
│  GpuTensor (cudarc)  ←→  tvm-ffi Tensor (DLPack)  ←→  TileLang .so │
│       │                        │                                   │
│  CudaSlice<T>              DLTensor                               │
│  CUdeviceptr      ──零拷贝──→  data = 同一 GPU 指针               │
│  shape/stride/dtype        shape/stride/dtype                     │
│                                                                │
│  cuBLAS (cudarc)          tvm-ffi Function                      │
│  BF16/FP32 GEMM           TileLang 内核调用                      │
└──────────────────────────────────────────────────────┘
```

---

## 实施阶段

### 阶段 0：最小 PoC — 验证 tvm-ffi + DLPack 零拷贝通路

**目标**：在正式开发前，用最小代价验证 Rust→TileLang 数据通路是否可行。若 tvm-ffi 不可用，立即切换到 libloading 方案。

**验证步骤**：

```
阶段 0: tvm-ffi + DLPack 零拷贝 PoC
  1. 在容器内用 TileLang JIT 编译 1 个 fp8_gemm 内核 (K3, N=4096, K=4096)
  2. 导出为 .so 文件
  3. Rust 通过 tvm-ffi (或 libloading) 加载 .so
  4. 构造 GpuTensor (cudarc CudaSlice) → 提取 CUdeviceptr → 构造 DLPack DLTensor
  5. 调用 fp8_gemm kernel，传入 5 个 DLTensor (A, B, C, scales_a, scales_b)
  6. 读回结果，与 PyTorch 参考结果对比
  7. 验证 CUDA stream 管理正确性
  ✓ 若成功：确认 tvm-ffi 可用，进入阶段 1
  ✓ 若失败：切换到 libloading + TVM C ABI (TVMModLoadFromFile/TVMModGetFunction/TVMFuncCall)
```

**PoC 关键验证点**：

| 验证项 | 预期结果 | 失败回退 |
|---|---|---|
| `Module::load_from_file()` 加载 .so | 成功加载 | libloading `dlopen` |
| `get_function()` 获取 PackedFunc | 成功获取 | `TVMModGetFunction` C API |
| `into_typed_fn!` 宏类型转换 | 编译通过 | 手动构造 TVMValue 数组 |
| DLPack FP8 (e4m3fn) dtype | 正确传递 | `kDLUInt`(8bit) + 语义约定 |
| DLPack E8M0 scale dtype | 正确传递 | `kDLUInt`(8bit) + 语义约定 |
| CUdeviceptr 零拷贝 | 数据一致 | memcpy 备用路径 |
| CUDA stream `with_stream()` | 异步执行 | `TVMSetStream` C API |

**DLPack 扩展类型映射**（FP4/E8M0 无标准 DLPack 类型码）：

| ds4rs DType | DLPack type_code | DLDataType bits | 说明 |
|---|---|---|---|
| FP8E4M3 | kDLFloat | 8 | 标准 FP8 |
| FP4E2M1 (packed as I8) | kDLInt | 8 | 物理存储为 I8，shape 用 [out, in//2] |
| FP8E8M0 | kDLUInt | 8 | 无符号 8bit，语义为共享指数 |
| BF16 | kDLFloat | 16 | 标准 BF16 |

**PoC 代码结构**（不进入正式项目，独立验证）：

```rust
fn poc_fp8_gemm() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let module = tvm_ffi::Module::load_from_file("tilelang/build/fp8_gemm_N4096_K4096.so")?;
    let func = module.get_function("fp8_gemm_kernel_")?;

    let a = device.alloc_zeros::<u8>(M * K)?;
    let b = device.alloc_zeros::<u8>(N * K)?;
    let c = device.alloc_zeros::<u16>(M * N)?;
    let scales_a = device.alloc_zeros::<u8>(M * K / 128)?;
    let scales_b = device.alloc_zeros::<u8>((N / 128) * (K / 128))?;

    let a_tensor = make_dlpack_tensor(&a, &[M, K], DType::FP8E4M3, &device);
    // ... 构造其余 DLTensor ...

    tvm_ffi::device::with_stream(stream, || {
        func(&a_tensor, &b_tensor, &c_tensor, &scales_a_tensor, &scales_b_tensor)?;
        Ok(())
    })?;

    // 读回 c，与 PyTorch 参考结果对比
    Ok(())
}
```

**PoC 完成标准**：fp8_gemm 计算结果与 PyTorch 参考误差 < 1e-2（FP8 精度范围内）

### 阶段 1：项目脚手架与基础设施工具

**目标**：建立 Rust 项目结构、构建系统、基础数据类型。

#### 1.1 项目结构初始化

```
ds4rs/
├── Cargo.toml              # workspace 根
├── Makefile                # 构建/测试入口
├── src/
│   ├── lib.rs              # 库入口
│   ├── main.rs             # → ds4_cli 二进制
│   ├── config.rs           # ModelConfig 反序列化
│   ├── tensor.rs           # CpuTensor / GpuTensor / DType / DLPack 桥接
│   ├── weights.rs          # safetensors 权重加载
│   ├── quant.rs            # FP4/FP8 量化工具（CPU 参考）
│   ├── model.rs            # 模型组件（RMSNorm, Attention, MoE 等）
│   ├── kv_cache.rs         # KV 缓存管理
│   ├── expert_cache.rs     # 专家三级缓存
│   ├── scheduler.rs        # 推理调度
│   ├── sampling.rs         # 采样策略
│   ├── tokenizer.rs        # 分词器封装
│   ├── plugin.rs           # 插件机制
│   └── bin/
│       └── ds4_server.rs   # HTTP 服务二进制
├── tilelang/               # TileLang 内核
│   ├── compile_kernels.py  # Python 编译脚本（容器内运行）
│   ├── build/              # 编译输出 .so 目录
│   └── src/                # Rust tvm-ffi 绑定
│       ├── mod.rs
│       ├── registry.rs     # 内核注册表 + tvm-ffi 封装
│       ├── act_quant.rs    # K1, K2: FP8/FP4 激活量化
│       ├── gemm.rs         # K3, K4: FP8/FP4 GEMM
│       ├── sparse_attn.rs  # K5: 稀疏注意力
│       ├── hc_sinkhorn.rs  # K6: HC Sinkhorn
│       ├── rmsnorm.rs      # K7: RMSNorm
│       ├── rotary_emb.rs   # K8: 旋转位置编码
│       ├── swiglu.rs       # K9: SwiGLU 激活
│       ├── gate_topk.rs    # K10: MoE 门控 Top-K
│       ├── indexer_score.rs # K11: Indexer 评分
│       ├── hadamard.rs     # K12: Hadamard 旋转
│       ├── compressor.rs   # K13: KV 压缩门控池化
│       ├── hc_pre.rs       # K14: HC 前置混合
│       ├── hc_post.rs      # K15: HC 后置混合
│       ├── hc_head.rs      # K16: HC Head 混合
│       ├── simple_block.rs # MK0: 简化版融合内核(Attention+FFN无MoE) — 阶段A
│       ├── hybrid_attn.rs  # MK1: SWA+CSA+HCA 融合注意力
│       ├── moe_gemm.rs     # MK2: MoE路由+FP4 GEMM+SwiGLU 融合
│       ├── kv_compress.rs  # MK3: KV压缩+Norm+RoPE+Quant 融合
│       └── block_mega.rs   # MK4: 全层 Mega-Kernel（远期）
├── tests/                  # 集成测试
└── misc/                   # 实验备忘录（gitignore）
```

#### 1.2 Cargo.toml 依赖规划

核心依赖：
- `cudarc`：CUDA Runtime API 绑定（设备管理、内存分配、流/事件、cuBLAS）
- `tvm-ffi`：Apache TVM 官方 Rust FFI 绑定，加载 TileLang .so + DLPack 零拷贝（回退：`libloading`）
- `safetensors`：权重文件解析
- `serde` / `serde_json`：配置反序列化
- `tokenizers`（huggingface）：分词器
- `tokio` + `axum`：HTTP 服务（仅 ds4_server）
- `clap`：CLI 参数解析
- `bytemuck`：安全字节转换
- `memmap2`：SSD MMAP 预取

#### 1.3 Makefile

```makefile
build:
	cargo build --release

test:
	cargo test

serve:
	cargo run --release --bin ds4_server

cli:
	cargo run --release --bin ds4

kernels:
	docker exec ds4rs-dev python /workspace/tilelang/compile_kernels.py
```

#### 1.4 基础张量类型

```rust
struct CpuTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
}

struct GpuTensor {
    slice: CudaSlice<u8>,  // cudarc GPU 内存
    shape: Vec<usize>,
    dtype: DType,
    device: CudaDevice,
}

enum DType { BF16, FP32, FP8E4M3, FP4E2M1, FP8E8M0, INT32, INT64 }

impl GpuTensor {
    fn to_dlpack_tensor(&self) -> tvm_ffi::Tensor;
    fn from_dlpack_tensor(tensor: &tvm_ffi::Tensor) -> Self;
    fn copy_h2d(&self, device: &CudaDevice) -> GpuTensor;
    fn copy_d2h(&self) -> CpuTensor;
    fn copy_d2d_async(&self, stream: &CuStream) -> GpuTensor;
}
```

---

### 阶段 2：TileLang 内核编译与 FFI 绑定

**目标**：将 K1-K16 + MK1-MK3 内核编译为 .so，在 Rust 中通过 tvm-ffi 调用。

#### 2.1 TileLang 内核编译脚本

`tilelang/compile_kernels.py` 在容器内运行：

```python
# 编译流程：
# 1. 导入 kernel.py 中的 6 个已有内核 (K1-K6)
# 2. 定义 10 个新增内核 (K7-K16)
# 3. 定义 4 个融合内核 (MK0-MK3)
# 4. 按上表 D 中的参数实例化每个内核
# 5. 调用 tilelang.jit 编译为 .so (AOT 模式)
# 6. 输出到 tilelang/build/ 目录
#
# JIT 模式（测试阶段）：
#   - 在 Python 中直接调用 tilelang.jit 装饰的函数
#   - 首次调用时自动编译并缓存
#   - 适合快速迭代验证融合效果
#
# AOT 模式（生产阶段）：
#   - 通过本脚本预编译所有 .so
#   - Rust 通过 tvm-ffi 加载预编译的 .so
#   - 避免运行时编译延迟
```

编译输出命名规范：`{kernel_name}_{param1}_{param2}..._{hash}.so`

#### 2.2 tvm-ffi 封装设计

```rust
use tvm_ffi::{Module, Function, Tensor, into_typed_fn};

struct TlKernel {
    module: Module,
    func: Function,
}

impl TlKernel {
    fn load(so_path: &str, func_name: &str) -> Result<Self> {
        let module = Module::load_from_file(so_path)?;
        let func = module.get_function(func_name)?;
        Ok(Self { module, func })
    }

    fn call(&self, stream: &CuStream, args: &[GpuTensor]) -> Result<()> {
        let tl_args: Vec<Tensor> = args.iter()
            .map(|t| t.to_dlpack_tensor())
            .collect();
        tvm_ffi::device::with_stream(stream, || {
            let typed = into_typed_fn!(&self.func, Fn(&Tensor, ...) -> Result<()>);
            typed(&tl_args[0], ...)?;
            Ok(())
        })
    }
}
```

#### 2.3 内核注册表

```rust
struct KernelRegistry {
    kernels: HashMap<String, TlKernel>,
}

impl KernelRegistry {
    fn load_dir(dir: &Path) -> Result<Self>;
    fn get(&self, name: &str) -> Option<&TlKernel>;
}
```

---

### 阶段 3：权重加载与量化格式支持

**目标**：加载 safetensors 权重，支持 FP8/FP4/BF16 多种格式。

#### 3.1 权重加载器

```rust
struct WeightLoader { model_dir: PathBuf }

impl WeightLoader {
    fn load_global(&self) -> GlobalWeights;
    fn load_layer(&self, layer_id: usize) -> LayerWeights;
    fn load_expert(&self, layer_id: usize, expert_id: usize) -> ExpertWeights;
}
```

- 优先加载 `fp8_weights/` 目录下已转换的分层权重（如存在）
- 回退加载原始 `model-*.safetensors`
- 路由专家权重：当前模型为 FP4 (I8 打包, shape [out, in//2]) + E8M0 scale (shape [out, in//32])；可选 `--expert-dtype fp8` 转换后为 FP8 (e4m3fn, shape [out, in]) + E8M0 scale (shape [out//128, in//128])
- FP4 权重：safetensors 中 dtype=I8，物理 shape [out, in//2]，每元素 1 字节含 2 个 FP4 值（float4_e2m1fn_x2 打包格式）
- FP4 scale：safetensors 中 dtype=F8_E8M0，shape [out, in//32]，per-row group_size=32 量化
- FP8 权重：e4m3fn + e8m0fnu 缩放因子，2D block [128, 128] 量化
- **wo_a 特殊处理**：当前 safetensors 中 wo_a 仍为 FP8 格式 `[8192, 4096]` + scale `[64, 32]`。运行 convert.py 后反量化为 BF16 并 reshape 为 `[o_groups, o_lora_rank, dim]` = `[8, 1024, 4096]`。权重加载器需同时支持两种格式：
  - **FP8 wo_a**（未转换模型）：加载 weight + scale → FP8 GEMM 路径（K3 `fp8_gemm_N8192_K4096`）
  - **BF16 wo_a**（已转换模型）：加载 weight → cuBLAS batched GEMM einsum 路径
  - 格式检测：检查 safetensors 中 `wo_a.weight` 的 dtype（FP8 vs BF16）和 shape（`[8192, 4096]` vs `[8, 1024, 4096]`）

#### 3.2 量化格式工具

```rust
fn dequant_fp4_to_bf16(data: &[u8], scales: &[u8], shape: &[usize]) -> Vec<u16>;
fn dequant_fp8_to_bf16(data: &[u8], scales: &[u8], shape: &[usize]) -> Vec<u16>;
```

- CPU 参考实现用于正确性验证
- GPU 路径直接传递量化权重给 TileLang GEMM 内核

---

### 阶段 4：模型架构实现（核心）

**目标**：复刻 `inference/model.py` 的模型前向传播，使用 TileLang 内核替代所有 PyTorch 计算。

**两阶段实现**：
- **阶段 4A（简化版）**：实现 MK0 简化版 Block（Attention + 共享专家 FFN，无 MoE/CSA），验证 Rust→TileLang 数据通路
- **阶段 4B（完整版）**：添加 MoE 路由、KV 压缩、Indexer 等完整逻辑

#### 4.1 配置数据结构

```rust
struct ModelConfig {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    n_kv_heads: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    o_groups: usize,
    rope_head_dim: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    n_activated_experts: usize,
    moe_inter_dim: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    vocab_size: usize,
    window_size: usize,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    scoring_func: ScoringFunc,
    route_scale: f32,
    swiglu_limit: f32,
    rope_theta: f32,
    compress_rope_theta: f32,
    yarn_factor: f32,
    original_seq_len: usize,
    beta_fast: usize,
    beta_slow: usize,
    compress_ratios: Vec<usize>,
    n_hash_layers: usize,
    n_mtp_layers: usize,
    rms_norm_eps: f32,
    expert_dtype: ExpertDtype,
    scale_fmt: ScaleFmt,
    topk_method: TopkMethod,
    norm_topk_prob: bool,
}
```

从 `config.json` 反序列化。

#### 4.2 模型组件实现 — 对应 TileLang 内核映射

| 组件 | model.py 行号 | 使用的 TileLang 内核 | cuBLAS 操作 |
|---|---|---|---|
| RMSNorm | L183-196 | K7 `rmsnorm` | — |
| Q norm (rsqrt) | L498 | K7 `rmsnorm_no_weight` | — |
| apply_rotary_emb | L232-244 | K8 `rotary_emb` | — |
| rotate_activation | L247-251 | K12 `hadamard` | — |
| linear (FP4) | L113-115 | K1 `act_quant` + K4 `fp4_gemm` | — |
| linear (FP8) | L116-118 | K1 `act_quant` + K3 `fp8_gemm` | — |
| linear (BF16) | L120 | — | cuBLAS SGEMM |
| Compressor | L316-377 | K13 `compressor` + K1 `act_quant` (KV FP8模拟 L372) + K2 `fp4_quant` (KV FP4模拟 L370) + K8 `rotary_emb` + K7 `rmsnorm` | cuBLAS (wkv, wgate BF16→FP32) |
| Indexer | L402-433 | K11 `indexer_score` + K12 `hadamard` + K2 `fp4_quant` + K8 `rotary_emb` | K3 fp8_gemm (wq_b) + cuBLAS (weights_proj BF16) |
| Attention | L484-543 | K5 `sparse_attn` + K7 `rmsnorm` (attn_norm, q_norm, kv_norm) + K8 `rotary_emb` + K1 `act_quant` | K3 fp8_gemm (wq_a, wq_b, wkv, wo_b) + cuBLAS batched GEMM (wo_a BF16 einsum, convert.py 已反量化为 BF16) |
| Gate (评分路由) | L564-584 | K10 `gate_topk` | cuBLAS (weight GEMM, BF16→FP32) |
| Gate (哈希路由, L0-2) | L546-563 | — (indices 由 tid2eid [129280,6] I64 查表) | cuBLAS (weight GEMM, BF16→FP32, 仅计算 weights) |
| Expert | L596-606 | K9 `swiglu` + K4 `fp4_gemm` (路由专家, FP4模型) / K3 `fp8_gemm` (共享专家 + FP8模型路由专家) | — |
| Block.hc_pre | L673-681 | K14 `hc_pre` + K6 `hc_sinkhorn` | cuBLAS (hc_fn GEMM) |
| Block.hc_post | L683-686 | K15 `hc_post` | — |
| ParallelHead.hc_head | L728-735 | K16 `hc_head` | cuBLAS (hc_fn GEMM) |
| ParallelHead.get_logits | L715-716 | — | cuBLAS (head GEMM, BF16→FP32) |
| Embedding | L96-105 | — | GPU embedding 查表 |
| MTPBlock | L738-766 | K7 `rmsnorm` + K8 `rotary_emb` + K5 `sparse_attn` + K1 `act_quant` | K3 fp8_gemm + cuBLAS (同 Attention 结构) |

#### 4.3 KV 缓存管理

```rust
struct KvCache {
    window_kv: Vec<GpuTensor>,     // 每层滑动窗口 KV [seq, 512]
    compressed_kv: Vec<GpuTensor>, // 每层压缩 KV [compressed_seq, 512]
    window_idx: Vec<usize>,
    compressed_idx: Vec<usize>,
}
```

- 滑动窗口大小 128，环形缓冲区
- 压缩 KV 按 compress_ratios 逐层管理
- 支持磁盘 KV 检查点（序列化/反序列化）

#### 4.4 MTP (Multi-Token Prediction) 层实现

MTPBlock (model.py L738-766) 是独立的预测头，结构与普通 Block 类似但有额外投影层：

```rust
struct MtpBlock {
    e_proj: Linear,       // [dim, dim] FP8 — 嵌入投影
    h_proj: Linear,       // [dim, dim] FP8 — 隐藏状态投影
    enorm: RmsNorm,       // 嵌入归一化
    hnorm: RmsNorm,       // 隐藏状态归一化
    block: Block,         // 标准Transformer Block（含Attention + MoE）
    hc_head_fn: GpuTensor,  // [4, 16384] FP32 — MTP专用HC head
    hc_head_scale: f32,     // HC head scale
    hc_head_base: GpuTensor, // [4] FP32 — HC head base
    norm: RmsNorm,          // 输出归一化
}
```

**MTP 前向流程**：
```
e = embed(input_ids) → enorm(e)
x = hnorm(x)  // x 来自上一层的隐藏状态
x = e_proj(e).unsqueeze(2) + h_proj(x)  // 嵌入投影 + 隐藏投影，扩展为 [b,s,hc,d]
x = block.forward(x)  // 标准Block前向（含完整MoE逻辑）
logits = head(x, hc_head_fn, hc_head_scale, hc_head_base, norm)  // HC head混合 + logits
```

**MTP 与普通层的差异**：
- MTP 有独立的 `e_proj` / `h_proj` 投影层（FP8 GEMM）
- MTP 使用独立的 `hc_head_fn` / `hc_head_scale` / `hc_head_base`（不同于主模型的 HC head）
- MTP 的 Block 内部是完整 MoE 逻辑（非简化版）
- MTP 层的 compress_ratio=0（纯 SWA，无 KV 压缩）

#### 4.5 Prefill vs Decode 双路径

DS-V4 的多个组件在 prefill 和 decode 阶段行为完全不同，必须实现两套路径：

| 组件 | Prefill 路径 | Decode 路径 | 差异点 |
|---|---|---|---|
| **Attention** | 多 token 并行，QKV 批量计算 | 单 token，QKV 增量更新 | KV cache 写入模式不同 |
| **KV Cache (窗口)** | 批量写入，环形缓冲截断 | 逐 token 写入，环形覆盖 | `start_pos=0` vs `start_pos>0` |
| **Compressor** | 按 ratio 分组批量压缩 | 增量式：每 ratio 个 token 触发一次 | state 缓冲区管理不同 |
| **Indexer** | 全量压缩 KV 评分 | 增量评分（新压缩 KV 加入后） | topk 索引集合变化 |
| **sparse_attn** | topk 索引包含全量窗口+压缩 | topk 索引动态更新 | 索引构建逻辑不同 |
| **MoE** | 多 token 可按专家分组并行 | 单 token，6 个专家串行或小 batch | dispatch/归约策略不同 |

```rust
enum InferenceMode {
    Prefill { seq_len: usize },
    Decode,
}

impl Attention {
    fn forward(&mut self, x: &GpuTensor, mode: InferenceMode) -> GpuTensor;
}
impl Compressor {
    fn forward(&mut self, x: &GpuTensor, mode: InferenceMode) -> Option<GpuTensor>;
}
```

**Prefill 阶段 Compressor 特殊逻辑**：
- `seqlen >= ratio` 时才压缩，remainder = seqlen % ratio
- 重叠窗口 (ratio=4)：`overlap_transform` 交错相邻 token 组
- 批量写入压缩 KV cache：`kv_cache[:bsz, :seqlen//ratio] = compressed_kv`

**Decode 阶段 Compressor 特殊逻辑**：
- 维护 `kv_state[bsz, coff*ratio, coff*d]` 和 `score_state[bsz, coff*ratio, coff*d]` (FP32)
- 每 ratio 个 token 触发一次压缩
- 重叠模式：同时维护两个窗口（重叠窗口 + 当前窗口）

#### 4.6 数值验证方案

**目标**：确保 Rust 推理结果与官方 Python 推理逐层一致。

**验证策略**：

```
1. 单算子验证（阶段 A）
   - 每个新内核 (K7-K16) 独立测试
   - Python 端导出输入/输出 tensor (numpy 格式)
   - Rust 端加载相同输入，调用 TileLang 内核，对比输出
   - 容忍度: FP8/FP4 路径 max_abs_err < 1e-2, BF16/FP32 路径 < 1e-3

2. 逐层验证（阶段 A 完成时）
   - Python 端在 model.py 每层 Block 输出处插入 hook
   - 导出每层输入/输出 tensor (BF16, 保存为 .npy)
   - Rust 端逐层前向，每层输出与 Python 对比
   - 容忍度: 累积误差 max_abs_err < 5e-2 (43层后)

3. 端到端验证（里程碑 A/B）
   - 相同输入 prompt，对比最终 logits
   - top-1 token 一致率 > 99%
   - top-5 token 集合 Jaccard 相似度 > 0.95

4. 数值漂移检测
   - 每 10 层检查一次隐藏状态范数
   - 若范数偏差 > 10%，标记为潜在精度问题
   - 特别关注: KV cache 压缩/解压、FP4 GEMM、inplace 量化
```

**Python 端导出脚本**（容器内运行）：

```python
import numpy as np
import torch

def export_layer_io(model, input_ids, layer_id, output_dir):
    hooks = []
    def hook_fn(module, input, output):
        np.save(f"{output_dir}/layer{layer_id}_input.npy", input[0].detach().cpu().float().numpy())
        np.save(f"{output_dir}/layer{layer_id}_output.npy", output.detach().cpu().float().numpy())
    hooks.append(model.layers[layer_id].register_forward_hook(hook_fn))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
```

**Rust 端验证辅助**：

```rust
fn validate_layer_output(rust_output: &GpuTensor, reference_path: &Path, tolerance: f32) -> Result<bool> {
    let cpu_output = rust_output.copy_d2h();
    let reference: Vec<f32> = load_npy(reference_path)?;
    let max_abs_err = cpu_output.data_f32()
        .iter().zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Ok(max_abs_err < tolerance)
}
```

---

### 阶段 5：路由专家三级缓存系统

**目标**：实现 GPU → CPU → SSD 三级缓存，优化 256 个路由专家的调度。

#### 5.0 专家权重大小与缓存容量计算

**关键发现**：路由专家权重存储为 **FP4 (e2m1fn_x2 packed as I8)** + **E8M0 scale**。safetensors 中 dtype 为 I8（物理存储 shape [out, in//2]，每元素 1 字节含 2 个 FP4 值）。可用 convert.py `--expert-dtype fp8` 转为 FP8。

**实际权重大小（从 safetensors 文件头验证）**：

| 张量 | FP4 存储形状 | FP4 大小 | FP8 存储形状 | FP8 大小 |
|---|---|---|---|---|
| w1.weight | [2048, 2048] I8 | 4.00 MB | [2048, 4096] F8 | 8.00 MB |
| w1.scale | [2048, 128] F8_E8M0 | 0.25 MB | [16, 32] F8_E8M0 | 512 B |
| w3.weight | [2048, 2048] I8 | 4.00 MB | [2048, 4096] F8 | 8.00 MB |
| w3.scale | [2048, 128] F8_E8M0 | 0.25 MB | [16, 32] F8_E8M0 | 512 B |
| w2.weight | [4096, 1024] I8 | 4.00 MB | [4096, 2048] F8 | 8.00 MB |
| w2.scale | [4096, 64] F8_E8M0 | 0.25 MB | [32, 16] F8_E8M0 | 512 B |
| **单专家合计** | | **12.75 MB** | | **~24.0 MB** |

| 项目 | FP4 模型 | FP8 模型 | 说明 |
|---|---|---|---|
| 单专家权重大小 | **~12.75 MB** | **~24.0 MB** | FP4: 3×4MB权重 + 3×0.25MB scale; FP8: 3×8MB权重 + 微量scale |
| 单层 256 专家 | **~3.19 GB** | **~6.0 GB** | |
| 全部 11008 专家 (43层) | **~137 GB** | **~258 GB** | 实测 safetensors 专家权重总计 ~140.3 GB (含MTP层) |
| 每 token 每层需加载 | **~76.5 MB** | **~144 MB** | 6 专家 × 单专家大小 |

**非专家权重 GPU 常驻占用**（从 safetensors 实测）：

| 类别 | 每层大小 | 说明 |
|---|---|---|
| Attention (wq_a, wq_b, wkv, wo_a, wo_b + scales) | ~102 MB | wq_b 32MB + wo_a 32MB + wo_b 32MB 为主 |
| Shared Expert (w1, w2, w3 + scales) | ~24 MB | |
| HC (hc_attn_fn, hc_ffn_fn, bases, scales) | ~3.5 MB | hc_fn [24, 16384] FP32 |
| Gate (weight + bias/tid2eid) | ~8 MB | 哈希层: tid2eid [129280,6] I64 = 5.92MB; 评分层: bias [256] FP32 |
| Compressor + Indexer (CSA层) | ~20 MB | 仅 ratio=4 的层 |
| Norms | ~0.1 MB | |
| **每层平均** | **~137 MB (非CSA) / ~159 MB (CSA)** | |
| **43层合计** | **~6.4 GB** | |
| Embed + Head + HC Head | **~2.0 GB** | embed [129280,4096] BF16 + head [129280,4096] BF16 |
| **非专家总计** | **~8.4 GB** | 必须常驻 GPU |

**GPU 显存预算（RTX 5060 Ti 16 GB）**：

| 用途 | 占用 | 说明 |
|---|---|---|
| 非专家权重 (常驻) | ~8.4 GB | 43层 + embed + head |
| KV Cache | ~1.5 GB | 滑动窗口128 + 压缩KV + Indexer KV；ratio=4 层 KV cache = 1152×512×2×BF16 ≈ 2.25MB/层/batch，43层×batch=4 ≈ 386MB；加上 Indexer KV 另计 |
| 激活/临时缓冲 | ~0.5 GB | |
| CUDA 上下文 + 框架开销 | ~0.5 GB | |
| tvm-ffi 运行时 | ~0.2 GB | libtvm_ffi.so + 内核模块 |
| **可用于专家缓存** | **~4.9 GB** | 16 - 8.4 - 1.5 - 0.5 - 0.5 - 0.2 |

**各级缓存容量（FP4 模型）**：

| 缓存级 | 可用容量 | 可容纳专家数 | 等效层数 |
|---|---|---|---|
| GPU VRAM (RTX 5060Ti 16G) | ~4.9 GB (扣除非专家+KV+激活+运行时) | **~384 个** | ~1.5 层 |
| CPU DDR5 (96G) | ~80 GB | **~6,275 个** | ~24.5 层 |
| SSD | 无限 | **11,008 个** | 43 层 (全量) |

**各级缓存容量（FP8 模型）**：

| 缓存级 | 可用容量 | 可容纳专家数 | 等效层数 |
|---|---|---|---|
| GPU VRAM (RTX 5060Ti 16G) | ~4.9 GB | **~204 个** | ~0.8 层 |
| CPU DDR5 (96G) | ~80 GB | **~3,333 个** | ~13 层 |
| SSD | 无限 | **11,008 个** | 43 层 (全量) |

**传输带宽与延迟**：

| 路径 | 带宽 | FP4 单专家 | FP4 6专家 | FP8 单专家 | FP8 6专家 |
|---|---|---|---|---|---|
| SSD → CPU (PCIe 5.0 x8 NVMe) | **~15 GB/s** | 0.85 ms | 5.10 ms | 1.60 ms | 9.60 ms |
| CPU → GPU (PCIe 5.0 x8) | ~16 GB/s | 0.80 ms | 4.78 ms | 1.50 ms | 9.00 ms |
| SSD → CPU → GPU 合计 | — | 1.65 ms | 9.88 ms | 3.10 ms | 18.60 ms |

**关键洞察**：
- SSD 为 PCIe 5.0 x8 独立缓存（~15 GB/s），读取速度接近 PCIe 带宽，SSD→CPU 延迟与 CPU→GPU 延迟几乎相等
- FP4 模型：每层每 token 需 PCIe 传输 6 专家 = 76.5 MB，PCIe 5.0 x8 传输需 ~4.78 ms
- FP4 模型 SSD 全路径：6 专家 SSD→CPU→GPU 需 ~9.88 ms，需提前 2-3 层预取才能完全隐藏
- FP8 模型：6 专家 SSD→CPU→GPU 需 ~18.6 ms，需提前 4-5 层预取，实际不可行
- **强烈推荐使用 FP4 模型**：单专家 12.75 MB，GPU 可缓存 ~384 专家（~1.5 层），CPU 可缓存 ~24.5 层，SSD 全路径延迟可接受
- **GPU 缓存有限但关键**：~384 专家虽仅覆盖 ~1.5 层，但 MoE 路由中高频专家跨层复用率高（如常见 token 模式），LFU 策略可有效提升命中率

#### 5.1 GPU 缓存层（热点专家）— LFU + Top Segment

**设计原则**：权重流式经过 L2 cache，shared memory 仅用于高频访问的 activations 与临时 buffers。

```rust
struct GpuExpertCache {
    slots: Vec<Option<(usize, GpuTensor)>>, // (expert_id, weight_gpu)
    lfu: LfuCounter,
    capacity: usize,                        // ~384 个专家 (FP4) / ~204 个专家 (FP8)
    free_slots: VecDeque<usize>,            // 空闲槽位索引
}
```

- **LFU 策略**：统计每个专家的使用频率，保留热点路由专家常驻 GPU
- **Top Segment 缓存**：按段（连续 token 序列）统计专家使用频率，段级粒度优于全局粒度
- **DMA 异步读取**：当缓存未命中时，通过 pinned memory + async DMA 从 CPU 缓存读取 Bottom Segment
- **淘汰策略**：LFU 最低的专家被淘汰，其 GPU 槽位释放给新专家
- **权重不进 shared memory**：专家权重通过 L2 cache 流式加载到寄存器（GEMM 的 T.copy 路径），shared memory 仅用于 activations 和 GEMM 的 A_shared/B_shared tiles

**专家权重 L2 流式加载原理**：
- TileLang GEMM 内核中，权重矩阵 B 通过 T.copy 从全局内存加载到 B_shared（shared memory）
- 但对于 MoE 场景，权重不在 GPU 常驻，而是从 CPU 动态加载
- 加载路径：SSD → CPU pinned memory → PCIe DMA → GPU 全局内存 → L2 cache → shared memory → 寄存器
- 关键优化：权重只需到达 GPU 全局内存即可被 GEMM 内核使用，L2 cache 自动缓存热点权重 tile

#### 5.2 跨层预取 — 隐藏 PCIe 传输延迟

```rust
struct LayerPrefetcher {
    compute_stream: CuStream,      // 计算流
    transfer_stream: CuStream,     // 传输流（独立于计算流）
    pinned_pool: PinnedMemoryPool, // 预分配 pinned memory 缓冲区池
    gpu_free_slots: VecDeque<GpuTensor>, // GPU 空闲槽位
    next_layer_experts: Vec<usize>,      // 预测的下一层专家集合
}

struct PinnedMemoryPool {
    buffers: Vec<PinnedHostSlice<u8>>, // 预分配的 pinned buffer
    available: VecDeque<usize>,        // 可用 buffer 索引
    buffer_size: usize,                // 每个 buffer 大小 = 12.75 MB (FP4) / 24 MB (FP8)
}
```

- **预测机制**：L 层计算 Gate TopK 后，得到当前 token 选择的 6 个专家。基于历史统计，预测 L+1 层可能选择的专家集合（概率加权 Top-K 扩展）
- **双流并行**：
  - `compute_stream`：执行当前层的 GEMM + Attention 计算
  - `transfer_stream`：异步 DMA 传输 L+1 层预测专家权重到 GPU 空闲槽位
- **Pinned Memory 池**：预分配 ~12 个 pinned buffer（FP4: 12 × 12.75 MB = 153 MB; FP8: 12 × 24 MB = 288 MB），避免运行时分配开销。每个 buffer 容纳 1 个专家的完整权重（weight + scale），6 个 buffer 为 1 组用于单层 6 专家预取
- **Write-Combined 优化**：对只写（CPU→GPU 方向）的权重缓冲区，使用 `cudaHostAllocWriteCombined` 提升约 40% 传输性能（需通过 cudarc `sys` 层封装）
- **同步点**：L+1 层计算开始前，`compute_stream.wait(transfer_stream.event)` 确保预取完成

**cudarc 支持情况**：

| 功能 | cudarc API | 说明 |
|---|---|---|
| Pinned memory 分配 | `CudaContext::alloc_pinned()` | ✅ 开箱即用 |
| 异步 H2D 拷贝 | `memcpy_htod` (内部 async) | ✅ 自动 DMA |
| 多流管理 | `new_stream()` + event 追踪 | ✅ 自动跨流同步 |
| 计算/传输重叠 | 双流 + pinned memory | ✅ Blackwell 架构支持 |
| Write-Combined | 需通过 `sys` 层封装 | ⚠️ 自行实现 |
| Pinned Memory 池 | 无内置 | ⚠️ 自行实现 |

#### 5.3 CPU 缓存层（SLRU）— 内存划分 + SSD MMAP 预取

```rust
struct CpuExpertCache {
    slru: SlruCache<usize, CpuTensor>, // SLRU: (layer*256+expert_id) → 权重数据
    total_memory: usize,               // ~80 GB 可用
    protected_ratio: f32,              // 动态调整
    ssd_prefetcher: SsdPrefetcher,     // SSD MMAP 预取器
}

struct SlruCache<K, V> {
    protected: LinkedHashMap<K, V>,    // 保护段：热点专家，不被轻易淘汰
    probation: LinkedHashMap<K, V>,    // 试用段：新加载专家，需二次访问才晋升
    protected_capacity: usize,
    probation_capacity: usize,
}
```

- **SLRU 策略**：
  - Protected 段：存放高频访问专家，仅当 protected 满时才淘汰到 probation
  - Probation 段：存放新加载/低频专家，LRU 淘汰
  - 动态划分：根据推理效率调整 protected/probation 比例
- **SSD MMAP 预取流水线**：
  - 使用 `memmap2` 将 SSD 上的专家权重文件映射到虚拟地址空间
  - 预取策略：L1-N 层路由专家按使用概率排序，高概率专家提前 MMAP 到内存
  - `cudaHostRegister`：将 MMAP 的内存注册为锁页内存，使 DMA 引擎可直接访问（需通过 cudarc `sys` 层封装）
  - **MMAP 预取三阶段**：
    1. **冷启动**：按层序 MMAP 前 N 层全部专家（N = CPU 可容纳层数），`madvise(MADV_SEQUENTIAL)` 预读
    2. **稳态推理**：跨层预取器根据 Gate 预测结果，提前 MMAP 下 2-3 层可能需要的专家
    3. **热切换**：SLRU 淘汰的专家 `madvise(MADV_DONTNEED)` 释放物理页，保留虚拟映射
- **推理效率优先动态划分**：
  - 监控缓存命中率和推理延迟
  - 命中率高 → 扩大 protected 段，减少淘汰
  - 命中率低 → 扩大 probation 段，增加新专家加载

#### 5.4 SSD 缓存层 — PCIe 5.0 x8 独立缓存，全量路由专家

```rust
struct SsdExpertCache {
    base_dir: PathBuf,
    mmap_handles: HashMap<(usize, usize), Mmap>, // (layer, expert) → mmap 句柄
    expert_files: Vec<PathBuf>,                   // 预计算的专家文件路径
    registered_regions: Vec<(usize, usize, usize)>, // (layer, expert, size) 已 cudaHostRegister 的区域
}
```

- **PCIe 5.0 x8 独立 NVMe SSD**，顺序读 ~15 GB/s，专用于路由专家权重存储
- 全量 256×43 = 11,008 路由专家存储在 SSD
- FP4 模型：全量专家 ~137 GB；FP8 模型：全量专家 ~258 GB
- 每个专家一个文件：`experts/L{layer:02d}_E{expert:03d}.bin`（含 weight + scale 连续存储）
- 使用 `memmap2` MMAP 预取，按需加载到 CPU 内存
- **SSD→CPU 延迟**：FP4 单专家 0.85 ms，6 专家 5.10 ms；与 CPU→GPU 延迟（0.80 ms / 4.78 ms）相当
- **SSD 独立缓存优势**：不与 OS/应用共享 I/O 带宽，延迟可预测；15 GB/s 接近 PCIe 5.0 x4 SSD 上限（PCIe 5.0 x8 为 SSD 提供双倍通道冗余）
- `cudaHostRegister`：将 MMAP 的内存注册为锁页内存，DMA 引擎可直接访问
- **cuFile (GPU Direct Storage)**：cudarc 支持 `memcpy_ftod` / `memcpy_dtof`，可直接从 SSD 到 GPU，绕过 CPU 内存。SSD PCIe 5.0 x8 独立缓存是 cuFile 的理想场景，可降低延迟至 ~5.10 ms (FP4 6专家，SSD→GPU 直读)
- **SSD 文件布局优化**：
  - 每个专家文件对齐到 4KB 边界（NVMe 页大小），避免跨页读取
  - 同层 256 专家可选择性合并为单文件（~3.19 GB），减少文件打开开销
  - 使用 `O_DIRECT` + `io_uring` 异步读取，绕过页缓存，降低延迟

#### 5.5 三级缓存协同工作流

**GPU 专家槽位生命周期管理**：

```
GPU 专家槽位状态机:
  FREE ──(预取写入)──→ PREFETCHING ──(DMA完成)──→ READY ──(计算使用)──→ ACTIVE
    ↑                                                              │
    └──────────────────(LFU淘汰 / 计算完成释放)────────────────────┘

槽位分配策略:
  - 总槽位数: ~384 (FP4) / ~204 (FP8)
  - 常驻槽位: LFU Top-K 热点专家，不参与淘汰（约 60% 槽位 = ~230 专家）
  - 动态槽位: 预取目标 + 当前层计算使用（约 40% 槽位 = ~154 专家）
  - 预取预留: 每层预留 6 个槽位给预取专家（154 / 6 ≈ 25 层预取窗口）
  - 淘汰触发: 动态槽位不足时，淘汰 ACTIVE 中 LFU 计数最低的专家
```

**推理工作流**：

```
推理 L 层:
  1. Gate TopK → 确定 6 个专家 ID
  2. 检查 GPU 缓存命中
     - 命中: 直接使用 GPU 权重，标记 ACTIVE
     - 未命中: 触发 CPU → GPU DMA 传输
  3. 检查 CPU 缓存命中
     - 命中: pinned memory → async DMA → GPU 动态槽位
     - 未命中: SSD MMAP → CPU 内存 → 注册 pinned → async DMA → GPU
  4. 同时: 在 transfer_stream 上预取 L+1, L+2 层预测专家
     - L+1 层: CPU 缓存命中 → 直接 DMA (4.78 ms)
     - L+2 层: SSD → CPU → GPU (9.88 ms，与 L+1 计算并行)
  5. 执行当前层计算 (compute_stream)
  6. 等待 L+1 层预取完成 (compute_stream.wait)
  7. 更新 LFU/SLRU 统计
  8. 释放当前层非热点专家槽位 → FREE
```

**延迟预算**（decode 阶段，单 token，FP4 模型，SSD PCIe 5.0 x8）：

| 场景 | 延迟 | 是否可隐藏 |
|---|---|---|
| GPU 缓存命中 | 0 (已在 GPU) | — |
| CPU → GPU (PCIe 5.0 x8) | ~4.78 ms (6专家) | ✅ 跨层预取可隐藏（需提前1层） |
| SSD → CPU → GPU | ~9.88 ms (6专家) | ✅ 提前 2-3 层预取可隐藏 |
| SSD → GPU (cuFile 直读) | ~5.10 ms (6专家) | ✅ 跨层预取可隐藏（需提前1层），远期优化 |

**跨层预取时序分析**（FP4 模型，decode 阶段）：

```
假设每层计算耗时 ~3 ms (Attention + MoE GEMM):

时间线 (ms):  0     3     6     9    12    15
Layer L:      [====计算====]
Layer L+1:          [====计算====]
Layer L+2:                [====计算====]

预取 L+1 层 (CPU→GPU):
  transfer:    [==4.78ms==]
  ✓ 在 L+1 计算开始前完成 (4.78ms < 6ms 余量)

预取 L+2 层 (SSD→CPU→GPU):
  SSD→CPU:     [===5.10ms===]
  CPU→GPU:           [===4.78ms===]
  ✓ 在 L+2 计算开始前完成 (9.88ms < 12ms 余量)

结论: FP4 模型下，2 层提前预取即可完全隐藏 SSD 全路径延迟
```

---

### 阶段 6：推理调度与生成

**目标**：实现 prefill + decode 推理循环。

```rust
struct InferenceScheduler {
    model: Transformer,
    kv_cache: KvCache,
    expert_cache: ExpertCacheManager,
    stream: CuStream,
}

impl InferenceScheduler {
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32>;
    fn decode(&mut self, token: u32) -> Vec<f32>;
    fn generate(&mut self, prompt: &[u32], max_tokens: usize) -> Vec<u32>;
}
```

采样策略：Gumbel-max 采样（避免 GPU-CPU 同步）、top-p、top-k

KV 复用与磁盘检查点：推理过程中实时 KV 复用；磁盘 KV 检查点支持热启动

---

### 阶段 7：分词器

```rust
struct Tokenizer { inner: tokenizers::Tokenizer }

impl Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}
```

---

### 阶段 8：CLI 交互界面

`ds4_cli.rs`：clap 参数解析 + linenoise REPL + 流式输出 + 思维模式切换

---

### 阶段 9：HTTP 服务

`ds4_server.rs`：OpenAI/Anthropic 兼容 API

- `POST /v1/chat/completions`：聊天补全（流式/非流式）
- `POST /v1/completions`：文本补全
- 工具调用支持（DSML 格式，参考 `encoding/encoding_dsv4.py`）
- SSE 流式传输 + 提前终止

---

### 阶段 10：插件机制

```rust
trait Plugin {
    fn name(&self) -> &str;
    fn on_request(&self, request: &mut Request) -> Result<()>;
    fn on_response(&self, response: &mut Response) -> Result<()>;
    fn on_token(&self, token: u32, text: &str) -> Result<()>;
}
```

---

### 阶段 11：测试与验证

贯穿全程，正确性优先：

- 单元测试：每个组件独立测试，CPU 参考实现 vs GPU TileLang 内核结果对比
- 集成测试：完整前向传播，与官方 Python 推理结果逐层对比
- TileLang 冒烟测试：每个 .so 内核加载与基本功能验证

---

## 实施优先级

| 优先级 | 阶段 | 说明 |
|---|---|---|
| P0 | 阶段 0 | 最小 PoC：验证 tvm-ffi + DLPack 零拷贝通路 |
| P0 | 阶段 1 | 项目脚手架 |
| P0 | 阶段 2 | TileLang 内核编译与 FFI（阶段 A 内核子集） |
| P0 | 阶段 3 | 权重加载（非专家 + 共享专家） |
| P0 | 阶段 4 | 模型架构（简化版：无 MoE/CSA） |
| P0 | 阶段 11 | 测试（贯穿全程） |
| P1 | 阶段 5 | 专家三级缓存（阶段 B 启动） |
| P1 | 阶段 6 | 推理调度与生成 |
| P1 | 阶段 7 | 分词器 |
| P2 | 阶段 8 | CLI |
| P2 | 阶段 9 | HTTP 服务 |
| P3 | 阶段 10 | 插件机制 |

**两阶段里程碑**：
- **里程碑 A**：简化版端到端推理可用（MK0 + 共享专家 FFN + SWA Attention）
- **里程碑 B**：完整版 MoE 推理可用（MK1-MK4 + 三级缓存 + 跨层预取）

## TileLang 内核开发优先级

### 阶段 A：简化版验证通路（MK0 优先）

| 批次 | 内核 | 依赖 | 说明 |
|---|---|---|---|
| 批次 A1 | K1, K3, K5, K6 | 无 | 简化版核心：FP8 量化 + FP8 GEMM + 稀疏注意力 + HC Sinkhorn |
| 批次 A2 | K7, K8, K9 | 无 | 高频基础算子：RMSNorm + RoPE + SwiGLU |
| 批次 A3 | K14, K15, K16 | K6, K7 | HC 混合：hc_pre + hc_post + hc_head |
| 批次 A4 | **MK0 `simple_block`** | K1,K3,K5-K9,K14-K16 | **简化版 Mega-Kernel：融合 Attention+FFN(共享专家)，无 MoE** |

### 阶段 B：完整版 MoE Mega-Kernel

| 批次 | 内核 | 依赖 | 说明 |
|---|---|---|---|
| 批次 B1 | K2, K4 | K1 | FP4 量化 + FP4 GEMM（路由专家） |
| 批次 B2 | K10, K11 | K7 | MoE 路由核心：Gate TopK + Indexer 评分 |
| 批次 B3 | K12, K13 | K8, K2 | 压缩相关：Hadamard + Compressor |
| 批次 B4 | MK1 `hybrid_attn` | K5, K11, K8 | Mega-Kernel: SWA+CSA+HCA 融合注意力 |
| 批次 B5 | MK2 `moe_gemm` | K10, K4, K9, K1 | Mega-Kernel: MoE路由+FP4 GEMM+SwiGLU 融合 |
| 批次 B6 | MK3 `kv_compress` | K13, K7, K8, K1, K2 | Mega-Kernel: KV压缩+Norm+RoPE+Quant 融合 |
| 批次 B7 | MK4 `block_mega` | MK1, MK2, MK3 | Mega-Kernel: 全层融合（远期，需自研调度器或等待 ETC） |

**开发原则**：
- 阶段 A 优先完成，验证 Rust→TileLang 数据通路后再进入阶段 B
- 每个批次内先验证独立内核正确性，再开发融合内核
- 测试阶段使用 JIT 编译验证融合效果，生产阶段使用 AOT 预编译 .so

## 关键技术决策

1. **CUDA 绑定**：使用 `cudarc` crate，避免手写 FFI
2. **TileLang 集成**：Python 预编译 .so → Rust `tvm-ffi` crate 加载调用（回退：`libloading`）
3. **DLPack 零拷贝**：`GpuTensor`(cudarc) ↔ `tvm-ffi::Tensor`(DLPack) 共享同一 `CUdeviceptr`，无需 Candle
4. **GEMM 策略**：FP8/FP4 GEMM 用 TileLang 内核；BF16/FP32 GEMM 用 cuBLAS（通过 cudarc）
5. **内核融合策略**：算子级融合 > 半层融合 > 全层融合，逐步递进
6. **Mega-Kernel 路线**：MK0 (简化版验证通路) → MK1-MK3 (算子级融合) → MK4 (全层融合，远期)
7. **两阶段策略**：先跑通简化版+层间专家预取框架，验证 Rust→TileLang 通路；再扩展为完整版 MoE Mega-Kernel
8. **编译策略**：测试阶段 JIT 编译验证融合效果，生产阶段 AOT 预编译 .so
9. **内存管理**：GPU 显存由 cudarc 管理，CPU 端使用标准 Rust 分配器
10. **异步 I/O**：专家预取使用 CUDA stream + pinned memory，不阻塞计算流
11. **量化格式**：FP4/FP8 权重直接传 GPU，CPU 端仅实现反量化用于验证
12. **KV 缓存**：磁盘检查点使用 serde + bincode 序列化
13. **HTTP 框架**：axum（轻量、异步、类型安全）
14. **无 C/C++ 代码**：严格遵守 AGENTS.md 规则
15. **不使用 Candle**：Candle 不支持 DLPack，改用 cudarc + tvm-ffi 自建轻量张量层
16. **专家权重 FP4 优先**：单专家 12.75 MB (FP4) vs 24 MB (FP8)，FP4 模型 GPU 可缓存 ~384 专家，SSD 全路径延迟 ~9.88 ms 可接受；FP8 模型 GPU 仅缓存 ~204 专家，SSD 全路径 ~18.6 ms 难以隐藏
17. **wo_a 双格式支持**：当前 safetensors 中 wo_a 为 FP8 `[8192, 4096]`，convert.py 转换后为 BF16 `[8, 1024, 4096]`；权重加载器按 dtype/shape 自动检测格式，FP8 走 TileLang GEMM，BF16 走 cuBLAS batched GEMM einsum
18. **GPU 槽位分区**：~60% 常驻 (LFU Top-K 热点) + ~40% 动态 (预取+计算)，避免常驻专家被淘汰
19. **跨层预取窗口**：FP4 模型需提前 2 层预取以隐藏 SSD 全路径延迟（9.88 ms < 2×3ms 层计算时间）
20. **Prefill/Decode 双路径**：Attention、Compressor、KV Cache、MoE 在 prefill 和 decode 阶段行为完全不同，必须实现两套路径（InferenceMode 枚举区分）
21. **MTP 独立实现**：MTPBlock 有独立的 e_proj/h_proj/hc_head_fn，不与普通 Block 共享权重，需单独实现
22. **数值验证贯穿全程**：单算子验证 (max_abs_err < 1e-2) → 逐层验证 (max_abs_err < 5e-2) → 端到端验证 (top-1 一致率 > 99%)
23. **DLPack 扩展类型映射**：FP4 (e2m1fn_x2 packed as I8) 用 kDLInt(8bit) + shape [out, in//2]；E8M0 scale 用 kDLUInt(8bit)；BF16 用标准 kDLFloat(16bit)
24. **KV 非 RoPE 维度 FP8 模拟**：官方 model.py L506 `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` 对 KV 的非 RoPE 维度做 FP8 量化+反量化（QAT 模拟），影响数值精度，必须实现
25. **Compressor RoPE 使用实际 layer_id**：Compressor 的 RoPE 预计算必须传入实际 layer_id，否则 `rope_theta_for_layer(0)` 返回 `rope_theta` 而非 `compress_rope_theta`
26. **attn_sink 为 FP32**：官方 `attn_sink = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))`，必须以 FP32 加载和使用
27. **滑动窗口索引必须匹配环形缓冲区布局**：`get_window_topk_idxs` 在 decode 阶段返回环形缓冲区的物理位置索引，不是逻辑位置

---

## 代码审查问题清单（2026-05-14）

对照 `inference/model.py` (827行) + `inference/kernel.py` (536行) 逐行审查当前 Rust 实现发现的所有问题。

### P0 — 正确性（必须修复，否则推理结果错误）

| # | 模块 | 问题 | 官方代码参考 | 修复方案 |
|---|---|---|---|---|
| P0-1 | layer.rs `compute_topk_idxs` | 滑动窗口索引不正确：当前生成简单序列 `0,1,2,...`，未处理环形缓冲区布局、-1 填充、prefill/decode 差异 | model.py L254-265 `get_window_topk_idxs` | 完整复刻官方逻辑：decode 时 `start_pos % win` 环形索引；prefill 时 causal mask + -1 填充 |
| P0-2 | layer.rs `attention` | 缺少 KV 非 RoPE 维度 FP8 模拟：官方在 kv_norm+RoPE 后对 `kv[..., :-rd]` 做 `act_quant(block_size=64, inplace=True)` | model.py L506 | 添加 `act_quant_inplace` 对 KV 前 `head_dim - rope_dim` 维度做 FP8 量化+反量化 |
| P0-3 | compressor.rs | RoPE 使用 `layer_id=0` 而非实际 layer_id：`rope_theta_for_layer(0)` 返回 `rope_theta` 而非 `compress_rope_theta` | model.py L473-474, L491 | Compressor 初始化时传入实际 layer_id，使 `rope_theta_for_layer` 正确返回 `compress_rope_theta` |
| P0-4 | layer.rs `sparse_attention` | attn_sink 按 BF16 处理，但官方存储为 FP32 | model.py L477 `nn.Parameter(torch.empty(n_heads, dtype=torch.float32))` | 加载 attn_sink 时保持 FP32，sparse_attention 中以 FP32 读取 |
| P0-5 | layer.rs `get_compress_topk_uniform` | 压缩索引 causal mask 逻辑不完整：官方 prefill 时 `mask = matrix >= torch.arange(1, seqlen+1).unsqueeze(1) // ratio` | model.py L268-276 | 复刻官方 `get_compress_topk_idxs` 的 causal mask 逻辑 |
| P0-6 | model.rs | n_layers 应为 43（config.json `num_hidden_layers`），需确认 config.rs 正确读取 | config.json | 验证 `num_hidden_layers` 字段映射正确 |

### P1 — 性能（必须修复，否则推理速度不可接受）

| # | 模块 | 问题 | 影响 | 修复方案 |
|---|---|---|---|---|
| P1-1 | gate.rs `hash_routing` | `tid2eid.to_host()` 在 `for t in 0..total` 循环内调用，每个 token 触发一次 GPU→CPU 传输 | O(total) 次 GPU→CPU 传输 | 将 `tid2eid.to_host()` 移到循环外 |
| P1-2 | kv_cache.rs | `update_prefill`/`update_decode` 下载整个 cache 到 CPU 修改后重新上传 | decode 单 token 时复制整个 cache | 使用 `copy_d2d_async` 直接在 GPU 上写入单个位置 |
| P1-3 | layer.rs `fp8_gemm_expert` | 每次调用创建新 `WeightLoader`，重新打开 safetensors 文件 | 每个专家重新读索引+打开文件 | 实现 ExpertScheduler 缓存已加载的专家权重 |
| P1-4 | layer.rs 全局 | 所有计算（rmsnorm, rope, attention, hc_pre/post, swiglu 等）都在 CPU 上执行：下载→计算→上传 | 每步操作 2 次 GPU↔CPU 传输 | 逐步替换为 TileLang GPU 内核（K7-K16） |
| P1-5 | layer.rs `fp8_gemm_act_quant` | FP8 权重先反量化为 BF16 再用 cuBLAS BF16 GEMM，未利用 FP8 精度优势 | 2x 权重内存 + 无 FP8 加速 | 使用 TileLang K3 `fp8_gemm` 内核，直接传 FP8 权重+scale |

### P2 — 架构（应修复，影响生产可用性）

| # | 模块 | 问题 | 修复方案 |
|---|---|---|---|
| P2-1 | 全局 | 缺少 TileLang sparse_attn (K5) 内核集成：当前 CPU 实现太慢 | 集成 K5 `sparse_attn_kernel` |
| P2-2 | 全局 | 缺少 TileLang FP8/FP4 GEMM (K3/K4) 内核集成 | 集成 K3/K4，替代 cuBLAS BF16 路径 |
| P2-3 | 全局 | 缺少 TileLang act_quant (K1) 内核集成 | 集成 K1，替代 CPU FP8 模拟 |
| P2-4 | 全局 | 缺少 TileLang hc_split_sinkhorn (K6) 内核集成 | 集成 K6，替代 CPU Sinkhorn |
| P2-5 | 全局 | 缺少 TileLang RMSNorm (K7) 内核集成 | 集成 K7，替代 CPU RMSNorm |
| P2-6 | indexer.rs | 缺少 Hadamard 旋转：官方 Indexer 使用 `rotate_activation(q)` + FP4 模拟 | 实现 K12 `hadamard` 内核 + FP4 量化模拟 |
| P2-7 | 全局 | 缺少专家权重三级缓存系统 | 实现阶段 5 的 GPU/CPU/SSD 三级缓存 |

### P3 — 次要

| # | 模块 | 问题 | 修复方案 |
|---|---|---|---|
| P3-1 | layer.rs | `q_l2_normalize` 命名误导：实际是无权重 RMSNorm | 重命名为 `rmsnorm_no_weight` |
| P3-2 | layer.rs | 多处 `cast_to_f32` + `rmsnorm` 实现重复 | 统一为通用 `rmsnorm(x, weight, eps)` 函数 |
| P3-3 | layer.rs | `output_proj` 每次调用都反量化 wo_a 权重 | 预缓存反量化后的 wo_a 权重 |

---

## 修正路线图

基于审查结果，当前代码处于 **"CPU 参考实现完成，GPU 内核集成待启动"** 阶段。修正优先级：

### 第一步：修复 P0 正确性问题（1-2 天）

1. **P0-1**: 重写 `compute_topk_idxs` → 完整复刻 `get_window_topk_idxs`
2. **P0-2**: 添加 KV 非 RoPE 维度 FP8 模拟（`act_quant_inplace`）
3. **P0-3**: 修复 Compressor RoPE layer_id
4. **P0-4**: 修复 attn_sink FP32 处理
5. **P0-5**: 修复压缩索引 causal mask
6. **P0-6**: 验证 n_layers=43 正确读取

### 第二步：修复 P1 性能问题（2-3 天）

1. **P1-1**: Gate tid2eid 移到循环外
2. **P1-2**: KV Cache 使用 d2d memcpy
3. **P1-3**: ExpertScheduler 缓存专家权重
4. **P1-4/P1-5**: 开始 TileLang 内核集成（K7 RMSNorm → K8 RoPE → K5 sparse_attn → K3/K4 GEMM）

### 第三步：TileLang 内核集成（按计划阶段 A 批次执行）

按原计划批次 A1-A4 执行，逐步将 CPU 计算替换为 GPU 内核。

### 第四步：数值验证

每完成一个内核集成，立即与 Python 参考结果对比验证。

---

## 当前实现状态总结

| 组件 | 状态 | 说明 |
|---|---|---|
| config.rs | ✅ 完成 | 正确读取 config.json，含 compress_ratios/YaRN 参数 |
| tensor.rs | ✅ 完成 | CpuTensor/GpuTensor/DType/DLPack 桥接 + from_host_pinned + gather_rows + scatter_add_rows |
| weight.rs | ✅ 完成 | safetensors 权重加载，mmap模式替代fs::read |
| quant.rs | ✅ 完成 | FP8/FP4 反量化（CPU 参考），128×128 block scale |
| rope.rs | ✅ 完成 | RoPE + YaRN 缩放，per-layer theta |
| kv_cache.rs | ✅ 基础完成 | 环形缓冲 + d2d memcpy + 压缩 KV 读写 |
| gate.rs | ✅ 完成 | hash/score 双路径，tid2eid 循环外传输 |
| compressor.rs | ✅ 已确认正确 | RoPE layer_id正确传入，compress_rope_theta正确使用 |
| indexer.rs | ⚠️ 需完善 | 缺少 Hadamard 旋转 |
| layer.rs | ✅ 大幅优化 | D2D gather/slice/concat; RoPE/cast/act_quant/rmsnorm/swiglu GPU路径; output_proj batched GEMM |
| model.rs | ✅ 大幅优化 | embed_lookup D2D gather; hc_expand D2D; rmsnorm/cast GPU路径; head_forward D2D |
| expert.rs | ✅ 三级缓存 | GPU LFU + RAM SLRU + SSD MMAP + PinnedPool + 双流预取 |
| cache.rs | ✅ 完成 | ThreeLevelCache (GPU/RAM/SSD) |
| pinned.rs | ✅ 新增 | PinnedBuffer(cuMemAllocHost) + PinnedPool |
| tvm_ffi.rs | ✅ 修复 | KernelRegistry key格式修复（短名匹配调用点） |
| TileLang 内核 | ✅ 29个.so已编译 | K1-K9+rope_interleaved+cast已编译; Rust集成完成 |
| 三级缓存 | ✅ 完成 | GPU LFU + RAM SLRU + SSD MMAP + 跨层异步预取 |

---

## 进度追踪（2026-05-15 更新）

### 已完成

| 日期 | 任务 | 影响 |
|------|------|------|
| 05-14 | P0-1: compute_topk_idxs 环形缓冲区索引修复 | 正确性 |
| 05-14 | P0-2: act_quant_inplace_nope KV FP8模拟 | 正确性 |
| 05-14 | P0-4: attn_sink FP32处理 | 正确性 |
| 05-14 | P0-6: n_layers=43 验证 | 正确性 |
| 05-14 | P1-1: Gate tid2eid 移到循环外 | 性能 |
| 05-14 | P1-2: KV Cache d2d memcpy | 性能 |
| 05-14 | P1-3: ExpertScheduler 缓存专家权重 | 性能 |
| 05-14 | P2-1: sparse_attn (K5) TileLang内核集成 | 性能 |
| 05-14 | P2-2: FP8/FP4 GEMM (K3/K4) TileLang内核集成 | 性能 |
| 05-14 | P2-3: act_quant (K1) TileLang内核集成 | 性能 |
| 05-14 | P2-4: hc_sinkhorn (K6) TileLang内核集成 | 性能 |
| 05-14 | P2-5: rmsnorm (K7) TileLang内核集成 | 性能 |
| 05-14 | hc_post GPU路径 (cuBLAS strided batched GEMM) | 性能 |
| 05-14 | hc_reduce 修复 (C=A@B 正确GEMM约定) | 正确性+性能 |
| 05-14 | hc_head_reduce GPU路径 (cuBLAS GEMM) | 性能 |
| 05-14 | hc_pre 无条件to_host传输修复 | 性能(消除129次无效传输) |
| 05-14 | K8 rotary_emb TileLang内核定义+Rust GPU路径 | 性能(消除129次GPU↔CPU往返) |
| 05-14 | cast_bf16_f32 TileLang内核定义+Rust GPU路径 | 性能(消除172次GPU↔CPU往返) |
| 05-14 | gemm_bf16_nn_strided_batched cuBLAS方法 | 基础设施 |
| 05-15 | WeightLoader mmap模式替代fs::read全量读入 | 内存(3.2GB→按需分页) |
| 05-15 | PinnedBuffer/PinnedPool (cuMemAllocHost_v2) | DMA传输带宽翻倍 |
| 05-15 | GpuTensor::from_host_pinned (cuMemcpyAsync) | H2D异步DMA传输 |
| 05-15 | GpuTensor::gather_rows (D2D行拷贝) | 消除MoE FFN的D2H+H2D |
| 05-15 | ExpertScheduler集成PinnedPool | 专家权重H2D走DMA |
| 05-15 | MoE FFN D2D gather优化 | 消除完整x的D2H+expert_tokens H2D |
| 05-15 | rope_interleaved_kernel (TileLang) | RoPE支持interleaved格式+cos/sin分离 |
| 05-15 | RoPE GPU路径 (Q/KV/inverse三方法) | 消除3次/层GPU↔CPU往返 |
| 05-15 | KernelRegistry key格式修复 | 解锁全部已定义TileLang内核 |
| 05-15 | act_quant_inplace_nope GPU路径 | KV nope维度FP8量化走GPU |
| 05-15 | output_proj batched GEMM | 消除分组GEMM CPU重排 |
| 05-15 | 跨层预取异步化 (双流+prefetch_pending) | 隐藏PCIe传输延迟 |
| 05-15 | P0-3/P0-5: 审查确认已正确实现 | 正确性(Compressor RoPE+causal mask) |
| 05-15 | slice_columns/concat_columns D2D优化 | 消除act_quant列切片D2H/H2D |
| 05-15 | embed_lookup D2D gather_rows优化 | 消除embedding查表D2H+H2D |
| 05-15 | hc_expand D2D优化 | 消除HC扩展D2H+H2D |
| 05-15 | TileLang内核编译 (30个.so) | 解锁全部GPU路径 |
| 05-15 | rope_interleaved内核修复 (分离nope/rope) | 解决TileLang布局器限制 |
| 05-15 | model.rs rmsnorm/cast GPU路径 | 消除head_forward等D2H |
| 05-15 | head_forward D2D gather优化 | 消除last_token提取D2H |
| 05-15 | rmsnorm_no_weight GPU路径 | 消除HC归一化D2H |
| 05-15 | scatter_add_D4096 TileLang内核+Rust集成 | 消除MoE FFN专家输出合并D2H（最大瓶颈） |

### 待完成（按优先级排序）

| 优先级 | 任务 | 预期收益 | 状态 |
|--------|------|----------|------|
| **P0** | ~~编译 .so 文件~~ | ✅ 30个已编译 | 已完成 |
| **P1** | ~~MoE FFN scatter-add GPU内核~~ | ✅ scatter_add_D4096已编译+集成 | 已完成 |
| **P1** | fp4_gemm内核集成 (路由专家FP4 GEMM) | 路由专家GEMM直接FP4→BF16 | ✅ compute_expert已有FP4路径 |
| **P1** | K10 gate_topk TileLang内核 | MoE路由GPU化 | 待实现 |
| **P2** | K12 hadamard TileLang内核 | Indexer旋转GPU化 | 待实现 |
| **P2** | K13 compressor TileLang内核 | KV压缩GPU化 | 待实现 |
| **P2** | ~~model.rs rmsnorm/cast GPU路径~~ | ✅ 已完成 | 已完成 |
| **P3** | q_l2_normalize 重命名 | 代码清晰度 | 待实现 |

### 每前向传播GPU↔CPU往返次数估算

| 场景 | 往返次数 | 说明 |
|------|----------|------|
| 全CPU回退（.so未编译） | ~6100 | 所有TileLang kernel回退CPU |
| .so编译后（已有kernel命中） | ~1800 | RoPE/cast/act_quant/rmsnorm/swiglu已消除 |
| fp4_gemm+gate_topk完成 | ~600 | 大部分计算在GPU |
| 全部P1完成 | ~200 | 仅少量控制流回CPU |

### TileLang 内核实例清单（32个，31个已编译）

| 类别 | 实例数 | 实例名 | 状态 |
|------|--------|--------|------|
| act_quant | 5 | N4096/N8192/N2048/N1024_bs128, N448_bs64_inplace | ✅ 已编译 |
| fp8_gemm | 7 | N32768_K1024, N512_K4096, N1024_K4096, N4096_K8192, N2048_K4096, N4096_K2048, N8192_K1024 | ✅ 已编译 |
| sparse_attn | 1 | h64_d512 | ✅ 已编译 |
| hc_sinkhorn | 1 | hc4_it20 | ✅ 已编译 |
| rmsnorm | 4 | N4096, N1024, N512, no_weight_N1024 | ✅ 已编译 |
| swiglu | 1 | N2048 | ✅ 已编译 |
| fp4_gemm | 2 | N2048_K4096, N4096_K2048 | ✅ 已编译 |
| fp4_quant | 1 | N448_bs32_inplace | ✅ 已编译 |
| rotary_emb | 2 | rope_forward_D64_TD512, rope_inverse_D64_TD512 | ✅ 已编译 |
| rope_interleaved | 2 | fwd_D64, inv_D64 | ✅ 已编译 |
| scatter_add | 1 | D4096 | ✅ 已编译 |
| cast | 4 | cast_bf16_to_f32_N4096/N16384, cast_f32_to_bf16_N4096/N16384 | ✅ 已编译 |
| **合计** | **32** | | **31个已编译** |
