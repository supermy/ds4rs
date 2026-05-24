import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
from iq2xs_gemm_tilelang import iq2xs_gemm_optimized


# =============================================================================
# DeepSeek V4 PyTorch 模型核心实现
# =============================================================================
# 本文件包含 DeepSeek V4 的完整 PyTorch 模型定义，核心机制包括：
#   - MLA (Multi-head Latent Attention): 低秩压缩的注意力机制，降低 KV Cache 显存占用
#   - MoE (Mixture-of-Experts): 混合专家模型，每层包含大量路由专家 + 共享专家
#   - Hyper-Connections (HC): 超连接机制，维护多个隐状态副本并通过 Sinkhorn 分配混合
#   - KV Cache 压缩 (Compressor/Indexer): 对历史 KV 进行 gated pooling 压缩，支持稀疏注意力
#   - FP4/FP8 量化: 权重和激活的低位宽量化，提升推理吞吐
#   - YaRN 长度外推: 基于 RoPE 的频率插值，支持超长上下文
# =============================================================================

# 全局分布式与量化配置状态（由 Transformer.__init__ 根据 ModelArgs 初始化时设置）
world_size = 1       # 张量并行的世界大小（GPU 数量）
rank = 0             # 当前进程在 TP 组中的 rank
block_size = 128     # FP8 量化的分块大小（每 block_size 个元素共享一个 scale）
fp4_block_size = 32  # FP4 量化的分块大小（每 32 个 FP4 元素一组 scale）
default_dtype = torch.bfloat16  # 默认权重数据类型（非量化路径）
scale_fmt = None     # 激活量化 scale 的格式（None 或 "ue8m0"）
scale_dtype = torch.float32     # 激活量化 scale 的数据类型


@contextmanager
def set_dtype(dtype):
    """临时覆盖 torch 默认数据类型的上下文管理器，退出时自动恢复（即使发生异常）。"""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    """DeepSeek V4 模型超参数配置类。字段名与官方 config JSON 的键保持一致。"""

    # -------------------- 基础配置 --------------------
    max_batch_size: int = 4          # 最大批处理大小（影响 KV Cache 等 buffer 的预分配）
    max_seq_len: int = 4096          # 最大序列长度（用于预计算位置编码、缓存尺寸等）
    dtype: Literal["bf16", "fp8"] = "fp8"          # 权重默认精度：bf16 或 fp8
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"    # 激活量化 scale 格式：None 或 ue8m0
    expert_dtype: Literal[None, "fp4"] = None      # 专家权重精度：None(默认bf16) 或 fp4
    scale_dtype: Literal["fp32", "fp8"] = "fp8"    # scale 数据类型：fp32 或 fp8
    vocab_size: int = 129280         # 词表大小（含预留的特殊 token）
    dim: int = 4096                  # 模型隐藏层维度（d_model）
    moe_inter_dim: int = 4096        # MoE 专家 FFN 的中间层维度
    n_layers: int = 7                # Transformer 层数（总层数，含 MTP 层）
    n_hash_layers: int = 0           # 使用 hash 路由的层数（前 n_hash_layers 层用 token-id 映射路由）
    n_mtp_layers: int = 1            # MTP (Multi-Token Prediction) 层数
    n_heads: int = 64                # 注意力头总数

    # -------------------- MoE 配置 --------------------
    n_routed_experts: int = 8        # 每层路由专家总数（所有 TP rank 合计）
    n_shared_experts: int = 1        # 每层共享专家数量（所有 token 都会经过）
    n_activated_experts: int = 2     # 每个 token 激活的路由专家数量（top-k）
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"  # 路由分数归一化函数
    route_scale: float = 1.          # 路由权重的缩放系数
    swiglu_limit: float = 0.         # SwiGLU 激活裁剪阈值（0 表示不裁剪）

    # -------------------- MLA / MQA 配置 --------------------
    q_lora_rank: int = 1024          # Query 投影的低秩维度（wq_a 输出维度）
    head_dim: int = 512              # 每个注意力头的总维度（含 rope 部分）
    rope_head_dim: int = 64          # 每个头中应用 RoPE 的维度数（剩余为 nope 维度）
    norm_eps: float = 1e-6           # RMSNorm 的 epsilon，防止除零
    o_groups: int = 8                # 输出投影的分组数（用于低秩 O 投影分组）
    o_lora_rank: int = 1024          # 输出投影的低秩维度（wo_a 输出维度）
    window_size: int = 128           # 滑动窗口注意力窗口大小（局部注意力范围）
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)  # 每层 KV 压缩比例（0 表示不压缩）

    # -------------------- YaRN / RoPE 配置 --------------------
    compress_rope_theta: float = 40000.0  # 压缩 KV 使用的 RoPE 基频
    original_seq_len: int = 0             # 训练时的原始序列长度（>0 时启用 YaRN 插值）
    rope_theta: float = 10000.0           # 标准 RoPE 基频
    rope_factor: float = 40               # YaRN 频率缩放因子
    beta_fast: int = 32                   # YaRN 高频修正范围（旋转次数阈值）
    beta_slow: int = 1                    # YaRN 低频修正范围（旋转次数阈值）

    # -------------------- Indexer (稀疏注意力索引) 配置 --------------------
    index_n_heads: int = 64          # Indexer 的注意力头数（用于压缩 KV 的 top-k 打分）
    index_head_dim: int = 128        # Indexer 每个头的维度
    index_topk: int = 512            # Indexer 选择的压缩 KV top-k 位置数

    # -------------------- Hyper-Connections (HC) 配置 --------------------
    hc_mult: int = 4                 # HC 隐状态副本数量（超连接分支数）
    hc_sinkhorn_iters: int = 20      # Sinkhorn 迭代次数（用于 HC 预权重分配）
    hc_eps: float = 1e-6             # HC 计算的 epsilon（防止除零）


class ParallelEmbedding(nn.Module):
    """词表维度切分的并行 Embedding 层（张量并行）。

    每个 TP rank 仅持有 vocab_size // world_size 行嵌入向量。
    前向时，超出当前 rank 词表范围的输入索引被置零掩码，
    再通过 all_reduce 汇总各 rank 的部分嵌入结果，得到完整嵌入。
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"词表大小必须能被 world_size 整除 (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size      # 当前 rank 负责的词表起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size  # 结束索引（不含）
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            # 标记不属于当前 rank 词表范围的索引，后续置零避免错误查找
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx   # 将全局索引转为当前 rank 的局部索引
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            # 将非本 rank 的嵌入结果置零，然后通过 all_reduce 求和得到完整嵌入
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """量化线性层的前向分发函数。

    根据 weight 的数据类型自动路由到对应的 GEMM 实现：
      - IQ2_XS (dict with __iq2xs__) → iq2xs_gemm_optimized
      - FP4 (torch.float4_e2m1fn_x2) → fp4_gemm
      - FP8 (torch.float8_e4m3fn)    → fp8_gemm
      - 其他（如 BF16）              → F.linear（标准 PyTorch 线性运算）

    对于量化权重，输入激活 x 会先通过 act_quant 量化为 FP8 格式及对应 scale。
    对于 IQ2_XS 权重，直接使用 BF16 激活，由 iq2xs_gemm_optimized 内部处理。
    """
    assert bias is None

    # IQ2_XS 直接 GEMM：直接使用 BF16 激活，不量化
    if isinstance(weight, dict) and weight.get("__iq2xs__"):
        # 确保输入连续（iq2xs_gemm_optimized 内部有断言检查）
        if not x.is_contiguous():
            x = x.contiguous()
        d = weight["d"]
        qs = weight["qs"]
        scales = weight["scales"]
        shape = weight["shape"]  # (out_dim, in_dim)

        # 归档存储扁平形状: d=[total_blocks], qs=[total_blocks, 32], scales=[total_blocks, 8]
        # iq2xs_gemm_optimized 需要 3D 形状: [N, n_blocks_per_row, ...]
        # N = out_dim, n_blocks_per_row = in_dim / 256
        if qs.dim() == 2 and shape is not None:
            out_dim, in_dim = shape
            n_blocks_per_row = in_dim // 256
            qs = qs.view(out_dim, n_blocks_per_row, 32)
            scales = scales.view(out_dim, n_blocks_per_row, 8)
            d = d.view(out_dim, n_blocks_per_row)

        # iq2xs_gemm_optimized 直接接受 BF16 输入，无需量化
        return iq2xs_gemm_optimized(x, qs, scales, d)

    if weight.dtype == torch.float4_e2m1fn_x2:
        # FP4 权重：先将激活量化为 FP8（x, scale），再调用 FP4 GEMM
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        # FP8 权重：先将激活量化为 FP8，再调用 FP8 GEMM
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        # 非量化路径（如 BF16），直接使用 PyTorch 标准线性运算
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        return F.linear(x, weight)


class Linear(nn.Module):
    """支持 BF16、FP8、FP4 三种权重格式的线性层，均使用 per-block scaling。

    权重布局说明：
      - FP4: weight 形状为 [out_features, in_features // 2]，dtype 为 float4_e2m1fn_x2
             （逻辑上为 [out, in]，每 2 个 FP4 值打包为 1 个 int8）
             scale 形状为 [out_features, in_features // 32]，dtype 为 float8_e8m0fnu
             （沿 K 维度每 32 个 FP4 元素共享一个 scale）
      - FP8: weight 形状为 [out_features, in_features]，dtype 为 float8_e4m3fn
             scale 形状为 [ceil(out/block_size), ceil(in/block_size)]，dtype 为 float8_e8m0fnu
      - BF16/其他: weight 为标准 [out, in]，无 scale
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # FP4 权重存储：物理形状 [out, in//2]，逻辑形状 [out, in]
            # scale 按每 32 个 FP4 元素（即 in 方向上每 32 列）一组
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        elif dtype == torch.float8_e4m3fn:
            # FP8 权重：标准形状 [out, in]
            # scale 按 block_size x block_size 的二维分块，每块一个 scale
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        else:
            # 非量化路径（如 BF16）
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """列并行线性层：沿输出维度（out_features）切分至各 TP rank。

    每个 rank 只负责计算部分输出通道，因此前向输出无需 all_reduce。
    典型应用：Q/K/V 投影、gate 投影等。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"输出维度必须能被 world_size 整除 (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """行并行线性层：沿输入维度（in_features）切分至各 TP rank。

    每个 rank 只持有部分输入通道的权重，计算得到部分结果后，
    需要通过 all_reduce 对各 rank 的部分结果求和，才能得到最终输出。
    偏置仅在 all_reduce 后添加一次。
    典型应用：输出投影（如 o_proj、ffn 输出）。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"输入维度必须能被 world_size 整除 (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            # all_reduce 前转为 float32，保证求和精度
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    """均方根层归一化（RMSNorm）。

    与 LayerNorm 不同，RMSNorm 不去均值，仅对输入做缩放：
        output = x / sqrt(mean(x^2) + eps) * weight

    其中 weight 是可学习的逐通道缩放参数。
    注意：checkpoint 中 RMSNorm 的 weight 以 bf16 存储，但此处参数使用 fp32，
    便于后续 logits 计算等需要高精度的场景。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # checkpoint 中 RMSNorm 以 bf16 存储，但此处用 fp32 存储参数，便于后续高精度计算
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        # 归一化计算在 fp32 中进行，避免低精度下数值不稳定
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """预计算带 YaRN 长度外推的旋转位置编码复指数（complex exponentials）。

    当 original_seq_len > 0 时，启用 YaRN 频率插值：
      - 对高频部分（旋转次数多，对应低维索引）进行频率缩放（除以 factor）
      - 对低频部分（旋转次数少，对应高维索引）保持原始频率
      - 中间通过线性斜坡（linear ramp）平滑过渡，过渡范围由 beta_fast/beta_slow 控制

    参数:
      dim: RoPE 维度（通常为 rope_head_dim）
      seqlen: 需要预计算的序列长度
      original_seq_len: 训练时的原始序列长度（>0 启用 YaRN）
      base: RoPE 基频（rope_theta）
      factor: YaRN 频率缩放因子
      beta_fast: 高频修正阈值（旋转次数）
      beta_slow: 低频修正阈值（旋转次数）

    返回:
      freqs_cis: [seqlen, dim//2] 的复数张量，用于 apply_rotary_emb
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """根据目标旋转次数计算对应的维度索引。"""
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """计算需要修正的频率范围（维度索引的 low-high 区间）。"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """生成 [0,1] 之间的线性斜坡，用于平滑过渡高频/低频修正。"""
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 基础 RoPE 频率：1 / (base^(i/dim)), i = 0, 2, 4, ..., dim-2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # 启用 YaRN：计算需要插值的频率范围
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 高频部分（smooth 接近 0）频率除以 factor，低频部分保持原频率
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 为每个位置 t 计算外积，得到 [seqlen, dim//2] 的角度矩阵
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    # 转换为复数形式：e^(j*theta) = cos(theta) + j*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """原地应用旋转位置编码（RoPE）。

    将 x 的最后两维视为复数实部/虚部，与 freqs_cis 做复数乘法实现旋转。
    inverse=True 时使用 freqs_cis 的共轭，实现反旋转（de-rotation）。

    参数:
      x: [..., seq_len, rope_dim] 或 [..., seq_len, heads, rope_dim]，最后维度必须是偶数
      freqs_cis: [seq_len, rope_dim//2] 的复数张量（来自 precompute_freqs_cis）
      inverse: 是否应用反旋转（用于输出投影后的反旋转）
    """
    y = x
    # 将最后一维拆分为 (复数维度, 2)，并视为复数
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()   # 反旋转：取共轭复数
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 复数乘法后展平回实数张量，原地拷贝到输入张量
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """应用随机 Hadamard 旋转，将信息均匀分散到各个维度。

    在 FP8/FP4 量化前使用，可减少量化误差（避免某些维度信息过于集中而被量化截断）。
    缩放因子为 1/sqrt(dim)，保持能量守恒。
    """
    assert x.dtype == torch.bfloat16
    try:
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=x.size(-1) ** -0.5)
    except ImportError:
        return _hadamard_transform_fallback(x)


def _hadamard_transform_fallback(x: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch Hadamard 变换回退实现。

    使用 Sylvester 构造法递归生成 Hadamard 矩阵，
    然后与输入相乘。性能低于 fast_hadamard_transform 但功能等价。
    """
    d = x.size(-1)
    assert d > 0 and (d & (d - 1)) == 0, f"dim must be power of 2, got {d}"
    scale = d ** -0.5
    if d == 1:
        return x * scale
    # Sylvester 构造: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    # 递归展开为蝶形运算
    orig_shape = x.shape
    x = x.reshape(-1, d)
    h = 1
    while h < d:
        x = x.view(-1, d // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, d)
        h *= 2
    return x.view(*orig_shape) * scale


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """生成滑动窗口注意力的 top-k 索引矩阵。

    对于每个查询位置，返回其滑动窗口内可访问的 KV 缓存索引。
    超出窗口范围或未来位置用 -1 填充（sparse_attn 中会忽略）。

    参数:
      window_size: 滑动窗口大小
      bsz: batch size
      seqlen: 当前序列长度
      start_pos: 当前解码位置（0 表示预填充阶段）

    返回:
      matrix: [bsz, seqlen, window_size] 的索引张量
    """
    if start_pos >= window_size - 1:
        # 解码阶段：KV 缓存以环形缓冲区方式存储，需要重排索引
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),  torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        # 解码阶段但缓存未满：右侧用 -1 填充
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        # 预填充阶段：为每个位置生成其滑动窗口内的合法索引
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    """生成压缩 KV 缓存的 top-k 索引矩阵。

    压缩 KV 每 ratio 个原始 token 合并为 1 个，因此索引指向压缩后的位置。
    超出当前序列长度的未来位置用 -1 填充。

    参数:
      ratio: 压缩比例（如 4 表示每 4 个 token 压缩为 1 个）
      bsz: batch size
      seqlen: 当前序列长度
      start_pos: 当前解码位置（0 表示预填充阶段）
      offset: 索引偏移量（用于区分滑动窗口 KV 和压缩 KV 的索引空间）

    返回:
      matrix: [bsz, seqlen, compress_len] 的索引张量
    """
    if start_pos > 0:
        # 解码阶段：所有已压缩的位置都可访问
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        # 预填充阶段：每个位置只能访问其之前的压缩位置（因果掩码）
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """KV Cache 压缩器：通过可学习的门控池化（gated pooling）将连续 compress_ratio 个 token 的 KV 压缩为 1 个。

    核心机制：
      - 对每个 token 通过 wkv 和 wgate 两个线性层分别生成 KV 候选值和门控分数
      - 在 compress_ratio 个连续 token 上做 softmax 加权求和，得到压缩后的 KV
      - 当 overlap=True（ratio==4 时），使用重叠窗口以获得更平滑的压缩边界
      - 压缩后的 KV 经过 RMSNorm、RoPE 位置编码、量化后写入 kv_cache

    状态管理：
      - 预填充阶段（start_pos==0）：一次性处理整个序列，末尾不足 ratio 的 token 存入 state buffer
      - 解码阶段（start_pos>0）：逐 token 增量处理，每凑齐 ratio 个 token 执行一次压缩
      - kv_state / score_state：用于解码阶段暂存未凑齐的 token 的 KV 和分数
    """

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4   # ratio==4 时启用重叠窗口模式
        self.rotate = rotate                 # 是否启用 Hadamard 旋转（Indexer 使用）
        coff = 1 + self.overlap              # 系数：overlap 模式下维度翻倍

        # 绝对位置编码（APE），用于给门控分数添加位置偏置
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv 和 wgate 在 checkpoint 中以 bf16 存储，但此处用 fp32 参数便于训练/微调
        # overlap 模式下，前半维度用于重叠压缩，后半维度用于正常压缩
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None   # 延迟绑定，由 Attention 层传入其 kv_cache 的后半部分
        # 解码阶段增量压缩的状态缓冲区
        # overlap 模式下：state[:, :ratio] 为重叠窗口，state[:, ratio:] 为当前窗口
        self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: torch.Tensor = None  # 延迟绑定，来自 Attention 层预计算的 RoPE

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        """重叠窗口变换：将 [b,s,r,2d] 转换为重叠的 [b,s,2r,d] 布局。

        将正常压缩的下半部分（[:,:,:,d:]）放到新张量的后半段（[:,:,r:]），
        将重叠压缩的上半部分（[:,:,:,:d]）通过时间平移放到新张量的前半段（[:,1:,:r]）。
        """
        # tensor: [b, s, ratio, 2*head_dim]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]       # 正常压缩部分
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]    # 重叠部分（时间平移）
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        """执行 KV 压缩并写入 kv_cache。

        参数:
          x: [bsz, seqlen, dim] 当前层的隐藏状态输入
          start_pos: 当前序列位置（0 表示预填充，>0 表示增量解码）

        返回:
          预填充阶段返回压缩后的 KV（用于 Attention 拼接）；
          解码阶段若未凑齐 ratio 个 token 则返回 None。
        """
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        # 压缩计算需要 fp32 精度，避免数值不稳定
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            # ==================== 预填充阶段 ====================
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                # 保存最后 ratio 个 token 用于重叠窗口的初始状态
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
            if remainder > 0:
                # 末尾不足 ratio 的 token 存入 state buffer，等待后续解码时凑齐
                kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            # 将序列按 ratio 分组：[bsz, num_groups, ratio, dim]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                # 应用重叠窗口变换
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            # 在 ratio 维度上做 softmax 加权求和，得到压缩后的 KV
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            # ==================== 解码阶段（增量） ====================
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                # 将当前 token 的 KV 和分数存入当前窗口
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    # 拼接重叠窗口和当前窗口，执行压缩
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    # 滑动窗口：当前窗口变为下一组的重叠窗口
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                # 非重叠模式：直接存入环形缓冲区
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        # 压缩后的 KV 经过归一化
        kv = self.norm(kv.to(dtype))
        # 对 rope 维度应用位置编码
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # 量化：Indexer 使用 FP4（含 Hadamard 旋转），普通 Compressor 使用 FP8
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        # 写入 KV Cache
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """压缩 KV 的稀疏注意力索引器：通过可学习的打分机制选择 top-k 压缩 KV 位置。

    Indexer 拥有独立的 Compressor（带 Hadamard 旋转），用于构建压缩 KV 供打分使用。
    核心流程：
      1. 使用低秩 Query 投影（wq_b）生成查询向量 q
      2. 对 q 应用 RoPE 和 Hadamard 旋转，并量化为 FP4（QAT 模拟）
      3. 通过独立 Compressor 构建压缩 KV（同样经过旋转和 FP4 量化）
      4. 计算 q 与压缩 KV 的注意力分数，结合 weights_proj 的权重进行加权
      5. 选择 top-k 压缩位置，返回索引供 Attention 层的 sparse_attn 使用

    注意：Indexer 的 kv_cache 存储的是用于打分的压缩 KV，与 Attention 的 kv_cache 分开。
    """

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        # Query 投影：从低秩表示生成 Indexer 专用的查询向量
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        # 权重投影：为每个头生成一个标量权重，用于加权注意力分数
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        # Indexer 使用独立的 Compressor，启用 Hadamard 旋转和 FP4 量化
        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)
        self.freqs_cis = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        """计算压缩 KV 的 top-k 索引。

        参数:
          x: [bsz, seqlen, dim] 当前层隐藏状态
          qr: [bsz, seqlen, q_lora_rank] Query 的低秩表示（来自 Attention.wq_a）
          start_pos: 当前序列位置
          offset: 索引偏移量（用于区分滑动窗口 KV 和压缩 KV 的索引空间）

        返回:
          topk_idxs: [bsz, seqlen, topk] 压缩 KV 的 top-k 位置索引
        """
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        # 延迟绑定 Compressor 的缓存和位置编码
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        # 生成 Indexer 专用的 Query 向量
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # QAT 模拟：将 q 量化为 FP4（与生产环境保持一致）
        fp4_act_quant(q, fp4_block_size, True)
        # 通过 Compressor 构建压缩 KV（会自动写入 self.kv_cache）
        self.compressor(x, start_pos)
        # 计算每个头的权重（用于加权注意力分数）
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # 计算 q 与压缩 KV 的注意力分数（QAT 模拟，kv 当前为 bf16，也可使用 fp8）
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        # ReLU 激活后按头权重加权求和，得到每个查询位置对每个压缩位置的总体分数
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            dist.all_reduce(index_score)
        # 预填充阶段：应用因果掩码（只能访问当前位置之前的压缩 KV）
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        # 选择 top-k 压缩位置
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        # 预填充阶段：将未来位置的索引置为 -1（sparse_attn 会忽略）
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """多头隐式注意力（Multi-head Latent Attention, MLA）。

    MLA 核心设计：
      - Query 低秩压缩：dim -> q_lora_rank -> n_heads * head_dim，减少 Q 投影参数量
      - KV 统一投影：单个 wkv 将输入投影为 head_dim 维的 KV 向量（而非传统 K/V 分离）
      - 滑动窗口注意力：仅关注最近的 window_size 个 token，降低 KV Cache 显存
      - 可选 KV 压缩：通过 Compressor 将历史 KV 按 compress_ratio 压缩，Indexer 选择 top-k 压缩位置
      - 输出低秩分解：n_heads * head_dim // n_groups -> n_groups * o_lora_rank -> dim
      - QAT 模拟：nope（非 rope）维度使用 FP8 量化，rope 维度保持 bf16 保证位置精度

    KV Cache 布局（当启用压缩时）：
      - [:, :window_size]          : 滑动窗口 KV（环形缓冲区）
      - [:, window_size:]          : 压缩 KV（由 Compressor 写入）
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        # 注意力池化（attention sink）参数，用于稳定长序列注意力
        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        # Query 低秩投影：dim -> q_lora_rank -> n_heads * head_dim
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        # KV 统一投影：将输入直接投影为 KV 表示（head_dim 维）
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # 输出低秩分解：先按组降维，再升维回 dim
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5

        # 根据配置初始化 KV 压缩器和索引器
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                # ratio==4 时使用 Indexer 进行智能 top-k 选择；ratio==128 时直接用固定索引
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        # KV Cache 预分配：滑动窗口部分 + 压缩部分（如果启用压缩）
        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim), persistent=False)
        # 预计算 RoPE 复指数：压缩 KV 使用 YaRN 外推，纯滑动窗口使用标准 RoPE
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            # 纯滑动窗口注意力禁用 YaRN，使用基础 rope_theta
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                         rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        """MLA 前向传播。

        参数:
          x: [bsz, seqlen, dim] 输入隐藏状态
          start_pos: 当前序列位置（0 表示预填充，>0 表示增量解码）

        返回:
          [bsz, seqlen, dim] 注意力输出
        """
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        # 延迟绑定 Compressor 和 Indexer 的缓存与位置编码
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # ==================== Query 投影 ====================
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        # Query 归一化（除 RMSNorm 外的额外缩放，稳定注意力分数）
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # ==================== KV 投影与滑动窗口索引 ====================
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # QAT 模拟：nope 维度量化为 FP8，rope 维度保持 bf16 以保证位置精度
        act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        # 生成滑动窗口的 top-k 索引（每个查询位置可访问的最近 window_size 个 KV）
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        # 如果启用压缩，追加压缩 KV 的索引
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                # 使用 Indexer 智能选择 top-k 压缩位置
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                # ratio==128 时使用固定压缩索引
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # ==================== KV 缓存更新与稀疏注意力 ====================
        if start_pos == 0:
            # 预填充阶段
            if seqlen <= win:
                # 序列较短，全部存入滑动窗口
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                # 序列较长，仅保留最后 window_size 个 token 到环形缓冲区
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                # 执行 KV 压缩，将压缩结果拼接到 KV 后供 sparse_attn 使用
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            # QAT 模拟：kv 当前为 bf16，也可使用 fp8 格式
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            # 解码阶段：将当前 token 的 KV 写入滑动窗口环形缓冲区
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                # 增量压缩（若凑齐 ratio 个 token 则执行压缩）
                self.compressor(x, start_pos)
            # 从预分配的 kv_cache 中读取（包含滑动窗口和压缩 KV）
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        # 反旋转：对输出应用逆 RoPE，抵消 Query 旋转的影响
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # ==================== 输出投影 ====================
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # 注：checkpoint 中 wo_a 为 FP8，此处为简化使用 BF16 计算
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE 路由门控：计算每个 token 到各专家的路由分数，并选择 top-k 个专家。

    支持两种路由模式：
      - Hash 路由（前 n_hash_layers 层）：专家索引由 token ID 直接查表确定，
        适用于早期层，避免计算开销。
      - 分数路由（剩余层）：通过线性投影计算分数，选择 top-k 专家。

    分数归一化函数：
      - "softmax": 标准 softmax 归一化
      - "sigmoid": sigmoid 后不归一化
      - "sqrtsoftplus": sqrt(softplus(x))，默认选项，训练更稳定

    注意：bias 仅用于专家选择（影响 topk），不影响最终路由权重（使用原始分数）。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            # Hash 路由：token ID -> 专家索引的查找表
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            # 分数路由：可学习的偏置，用于调整专家负载均衡
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算路由分数并选择专家。

        参数:
          x: [num_tokens, dim] 展平后的隐藏状态
          input_ids: [num_tokens] token ID（仅 hash 路由时使用）

        返回:
          weights: [num_tokens, topk] 路由权重
          indices: [num_tokens, topk] 选中的专家索引
        """
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # bias 仅影响专家选择（topk），不影响最终路由权重（使用原始分数）
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            # Hash 路由：直接查表获取专家索引
            indices = self.tid2eid[input_ids]
        else:
            # 分数路由：选择分数最高的 top-k 个专家
            indices = scores.topk(self.topk, dim=-1)[1]
        # 使用原始分数（未加 bias）作为路由权重
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            # 非 softmax 模式下，对权重做归一化（使 top-k 权重之和为 1）
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """单个 MoE 专家：SwiGLU 前馈网络（FFN）。

    SwiGLU 结构：
      - w1 (gate):  dim -> inter_dim，控制门控
      - w3 (up):    dim -> inter_dim，提供上采样激活
      - w2 (down):  inter_dim -> dim，输出投影
      - 输出 = w2( silu(w1(x)) * w3(x) )

    计算在 float32 中进行以保证数值稳定性，最后转回输入精度。
    可选的 swiglu_limit 用于裁剪激活值，防止极端值。
    """
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)   # gate 投影
        self.w2 = Linear(inter_dim, dim, dtype=dtype)   # down 投影
        self.w3 = Linear(dim, inter_dim, dtype=dtype)   # up 投影
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """专家前向传播。

        参数:
          x: [num_tokens, dim] 输入隐藏状态
          weights: [num_tokens, 1] 路由权重（若提供则对输出做加权）

        返回:
          [num_tokens, dim] 专家输出
        """
        dtype = x.dtype
        # SwiGLU 计算在 fp32 中进行，避免低精度下数值不稳定
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            # 裁剪激活值，防止极端值影响训练/推理稳定性
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            # 应用路由权重（来自 Gate 的 top-k 权重）
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """混合专家模型（Mixture-of-Experts）：将每个 token 路由到 top-k 个路由专家 + 1 个共享专家。

    张量并行策略：
      - 路由专家按 TP rank 切分：每个 rank 只持有 n_routed_experts // world_size 个专家
      - 共享专家不跨 rank 切分（所有 rank 都持有完整共享专家）
      - 路由计算后通过 all_reduce 汇总各 rank 的部分结果

    前向流程：
      1. Gate 计算每个 token 的路由分数和专家索引
      2. 每个 rank 仅计算分配给它的专家
      3. all_reduce 汇总所有 rank 的专家输出
      4. 加上共享专家的输出（所有 token 都会经过）
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"专家数量必须能被 world_size 整除 (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        # 懒初始化：路由专家不预分配权重，仅记录元信息
        # 256 个路由专家 × 43 层 ≈ 143GB 内存，远超 90GB 限制
        # 推理时通过 _on_experts_needed 回调按需从 safetensors 加载到 GPU
        self._expert_dim = args.dim
        self._expert_inter_dim = args.moe_inter_dim
        self._expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self._expert_swiglu_limit = args.swiglu_limit
        self.experts = nn.ModuleList([None] * args.n_routed_experts)
        # 流式加载回调：由 generate.py 的 load_weights_streaming 设置
        # _on_experts_needed(activated_indices): Gate 计算后、专家计算前调用，加载激活专家权重到 GPU
        # _on_experts_done(activated_indices): 专家计算后调用，卸载专家释放 GPU 显存
        self._on_experts_needed = None
        self._on_experts_done = None
        # CPU 专家推理运行器：GPU 缓存未命中时走 CPU 推理
        self._cpu_expert_runner = None
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def _ensure_expert(self, idx: int) -> Expert:
        """确保第 idx 个路由专家已加载，若未加载则创建空壳 Expert。

        实际权重由 generate.py 的流式加载钩子注入，
        此方法仅保证 self.experts[idx] 非 None 以支持 forward 调用。
        空壳 Expert 在 CPU 上创建，避免占用 GPU 显存。
        """
        if self.experts[idx] is None:
            with torch.device('cpu'):
                self.experts[idx] = Expert(
                    self._expert_dim, self._expert_inter_dim,
                    dtype=self._expert_dtype, swiglu_limit=self._expert_swiglu_limit
                )
        return self.experts[idx]

    def _get_compute_streams(self, n: int):
        """获取 CUDA Stream 池（复用，避免每次调用重新创建）。"""
        if not hasattr(self, '_compute_streams'):
            self._compute_streams = [torch.cuda.Stream() for _ in range(6)]  # top-k=6
        return self._compute_streams[:n]

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """MoE 前向传播。

        参数:
          x: [bsz, seqlen, dim] 输入隐藏状态
          input_ids: [bsz, seqlen] token ID（用于 hash 路由）

        返回:
          [bsz, seqlen, dim] MoE 输出
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        # 计算路由分数和专家索引
        weights, indices = self.gate(x, input_ids.flatten())
        # 累加各专家输出的缓冲区（fp32 保证精度）
        y = torch.zeros_like(x, dtype=torch.float32)
        # 统计每个专家被激活的次数，仅计算有 token 路由到的专家
        # 使用 GPU nonzero 替代 bincount+tolist，避免 GPU→CPU 同步阻塞
        flat_indices = indices.flatten()
        unique_experts = torch.unique(flat_indices)
        activated = unique_experts.tolist()

        # 流式加载回调：仅加载激活的 top-k 专家到 GPU（而非全部 256 个）
        # 由 generate.py 的 load_weights_streaming 设置，避免 GPU OOM
        # _on_experts_needed 内部调用 _load_activated_experts，负责：
        #   1. 创建 Expert 空壳（CPU 上）
        #   2. 从缓存/SSD 加载权重到 GPU
        #   3. 将 GPU 参数设置到 Expert 对象上
        # 因此此处不再调用 _ensure_expert，避免创建 CPU 空壳导致 fp4_gemm 收到 CPU 权重
        if self._on_experts_needed is not None:
            self._on_experts_needed(activated)

        # 在默认流上预计算专家分配，避免并行流对 indices 的竞争
        expert_inputs = {}
        expert_weights = {}
        for i in activated:
            expert = self.experts[i]
            if expert is None:
                continue
            idx, top = torch.where(indices == i)
            expert_inputs[i] = (x[idx], idx, top)

        # 逐专家计算：GPU 命中走 GPU，未命中走 CPU（混合推理）
        gpu_count = 0
        cpu_count = 0
        for i in expert_inputs:
            expert = self.experts[i]
            xi, idx, top = expert_inputs[i]

            if expert is not None:
                # GPU 路径：专家权重已在 GPU 上
                y[idx] += expert(xi, weights[idx, top, None])
                gpu_count += 1
            else:
                # CPU 路径：专家权重不在 GPU 缓存中，走 CPU 推理
                if self._cpu_expert_runner is not None:
                    for k in range(xi.shape[0]):
                        cpu_out = self._cpu_expert_runner.compute_expert_cpu(
                            self.layer_id, i, xi[k], route_weight=weights[idx[k], top[k], 0].item(),
                            swiglu_limit=self._expert_swiglu_limit
                        )
                        y[idx[k]] += cpu_out
                    cpu_count += 1
                # else: 无 CPU 推理能力，跳过（输出为 0）

        # 混合推理统计（每 10 层打印一次）
        if cpu_count > 0 and self.layer_id % 10 == 0:
            total = gpu_count + cpu_count
            print(f"[MoE] L{self.layer_id}: GPU={gpu_count}/{total}, CPU={cpu_count}/{total}")

        # 预取下一层热专家到 CPU Rust SLRU（在当前层计算完成后）
        if self._cpu_expert_runner is not None:
            self._cpu_expert_runner.prefetch_layer(self.layer_id + 1)

        # 流式卸载回调：专家计算完毕后释放 GPU 显存
        if self._on_experts_done is not None:
            self._on_experts_done(activated)

        # 汇总所有 rank 的专家输出
        if world_size > 1:
            dist.all_reduce(y)
        # 加上共享专家的输出（所有 token 都经过）
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer 块，集成 Hyper-Connections (HC) 混合机制。

    与传统残差连接不同，HC 维护 hc_mult 个隐状态副本：
      - hc_pre:  将 hc_mult 个副本通过可学习的加权求和压缩为 1 个（预权重通过 Sinkhorn 分配）
      - hc_post: 将 1 个输出通过可学习的后权重和组合矩阵扩展回 hc_mult 个副本

    每个 Block 包含两个子层（Attention 和 MoE），每个子层前后都有 HC 处理。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            # HC 混合函数参数：用于 Attention 子层和 FFN 子层
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """HC 预处理：将 hc_mult 个隐状态副本压缩为 1 个。

        参数:
          x: [b, s, hc, d] 输入的多个隐状态副本
          hc_fn: [mix_hc, hc*d] 混合函数权重
          hc_scale: [3] Sinkhorn 分配的尺度参数
          hc_base: [mix_hc] 混合偏置

        返回:
          y: [b, s, d] 压缩后的单个隐状态
          post: [b, s, hc] 后处理权重（供 hc_post 使用）
          comb: [b, s, hc, hc] 组合矩阵（供 hc_post 使用）
        """
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """HC 后处理：将 1 个输出扩展回 hc_mult 个隐状态副本。

        参数:
          x: [b, s, d] 子层输出（单个隐状态）
          residual: [b, s, hc, d] 预处理前的多个隐状态副本
          post: [b, s, hc] 后处理权重（来自 hc_pre）
          comb: [b, s, hc, hc] 组合矩阵（来自 hc_pre）

        返回:
          y: [b, s, hc, d] 扩展后的多个隐状态副本
        """
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Transformer 块前向传播。

        参数:
          x: [bsz, seqlen, hc_mult, dim] 输入的多个隐状态副本
          start_pos: 当前序列位置
          input_ids: token ID（用于 MoE 的 hash 路由）

        返回:
          [bsz, seqlen, hc_mult, dim] 输出的多个隐状态副本
        """
        # Attention 子层
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        # FFN (MoE) 子层
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):
    """并行输出头：将 HC 混合后的隐状态映射到词表维度的 logits。

    与 ParallelEmbedding 类似，沿词表维度切分至各 TP rank。
    前向时先通过 hc_head 将 hc_mult 个副本压缩为 1 个，再计算 logits。
    checkpoint 中 lm_head 以 bf16 存储，但此处用 fp32 便于后续高精度 logits 计算。
    """

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = (vocab_size // world_size)
        # lm_head 在 checkpoint 中以 bf16 存储，但此处用 fp32 便于后续 logits 高精度计算
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def get_logits(self, x):
        """计算最后一个位置的 logits（用于自回归生成）。

        分块计算避免一次性分配整个词表的 float32 权重（~2GB），
        每次处理 16384 个词表条目，峰值显存仅 ~256MB。

        参数:
          x: [bsz, dim] 最后一个位置的隐藏状态

        返回:
          [bsz, part_vocab_size] 当前 rank 负责的部分 logits
        """
        x_last = x[:, -1].float()
        chunk = 16384
        parts = []
        for i in range(0, self.weight.shape[0], chunk):
            w = self.weight[i:i + chunk].float()
            parts.append(F.linear(x_last, w))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        """输出头前向传播。

        参数:
          x: [bsz, seqlen, hc_mult, dim] HC 混合后的多个隐状态副本
          hc_fn, hc_scale, hc_base: HC 头混合参数
          norm: 输出前的 RMSNorm

        返回:
          [bsz, vocab_size] 完整 logits（多 rank 时通过 all_gather 拼接）
        """
        # x: [b,s,hc,d]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x))
        if world_size > 1:
            # 收集所有 rank 的部分 logits，拼接为完整词表
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """输出头的 HC 压缩：将 hc_mult 个副本压缩为 1 个。

        与 Block 中的 hc_pre 类似，但使用 sigmoid 而非 Sinkhorn 分配。
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class MTPBlock(Block):
    """多 Token 预测（Multi-Token Prediction, MTP）块。

    MTP 在标准 Transformer 块基础上增加：
      - e_proj: 将输入 token 的嵌入投影到与隐藏状态相同的空间
      - h_proj: 将前一层的隐藏状态投影
      - 两者相加后通过标准 Block 处理，最后经 ParallelHead 输出下一个 token 的 logits

    注意：MTP 收益在 DeepSeek V4 中不高，生产环境不建议使用。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim)   # 输入嵌入投影
        self.h_proj = Linear(args.dim, args.dim)   # 隐藏状态投影
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        """MTP 块前向传播。

        参数:
          x: [bsz, seqlen, hc_mult, dim] 前一层的隐藏状态（HC 格式）
          start_pos: 当前序列位置
          input_ids: [bsz, seqlen] 输入 token ID

        返回:
          [bsz, vocab_size] 预测的下一个 token 的 logits
        """
        # x: [b,s,hc,d]
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        # 将输入嵌入和隐藏状态投影后相加，再通过标准 Block
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


class Transformer(nn.Module):
    """完整的 DeepSeek-V4 模型。

    前向流程：
      1. embed: 词嵌入
      2. HC 扩展：将单个嵌入扩展为 hc_mult 个副本
      3. N 个 Block: 交替执行 Attention 和 MoE，每层都有 HC 混合
      4. HC 头压缩：将 hc_mult 个副本压缩为 1 个
      5. ParallelHead: 计算 logits

    全局状态设置：
      __init__ 中根据 ModelArgs 设置 world_size, rank, default_dtype, scale_fmt, scale_dtype
    """
    def __init__(self, args: ModelArgs):
        global world_size, rank, default_dtype, scale_fmt, scale_dtype
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            self.mtp.append(MTPBlock(args.n_layers + layer_id, args))
            self.mtp[-1].embed = self.embed
            self.mtp[-1].head = self.head
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        """模型前向传播。

        参数:
          input_ids: [bsz, seqlen] 输入 token 索引
          start_pos: 当前序列位置（0 表示预填充，>0 表示增量解码）

        返回:
          [bsz, vocab_size] 最后一个位置的 logits（自回归生成）
        """
        h = self.embed(input_ids)
        # 将单个嵌入扩展为 hc_mult 个副本，供 Hyper-Connections 使用
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(n_hash_layers=0)
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)

    print(model(x).size())
    for i in range(128, 150):
        print(i, model(x[:, 0:1], i).size())

    h = torch.randn(2, 128, args.hc_mult, args.dim)
    mtp = model.mtp[0]
    print(mtp(h, 0, x).size())
    print(mtp(h[:, 0:1], 1, x[:, 0:1]).size())
