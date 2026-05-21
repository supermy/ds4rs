import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


# FP4 e2m1fn 到 float32 的完整解码表（共16个条目）。
# FP4 格式：1 位符号 + 2 位指数 + 1 位尾数（非规格化）。
# 索引 0~7 对应符号位为 0（正数），8~15 对应符号位为 1（负数）。
# 该表用于将打包在 int8 低4位/高4位中的 FP4 值直接映射为 float32，以便后续无损转换到 FP8。
FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 FP4(e2m1fn) 打包格式的权重无损转换为 FP8(e4m3fn) 格式，并重新计算 scale。

    转换原理：
    1. 输入 x 为 int8 类型，每个字节打包 2 个 FP4 值（低4位和高4位）。
    2. 通过 FP4_TABLE 将每个4位索引解码为 float32 数值，恢复原始权重矩阵。
    3. 原始 FP4 的 scale 是按 "每行、每32列一组" 存储的 FP8(e8m0fnu)。
       由于 FP4 的动态范围（最大绝对值 6.0）远小于 FP8 e4m3fn 的动态范围（最大 448），
       我们可以将原 scale 拆分为一个 "基础缩放因子" 和一个 "偏移量"，
       使得：新权重 = 解码后的 FP4 值 × 偏移量，新 scale = 基础缩放因子。
    4. 偏移量必须保证新权重的绝对值不超过 FP8 e4m3fn 的最大值 448。
       计算得 6.0 × 2^6 = 384 < 448，因此偏移量最大可用 2^6 = 64，即 MAX_OFFSET_BITS = 6。

    参数:
        x (torch.Tensor): FP4 打包权重，dtype=torch.int8，shape=(out_dim, in_dim//2)。
                          注意：in_dim//2 是因为每字节存2个 FP4。
        scale (torch.Tensor): 原始 FP4 scale，dtype 通常为 FP8 e8m0fnu，
                              shape=(out_dim, in_dim//32)，即每行每32列一组 scale。

    返回:
        tuple[torch.Tensor, torch.Tensor]:
            - 转换后的 FP8 e4m3fn 权重，shape=(out_dim, in_dim)。
            - 重新计算后的 FP8 e8m0fnu scale，shape=(out_dim//128, in_dim//128)。
              新 scale 的粒度为 128×128 块（FP8 标准块大小）。
    """
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim_half = x.size()
    in_dim = in_dim_half * 2  # 实际输入维度（每字节2个 FP4，所以展开后翻倍）
    fp8_block_size = 128      # FP8 量化标准块大小
    fp4_block_size = 32       # FP4 原始 scale 对应的列分组大小
    # 确保维度能被 FP8 块大小整除，以便按 128×128 块重新量化
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    # 验证原始 scale 的形状：每行、每32列一组
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    # 将 int8 视为 uint8，提取低4位和高4位，分别查表解码为 float32
    x = x.view(torch.uint8)
    low  = x & 0x0F           # 低4位：第一个 FP4 值
    high = (x >> 4) & 0x0F    # 高4位：第二个 FP4 值
    # stack 后在最后一维展开，再 flatten，恢复为 (out_dim, in_dim) 的 float32 矩阵
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    # FP4 最大绝对值为 6.0，FP8 e4m3fn 最大可表示值为 448。
    # 需要满足：max_fp4 * MAX_OFFSET <= 448
    # 6.0 * 2^6 = 384 < 448，安全；6.0 * 2^7 = 768 > 448，溢出。
    # 因此偏移量比特数取 6，即偏移量最大为 64。
    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size   # 输出方向块数
    bIn = in_dim // fp8_block_size     # 输入方向块数
    # 将权重重塑为 (bOut, 128, bIn, 128)，然后转置为 (bOut, bIn, 128, 128)，
    # 以便按 128×128 块处理
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)

    # 将原始 scale 同样重塑为块结构：(bOut, 128, bIn, 4) 因为 128/32=4
    # 转置为 (bOut, bIn, 128, 4)，再 flatten 最后两维为 (bOut, bIn, 128*4)
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)

    # 计算每个 128×128 块内原始 scale 的最大值，并除以 2^6，
    # 得到新的基础缩放因子 scale_max_offset_bits（即新的 FP8 scale）
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)

    # 计算偏移量：原始 scale / 新 scale，用于将 FP4 解码值放大到 FP8 可表示范围
    offset = scale / scale_max_offset_bits

    # 将偏移量从 (bOut, bIn, 128*4) 重塑回 (bOut, bIn, 128, 4)，
    # 然后对最后一维（4个 scale 组）进行 repeat_interleave，每组重复32次，
    # 最终得到每个元素对应的偏移量 (bOut, bIn, 128, 128)
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)

    # 应用偏移量，恢复块布局，重塑为最终权重矩阵
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    # 返回 FP8 e4m3fn 权重，以及 squeeze 后的新 scale（FP8 e8m0fnu）
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


# HuggingFace 权重名称到自定义命名规范的映射表。
# 键：HF checkpoint 中的权重名称关键字（通常是倒数第二个点后的字段）。
# 值：元组 (new_key, dim)，其中 new_key 是自定义命名，dim 是模型并行切分维度。
#   - dim=0：按第0维（输出维度）切分，例如 Linear 的 weight[out_features, in_features]。
#   - dim=1：按第1维（输入维度）切分。
#   - dim=None：不切分，所有 MP  rank 保留完整副本。
mapping = {
    "embed_tokens": ("embed", 0),           # 词嵌入，按词表维度切分
    "input_layernorm": ("attn_norm", None), # Attention 前的 RMSNorm，不切分
    "post_attention_layernorm": ("ffn_norm", None),  # FFN 前的 RMSNorm，不切分
    "q_proj": ("wq", 0),                    # Q 投影，按输出维度切分
    "q_a_proj": ("wq_a", None),             # Q 压缩投影（ MLA ），不切分
    "q_a_layernorm": ("q_norm", None),      # Q 压缩后的 Norm，不切分
    "q_b_proj": ("wq_b", 0),                # Q 解压投影，按输出维度切分
    "kv_a_proj_with_mqa": ("wkv_a", None),  # KV 压缩投影（带 MQA ），不切分
    "kv_a_layernorm": ("kv_norm", None),    # KV 压缩后的 Norm，不切分
    "kv_b_proj": ("wkv_b", 0),              # KV 解压投影，按输出维度切分
    "o_proj": ("wo", 1),                    # Attention 输出投影，按输入维度切分
    "gate_proj": ("w1", 0),                 # FFN gate 投影，按输出维度切分
    "down_proj": ("w2", 1),                 # FFN down 投影，按输入维度切分
    "up_proj": ("w3", 0),                   # FFN up 投影，按输出维度切分
    "lm_head": ("head", 0),                 # 语言模型头，按词表维度切分

    # 以下可能是已经转换过的名称或特殊权重，保持原样或指定切分维度
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def main(hf_ckpt_path, save_path, n_experts, mp, expert_dtype):
    """
    将 HuggingFace 格式的 safetensors checkpoint 转换为自定义格式，支持模型并行切分和专家权重格式转换。

    主要处理流程：
    1. 分片加载 HF checkpoint 中的所有 safetensors 文件。
    2. 根据 mapping 表重命名权重，并去除 "model." 前缀等 HF 特有命名。
    3. 跳过 MTP（Multi-Token Prediction）相关的不必要权重。
    4. 对非专家权重按模型并行维度 mp 进行切分；对专家权重按专家索引分配到对应 rank。
    5. 对 MLA 的 wo_a 权重进行反量化（解压 scale 并转为 bfloat16）。
    6. 对 MoE 专家权重进行格式转换：若 expert_dtype="fp8" 则调用 cast_e2m1fn_to_e4m3fn
       转为 FP8；否则保留为 FP4 打包格式（view 为 float4_e2m1fn_x2）。
    7. 将每个 rank 的 state_dict 保存为独立的 safetensors 文件。
    8. 复制 tokenizer 相关文件到输出目录。

    参数:
        hf_ckpt_path (str): HF checkpoint 目录路径，包含 *.safetensors 和 tokenizer 文件。
        save_path (str): 转换后文件的保存目录。
        n_experts (int): 模型中专家总数（MoE 层）。
        mp (int): 模型并行数（Model Parallelism），即切分为多少个 rank。
        expert_dtype (str|None): 专家权重目标类型，"fp8" 表示转为 FP8，"fp4" 或 None 保留 FP4。

    返回:
        None
    """
    torch.set_num_threads(8)          # 限制 PyTorch CPU 线程数，避免多进程竞争
    n_local_experts = n_experts // mp  # 每个 rank 负责的专家数量
    state_dicts = [{} for _ in range(mp)]  # 为每个 MP rank 准备一个 state_dict

    # 遍历所有 safetensors 分片文件，逐个加载权重
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                param: torch.Tensor = f.get_tensor(name)

                # 去除 "model." 前缀（HF 习惯在权重名前加 model.）
                if name.startswith("model."):
                    name = name[len("model."):]

                # 跳过 MTP 模块的嵌入和输出头权重，这些在推理主路径中不需要
                if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
                    continue

                # 统一命名风格：HF 的 self_attn -> attn，mlp -> ffn
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                # HF 使用 weight_scale_inv 表示 scale，我们简化为 scale
                name = name.replace("weight_scale_inv", "scale")
                # e_score_correction_bias 是 MoE 路由的校正偏置，简化为 bias
                name = name.replace("e_score_correction_bias", "bias")

                # 提取关键字用于 mapping 查找。
                # 特殊名称（如 attn_sink、tie2eid、ape）直接取最后一段；其他取倒数第二段（通常是层内模块名）
                if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
                    key = name.split(".")[-1]
                else:
                    key = name.split(".")[-2]

                # 根据 mapping 表替换为自定义命名，并获取切分维度
                if key in mapping:
                    new_key, dim = mapping[key]
                else:
                    new_key, dim = key, None
                name = name.replace(key, new_key)

                # 将当前权重分配到各个 MP rank
                for i in range(mp):
                    new_param = param
                    # 专家权重：根据专家索引决定归属哪个 rank
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])  # 专家索引在名称中的位置
                        # 若当前专家不属于该 rank 的区间，则跳过
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    # 非专家权重：按指定维度进行模型并行切分
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        # narrow 切片后 contiguous 保证内存连续
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    # 对每个 rank 的后处理：反量化特殊权重、转换专家格式，然后保存
    for i in trange(mp):
        names = list(state_dicts[i].keys())
        for name in names:
            if name.endswith("wo_a.weight"):
                # wo_a 是 MLA 中的低秩分解权重，存储时与 scale 一起打包。
                # 这里将其解压：先 reshape 为 (rank, 128, rank, 128)，乘以对应 scale，
                # 再 flatten 并转为 bfloat16。
                weight = state_dicts[i][name]
                scale = state_dicts[i].pop(name.replace("weight", "scale"))
                weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float()
                state_dicts[i][name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
            elif "experts" in name and state_dicts[i][name].dtype == torch.int8:
                # 专家权重为 int8 类型，表示 FP4 打包格式（每字节2个 FP4）
                if expert_dtype == "fp8":
                    # 目标格式为 FP8：调用无损转换函数，同时替换 weight 和 scale
                    scale_name = name.replace("weight", "scale")
                    weight = state_dicts[i].pop(name)
                    scale = state_dicts[i].pop(scale_name)
                    state_dicts[i][name], state_dicts[i][scale_name] = cast_e2m1fn_to_e4m3fn(weight, scale)
                else:
                    # 保留 FP4：将 int8 view 为 PyTorch 的 float4_e2m1fn_x2 类型（2个 FP4 打包）
                    state_dicts[i][name] = state_dicts[i][name].view(torch.float4_e2m1fn_x2)
        # 保存当前 rank 的 checkpoint
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # 复制 tokenizer 配置文件到输出目录，保持推理时可直接使用
    for file in ["tokenizer.json", "tokenizer_config.json"]:
        old_file_path = os.path.join(hf_ckpt_path, file)
        new_file_path = os.path.join(save_path, file)
        if os.path.exists(old_file_path):
            shutil.copyfile(old_file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="HF checkpoint 转换为自定义格式工具")
    parser.add_argument("--hf-ckpt-path", type=str, required=True, help="HF checkpoint 目录路径")
    parser.add_argument("--save-path", type=str, required=True, help="转换后保存目录路径")
    parser.add_argument("--n-experts", type=int, required=True, help="专家总数")
    parser.add_argument("--model-parallel", type=int, required=True, help="模型并行数（MP）")
    parser.add_argument("--expert-dtype", type=str, choices=["fp8", "fp4"], required=False, default=None,
                        help="专家权重目标类型：fp8 转为 FP8 e4m3fn，fp4 或默认保留 FP4 打包格式")
    args = parser.parse_args()
    # 专家数必须能被模型并行数整除，确保每个 rank 分配相同数量的专家
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel, args.expert_dtype)
