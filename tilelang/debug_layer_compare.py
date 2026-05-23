"""Python 官方推理 - 只做 embed + L0 attention，避免 OOM"""
import os, sys, json
import torch

sys.path.insert(0, "/models/inference")
from model import Transformer, ModelArgs

from transformers import AutoTokenizer
from safetensors.torch import load_model

CKPT = "/models"
CONFIG = "/models/inference/config.json"

def ts(name, t):
    if t is None: return
    t = t.detach().float()
    print(f"  [PY] {name}: shape={list(t.shape)} mean={t.mean().item():.6f} min={t.min().item():.6f} max={t.max().item():.6f}")

@torch.inference_mode()
def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.manual_seed(33377335)

    with open(CONFIG) as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = 1
    args.max_seq_len = 128
    args.compress_ratios = [0] * args.n_layers
    args.n_mtp_layers = 0
    print(f"[config] n_layers={args.n_layers}")

    # CPU 上创建模型
    with torch.device("cpu"):
        model = Transformer(args)
    load_model(model, os.path.join(CKPT, "model0-mp1.safetensors"), strict=False)
    print(f"[load] model loaded")

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    import sys
    sys.path.insert(0, '/models/encoding')
    from encoding_dsv4 import encode_messages
    messages = [{'role': 'user', 'content': 'hello,world'}]
    prompt = encode_messages(messages, thinking_mode='chat')
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"[tokenizer] prompt_tokens={prompt_tokens}")

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")

    # Embed on GPU
    model.embed = model.embed.cuda()
    h = model.embed(input_ids)
    ts("embed_output", h)
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    ts("hc_expand_output", h)
    model.embed = model.embed.cpu()
    torch.cuda.empty_cache()

    # L0 attention only
    layer = model.layers[0]
    layer.attn = layer.attn.cuda()
    layer.attn_norm = layer.attn_norm.cuda()
    layer.hc_attn_fn = layer.hc_attn_fn.cuda()
    layer.hc_attn_scale = layer.hc_attn_scale.cuda()
    layer.hc_attn_base = layer.hc_attn_base.cuda()
    layer.hc_ffn_fn = layer.hc_ffn_fn.cuda()
    layer.hc_ffn_scale = layer.hc_ffn_scale.cuda()
    layer.hc_ffn_base = layer.hc_ffn_base.cuda()

    residual = h
    x, post, comb = layer.hc_pre(h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
    ts("L0_hc_pre_y", x)
    ts("L0_hc_pre_post", post)
    ts("L0_hc_pre_comb", comb)

    x = layer.attn_norm(x)
    ts("L0_attn_norm_out", x)

    x = layer.attn(x, 0)
    ts("L0_attn_out", x)

    h = layer.hc_post(x, residual, post, comb)
    ts("L0_after_hc_post_attn", h)

    print("\n--- Python L0 attention done ---")

if __name__ == "__main__":
    main()
