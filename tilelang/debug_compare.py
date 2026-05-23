"""对比官方 Python 推理中间值，定位 Rust 推理数值发散根因"""
import sys, os, json
sys.path.insert(0, "/models/inference")
sys.path.insert(0, "/models/encoding")
import torch
import numpy as np

from model import Transformer, ModelArgs
from encoding_dsv4 import encode_messages
from safetensors import safe_open

def main():
    with open("/models/inference/config.json") as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = 1

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    with torch.device("cuda"):
        model = Transformer(args)

    print("Loading weights...")
    index_file = "/models/model.safetensors.index.json"
    state_dict = {}
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    files_needed = sorted(set(weight_map.values()))
    for fname in files_needed:
        fpath = os.path.join("/models", fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"  first 10 missing: {missing[:10]}")
    del state_dict
    model.eval()
    print("Model loaded.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/models")

    prompt = "hello,world!"
    messages = [{"role": "user", "content": prompt}]
    chat_text = encode_messages(messages, thinking_mode="chat")
    prompt_tokens = tokenizer.encode(chat_text)
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Prompt text: {chat_text!r}")

    with torch.no_grad():
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
        bsz, seqlen = tokens.shape

        h = model.embed(tokens)
        print(f"\n[embed] shape={h.shape} mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")

        hc = model.hc_mult
        h = h.unsqueeze(2).repeat(1, 1, hc, 1)
        print(f"[hc_expand] shape={h.shape} mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")

        for i, layer in enumerate(model.layers):
            residual = h
            x_attn, post, comb = layer.hc_pre(h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
            if i < 3 or i >= 40:
                print(f"\n[L{i:02d}] after hc_pre(attn) mean={x_attn.float().mean():.6e} min={x_attn.float().min():.6e} max={x_attn.float().max():.6e}")
                print(f"[L{i:02d}] post(attn) mean={post.float().mean():.6e} min={post.float().min():.6e} max={post.float().max():.6e}")
                print(f"[L{i:02d}] comb(attn) mean={comb.float().mean():.6e} min={comb.float().min():.6e} max={comb.float().max():.6e}")

            x_norm = layer.attn_norm(x_attn)
            attn_out = layer.attn(x_norm, 0)

            if i < 3 or i >= 40:
                print(f"[L{i:02d}] attn_out mean={attn_out.float().mean():.6e} min={attn_out.float().min():.6e} max={attn_out.float().max():.6e}")

            h = layer.hc_post(attn_out, residual, post, comb)

            if i < 3 or i >= 40:
                print(f"[L{i:02d}] after hc_post(attn) mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")

            residual = h
            x_ffn, post, comb = layer.hc_pre(h, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
            x_norm = layer.ffn_norm(x_ffn)
            ffn_out = layer.ffn(x_norm, tokens)
            h = layer.hc_post(ffn_out, residual, post, comb)

            if i < 3 or i >= 40:
                print(f"[L{i:02d}] after hc_post(ffn) mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")
            elif i % 10 == 0:
                print(f"[L{i:02d}] after layer mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")

        h_reduced = model.head.hc_head(h, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
        print(f"\n[hc_head] mean={h_reduced.float().mean():.6e} min={h_reduced.float().min():.6e} max={h_reduced.float().max():.6e}")

        h_normed = model.norm(h_reduced)
        print(f"[norm] mean={h_normed.float().mean():.6e} min={h_normed.float().min():.6e} max={h_normed.float().max():.6e}")

        logits = model.head.get_logits(h_normed)
        print(f"[logits] mean={logits.mean():.6e} min={logits.min():.6e} max={logits.max():.6e}")

        topk = torch.topk(logits[0], 10)
        print(f"[top-10]: {list(zip(topk.indices.cpu().tolist(), [f'{v:.4f}' for v in topk.values.cpu().tolist()]))}")

        next_token = logits[0].argmax().item()
        decoded = tokenizer.decode([next_token])
        print(f"\nNext token: {next_token} = {decoded!r}")

if __name__ == "__main__":
    main()
