import torch
import sys
import json
import os
import time

sys.path.insert(0, "/models/inference")

model_dir = "/models"
config_path = os.path.join(model_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

from model import ModelArgs, Transformer
known_keys = set(ModelArgs.__dataclass_fields__.keys()) if hasattr(ModelArgs, '__dataclass_fields__') else set()
filtered = {k: v for k, v in config.items() if k in known_keys} if known_keys else config
args = ModelArgs(**filtered)
args.max_batch_size = 1
args.max_seq_len = 128

model = Transformer(args)

from safetensors import safe_open
index_file = os.path.join(model_dir, "model.safetensors.index.json")
with open(index_file) as f:
    index = json.load(f)
weight_map = index.get("weight_map", {})

state_dict = {}
files_needed = sorted(set(weight_map.values()))
for fname in files_needed:
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

size_mismatch = [k for k in missing if k in state_dict]
if size_mismatch:
    print(f"Size mismatch keys: {len(size_mismatch)}")
    for k in size_mismatch[:3]:
        print(f"  {k}: checkpoint={state_dict[k].shape}, model={dict(model.named_parameters())[k].shape if k in dict(model.named_parameters()) else 'N/A'}")

del state_dict
model.eval().cuda().to(torch.bfloat16)
print("Model loaded")

from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

prompt = "hello world"
encoded = tok.encode(prompt)
token_ids = torch.tensor([encoded.ids], dtype=torch.long).cuda()
print(f"token_ids: {encoded.ids}")

with torch.no_grad():
    h = model.embed(token_ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    print(f"after embed+hc_expand: shape={h.shape} mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")

    for i, layer in enumerate(model.layers):
        residual = h
        x, post, comb = layer.hc_pre(h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        if i == 0:
            print(f"[L{i:02d}] after hc_pre(attn): shape={x.shape} mean={x.float().mean():.6e} min={x.float().min():.6e} max={x.float().max():.6e}")

        x = layer.attn_norm(x)
        x = layer.attn(x, 0)
        if i == 0:
            print(f"[L{i:02d}] after attn: shape={x.shape} mean={x.float().mean():.6e} min={x.float().min():.6e} max={x.float().max():.6e}")

        h = layer.hc_post(x, residual, post, comb)
        if i == 0:
            print(f"[L{i:02d}] after hc_post(attn): shape={h.shape} mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")

        residual = h
        x, post, comb = layer.hc_pre(h, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
        x = layer.ffn_norm(x)
        x = layer.ffn(x, token_ids)
        h = layer.hc_post(x, residual, post, comb)

        if i < 3 or i >= 40:
            print(f"[L{i:02d}] after layer: shape={h.shape} mean={h.float().mean():.6e} min={h.float().min():.6e} max={h.float().max():.6e}")

    logits = model.head(h, model.hc_head_fn, model.hc_head_scale, model.hc_head_base, model.norm)
    print(f"logits: shape={logits.shape} mean={logits.float().mean():.6e} min={logits.float().min():.6e} max={logits.float().max():.6e}")

    topk = logits[0, -1].float().topk(10)
    print(f"top-10 tokens: {list(zip(topk.indices.tolist(), [f'{v:.4f}' for v in topk.values.tolist()]))}")

    next_token = logits[0, -1].argmax().item()
    print(f"next token: {next_token} = {tok.decode([next_token])}")
