import torch
import sys
import json
import os

sys.path.insert(0, "/models/inference")
from safetensors import safe_open

model_dir = "/models"
index_file = os.path.join(model_dir, "model.safetensors.index.json")
state_dict = {}
with open(index_file) as f:
    index = json.load(f)
weight_map = index.get("weight_map", {})

needed = ["embed.weight"]
for k in weight_map:
    if k.startswith("layers.0.hc_attn"):
        needed.append(k)

files_needed = sorted(set(weight_map[k] for k in needed if k in weight_map))
for fname in files_needed:
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k in needed:
                state_dict[k] = f.get_tensor(k)

from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
prompt = "hello world"
encoded = tok.encode(prompt)
token_ids = encoded.ids

embed = state_dict["embed.weight"]
x = embed[token_ids].unsqueeze(0)
hc_mult = 4
x_hc = x.unsqueeze(2).repeat(1, 1, hc_mult, 1)
x_flat = x_hc.flatten(2).float()
rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + 1e-6)

hc_fn = state_dict["layers.0.hc_attn_fn"].float()
hc_scale = state_dict["layers.0.hc_attn_scale"].float()
hc_base = state_dict["layers.0.hc_attn_base"].float()

mixes = torch.nn.functional.linear(x_flat, hc_fn) * rsqrt
print(f"mixes[0, 0, :4] (pre part): {mixes[0, 0, :4].tolist()}")
print(f"mixes[0, 1, :4] (pre part): {mixes[0, 1, :4].tolist()}")
print(f"mixes[0, 0, 4:8] (post part): {mixes[0, 0, 4:8].tolist()}")
print(f"mixes[0, 0, 8:] (comb part): {mixes[0, 0, 8:].tolist()}")

print(f"\nhc_attn_scale: {hc_scale.tolist()}")
print(f"hc_attn_base[:4] (pre base): {hc_base[:4].tolist()}")
print(f"hc_attn_base[4:8] (post base): {hc_base[4:8].tolist()}")
print(f"hc_attn_base[8:] (comb base): {hc_base[8:].tolist()}")

pre_input = mixes[:, :, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
print(f"\npre_input (before sigmoid): {pre_input[0, 0, :].tolist()}")
print(f"pre_input (before sigmoid) pos 1: {pre_input[0, 1, :].tolist()}")

pre = torch.sigmoid(pre_input) + 1e-6
print(f"pre: {pre[0, 0, :].tolist()}")
print(f"pre pos 1: {pre[0, 1, :].tolist()}")

# 对比 Rust 中的 mixes 值
# Rust: hc_pre_mixes shape=[5, 24] mean=-7.820128e0 min=-4.419124e2 max=3.547140e1
# Python: mixes shape=[1, 2, 24] mean=-4.821727e+01 min=-5.782585e+02 max=3.897438e+01
# 注意: Rust 的 seqlen=5 (包含 chat template tokens), Python seqlen=2 (仅 hello world)
print(f"\nPython mixes stats: mean={mixes.mean():.6e} min={mixes.min():.6e} max={mixes.max():.6e}")
