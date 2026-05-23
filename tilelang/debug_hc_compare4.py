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

# 使用与 Rust 相同的 token IDs
token_ids = [0, 128803, 33310, 2058, 128804]
embed = state_dict["embed.weight"]
x = embed[token_ids].unsqueeze(0).float()  # [1, 5, 4096]
print(f"embed output: shape={x.shape} mean={x.mean():.6e} min={x.min():.6e} max={x.max():.6e}")

hc_mult = 4
x_hc = x.unsqueeze(2).repeat(1, 1, hc_mult, 1)  # [1, 5, 4, 4096]
x_flat = x_hc.flatten(2).float()  # [1, 5, 16384]
rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + 1e-6)
print(f"rsqrt: shape={rsqrt.shape} mean={rsqrt.mean():.6e} min={rsqrt.min():.6e} max={rsqrt.max():.6e}")

hc_fn = state_dict["layers.0.hc_attn_fn"].float()
hc_scale = state_dict["layers.0.hc_attn_scale"].float()
hc_base = state_dict["layers.0.hc_attn_base"].float()

mixes = torch.nn.functional.linear(x_flat, hc_fn) * rsqrt
print(f"mixes: shape={mixes.shape} mean={mixes.mean():.6e} min={mixes.min():.6e} max={mixes.max():.6e}")

pre = torch.sigmoid(mixes[:, :, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + 1e-6
post = 2 * torch.sigmoid(mixes[:, :, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult])
print(f"pre: shape={pre.shape} mean={pre.mean():.6e} min={pre.min():.6e} max={pre.max():.6e}")
print(f"post: shape={post.shape} mean={post.mean():.6e} min={post.min():.6e} max={post.max():.6e}")

y = torch.sum(pre.unsqueeze(-1) * x_hc, dim=2)
print(f"hc_pre output: shape={y.shape} mean={y.mean():.6e} min={y.min():.6e} max={y.max():.6e}")

# 打印每个 token 位置的 pre 值
for t in range(5):
    print(f"  token {t} pre: {pre[0, t, :].tolist()}")
    print(f"  token {t} post: {post[0, t, :].tolist()}")
