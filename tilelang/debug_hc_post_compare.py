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

token_ids = [0, 128803, 33310, 2058, 128804]
embed = state_dict["embed.weight"]
x = embed[token_ids].unsqueeze(0).float()
hc_mult = 4
x_hc = x.unsqueeze(2).repeat(1, 1, hc_mult, 1)
x_flat = x_hc.flatten(2).float()
rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + 1e-6)

hc_fn = state_dict["layers.0.hc_attn_fn"].float()
hc_scale = state_dict["layers.0.hc_attn_scale"].float()
hc_base = state_dict["layers.0.hc_attn_base"].float()

mixes = torch.nn.functional.linear(x_flat, hc_fn) * rsqrt

pre = torch.sigmoid(mixes[:, :, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + 1e-6
post = 2 * torch.sigmoid(mixes[:, :, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult])

comb_raw = mixes[:, :, 2*hc_mult:] * hc_scale[2] + hc_base[2*hc_mult:]
comb_4d = comb_raw.view(1, 5, hc_mult, hc_mult)

for _ in range(20):
    comb_4d = comb_4d - torch.logsumexp(comb_4d, dim=-1, keepdim=True)
    comb_4d = comb_4d - torch.logsumexp(comb_4d, dim=-2, keepdim=True)
comb = torch.exp(comb_4d)

# 模拟 attn_out ≈ 0
attn_out = torch.zeros(1, 5, 4096)

# hc_post: y = post * attn_out + comb * residual
# 方法1: comb * residual (正确)
y1 = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * x_hc.unsqueeze(-2), dim=2)
# y1 shape: [1, 5, hc, dim]

# 方法2: comb^T * residual (错误)
comb_T = comb.transpose(-2, -1)
y2 = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.sum(comb_T.unsqueeze(-1) * x_hc.unsqueeze(-2), dim=2)

print("comb * residual (correct):")
print(f"  mean={y1.mean():.6e} min={y1.min():.6e} max={y1.max():.6e}")
print(f"  [0,0,:,:] mean={y1[0,0].mean():.6e} min={y1[0,0].min():.6e} max={y1[0,0].max():.6e}")

print("comb^T * residual (wrong):")
print(f"  mean={y2.mean():.6e} min={y2.min():.6e} max={y2.max():.6e}")
print(f"  [0,0,:,:] mean={y2[0,0].mean():.6e} min={y2[0,0].min():.6e} max={y2[0,0].max():.6e}")

# Rust 输出: after hc_post(attn) mean=1.102629e-3 min=-6.601562e-1 max=6.835938e-1
# 但 attn_out 不是 0，所以不能直接对比

# 对比 comb 矩阵
print(f"\ncomb[0,0] = {comb[0,0].tolist()}")
print(f"comb^T[0,0] = {comb_T[0,0].tolist()}")
