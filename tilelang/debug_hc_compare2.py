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
    if k.startswith("layers.0.") or k.startswith("layers.1."):
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
print(f"token_ids: {token_ids}")

embed = state_dict["embed.weight"]
x = embed[token_ids].unsqueeze(0)  # [1, seqlen, 4096]
print(f"embed output: shape={x.shape} mean={x.float().mean():.6e} min={x.float().min():.6e} max={x.float().max():.6e}")

hc_mult = 4
x_hc = x.unsqueeze(2).repeat(1, 1, hc_mult, 1)  # [1, seqlen, 4, 4096]
print(f"x_hc: shape={x_hc.shape} mean={x_hc.float().mean():.6e} min={x_hc.float().min():.6e} max={x_hc.float().max():.6e}")

x_flat = x_hc.flatten(2).float()  # [1, seqlen, 16384]
rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + 1e-6)
print(f"rsqrt: shape={rsqrt.shape} mean={rsqrt.mean():.6e} min={rsqrt.min():.6e} max={rsqrt.max():.6e}")

hc_fn = state_dict["layers.0.hc_attn_fn"].float()
hc_scale = state_dict["layers.0.hc_attn_scale"].float()
hc_base = state_dict["layers.0.hc_attn_base"].float()
print(f"hc_attn_scale: {hc_scale.tolist()}")
print(f"hc_attn_base: mean={hc_base.mean():.6e} min={hc_base.min():.6e} max={hc_base.max():.6e}")

mixes = torch.nn.functional.linear(x_flat, hc_fn) * rsqrt
print(f"mixes: shape={mixes.shape} mean={mixes.mean():.6e} min={mixes.min():.6e} max={mixes.max():.6e}")

pre = torch.sigmoid(mixes[:, :, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + 1e-6
post = 2 * torch.sigmoid(mixes[:, :, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult])
print(f"pre: shape={pre.shape} mean={pre.mean():.6e} min={pre.min():.6e} max={pre.max():.6e}")
print(f"post: shape={post.shape} mean={post.mean():.6e} min={post.min():.6e} max={post.max():.6e}")

y = torch.sum(pre.unsqueeze(-1) * x_hc, dim=2)
print(f"hc_pre output: shape={y.shape} mean={y.mean():.6e} min={y.min():.6e} max={y.max():.6e}")

# hc_post: y = post * attn_out + comb @ residual
# 模拟 attn_out ≈ 0 (简化)
attn_out_sim = torch.zeros_like(y)
residual = x_hc
comb_raw = mixes[:, :, 2*hc_mult:] * hc_scale[2] + hc_base[2*hc_mult:]
comb_4d = comb_raw.view(1, x.size(1), hc_mult, hc_mult)

# Sinkhorn normalization
for _ in range(20):
    comb_4d = comb_4d - torch.logsumexp(comb_4d, dim=-1, keepdim=True)
    comb_4d = comb_4d - torch.logsumexp(comb_4d, dim=-2, keepdim=True)
comb = torch.exp(comb_4d)
print(f"comb (after sinkhorn): shape={comb.shape} mean={comb.mean():.6e} min={comb.min():.6e} max={comb.max():.6e}")

hc_post_out = post.unsqueeze(-1) * attn_out_sim.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
print(f"hc_post output (attn_out=0): shape={hc_post_out.shape} mean={hc_post_out.mean():.6e} min={hc_post_out.min():.6e} max={hc_post_out.max():.6e}")
