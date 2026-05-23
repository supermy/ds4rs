import torch
import sys
import json
import os

sys.path.insert(0, "/models/inference")
from model import Transformer

model_dir = "/models"
with open(os.path.join(model_dir, "config.json")) as f:
    config = json.load(f)

from safetensors import safe_open

index_file = os.path.join(model_dir, "model.safetensors.index.json")
state_dict = {}
with open(index_file) as f:
    index = json.load(f)
weight_map = index.get("weight_map", {})
files_needed = sorted(set(weight_map.values()))

l0_keys = [k for k in weight_map if k.startswith("layers.0.") or k in ["embed.weight", "hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight"]]
files_for_l0 = sorted(set(weight_map[k] for k in l0_keys))

for fname in files_for_l0:
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k in l0_keys or k.startswith("layers.0.hc_attn"):
                state_dict[k] = f.get_tensor(k)

hc_fn_key = "layers.0.hc_attn_fn"
if hc_fn_key in state_dict:
    t = state_dict[hc_fn_key]
    print(f"hc_attn_fn: shape={t.shape} dtype={t.dtype} mean={t.float().mean():.6e} min={t.float().min():.6e} max={t.float().max():.6e}")

hc_scale_key = "layers.0.hc_attn_scale"
if hc_scale_key in state_dict:
    t = state_dict[hc_scale_key]
    print(f"hc_attn_scale: shape={t.shape} dtype={t.dtype} values={t.tolist()}")

hc_base_key = "layers.0.hc_attn_base"
if hc_base_key in state_dict:
    t = state_dict[hc_base_key]
    print(f"hc_attn_base: shape={t.shape} dtype={t.dtype} mean={t.float().mean():.6e} min={t.float().min():.6e} max={t.float().max():.6e}")

embed_key = "embed.weight"
if embed_key in state_dict:
    embed = state_dict[embed_key]
    torch.manual_seed(42)
    x = torch.randn(1, 5, 4, 4096) * 0.01
    x_flat = x.flatten(2).float()

    hc_fn = state_dict[hc_fn_key].float()
    hc_scale = state_dict[hc_scale_key].float()
    hc_base = state_dict[hc_base_key].float()

    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + 1e-6)
    print(f"rsqrt: shape={rsqrt.shape} mean={rsqrt.mean():.6e} min={rsqrt.min():.6e} max={rsqrt.max():.6e}")

    mixes = torch.nn.functional.linear(x_flat, hc_fn) * rsqrt
    print(f"mixes: shape={mixes.shape} mean={mixes.mean():.6e} min={mixes.min():.6e} max={mixes.max():.6e}")

    x_normed = x_flat * rsqrt
    mixes_wrong = torch.nn.functional.linear(x_normed, hc_fn)
    print(f"mixes_wrong (norm first): shape={mixes_wrong.shape} mean={mixes_wrong.mean():.6e} min={mixes_wrong.min():.6e} max={mixes_wrong.max():.6e}")

    hc = 4
    pre = torch.sigmoid(mixes[:, :, :hc] * hc_scale[0] + hc_base[:hc]) + 1e-6
    print(f"pre: shape={pre.shape} mean={pre.mean():.6e} min={pre.min():.6e} max={pre.max():.6e}")

    y = torch.sum(pre.unsqueeze(-1) * x, dim=2)
    print(f"y (hc_pre output): shape={y.shape} mean={y.mean():.6e} min={y.min():.6e} max={y.max():.6e}")
