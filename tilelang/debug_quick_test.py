"""快速对比 Python 官方推理 - 只做 embed + L0 + L1"""
import os, sys, json, time
import torch

sys.path.insert(0, "/models/inference")

from transformers import AutoTokenizer

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

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    chat_text = "<｜begin▁of▁sentence｜><｜User｜>hello,world<｜Assistant｜>"
    prompt_tokens = tokenizer.encode(chat_text, add_special_tokens=False)
    print(f"[tokenizer] prompt_tokens={prompt_tokens}")
    for i, tid in enumerate(prompt_tokens):
        print(f"  token {i}: id={tid} text={repr(tokenizer.decode([tid]))}")

    # 直接用官方 generate.py 运行完整推理
    from generate import Generate
    from model import ModelArgs

    with open(CONFIG) as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = 1
    args.max_seq_len = 4096
    args.compress_ratios = [0] * args.n_layers
    args.n_mtp_layers = 0

    print(f"[config] n_layers={args.n_layers} compress_ratios={args.compress_ratios[:5]}...")

    gen = Generate(args, CKPT, device="cuda")

    # 生成
    output = gen.generate(prompt_tokens, max_new_tokens=20, temperature=0.0)
    print(f"[generate] output_tokens={output}")
    print(f"[generate] output_text={repr(tokenizer.decode(output))}")

if __name__ == "__main__":
    main()
