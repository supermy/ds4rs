from safetensors import safe_open
import os

for i in range(1, 47):
    path = f"/models/model-{i:05d}-of-00046.safetensors"
    if not os.path.exists(path):
        continue
    try:
        f = safe_open(path, framework="pt")
        keys = [k for k in f.keys() if "wo_b" in k and "0." in k]
        if keys:
            print(f"File {i}:")
            for k in sorted(keys):
                t = f.get_tensor(k)
                print(f"  {k}: shape={t.shape}, dtype={t.dtype}")
                if "scale" in k:
                    print(f"    sample: {t.flat[:4].tolist()}")
                    print(f"    stats: min={t.float().min():.6e}, max={t.float().max():.6e}, mean={t.float().mean():.6e}")
    except Exception as e:
        print(f"File {i}: error ({e})")
