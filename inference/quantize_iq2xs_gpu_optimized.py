"""GPU 加速的 IQ2_XS 量化（GGUF 官方算法对齐版）

核心修复（对齐 iq2_xs.h quantize_row_iq2_xs_impl）：
1. 量化搜索使用逻辑网格 {1,3,5} + kmap 查表（替代物理网格暴力搜索）
2. 反量化使用物理网格 {8,25,43}（与官方 dequantize 一致）
3. 19 候选值迭代 scale 优化（加权 sumqx²/sumq2 最大化，权重 = sqrt(sigma2 + x²)）
4. 符号奇偶性：翻转 weight*x² 最小元素使负数个数为偶数
5. off-grid 修正 + 负 scale 处理
6. 防止内存溢出（分批处理、及时释放）
7. GROUP_MAX_EPS = 1e-15（对齐 C 代码）

用法：
  python quantize_iq2xs_gpu_optimized.py --input /models --output /workspace/models_iq2xs.safetensors
  python quantize_iq2xs_gpu_optimized.py --test-shard /models/model-00002-of-00046.safetensors
"""
import os
import sys
import json
import struct
import time as _time  # 保留用于可能的扩展
import gc
from argparse import ArgumentParser
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm
from safetensors.torch import save_file

import numpy as np

# ============================================================================
# IQ2_XS 常量（从 csrc/iq2_xs.cuh 内联，不再依赖 convert_fp4_to_iq2xs_gguf_v2）
# ============================================================================

# 超级块大小 — 每个 block_iq2_xs 编码 256 个元素
QK_K = 256

# 物理网格查找表 — 512 项, 每项 8 字节 (8 个 int8)
# 来源: csrc/iq2_xs.cuh iq2xs_grid[512]
# 每个 uint64 小端序展开为 8 字节，值域 {8, 25, 43}（对应 L=0,1,2 的 2*L+1=1,3,5 乘以 8）
IQ2_XS_GRID = np.array([
    8, 8, 8, 8, 8, 8, 8, 8,
    43, 8, 8, 8, 8, 8, 8, 8,
    25, 25, 8, 8, 8, 8, 8, 8,
    8, 43, 8, 8, 8, 8, 8, 8,
    43, 43, 8, 8, 8, 8, 8, 8,
    25, 8, 25, 8, 8, 8, 8, 8,
    8, 25, 25, 8, 8, 8, 8, 8,
    43, 25, 25, 8, 8, 8, 8, 8,
    25, 43, 25, 8, 8, 8, 8, 8,
    8, 8, 43, 8, 8, 8, 8, 8,
    43, 8, 43, 8, 8, 8, 8, 8,
    25, 25, 43, 8, 8, 8, 8, 8,
    8, 43, 43, 8, 8, 8, 8, 8,
    25, 8, 8, 25, 8, 8, 8, 8,
    8, 25, 8, 25, 8, 8, 8, 8,
    43, 25, 8, 25, 8, 8, 8, 8,
    25, 43, 8, 25, 8, 8, 8, 8,
    8, 8, 25, 25, 8, 8, 8, 8,
    43, 8, 25, 25, 8, 8, 8, 8,
    25, 25, 25, 25, 8, 8, 8, 8,
    8, 43, 25, 25, 8, 8, 8, 8,
    25, 8, 43, 25, 8, 8, 8, 8,
    8, 25, 43, 25, 8, 8, 8, 8,
    8, 8, 8, 43, 8, 8, 8, 8,
    43, 8, 8, 43, 8, 8, 8, 8,
    25, 25, 8, 43, 8, 8, 8, 8,
    8, 43, 8, 43, 8, 8, 8, 8,
    25, 8, 25, 43, 8, 8, 8, 8,
    8, 25, 25, 43, 8, 8, 8, 8,
    25, 43, 25, 43, 8, 8, 8, 8,
    8, 8, 43, 43, 8, 8, 8, 8,
    25, 8, 8, 8, 25, 8, 8, 8,
    8, 25, 8, 8, 25, 8, 8, 8,
    43, 25, 8, 8, 25, 8, 8, 8,
    25, 43, 8, 8, 25, 8, 8, 8,
    8, 8, 25, 8, 25, 8, 8, 8,
    43, 8, 25, 8, 25, 8, 8, 8,
    25, 25, 25, 8, 25, 8, 8, 8,
    8, 43, 25, 8, 25, 8, 8, 8,
    43, 43, 25, 8, 25, 8, 8, 8,
    25, 8, 43, 8, 25, 8, 8, 8,
    8, 25, 43, 8, 25, 8, 8, 8,
    8, 8, 8, 25, 25, 8, 8, 8,
    43, 8, 8, 25, 25, 8, 8, 8,
    25, 25, 8, 25, 25, 8, 8, 8,
    8, 43, 8, 25, 25, 8, 8, 8,
    25, 8, 25, 25, 25, 8, 8, 8,
    8, 25, 25, 25, 25, 8, 8, 8,
    8, 8, 43, 25, 25, 8, 8, 8,
    8, 43, 43, 25, 25, 8, 8, 8,
    25, 8, 8, 43, 25, 8, 8, 8,
    8, 25, 8, 43, 25, 8, 8, 8,
    8, 8, 25, 43, 25, 8, 8, 8,
    8, 8, 8, 8, 43, 8, 8, 8,
    43, 8, 8, 8, 43, 8, 8, 8,
    25, 25, 8, 8, 43, 8, 8, 8,
    8, 43, 8, 8, 43, 8, 8, 8,
    25, 8, 25, 8, 43, 8, 8, 8,
    8, 25, 25, 8, 43, 8, 8, 8,
    8, 8, 43, 8, 43, 8, 8, 8,
    25, 8, 8, 25, 43, 8, 8, 8,
    8, 25, 8, 25, 43, 8, 8, 8,
    8, 8, 25, 25, 43, 8, 8, 8,
    25, 25, 25, 25, 43, 8, 8, 8,
    8, 8, 8, 43, 43, 8, 8, 8,
    43, 43, 8, 43, 43, 8, 8, 8,
    25, 8, 8, 8, 8, 25, 8, 8,
    8, 25, 8, 8, 8, 25, 8, 8,
    43, 25, 8, 8, 8, 25, 8, 8,
    25, 43, 8, 8, 8, 25, 8, 8,
    8, 8, 25, 8, 8, 25, 8, 8,
    43, 8, 25, 8, 8, 25, 8, 8,
    25, 25, 25, 8, 8, 25, 8, 8,
    8, 43, 25, 8, 8, 25, 8, 8,
    25, 8, 43, 8, 8, 25, 8, 8,
    8, 25, 43, 8, 8, 25, 8, 8,
    8, 8, 8, 25, 8, 25, 8, 8,
    43, 8, 8, 25, 8, 25, 8, 8,
    25, 25, 8, 25, 8, 25, 8, 8,
    8, 43, 8, 25, 8, 25, 8, 8,
    25, 8, 25, 25, 8, 25, 8, 8,
    8, 25, 25, 25, 8, 25, 8, 8,
    43, 25, 25, 25, 8, 25, 8, 8,
    8, 8, 43, 25, 8, 25, 8, 8,
    25, 8, 8, 43, 8, 25, 8, 8,
    8, 25, 8, 43, 8, 25, 8, 8,
    8, 8, 25, 43, 8, 25, 8, 8,
    8, 8, 8, 8, 25, 25, 8, 8,
    43, 8, 8, 8, 25, 25, 8, 8,
    25, 25, 8, 8, 25, 25, 8, 8,
    8, 43, 8, 8, 25, 25, 8, 8,
    25, 8, 25, 8, 25, 25, 8, 8,
    8, 25, 25, 8, 25, 25, 8, 8,
    8, 8, 43, 8, 25, 25, 8, 8,
    25, 8, 8, 25, 25, 25, 8, 8,
    8, 25, 8, 25, 25, 25, 8, 8,
    8, 8, 25, 25, 25, 25, 8, 8,
    25, 8, 43, 25, 25, 25, 8, 8,
    8, 8, 8, 43, 25, 25, 8, 8,
    25, 8, 8, 8, 43, 25, 8, 8,
    8, 25, 8, 8, 43, 25, 8, 8,
    8, 8, 25, 8, 43, 25, 8, 8,
    43, 25, 43, 8, 43, 25, 8, 8,
    8, 8, 8, 25, 43, 25, 8, 8,
    43, 8, 8, 25, 43, 25, 8, 8,
    8, 25, 8, 43, 43, 25, 8, 8,
    8, 8, 8, 8, 8, 43, 8, 8,
    43, 8, 8, 8, 8, 43, 8, 8,
    25, 25, 8, 8, 8, 43, 8, 8,
    8, 43, 8, 8, 8, 43, 8, 8,
    43, 43, 8, 8, 8, 43, 8, 8,
    25, 8, 25, 8, 8, 43, 8, 8,
    8, 25, 25, 8, 8, 43, 8, 8,
    8, 8, 43, 8, 8, 43, 8, 8,
    25, 25, 43, 8, 8, 43, 8, 8,
    25, 8, 8, 25, 8, 43, 8, 8,
    8, 25, 8, 25, 8, 43, 8, 8,
    8, 8, 25, 25, 8, 43, 8, 8,
    8, 43, 25, 25, 8, 43, 8, 8,
    8, 8, 8, 43, 8, 43, 8, 8,
    8, 8, 43, 43, 8, 43, 8, 8,
    43, 43, 43, 43, 8, 43, 8, 8,
    25, 8, 8, 8, 25, 43, 8, 8,
    8, 25, 8, 8, 25, 43, 8, 8,
    8, 8, 25, 8, 25, 43, 8, 8,
    8, 8, 8, 25, 25, 43, 8, 8,
    25, 8, 8, 43, 25, 43, 8, 8,
    25, 43, 8, 43, 25, 43, 8, 8,
    8, 8, 8, 8, 43, 43, 8, 8,
    8, 8, 43, 8, 43, 43, 8, 8,
    8, 43, 43, 8, 43, 43, 8, 8,
    43, 25, 25, 43, 43, 43, 8, 8,
    8, 8, 43, 43, 43, 43, 8, 8,
    25, 8, 8, 8, 8, 8, 25, 8,
    8, 25, 8, 8, 8, 8, 25, 8,
    43, 25, 8, 8, 8, 8, 25, 8,
    25, 43, 8, 8, 8, 8, 25, 8,
    8, 8, 25, 8, 8, 8, 25, 8,
    43, 8, 25, 8, 8, 8, 25, 8,
    25, 25, 25, 8, 8, 8, 25, 8,
    8, 43, 25, 8, 8, 8, 25, 8,
    25, 8, 43, 8, 8, 8, 25, 8,
    8, 25, 43, 8, 8, 8, 25, 8,
    8, 8, 8, 25, 8, 8, 25, 8,
    43, 8, 8, 25, 8, 8, 25, 8,
    25, 25, 8, 25, 8, 8, 25, 8,
    8, 43, 8, 25, 8, 8, 25, 8,
    25, 8, 25, 25, 8, 8, 25, 8,
    8, 25, 25, 25, 8, 8, 25, 8,
    8, 8, 43, 25, 8, 8, 25, 8,
    43, 43, 43, 25, 8, 8, 25, 8,
    25, 8, 8, 43, 8, 8, 25, 8,
    8, 25, 8, 43, 8, 8, 25, 8,
    8, 8, 25, 43, 8, 8, 25, 8,
    8, 8, 8, 8, 25, 8, 25, 8,
    43, 8, 8, 8, 25, 8, 25, 8,
    25, 25, 8, 8, 25, 8, 25, 8,
    8, 43, 8, 8, 25, 8, 25, 8,
    25, 8, 25, 8, 25, 8, 25, 8,
    8, 25, 25, 8, 25, 8, 25, 8,
    8, 8, 43, 8, 25, 8, 25, 8,
    25, 8, 8, 25, 25, 8, 25, 8,
    8, 25, 8, 25, 25, 8, 25, 8,
    8, 8, 25, 25, 25, 8, 25, 8,
    8, 8, 8, 43, 25, 8, 25, 8,
    8, 25, 25, 43, 25, 8, 25, 8,
    43, 25, 25, 43, 25, 8, 25, 8,
    25, 8, 8, 8, 43, 8, 25, 8,
    8, 25, 8, 8, 43, 8, 25, 8,
    43, 25, 8, 8, 43, 8, 25, 8,
    8, 8, 25, 8, 43, 8, 25, 8,
    8, 8, 8, 25, 43, 8, 25, 8,
    8, 8, 43, 25, 43, 8, 25, 8,
    8, 8, 8, 8, 8, 25, 25, 8,
    43, 8, 8, 8, 8, 25, 25, 8,
    25, 25, 8, 8, 8, 25, 25, 8,
    8, 43, 8, 8, 8, 25, 25, 8,
    25, 8, 25, 8, 8, 25, 25, 8,
    8, 25, 25, 8, 8, 25, 25, 8,
    8, 8, 43, 8, 8, 25, 25, 8,
    25, 8, 8, 25, 8, 25, 25, 8,
    8, 25, 8, 25, 8, 25, 25, 8,
    25, 43, 8, 25, 8, 25, 25, 8,
    8, 8, 25, 25, 8, 25, 25, 8,
    8, 25, 43, 25, 8, 25, 25, 8,
    8, 8, 8, 43, 8, 25, 25, 8,
    25, 8, 8, 8, 25, 25, 25, 8,
    8, 25, 8, 8, 25, 25, 25, 8,
    8, 8, 25, 8, 25, 25, 25, 8,
    8, 8, 8, 25, 25, 25, 25, 8,
    8, 8, 8, 8, 43, 25, 25, 8,
    8, 25, 25, 8, 43, 25, 25, 8,
    25, 43, 8, 25, 43, 25, 25, 8,
    25, 8, 8, 8, 8, 43, 25, 8,
    8, 25, 8, 8, 8, 43, 25, 8,
    8, 8, 25, 8, 8, 43, 25, 8,
    43, 8, 25, 8, 8, 43, 25, 8,
    8, 8, 8, 25, 8, 43, 25, 8,
    8, 25, 25, 25, 8, 43, 25, 8,
    43, 25, 8, 43, 8, 43, 25, 8,
    8, 8, 8, 8, 25, 43, 25, 8,
    25, 25, 8, 8, 25, 43, 25, 8,
    43, 25, 43, 25, 25, 43, 25, 8,
    25, 8, 25, 25, 43, 43, 25, 8,
    25, 43, 43, 43, 43, 43, 25, 8,
    8, 8, 8, 8, 8, 8, 43, 8,
    43, 8, 8, 8, 8, 8, 43, 8,
    25, 25, 8, 8, 8, 8, 43, 8,
    8, 43, 8, 8, 8, 8, 43, 8,
    43, 43, 8, 8, 8, 8, 43, 8,
    25, 8, 25, 8, 8, 8, 43, 8,
    8, 25, 25, 8, 8, 8, 43, 8,
    8, 8, 43, 8, 8, 8, 43, 8,
    25, 8, 8, 25, 8, 8, 43, 8,
    8, 25, 8, 25, 8, 8, 43, 8,
    8, 8, 25, 25, 8, 8, 43, 8,
    8, 8, 8, 43, 8, 8, 43, 8,
    8, 8, 43, 43, 8, 8, 43, 8,
    25, 8, 8, 8, 25, 8, 43, 8,
    8, 25, 8, 8, 25, 8, 43, 8,
    8, 8, 25, 8, 25, 8, 43, 8,
    8, 8, 8, 25, 25, 8, 43, 8,
    8, 43, 8, 25, 25, 8, 43, 8,
    25, 25, 43, 25, 25, 8, 43, 8,
    8, 8, 8, 8, 43, 8, 43, 8,
    43, 8, 43, 8, 43, 8, 43, 8,
    8, 8, 8, 43, 43, 8, 43, 8,
    8, 43, 43, 43, 43, 8, 43, 8,
    25, 8, 8, 8, 8, 25, 43, 8,
    8, 25, 8, 8, 8, 25, 43, 8,
    8, 8, 25, 8, 8, 25, 43, 8,
    25, 43, 43, 8, 8, 25, 43, 8,
    8, 8, 8, 25, 8, 25, 43, 8,
    8, 8, 8, 8, 25, 25, 43, 8,
    25, 8, 8, 25, 25, 25, 43, 8,
    43, 8, 25, 25, 25, 25, 43, 8,
    25, 43, 25, 43, 25, 25, 43, 8,
    25, 8, 8, 8, 43, 25, 43, 8,
    43, 43, 25, 8, 43, 25, 43, 8,
    43, 25, 43, 43, 43, 25, 43, 8,
    8, 8, 8, 8, 8, 43, 43, 8,
    8, 43, 8, 8, 8, 43, 43, 8,
    43, 43, 8, 8, 8, 43, 43, 8,
    8, 8, 43, 8, 8, 43, 43, 8,
    25, 25, 25, 25, 8, 43, 43, 8,
    8, 43, 8, 43, 8, 43, 43, 8,
    43, 8, 43, 43, 8, 43, 43, 8,
    8, 43, 43, 25, 25, 43, 43, 8,
    8, 8, 25, 43, 25, 43, 43, 8,
    8, 43, 8, 8, 43, 43, 43, 8,
    8, 8, 43, 8, 43, 43, 43, 8,
    43, 8, 8, 43, 43, 43, 43, 8,
    8, 43, 8, 43, 43, 43, 43, 8,
    43, 43, 8, 43, 43, 43, 43, 8,
    25, 8, 8, 8, 8, 8, 8, 25,
    8, 25, 8, 8, 8, 8, 8, 25,
    43, 25, 8, 8, 8, 8, 8, 25,
    25, 43, 8, 8, 8, 8, 8, 25,
    8, 8, 25, 8, 8, 8, 8, 25,
    43, 8, 25, 8, 8, 8, 8, 25,
    25, 25, 25, 8, 8, 8, 8, 25,
    8, 43, 25, 8, 8, 8, 8, 25,
    25, 8, 43, 8, 8, 8, 8, 25,
    8, 25, 43, 8, 8, 8, 8, 25,
    8, 8, 8, 25, 8, 8, 8, 25,
    43, 8, 8, 25, 8, 8, 8, 25,
    25, 25, 8, 25, 8, 8, 8, 25,
    8, 43, 8, 25, 8, 8, 8, 25,
    43, 43, 8, 25, 8, 8, 8, 25,
    25, 8, 25, 25, 8, 8, 8, 25,
    8, 25, 25, 25, 8, 8, 8, 25,
    8, 8, 43, 25, 8, 8, 8, 25,
    25, 25, 43, 25, 8, 8, 8, 25,
    25, 8, 8, 43, 8, 8, 8, 25,
    8, 25, 8, 43, 8, 8, 8, 25,
    8, 8, 25, 43, 8, 8, 8, 25,
    8, 8, 8, 8, 25, 8, 8, 25,
    43, 8, 8, 8, 25, 8, 8, 25,
    25, 25, 8, 8, 25, 8, 8, 25,
    8, 43, 8, 8, 25, 8, 8, 25,
    25, 8, 25, 8, 25, 8, 8, 25,
    8, 25, 25, 8, 25, 8, 8, 25,
    8, 8, 43, 8, 25, 8, 8, 25,
    25, 8, 8, 25, 25, 8, 8, 25,
    8, 25, 8, 25, 25, 8, 8, 25,
    8, 8, 25, 25, 25, 8, 8, 25,
    8, 8, 8, 43, 25, 8, 8, 25,
    25, 25, 8, 43, 25, 8, 8, 25,
    43, 8, 43, 43, 25, 8, 8, 25,
    25, 8, 8, 8, 43, 8, 8, 25,
    8, 25, 8, 8, 43, 8, 8, 25,
    8, 8, 25, 8, 43, 8, 8, 25,
    43, 8, 25, 8, 43, 8, 8, 25,
    25, 43, 43, 8, 43, 8, 8, 25,
    8, 8, 8, 25, 43, 8, 8, 25,
    8, 8, 8, 8, 8, 25, 8, 25,
    43, 8, 8, 8, 8, 25, 8, 25,
    25, 25, 8, 8, 8, 25, 8, 25,
    8, 43, 8, 8, 8, 25, 8, 25,
    25, 8, 25, 8, 8, 25, 8, 25,
    8, 25, 25, 8, 8, 25, 8, 25,
    25, 43, 25, 8, 8, 25, 8, 25,
    8, 8, 43, 8, 8, 25, 8, 25,
    25, 8, 8, 25, 8, 25, 8, 25,
    8, 25, 8, 25, 8, 25, 8, 25,
    8, 8, 25, 25, 8, 25, 8, 25,
    8, 8, 8, 43, 8, 25, 8, 25,
    8, 25, 25, 43, 8, 25, 8, 25,
    25, 8, 8, 8, 25, 25, 8, 25,
    8, 25, 8, 8, 25, 25, 8, 25,
    8, 8, 25, 8, 25, 25, 8, 25,
    8, 25, 43, 8, 25, 25, 8, 25,
    8, 8, 8, 25, 25, 25, 8, 25,
    43, 43, 25, 43, 25, 25, 8, 25,
    8, 8, 8, 8, 43, 25, 8, 25,
    43, 43, 8, 8, 43, 25, 8, 25,
    8, 25, 8, 25, 43, 25, 8, 25,
    8, 8, 25, 25, 43, 25, 8, 25,
    25, 8, 8, 8, 8, 43, 8, 25,
    8, 25, 8, 8, 8, 43, 8, 25,
    8, 8, 25, 8, 8, 43, 8, 25,
    8, 8, 8, 25, 8, 43, 8, 25,
    25, 25, 8, 25, 8, 43, 8, 25,
    8, 25, 25, 25, 8, 43, 8, 25,
    43, 8, 43, 25, 8, 43, 8, 25,
    8, 8, 8, 8, 25, 43, 8, 25,
    25, 8, 25, 8, 25, 43, 8, 25,
    8, 25, 8, 25, 25, 43, 8, 25,
    8, 8, 25, 25, 25, 43, 8, 25,
    25, 43, 43, 25, 25, 43, 8, 25,
    8, 25, 8, 8, 43, 43, 8, 25,
    8, 8, 8, 8, 8, 8, 25, 25,
    43, 8, 8, 8, 8, 8, 25, 25,
    25, 25, 8, 8, 8, 8, 25, 25,
    8, 43, 8, 8, 8, 8, 25, 25,
    25, 8, 25, 8, 8, 8, 25, 25,
    8, 25, 25, 8, 8, 8, 25, 25,
    8, 8, 43, 8, 8, 8, 25, 25,
    8, 43, 43, 8, 8, 8, 25, 25,
    25, 8, 8, 25, 8, 8, 25, 25,
    8, 25, 8, 25, 8, 8, 25, 25,
    8, 8, 25, 25, 8, 8, 25, 25,
    8, 8, 8, 43, 8, 8, 25, 25,
    25, 8, 8, 8, 25, 8, 25, 25,
    8, 25, 8, 8, 25, 8, 25, 25,
    8, 8, 25, 8, 25, 8, 25, 25,
    25, 25, 25, 8, 25, 8, 25, 25,
    8, 8, 8, 25, 25, 8, 25, 25,
    43, 8, 8, 25, 25, 8, 25, 25,
    8, 8, 8, 8, 43, 8, 25, 25,
    8, 25, 8, 25, 43, 8, 25, 25,
    43, 43, 43, 43, 43, 8, 25, 25,
    25, 8, 8, 8, 8, 25, 25, 25,
    8, 25, 8, 8, 8, 25, 25, 25,
    8, 8, 25, 8, 8, 25, 25, 25,
    25, 8, 43, 8, 8, 25, 25, 25,
    8, 8, 8, 25, 8, 25, 25, 25,
    8, 8, 43, 25, 8, 25, 25, 25,
    25, 8, 8, 43, 8, 25, 25, 25,
    25, 8, 43, 43, 8, 25, 25, 25,
    8, 8, 8, 8, 25, 25, 25, 25,
    8, 43, 8, 8, 25, 25, 25, 25,
    8, 8, 8, 43, 25, 25, 25, 25,
    8, 43, 8, 43, 25, 25, 25, 25,
    25, 8, 43, 8, 43, 25, 25, 25,
    8, 43, 43, 25, 43, 25, 25, 25,
    25, 8, 43, 43, 43, 25, 25, 25,
    8, 8, 8, 8, 8, 43, 25, 25,
    8, 25, 25, 8, 8, 43, 25, 25,
    25, 8, 8, 25, 8, 43, 25, 25,
    8, 8, 25, 25, 8, 43, 25, 25,
    25, 43, 25, 43, 8, 43, 25, 25,
    43, 43, 25, 8, 25, 43, 25, 25,
    8, 8, 8, 25, 25, 43, 25, 25,
    43, 8, 8, 25, 25, 43, 25, 25,
    25, 25, 8, 43, 43, 43, 25, 25,
    25, 8, 8, 8, 8, 8, 43, 25,
    8, 25, 8, 8, 8, 8, 43, 25,
    8, 8, 25, 8, 8, 8, 43, 25,
    8, 8, 8, 25, 8, 8, 43, 25,
    8, 25, 25, 25, 8, 8, 43, 25,
    43, 8, 43, 25, 8, 8, 43, 25,
    43, 25, 8, 43, 8, 8, 43, 25,
    25, 43, 43, 43, 8, 8, 43, 25,
    8, 8, 8, 8, 25, 8, 43, 25,
    8, 25, 43, 8, 43, 8, 43, 25,
    43, 43, 8, 25, 43, 8, 43, 25,
    43, 8, 25, 43, 43, 8, 43, 25,
    8, 8, 8, 8, 8, 25, 43, 25,
    43, 25, 25, 8, 8, 25, 43, 25,
    8, 8, 25, 8, 25, 25, 43, 25,
    8, 8, 8, 25, 25, 25, 43, 25,
    25, 25, 8, 25, 25, 25, 43, 25,
    8, 25, 43, 43, 25, 25, 43, 25,
    25, 8, 8, 8, 8, 43, 43, 25,
    43, 43, 43, 25, 8, 43, 43, 25,
    25, 25, 43, 8, 25, 43, 43, 25,
    43, 25, 8, 8, 43, 43, 43, 25,
    8, 25, 25, 25, 43, 43, 43, 25,
    43, 8, 43, 25, 43, 43, 43, 25,
    8, 8, 8, 8, 8, 8, 8, 43,
    43, 8, 8, 8, 8, 8, 8, 43,
    25, 25, 8, 8, 8, 8, 8, 43,
    8, 43, 8, 8, 8, 8, 8, 43,
    25, 8, 25, 8, 8, 8, 8, 43,
    8, 25, 25, 8, 8, 8, 8, 43,
    8, 8, 43, 8, 8, 8, 8, 43,
    43, 43, 43, 8, 8, 8, 8, 43,
    25, 8, 8, 25, 8, 8, 8, 43,
    8, 25, 8, 25, 8, 8, 8, 43,
    8, 8, 25, 25, 8, 8, 8, 43,
    8, 8, 8, 43, 8, 8, 8, 43,
    43, 8, 8, 43, 8, 8, 8, 43,
    8, 43, 43, 43, 8, 8, 8, 43,
    43, 43, 43, 43, 8, 8, 8, 43,
    25, 8, 8, 8, 25, 8, 8, 43,
    8, 25, 8, 8, 25, 8, 8, 43,
    43, 25, 8, 8, 25, 8, 8, 43,
    8, 8, 25, 8, 25, 8, 8, 43,
    8, 8, 8, 25, 25, 8, 8, 43,
    25, 8, 25, 25, 25, 8, 8, 43,
    25, 43, 25, 25, 25, 8, 8, 43,
    8, 8, 8, 8, 43, 8, 8, 43,
    8, 8, 43, 8, 43, 8, 8, 43,
    8, 8, 8, 43, 43, 8, 8, 43,
    43, 8, 8, 43, 43, 8, 8, 43,
    8, 8, 43, 43, 43, 8, 8, 43,
    8, 43, 43, 43, 43, 8, 8, 43,
    25, 8, 8, 8, 8, 25, 8, 43,
    8, 25, 8, 8, 8, 25, 8, 43,
    8, 8, 25, 8, 8, 25, 8, 43,
    43, 8, 25, 8, 8, 25, 8, 43,
    25, 25, 25, 8, 8, 25, 8, 43,
    8, 8, 8, 25, 8, 25, 8, 43,
    8, 8, 43, 25, 8, 25, 8, 43,
    25, 43, 8, 43, 8, 25, 8, 43,
    8, 8, 8, 8, 25, 25, 8, 43,
    8, 25, 8, 25, 25, 25, 8, 43,
    25, 25, 43, 43, 25, 25, 8, 43,
    8, 43, 25, 8, 43, 25, 8, 43,
    43, 43, 43, 25, 43, 25, 8, 43,
    8, 8, 8, 8, 8, 43, 8, 43,
    8, 43, 8, 8, 8, 43, 8, 43,
    25, 25, 43, 8, 8, 43, 8, 43,
    43, 43, 25, 25, 8, 43, 8, 43,
    8, 8, 8, 43, 8, 43, 8, 43,
    43, 8, 8, 43, 8, 43, 8, 43,
    8, 43, 43, 43, 8, 43, 8, 43,
    43, 25, 8, 8, 25, 43, 8, 43,
    43, 8, 43, 8, 43, 43, 8, 43,
    8, 8, 8, 43, 43, 43, 8, 43,
    8, 43, 8, 43, 43, 43, 8, 43,
    43, 25, 25, 43, 43, 43, 8, 43,
    8, 43, 43, 43, 43, 43, 8, 43,
    25, 8, 8, 8, 8, 8, 25, 43,
    8, 25, 8, 8, 8, 8, 25, 43,
    8, 8, 25, 8, 8, 8, 25, 43,
    8, 8, 8, 25, 8, 8, 25, 43,
    43, 25, 25, 25, 8, 8, 25, 43,
    8, 25, 8, 43, 8, 8, 25, 43,
    8, 8, 8, 8, 25, 8, 25, 43,
    43, 8, 43, 8, 25, 8, 25, 43,
    8, 25, 43, 25, 25, 8, 25, 43,
    43, 25, 25, 25, 43, 8, 25, 43,
    25, 43, 8, 43, 43, 8, 25, 43,
    8, 8, 8, 8, 8, 25, 25, 43,
    25, 25, 8, 8, 8, 25, 25, 43,
    8, 25, 8, 25, 8, 25, 25, 43,
    8, 8, 25, 25, 8, 25, 25, 43,
    8, 43, 25, 25, 8, 25, 25, 43,
    25, 43, 43, 8, 25, 25, 25, 43,
    8, 8, 25, 43, 25, 25, 25, 43,
    43, 8, 25, 43, 25, 25, 25, 43,
    25, 8, 8, 25, 43, 25, 25, 43,
    25, 8, 25, 25, 8, 43, 25, 43,
    43, 25, 43, 43, 8, 43, 25, 43,
    25, 43, 8, 25, 25, 43, 25, 43,
    25, 25, 25, 8, 43, 43, 25, 43,
    8, 8, 43, 25, 43, 43, 25, 43,
    8, 8, 8, 8, 8, 8, 43, 43,
    43, 8, 8, 8, 8, 8, 43, 43,
    8, 43, 8, 8, 8, 8, 43, 43,
    43, 43, 8, 8, 8, 8, 43, 43,
    8, 8, 43, 8, 8, 8, 43, 43,
    43, 43, 43, 8, 8, 8, 43, 43,
    8, 8, 43, 43, 8, 8, 43, 43,
    25, 8, 25, 25, 25, 8, 43, 43,
    25, 43, 25, 25, 25, 8, 43, 43,
    43, 25, 43, 43, 25, 8, 43, 43,
    8, 8, 8, 8, 43, 8, 43, 43,
    43, 8, 8, 8, 43, 8, 43, 43,
    8, 43, 8, 8, 43, 8, 43, 43,
    43, 43, 43, 8, 43, 8, 43, 43,
    8, 8, 8, 43, 43, 8, 43, 43,
    8, 8, 43, 43, 43, 8, 43, 43,
    8, 8, 8, 25, 8, 25, 43, 43,
    25, 25, 25, 43, 8, 25, 43, 43,
    25, 25, 43, 25, 43, 25, 43, 43,
    8, 43, 25, 43, 43, 25, 43, 43,
    43, 43, 8, 8, 8, 43, 43, 43,
    8, 8, 43, 43, 8, 43, 43, 43,
    43, 8, 43, 8, 8, 43, 43, 43,
    8, 43, 43, 8, 8, 43, 43, 43,
    8, 43, 43, 43, 8, 43, 43, 43,
    8, 43, 43, 43, 8, 43, 43, 43,
    8, 25, 8, 8, 25, 43, 43, 43,
    8, 25, 8, 43, 25, 43, 43, 43,
    43, 25, 8, 43, 25, 43, 43, 43,
    8, 43, 43, 8, 43, 43, 43, 43,
    43, 43, 43, 8, 43, 43, 43, 43,
    25, 8, 25, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43,
], dtype=np.float32).reshape(512, 8)

# 符号反转查找表 — 128 项, 用于 7-bit 符号编码
# 来源: csrc/iq2_xs.cuh ksigns_iq2xs[128]
KSIGNS = np.array([
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
], dtype=np.uint8)

_kgrid_2bit_512 = np.array([
        0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
       73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
      260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
      352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
      640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
     1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
     1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
     2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
     2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
     4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
     4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
     5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
     5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
     8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
     8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
    10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
    16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
    16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
    16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
    17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
    18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
    20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
    21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
    22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
    24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
    32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
    33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
    33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
    35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
    37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
    40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
    42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
], dtype=np.uint16)

# ============================================================================
# 构建 kmap 查找表（L 值编码 → grid 索引）
# ============================================================================
print("[初始化] 构建 kmap 查找表...")
_kmap_size = 4 ** 8  # 65536
_kmap = np.full(_kmap_size, -1, dtype=np.int32)

# 从 kgrid_2bit_512 填充 on-grid 映射
for _gi in range(512):
    _u = int(_kgrid_2bit_512[_gi])
    _kmap[_u] = _gi

# 为 off-grid 点找最近邻（L2 距离）
_grid_q = np.zeros((512, 8), dtype=np.float32)
for _i in range(512):
    _u = int(_kgrid_2bit_512[_i])
    for _k in range(8):
        _L = (_u >> (2 * _k)) & 3
        _grid_q[_i, _k] = 2 * _L + 1

for _u in range(_kmap_size):
    if _kmap[_u] >= 0:
        continue
    _L_vals = np.array([(_u >> (2 * _k)) & 3 for _k in range(8)], dtype=np.int32)
    _q_vals = 2 * _L_vals + 1
    _dist2 = np.sum((_grid_q - _q_vals) ** 2, axis=1)
    _kmap[_u] = np.argmin(_dist2)

# kmap 转为 GPU tensor
KMAP_TENSOR = torch.from_numpy(_kmap).to(torch.int32).cuda()

# 物理网格（反量化用，值域 {8, 25, 43}）
IQ2_XS_GRID_TENSOR = torch.from_numpy(IQ2_XS_GRID).cuda().float()
KSIGNS_TENSOR = torch.from_numpy(KSIGNS).cuda()

# 逻辑网格（量化用，值域 {1, 3, 5}）
GRID_LOGICAL_TENSOR = torch.from_numpy(_grid_q).cuda()

del _kmap, _grid_q, _kgrid_2bit_512, _L_vals, _q_vals, _dist2

FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32, device='cuda')

# KSIGNS_INV: sign_byte(0-255) → sign_index(0-127)
print("[初始化] 预计算 Sign 查找表...")
KSIGNS_INV = torch.zeros(256, dtype=torch.long, device='cuda')
for i in range(256):
    idx = (KSIGNS_TENSOR == i).nonzero()
    if idx.numel() > 0:
        KSIGNS_INV[i] = idx[0, 0]
    else:
        flipped = i ^ 1
        idx2 = (KSIGNS_TENSOR == flipped).nonzero()
        if idx2.numel() > 0:
            KSIGNS_INV[i] = idx2[0, 0]
        else:
            KSIGNS_INV[i] = 0

# kmask_iq2xs: [8] = {1, 2, 4, 8, 16, 32, 64, 128}
KMASK = (1 << torch.arange(8, device='cuda')).to(torch.uint8)  # [8]

print("[初始化] 完成")


def decode_fp4_to_float16_gpu(weight_i8: torch.Tensor) -> torch.Tensor:
    """GPU: 将 int8 打包的 FP4 解码为 float16。"""
    x = weight_i8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    decoded = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(-2)
    return decoded.to(torch.float16)


# 常量
kMaxQ = 3  # L 值范围 [0, kMaxQ-1] = [0, 2]
GROUP_MAX_EPS = 1e-15  # 对齐 C 代码 iq2_xs.h（原为 1e-10）


def compute_signs_and_xval(groups_8: torch.Tensor, weight_8: torch.Tensor = None) -> tuple:
    """计算符号和绝对值，确保每组 8 元素中负数个数为偶数。

    对齐官方 quantize_row_iq2_xs_impl 的符号处理：
    1. 计算每个元素的符号和绝对值
    2. 统计负数个数 nflip
    3. 若 nflip 为奇数，翻转 weight*x² 最小元素的符号（对齐 C 代码 iq2_xs.h 第586-590行）

    参数:
        groups_8: [N, 8] 原始值
        weight_8: [N, 8] 重要性权重（= sqrt(sigma2 + x²)），为 None 时退化为 |x| 最小

    返回:
        xval: [N, 8] 绝对值（可能有一个被翻转为负）
        sign_indices: [N] 符号索引（0-127）
    """
    N = groups_8.shape[0]

    # 计算符号位
    is_neg = (groups_8 < 0)  # [N, 8]
    nflip = is_neg.sum(dim=1)  # [N]

    # 计算 sign_byte
    sign_bits = is_neg.to(torch.uint8)  # [N, 8]
    bit_weights = (1 << torch.arange(8, device='cuda')).unsqueeze(0)  # [1, 8]
    sign_bytes = (sign_bits * bit_weights).sum(dim=1)  # [N]

    # 计算绝对值
    xval = groups_8.abs()  # [N, 8]

    # 对 nflip 为奇数的组，翻转 weight*x² 最小元素的符号（对齐 C 代码 iq2_xs.h 第586-590行）
    # C 代码: min(weight[8*k+i] * xb[8*k+i] * xb[8*k+i])，即 weight * x² 最小
    odd_mask = (nflip % 2 == 1)  # [N]

    if odd_mask.any():
        # 找每组中 weight * x² 最小的元素索引
        if weight_8 is not None:
            flip_criterion = weight_8 * groups_8 ** 2  # [N, 8]
            min_idx = flip_criterion.argmin(dim=1)  # [N]
        else:
            min_idx = xval.argmin(dim=1)  # [N]（无权重时退化为 |x| 最小）

        # 翻转该元素的符号位
        flip_bits = (1 << min_idx).to(torch.uint8)  # [N]
        sign_bytes_updated = sign_bytes ^ flip_bits  # XOR 翻转对应位

        # 只对 odd_mask=True 的组更新
        sign_bytes = torch.where(odd_mask, sign_bytes_updated, sign_bytes)

        # 对被翻转的元素，xval 取负（对齐 C 代码 iq2_xs.h 第592行）
        # C 代码: xval[8*k+imin] = -xval[8*k+imin];
        # 不管原来正负，翻转后 xval[imin] 都变成负值
        flip_mask = torch.zeros(N, 8, dtype=torch.bool, device='cuda')
        flip_mask[torch.arange(N, device='cuda'), min_idx] = odd_mask
        xval = torch.where(flip_mask, -xval, xval)

    # sign_bytes → sign_index via KSIGNS_INV
    sign_indices = KSIGNS_INV[sign_bytes.long()]

    return xval, sign_indices


def quantize_iq2_xs_batch(blocks: torch.Tensor, desc: str = "") -> dict:
    """批量量化多个 256 元素的 super-block（GGUF 官方算法）。

    对齐 ggml-quants.c quantize_row_iq2_xs_impl：
    1. 对每个 16 元素子块，计算符号（保证偶数负数）+ 绝对值 xval
    2. 尝试 19 个 scale 候选值，找最优 scale（sumqx²/sumq2 最大化）
    3. 用 L 值计算 + kmap 查找 grid 索引
    4. off-grid 修正 + 负 scale 处理
    5. d = max_scale / 31
    6. scale_4bit = nearest_int(0.5 * (id * scale - 1))

    参数:
        blocks: [N, 256] N 个 256 元素的 block
        desc: 进度条描述

    返回:
        d: [N] float16
        qs: [N, 32] uint16
        scales: [N, 8] uint8
    """
    N = blocks.shape[0]

    # 将 256 元素 block 重排为 32 个 8 元素组
    # blocks: [N, 256] → groups: [N, 32, 8]
    groups = blocks.reshape(N, 32, 8)

    # 计算重要性权重（对齐 C 代码 iq2_xs.h 第558-569行）
    # sigma2 = mean(x²) 对每个 256 元素 block
    # weight = sqrt(sigma2 + x²) 对每个元素（qw=1 时简化）
    # C: sumx2 = Σx²; sigma2 = sumx2/QK_K; weight[i] = qw[i] * sqrt(sigma2 + xb[i]²)
    sigma2 = (blocks ** 2).mean(dim=1)  # [N]
    weight_all = (sigma2.unsqueeze(1) + blocks ** 2).sqrt()  # [N, 256]
    weight_flat = weight_all.reshape(N * 32, 8)  # [N*32, 8]

    # 步骤 1: 对每个 8 元素组计算符号和绝对值
    # 展平为 [N*32, 8] 处理
    groups_flat = groups.reshape(N * 32, 8)
    xval_flat, sign_idx_flat = compute_signs_and_xval(groups_flat, weight_flat)
    xval_flat = xval_flat.reshape(N, 32, 8)       # [N, 32, 8]
    sign_idx_flat = sign_idx_flat.reshape(N, 32)   # [N, 32]

    # 步骤 2: 对每个 16 元素子块（2 个 8 元素组）进行迭代 scale 优化
    # 子块索引 ib: 0-15，对应 groups[:, 2*ib:2*ib+2, :]
    all_scales = torch.zeros(N, 16, dtype=torch.float32, device='cuda')
    all_L = torch.zeros(N, 32, 8, dtype=torch.int8, device='cuda')
    all_signs = sign_idx_flat.clone()  # [N, 32]

    for ib in range(16):
        # 2 个 8 元素组的 xval，拼成 [N, 16]
        xval_16 = xval_flat[:, 2 * ib: 2 * ib + 2, :].reshape(N, 16)

        # 重要性权重（对齐 C 代码: weight[i] = qw[i] * sqrt(sigma2 + xb[i]²)）
        weight_16 = weight_all[:, 16 * ib: 16 * ib + 16]  # [N, 16]

        # 每组的最大绝对值
        max_val = xval_16.amax(dim=1)  # [N]

        # 跳过全零组
        valid = max_val > GROUP_MAX_EPS  # [N]

        if not valid.any():
            continue

        # 初始化最优值
        best_scale = max_val.float() / (2 * kMaxQ - 1)  # [N]
        best_metric = torch.zeros(N, dtype=torch.float32, device='cuda')
        best_L = torch.zeros(N, 16, dtype=torch.int8, device='cuda')
        is_on_grid = torch.ones(N, 2, dtype=torch.bool, device='cuda')

        # 19 个候选值搜索
        for is_val in range(-9, 10):
            id_val = (2 * kMaxQ - 1 + is_val * 0.1) / max_val.clamp(min=GROUP_MAX_EPS)  # [N]

            # 计算 L 值: L = clamp(round(0.5 * (id * xval - 1)), 0, kMaxQ-1)
            L_16 = (0.5 * (id_val.unsqueeze(1) * xval_16 - 1)).round().clamp(0, kMaxQ - 1).to(torch.int8)  # [N, 16]

            # 检查 on-grid
            on_grid_aux = torch.ones(N, 2, dtype=torch.bool, device='cuda')
            for k in range(2):
                L_8 = L_16[:, 8 * k: 8 * (k + 1)]  # [N, 8]
                # 编码 u = sum(L[i] << (2*i))
                u = torch.zeros(N, dtype=torch.int32, device='cuda')
                for i in range(8):
                    u |= (L_8[:, i].int() << (2 * i))

                # kmap 查找
                grid_idx = KMAP_TENSOR[u.clamp(0, _kmap_size - 1)]
                # kmap 中所有值 >= 0（off-grid 已映射到最近邻），所以总是 on-grid

            # 计算加权最优 scale: scale = sumqx / sumq2（对齐 C 代码加权公式）
            # C: sumqx += w * xval[i] * q; sumq2 += w * q * q
            q = 2 * L_16.float() + 1  # [N, 16]
            sumqx = (weight_16 * xval_16 * q).sum(dim=1)  # [N]
            sumq2 = (weight_16 * q * q).sum(dim=1)         # [N]

            # 选择 metric = sumqx² / sumq2 最大的候选
            metric = torch.where(sumq2 > 0, sumqx * sumqx / sumq2.clamp(min=1e-10), torch.zeros_like(sumqx))

            better = (metric > best_metric) & valid & (sumq2 > 0)
            best_scale = torch.where(better, sumqx / sumq2.clamp(min=1e-10), best_scale)
            best_metric = torch.where(better, metric, best_metric)
            best_L = torch.where(better.unsqueeze(1), L_16, best_L)

        # 步骤 3: off-grid 修正（kmap 已映射所有点，此步简化为重新计算 scale）
        # 对有 off-grid 的组，用最优 scale 重新计算 L 值
        # 由于 kmap 已将所有 off-grid 映射到最近邻，此步主要确保 scale 最优
        id_recalc = 1.0 / best_scale.clamp(min=GROUP_MAX_EPS)
        L_recalc = (0.5 * (id_recalc.unsqueeze(1) * xval_16 - 1)).round().clamp(0, kMaxQ - 1).to(torch.int8)
        q_recalc = 2 * L_recalc.float() + 1
        sumqx_r = (weight_16 * xval_16 * q_recalc).sum(dim=1)
        sumq2_r = (weight_16 * q_recalc * q_recalc).sum(dim=1)
        scale_recalc = torch.where(sumq2_r > 0, sumqx_r / sumq2_r.clamp(min=1e-10), best_scale)

        # 只对 valid 组更新
        best_scale = torch.where(valid, scale_recalc, best_scale)
        best_L = torch.where(valid.unsqueeze(1), L_recalc, best_L)

        # 步骤 4: 处理负 scale
        neg_scale = (best_scale < 0) & valid
        if neg_scale.any():
            best_scale = torch.where(neg_scale, -best_scale, best_scale)
            # 翻转对应子块的两个 8 元素组的符号
            for k in range(2):
                all_signs[:, 2 * ib + k] = torch.where(
                    neg_scale,
                    (~all_signs[:, 2 * ib + k].to(torch.uint8)) & 127,
                    all_signs[:, 2 * ib + k]
                )

        all_scales[:, ib] = best_scale
        all_L[:, 2 * ib: 2 * ib + 2, :] = best_L.reshape(N, 2, 8)

    # 步骤 5: 编码全局 scale d 和子块 scale
    max_scale = all_scales.amax(dim=1)  # [N]
    zero_mask = max_scale < GROUP_MAX_EPS

    d = (max_scale / 31.0).to(torch.float16)
    scale_f = d.float()
    id_val = 1.0 / scale_f.clamp(min=1e-10)

    # 编码 4-bit scales: scale_4bit = nearest_int(0.5 * (id * scale - 1))
    scales_4bit = (0.5 * (id_val.unsqueeze(1) * all_scales - 1)).round().clamp(0, 15).to(torch.uint8)  # [N, 16]

    # 打包为 8 个 uint8（每 2 个 4-bit 共享一个 uint8）
    scales_packed = (scales_4bit[:, 1::2] << 4) | scales_4bit[:, 0::2]  # [N, 8]

    # 步骤 6: 编码 qs（grid 索引 + 符号索引）
    qs = torch.zeros(N, 32, dtype=torch.int32, device='cuda')

    for g in range(32):  # 32 个 8 元素组
        L_8 = all_L[:, g, :]  # [N, 8]

        # 编码 u = sum(L[i] << (2*i))
        u = torch.zeros(N, dtype=torch.int32, device='cuda')
        for i in range(8):
            u |= (L_8[:, i].int() << (2 * i))

        # kmap 查找 grid 索引
        grid_idx = KMAP_TENSOR[u.clamp(0, _kmap_size - 1)]

        # 符号索引
        sign_idx = all_signs[:, g]

        # 打包: 低 9-bit = grid_idx, 高 7-bit = sign_idx
        qs[:, g] = (grid_idx & 511) | ((sign_idx & 127) << 9)

    return {
        "d": d,
        "qs": qs.to(torch.uint16),
        "scales": scales_packed,
    }


def quantize_weight_gpu_optimized(weight: torch.Tensor, batch_size: int = 0, desc: str = "量化") -> dict:
    """GPU 优化版：将 FP4 打包权重量化为 GGUF 兼容的 IQ2_XS 格式。
    
    参数:
        weight: int8 FP4 打包权重
        batch_size: 批量处理的 block 数量（0=自动检测）
        desc: 进度条描述
    """
    weight_float = decode_fp4_to_float16_gpu(weight)
    shape = weight_float.shape
    
    flat = weight_float.flatten().float()
    del weight_float  # 提前释放
    K = flat.numel()
    n_blocks = (K + QK_K - 1) // QK_K
    
    # 自动计算最优 batch_size
    if batch_size == 0:
        batch_size = compute_optimal_batch_size(n_blocks, "quantize")
    
    n_elements = 1
    for s in shape:
        n_elements *= s
    
    if K % QK_K != 0:
        padded = torch.zeros(n_blocks * QK_K, dtype=torch.float32, device='cuda')
        padded[:K] = flat
        del flat  # 释放原始 tensor
        flat = padded
    
    flat = flat.reshape(n_blocks, QK_K)
    
    # 预分配输出，避免列表累积
    all_d = torch.zeros(n_blocks, dtype=torch.float16, device='cuda')
    all_qs = torch.zeros(n_blocks, 32, dtype=torch.uint16, device='cuda')
    all_scales = torch.zeros(n_blocks, 8, dtype=torch.uint8, device='cuda')
    
    n_batches = (n_blocks + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_blocks)
        
        batch_blocks = flat[start_idx:end_idx]
        
        result = quantize_iq2_xs_batch(batch_blocks)
        
        all_d[start_idx:end_idx] = result["d"]
        all_qs[start_idx:end_idx] = result["qs"]
        all_scales[start_idx:end_idx] = result["scales"]
        
        del batch_blocks, result
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    del flat
    torch.cuda.empty_cache()
    
    return {
        "d": all_d,
        "qs": all_qs,
        "scales": all_scales,
        "shape": shape,
    }


def dequantize_iq2_xs_batch(d: torch.Tensor, qs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """批量反量化 IQ2_XS 权重（完全向量化，无 .item() 调用）。
    
    参数:
        d: [N] float16
        qs: [N, 32] uint16
        scales: [N, 8] uint8
    
    返回:
        W: [N * 256] float32
    """
    N = d.shape[0]
    scale_f = d.float()
    
    # 解包 scales：低 4-bit 用于 l=0,1，高 4-bit 用于 l=2,3
    scales_low = (scales & 0x0F).float()   # [N, 8]
    scales_high = ((scales >> 4) & 0x0F).float()  # [N, 8]
    
    db_low = scale_f.unsqueeze(1) * (0.5 + scales_low) * 0.25   # [N, 8]
    db_high = scale_f.unsqueeze(1) * (0.5 + scales_high) * 0.25  # [N, 8]
    
    # 解包 qs：低 9-bit = grid_idx，高 7-bit = sign_idx
    # UInt16 不支持位运算，先转 int32
    qs_int = qs.int()
    grid_idx = qs_int & 511           # [N, 32]
    sign_idx = (qs_int >> 9) & 127    # [N, 32]
    
    W = torch.zeros(N, 256, dtype=torch.float32, device='cuda')
    
    sign_bits = torch.arange(8, device='cuda').unsqueeze(0)  # [1, 8]
    
    for ib32 in range(8):
        for l in range(4):
            qs_idx = ib32 * 4 + l
            
            g_idx = grid_idx[:, qs_idx]   # [N]
            s_idx = sign_idx[:, qs_idx]   # [N]
            
            grid_vals = IQ2_XS_GRID_TENSOR[g_idx]  # [N, 8]
            sign_bytes = KSIGNS_TENSOR[s_idx].unsqueeze(1)  # [N, 1]
            
            db = db_low[:, ib32] if l < 2 else db_high[:, ib32]  # [N]
            
            sign_mask = (sign_bytes >> sign_bits) & 1  # [N, 8]
            signs = 1.0 - sign_mask.float() * 2.0      # [N, 8]
            
            vals = db.unsqueeze(1) * grid_vals * signs  # [N, 8]
            
            start = ib32 * 32 + l * 8
            W[:, start:start+8] = vals
    
    return W.flatten()


def dequantize_weight_gpu_optimized(d: torch.Tensor, qs: torch.Tensor, scales: torch.Tensor, K: int, batch_size: int = 0, desc: str = "反量化") -> torch.Tensor:
    """GPU 优化版：反量化 IQ2_XS 权重（完全向量化）。
    
    参数:
        d: [n_blocks] float16
        qs: [n_blocks, 32] uint16
        scales: [n_blocks, 8] uint8
        K: 原始权重元素数
        batch_size: 批量处理的 block 数量（0=自动检测）
        desc: 进度条描述
    """
    n_blocks = d.numel()
    
    # 自动计算最优 batch_size
    if batch_size == 0:
        batch_size = compute_optimal_batch_size(n_blocks, "dequantize")
    
    n_batches = (n_blocks + batch_size - 1) // batch_size
    
    # 预分配输出，避免列表累积
    W = torch.zeros(n_blocks * QK_K, dtype=torch.float32, device='cuda')
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_blocks)
        
        batch_d = d[start_idx:end_idx]
        batch_qs = qs[start_idx:end_idx]
        batch_scales = scales[start_idx:end_idx]
        
        batch_W = dequantize_iq2_xs_batch(batch_d, batch_qs, batch_scales)
        
        W[start_idx * QK_K:end_idx * QK_K] = batch_W
        
        del batch_d, batch_qs, batch_scales, batch_W
        
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    return W[:K]


def get_shard_metadata(shard_path: str) -> dict:
    """读取 safetensors 元数据。"""
    with open(shard_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    metadata = {}
    for key, info in header.items():
        dtype_str = info["dtype"]
        shape = info["shape"]
        data_start, data_end = info["data_offsets"]
        dtype_map = {
            "I8": torch.int8,
            "INT8": torch.int8,
            "F16": torch.float16,
            "F32": torch.float32,
            "BF16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.uint8)
        metadata[key] = (dtype, shape, 8 + header_size + data_start, data_end - data_start)
    return metadata


def read_tensor(shard_path: str, key: str, metadata: dict) -> torch.Tensor:
    """从 safetensors 读取单个张量。"""
    dtype, shape, offset, length = metadata[key]
    with open(shard_path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
    return torch.frombuffer(data, dtype=dtype).reshape(shape).clone()


def get_memory_usage():
    """获取当前内存使用情况。"""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {gpu_allocated:.2f}GB / {gpu_reserved:.2f}GB"
    return ""


def detect_system_resources():
    """检测系统资源，返回 GPU 显存和 CPU 内存信息。
    
    返回:
        dict: {
            gpu_total_gb: GPU 总显存 (GB)
            gpu_free_gb: GPU 可用显存 (GB)
            ram_total_gb: 系统总内存 (GB)
            ram_available_gb: 系统可用内存 (GB)
            gpu_name: GPU 名称
        }
    """
    import subprocess
    
    info = {}
    
    # GPU 信息
    if torch.cuda.is_available():
        info['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['gpu_free_gb'] = (torch.cuda.get_device_properties(0).total_memory -
                               torch.cuda.memory_allocated(0)) / 1024**3
        info['gpu_name'] = torch.cuda.get_device_properties(0).name
    else:
        info['gpu_total_gb'] = 0
        info['gpu_free_gb'] = 0
        info['gpu_name'] = "N/A"
    
    # CPU 内存信息
    try:
        result = subprocess.run(['free', '-g'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            info['ram_total_gb'] = int(parts[1])
            info['ram_available_gb'] = int(parts[6])
        else:
            info['ram_total_gb'] = 0
            info['ram_available_gb'] = 0
    except Exception:
        info['ram_total_gb'] = 0
        info['ram_available_gb'] = 0
    
    return info


def compute_optimal_batch_size(n_blocks: int, mode: str = "quantize") -> int:
    """根据可用显存计算最优 batch_size。
    
    量化内存估算（每 block）：
      - 输入: 256 * 4 = 1KB (float32)
      - Grid 搜索中间: 512 * 8 * 4 = 16KB (float32)
      - 输出: 74 bytes
      - 总计每 block 约 20KB 峰值
    
    反量化内存估算（每 block）：
      - 输入: 74 bytes
      - 中间: 256 * 4 = 1KB (float32)
      - 输出: 256 * 4 = 1KB (float32)
      - 总计每 block 约 3KB 峰值
    
    参数:
        n_blocks: 总 block 数
        mode: "quantize" 或 "dequantize"
    
    返回:
        推荐的 batch_size
    """
    if not torch.cuda.is_available():
        return 4096
    
    gpu_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    gpu_free_gb = gpu_free / 1024**3
    
    # 预留 2GB 给 CUDA 内核和其他开销
    usable_gb = max(gpu_free_gb - 2.0, 0.5)
    
    if mode == "quantize":
        bytes_per_block = 20 * 1024  # ~20KB 峰值
    else:
        bytes_per_block = 3 * 1024   # ~3KB 峰值
    
    max_batch = int(usable_gb * 1024**3 / bytes_per_block)
    
    # 限制在合理范围
    batch_size = min(max_batch, n_blocks)
    batch_size = max(batch_size, 64)  # 最小 64
    
    # 对齐到 2 的幂次
    power = 1
    while power * 2 <= batch_size:
        power *= 2
    batch_size = power
    
    return batch_size


def main():
    parser = ArgumentParser(description="GPU 优化版 IQ2_XS 量化")
    parser.add_argument("--input", help="输入模型目录")
    parser.add_argument("--output", help="输出文件（.safetensors）")
    parser.add_argument("--test-shard", help="测试单个 shard（量化+反量化）")
    parser.add_argument("--test-quantize", help="仅测试量化", action="store_true")
    parser.add_argument("--test-dequantize", help="仅测试反量化", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0, help="批量大小（0=自动检测）")
    parser.add_argument("--limit", type=int, default=0, help="限制处理专家数（0=全部）")
    args = parser.parse_args()
    
    # 检测系统资源
    res = detect_system_resources()
    print("\n" + "=" * 70)
    print("系统资源检测")
    print("=" * 70)
    print(f"  GPU: {res['gpu_name']}")
    print(f"  GPU 显存: {res['gpu_total_gb']:.1f} GB (可用 {res['gpu_free_gb']:.1f} GB)")
    print(f"  系统内存: {res['ram_total_gb']} GB (可用 {res['ram_available_gb']} GB)")
    
    # 自动选择 batch_size
    auto_batch = args.batch_size == 0
    if auto_batch:
        args.batch_size = compute_optimal_batch_size(100000, "quantize")
        print(f"\n  [自动] batch_size = {args.batch_size} (基于 {res['gpu_free_gb']:.1f}GB 可用显存)")
    else:
        print(f"\n  [手动] batch_size = {args.batch_size}")
    
    # 内存建议
    if res['ram_available_gb'] < 20:
        print(f"  [警告] 可用内存仅 {res['ram_available_gb']}GB，建议关闭其他进程")
    if res['gpu_free_gb'] < 4:
        print(f"  [警告] 可用显存仅 {res['gpu_free_gb']:.1f}GB，量化速度可能受限")
    
    if args.test_shard:
        metadata = get_shard_metadata(args.test_shard)
        expert_keys = [k for k in metadata if ".ffn.experts." in k and ".weight" in k]
        
        if len(expert_keys) == 0:
            print("未找到专家权重")
            return
        
        limit = args.limit if args.limit > 0 else len(expert_keys)
        expert_keys = expert_keys[:limit]
        total = len(expert_keys)
        
        # 仅量化测试
        if args.test_quantize and not args.test_dequantize:
            print("\n" + "=" * 70)
            print(f"量化测试 ({total} 个专家)")
            print("=" * 70)
            
            start_time = _time.time()
            all_errors = []
            
            for idx, key in enumerate(expert_keys):
                weight = read_tensor(args.test_shard, key, metadata)
                weight_gpu = weight.cuda()
                del weight
                gc.collect()
                
                qdata = quantize_weight_gpu_optimized(weight_gpu, batch_size=args.batch_size, desc=f"量化")
                
                # 量化精度：对比原始 FP4 解码值
                original = decode_fp4_to_float16_gpu(weight_gpu).flatten().float()
                K = qdata['d'].numel() * QK_K
                dequant = dequantize_iq2_xs_batch(qdata['d'], qdata['qs'], qdata['scales'])
                
                if original.numel() < K:
                    padded = torch.zeros(K, dtype=torch.float32, device='cuda')
                    padded[:original.numel()] = original
                    original = padded
                
                mse = ((original[:K] - dequant[:K]) ** 2).mean().item()
                all_errors.append({"key": key, "mse": mse})
                
                del weight_gpu, qdata, original, dequant
                torch.cuda.empty_cache()
                gc.collect()
                
                # 整体进度
                if (idx + 1) % max(1, total // 20) == 0 or (idx + 1) == total:
                    elapsed = _time.time() - start_time
                    avg = elapsed / (idx + 1)
                    remaining = avg * (total - idx - 1)
                    print(f"  [量化] {idx + 1}/{total} ({(idx+1)/total*100:.0f}%), "
                          f"已用 {elapsed:.0f}s, 剩余 ~{remaining:.0f}s, {get_memory_usage()}")
            
            elapsed = _time.time() - start_time
            avg_mse = sum(e["mse"] for e in all_errors) / len(all_errors)
            max_mse = max(e["mse"] for e in all_errors)
            print(f"\n  [结果] 量化完成: {total} 个专家, {elapsed:.1f}s")
            print(f"  [结果] 平均 MSE: {avg_mse:.6f}, 最大 MSE: {max_mse:.6f}")
            return
        
        # 仅反量化测试
        if args.test_dequantize and not args.test_quantize:
            print("\n" + "=" * 70)
            print(f"反量化测试 ({total} 个专家)")
            print("=" * 70)
            
            start_time = _time.time()
            all_results = []
            
            for idx, key in enumerate(expert_keys):
                weight = read_tensor(args.test_shard, key, metadata)
                weight_gpu = weight.cuda()
                del weight
                gc.collect()
                
                qdata = quantize_weight_gpu_optimized(weight_gpu, batch_size=args.batch_size, desc="量化")
                
                K = qdata['d'].numel() * QK_K
                dequant = dequantize_weight_gpu_optimized(qdata['d'], qdata['qs'], qdata['scales'], K, batch_size=args.batch_size, desc="反量化")
                
                all_results.append({"key": key, "shape": qdata["shape"], "dequant_shape": dequant.shape})
                
                del weight_gpu, qdata, dequant
                torch.cuda.empty_cache()
                gc.collect()
                
                if (idx + 1) % max(1, total // 20) == 0 or (idx + 1) == total:
                    elapsed = _time.time() - start_time
                    avg = elapsed / (idx + 1)
                    remaining = avg * (total - idx - 1)
                    print(f"  [反量化] {idx + 1}/{total} ({(idx+1)/total*100:.0f}%), "
                          f"已用 {elapsed:.0f}s, 剩余 ~{remaining:.0f}s, {get_memory_usage()}")
            
            elapsed = _time.time() - start_time
            print(f"\n  [结果] 反量化完成: {total} 个专家, {elapsed:.1f}s")
            for r in all_results[:5]:
                print(f"    {r['key']}: {list(r['shape'])} → {list(r['dequant_shape'])}")
            if len(all_results) > 5:
                print(f"    ... 共 {len(all_results)} 个")
            return
        
        # 量化+反量化测试（默认）
        print("\n" + "=" * 70)
        print(f"量化+反量化测试 ({total} 个专家)")
        print("=" * 70)
        
        all_results = {}
        all_errors = []
        start_time = _time.time()
        
        for idx, key in enumerate(expert_keys):
            weight = read_tensor(args.test_shard, key, metadata)
            weight_gpu = weight.cuda()
            del weight
            gc.collect()
            
            qdata = quantize_weight_gpu_optimized(weight_gpu, batch_size=args.batch_size, desc="量化")
            
            K = qdata['d'].numel() * QK_K
            dequant = dequantize_weight_gpu_optimized(qdata['d'], qdata['qs'], qdata['scales'], K, batch_size=args.batch_size, desc="反量化")
            
            original = decode_fp4_to_float16_gpu(weight_gpu).flatten().float()
            if original.numel() < K:
                original_padded = torch.zeros(K, dtype=torch.float32, device='cuda')
                original_padded[:original.numel()] = original
                original = original_padded
            
            error = (original[:K] - dequant).abs()
            mse = (error ** 2).mean().item()
            
            all_errors.append({"key": key, "mse": mse})
            
            prefix = key.replace(".weight", "")
            all_results[f"{prefix}.d"] = qdata["d"].cpu()
            all_results[f"{prefix}.qs"] = qdata["qs"].cpu()
            all_results[f"{prefix}.scales"] = qdata["scales"].cpu()
            all_results[f"{prefix}.shape"] = torch.tensor(qdata["shape"], dtype=torch.int64)
            
            del weight_gpu, qdata, dequant, original
            torch.cuda.empty_cache()
            gc.collect()
            
            # 整体进度（约 20 次更新）
            if (idx + 1) % max(1, total // 20) == 0 or (idx + 1) == total:
                elapsed = _time.time() - start_time
                avg = elapsed / (idx + 1)
                remaining = avg * (total - idx - 1)
                print(f"  [进度] {idx + 1}/{total} ({(idx+1)/total*100:.0f}%), "
                      f"已用 {elapsed:.0f}s, 剩余 ~{remaining:.0f}s, {get_memory_usage()}")
        
        elapsed = _time.time() - start_time
        avg_mse = sum(e["mse"] for e in all_errors) / len(all_errors)
        max_mse = max(e["mse"] for e in all_errors)
        
        print(f"\n  [结果] 完成: {total} 个专家, {elapsed:.1f}s")
        print(f"  [结果] 平均 MSE: {avg_mse:.6f}, 最大 MSE: {max_mse:.6f}")
        
        return
    
    if not args.input or not args.output:
        print("需要 --input 和 --output 参数")
        return
    
    print("\n" + "=" * 70)
    print("GPU 优化版 IQ2_XS 量化")
    print("=" * 70)
    print(f"\n[输入] {args.input}")
    print(f"[输出] {args.output}")
    print(f"[批量大小] {args.batch_size}")
    
    shard_paths = sorted(glob(os.path.join(args.input, "*.safetensors")))
    print(f"\n[扫描] 找到 {len(shard_paths)} 个 shard")
    
    expert_keys = {}
    for shard_path in shard_paths:
        metadata = get_shard_metadata(shard_path)
        for key in metadata:
            if ".ffn.experts." in key and ".weight" in key:
                parts = key.split(".")
                try:
                    layer_idx = parts.index("layers") + 1
                    expert_idx = parts.index("experts") + 1
                    layer_id = int(parts[layer_idx])
                    expert_id = int(parts[expert_idx])
                    param_name = parts[-2]
                    expert_keys[(layer_id, expert_id, param_name)] = (shard_path, key)
                except (ValueError, IndexError):
                    continue
    
    print(f"[扫描] 找到 {len(expert_keys)} 个专家权重")
    
    layers = {}
    for (layer_id, expert_id, param_name), (shard_path, key) in expert_keys.items():
        if layer_id not in layers:
            layers[layer_id] = {}
        if expert_id not in layers[layer_id]:
            layers[layer_id][expert_id] = {}
        layers[layer_id][expert_id][param_name] = (shard_path, key)
    
    all_data = {}
    total_processed = 0
    total_size = 0
    
    shard_metadata_cache = {}
    
    limit = args.limit if args.limit > 0 else len(expert_keys)
    start_time = _time.time()
    
    for layer_id in sorted(layers.keys()):
        layer_data = {}
        
        for expert_id in sorted(layers[layer_id].keys()):
            if total_processed >= limit:
                break
            
            expert_meta = layers[layer_id][expert_id]
            
            for param_name in ["w1", "w2", "w3"]:
                if param_name not in expert_meta:
                    continue
                shard_path, key = expert_meta[param_name]
                
                if shard_path not in shard_metadata_cache:
                    shard_metadata_cache[shard_path] = get_shard_metadata(shard_path)
                metadata = shard_metadata_cache[shard_path]
                
                weight = read_tensor(shard_path, key, metadata)
                weight_gpu = weight.cuda()
                del weight
                gc.collect()
                
                qdata = quantize_weight_gpu_optimized(weight_gpu, batch_size=args.batch_size, desc=f"L{layer_id}E{expert_id}.{param_name}")
                
                prefix = f"layers.{layer_id}.experts.{expert_id}.{param_name}"
                layer_data[f"{prefix}.d"] = qdata["d"].cpu()
                layer_data[f"{prefix}.qs"] = qdata["qs"].cpu()
                layer_data[f"{prefix}.scales"] = qdata["scales"].cpu()
                layer_data[f"{prefix}.shape"] = torch.tensor(qdata["shape"], dtype=torch.int64)
                
                total_size += qdata["d"].numel() * 2
                total_size += qdata["qs"].numel() * 2
                total_size += qdata["scales"].numel() * 1
                total_size += len(qdata["shape"]) * 8
                
                del weight_gpu, qdata
                torch.cuda.empty_cache()
                gc.collect()
            
            total_processed += 1
        
        all_data.update(layer_data)
        del layer_data
        gc.collect()
        
        if (layer_id + 1) % 5 == 0:
            elapsed = _time.time() - start_time
            avg_time = elapsed / total_processed if total_processed > 0 else 0
            remaining = avg_time * (limit - total_processed)
            print(f"  [进度] L{layer_id}, {total_processed}/{limit} 专家 "
                  f"({total_processed / limit * 100:.1f}%), "
                  f"已用 {elapsed:.0f}s, 剩余 ~{remaining:.0f}s, "
                  f"大小 {total_size / 1024**3:.2f}GB, "
                  f"内存: {get_memory_usage()}")
    
    del shard_metadata_cache
    gc.collect()
    
    print(f"\n[统计] 处理了 {total_processed} 个专家")
    print(f"[统计] 总大小: {total_size / 1024**3:.2f} GB")
    
    print(f"\n[保存] 写入 {args.output}...")
    save_file(all_data, args.output)
    
    output_size = os.path.getsize(args.output) / 1024**3
    print(f"[完成] 输出: {args.output}")
    print(f"[完成] 文件大小: {output_size:.2f} GB")


if __name__ == "__main__":
    main()
