"""
P2: NHWC/Tile 内存布局优化

目标：提升 L1/L2 缓存命中率

当前布局（行主序 NCHW）：
  - qs: [n_blocks, 32] 行主序
  - 访问模式：qs[bi, g] 跳跃访问

优化布局（NHWC/Tile）：
  - qs_tile: [n_blocks_tile, 32, TILE_SIZE] Tile 主序
  - 访问模式：qs_tile[blk_tile][g] 连续访问

Tile 大小选择：
  - L1: 32KB → TILE_SIZE = 8 blocks (2KB/tile)
  - L2: 1MB → TILE_ROWS = 64 rows
"""

import numpy as np
from typing import Tuple
import time

TILE_SIZE = 8
TILE_ROWS = 64


def convert_to_tile_layout(
    d: np.ndarray,
    qs: np.ndarray,
    scales: np.ndarray,
    shape: Tuple[int, int],
) -> dict:
    """将行主序布局转换为 Tile 布局。

    参数：
      d: [n_blocks] float16
      qs: [n_blocks, 32] uint16
      scales: [n_blocks, 8] uint8
      shape: (n_rows, n_cols)

    返回：
      tile_data: {
        'd_tiles': [n_tiles, TILE_SIZE] float32
        'qs_tiles': [n_tiles, 32, TILE_SIZE] uint16
        'scales_tiles': [n_tiles, 8, TILE_SIZE] uint8
        'shape': (n_rows, n_cols)
        'n_tiles': int
      }
    """
    n_rows, n_cols = shape
    n_blocks_per_row = n_cols // 256
    n_blocks = n_rows * n_blocks_per_row

    d = d.ravel()[:n_blocks]
    qs = qs.reshape(n_blocks, 32)
    scales = scales.reshape(n_blocks, 8)

    n_tiles = (n_blocks + TILE_SIZE - 1) // TILE_SIZE

    d_tiles = np.zeros((n_tiles, TILE_SIZE), dtype=np.float32)
    qs_tiles = np.zeros((n_tiles, 32, TILE_SIZE), dtype=np.uint16)
    scales_tiles = np.zeros((n_tiles, 8, TILE_SIZE), dtype=np.uint8)

    for t in range(n_tiles):
        start = t * TILE_SIZE
        end = min(start + TILE_SIZE, n_blocks)
        size = end - start

        d_tiles[t, :size] = d[start:end].astype(np.float32)
        qs_tiles[t, :, :size] = qs[start:end].T
        scales_tiles[t, :, :size] = scales[start:end].T

    return {
        'd_tiles': d_tiles,
        'qs_tiles': qs_tiles,
        'scales_tiles': scales_tiles,
        'shape': shape,
        'n_tiles': n_tiles,
        'n_blocks': n_blocks,
        'n_blocks_per_row': n_blocks_per_row,
    }


def iq2xs_matvec_tile(
    tile_data: dict,
    x: np.ndarray,
    grid: np.ndarray,
    sign_table: np.ndarray,
    scale_table: np.ndarray,
) -> np.ndarray:
    """Tile 布局的 IQ2_XS 矩阵向量乘法。

    参数：
      tile_data: Tile 布局数据
      x: [n_cols] float32
      grid: [512, 8] int8
      sign_table: [128, 8] float32
      scale_table: [256, 2] float32

    返回：
      output: [n_rows] float32
    """
    d_tiles = tile_data['d_tiles']
    qs_tiles = tile_data['qs_tiles']
    scales_tiles = tile_data['scales_tiles']
    n_rows = tile_data['shape'][0]
    n_blocks_per_row = tile_data['n_blocks_per_row']

    output = np.zeros(n_rows, dtype=np.float32)
    x_groups = x.reshape(-1, 8)

    for row in range(n_rows):
        row_sum = 0.0
        row_offset = row * n_blocks_per_row

        for blk in range(n_blocks_per_row):
            bi = row_offset + blk
            tile_idx = bi // TILE_SIZE
            tile_offset = bi % TILE_SIZE

            d_val = d_tiles[tile_idx, tile_offset]
            block_sum = 0.0

            for g in range(32):
                q = qs_tiles[tile_idx, g, tile_offset]
                gi = q & 511
                si = (q >> 9) & 127

                ib32 = g >> 2
                within = g & 3
                sc_val = scales_tiles[tile_idx, ib32, tile_offset]
                ls = scale_table[sc_val, 0 if within < 2 else 1]

                x_base = blk * 32 + g
                group_dot = 0.0

                for j in range(8):
                    gv = grid[gi, j]
                    sm = sign_table[si, j]
                    group_dot += gv * sm * x_groups[x_base, j]

                block_sum += ls * group_dot

            row_sum += d_val * 0.125 * block_sum

        output[row] = row_sum

    return output


def benchmark_tile_layout():
    """Tile 布局性能测试。"""
    dim = 4096
    n_rows = 128
    n_blocks = n_rows * dim // 256

    rng = np.random.RandomState(42)
    d = rng.randn(n_blocks).astype(np.float16) * 0.01
    qs = rng.randint(0, 65535, (n_blocks, 32), dtype=np.uint16)
    scales = rng.randint(0, 256, (n_blocks, 8), dtype=np.uint8)

    tile_data = convert_to_tile_layout(d, qs, scales, (n_rows, dim))

    grid = np.zeros((512, 8), dtype=np.int8)
    sign_table = np.ones((128, 8), dtype=np.float32)
    scale_table = np.ones((256, 2), dtype=np.float32)

    x = np.random.randn(dim).astype(np.float32)

    n_iter = 10
    times = []

    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = iq2xs_matvec_tile(tile_data, x, grid, sign_table, scale_table)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000

    print(f"Tile layout matvec ({n_rows}×{dim}):")
    print(f"  avg: {avg_ms:.2f}ms, min: {min_ms:.2f}ms")

    tile_size_mb = (
        tile_data['d_tiles'].nbytes +
        tile_data['qs_tiles'].nbytes +
        tile_data['scales_tiles'].nbytes
    ) / (1024 * 1024)

    print(f"  Tile data size: {tile_size_mb:.2f}MB")


if __name__ == "__main__":
    benchmark_tile_layout()
