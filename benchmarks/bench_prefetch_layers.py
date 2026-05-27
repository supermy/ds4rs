#!/usr/bin/env python3
"""Predictive Prefetch 基准测试：prefetch_layers 1-6 命中率对比。

用法（容器内）:
  python3 benchmarks/bench_prefetch_layers.py

输出：
  每组配置的 GPU FFN 命中率、吞吐量、时间线分析。
"""

import subprocess
import sys
import re
import json

# 基准测试配置
PREFETCH_LAYERS_RANGE = range(1, 7)  # 1-6
PREFETCH_COUNT = 50
MAX_NEW_TOKENS = 30
PROMPT = "The capital of France is"

# Docker 执行命令模板
CMD_TEMPLATE = (
    'docker exec ds4rs-dev bash -c '
    '"cd /workspace && echo \'{prompt}\' | timeout 600 python3 inference/generate.py '
    '--ckpt-path /models --config /models/config.json --interactive '
    '--max-new-tokens {max_tokens} --quant-type iq2xxs_q2k '
    '--gguf-path /workspace/gguf/experts_iq2xxs_q2k.gguf '
    '--gpu-ffn --temperature 0.6 --top-p 0.95 --min-p 0.01 '
    '--prefetch-count {prefetch_count} --prefetch-layers {prefetch_layers}"'
)


def parse_stats(output: str) -> dict:
    """从推理输出中解析统计信息。"""
    stats = {}

    # 吞吐量: "[stats] 30 tokens in 60.00s → 0.5 t/s"
    m = re.search(r'\[stats\]\s+(\d+)\s+tokens\s+in\s+([\d.]+)s\s+→\s+([\d.]+)\s+t/s', output)
    if m:
        stats['total_tokens'] = int(m.group(1))
        stats['elapsed_s'] = float(m.group(2))
        stats['tokens_per_sec'] = float(m.group(3))

    # GPU FFN 命中率: "hits=100, misses=200, hit_rate=33.3%"
    m = re.search(r'hits=(\d+),\s+misses=(\d+),\s+hit_rate=([\d.]+)%', output)
    if m:
        stats['hits'] = int(m.group(1))
        stats['misses'] = int(m.group(2))
        stats['hit_rate'] = float(m.group(3))

    # GPU/CPU FFN 延迟: "gpu_ffn=1.24ms/hit, cpu_ffn=2.70ms/miss"
    m = re.search(r'gpu_ffn=([\d.]+)ms/hit,\s+cpu_ffn=([\d.]+)ms/miss', output)
    if m:
        stats['gpu_ffn_ms'] = float(m.group(1))
        stats['cpu_ffn_ms'] = float(m.group(2))

    # GPU/CPU FFN 总时间: "gpu_total=0.123s, cpu_total=0.456s"
    m = re.search(r'gpu_total=([\d.]+)s,\s+cpu_total=([\d.]+)s', output)
    if m:
        stats['gpu_total_s'] = float(m.group(1))
        stats['cpu_total_s'] = float(m.group(2))

    # 缓存大小: "cache=535/749"
    m = re.search(r'cache=(\d+)/(\d+)', output)
    if m:
        stats['cache_size'] = int(m.group(1))
        stats['cache_capacity'] = int(m.group(2))

    # 时间线: "attn  ...  50.0ms/step" 等
    timeline = {}
    for m in re.finditer(r'^\s+(\w+)\s+([\d.]+)ms/step\s+\(([\d.]+)%\)', output, re.MULTILINE):
        stage = m.group(1)
        ms_per_step = float(m.group(2))
        pct = float(m.group(3))
        timeline[stage] = {'ms_per_step': ms_per_step, 'pct': pct}
    if timeline:
        stats['timeline'] = timeline

    # 时间线总步数和延迟
    m = re.search(r'Timeline\s+\((\d+)\s+steps,\s+total=([\d.]+)s,\s+([\d.]+)ms/step\)', output)
    if m:
        stats['timeline_steps'] = int(m.group(1))
        stats['timeline_total_s'] = float(m.group(2))
        stats['timeline_ms_per_step'] = float(m.group(3))

    return stats


def run_benchmark(prefetch_layers: int) -> dict:
    """运行单次基准测试。"""
    cmd = CMD_TEMPLATE.format(
        prompt=PROMPT,
        max_tokens=MAX_NEW_TOKENS,
        prefetch_count=PREFETCH_COUNT,
        prefetch_layers=prefetch_layers,
    )
    print(f"\n{'='*60}")
    print(f"  Running: prefetch_layers={prefetch_layers}, prefetch_count={PREFETCH_COUNT}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    # 保存原始输出
    with open(f'/tmp/bench_prefetch_L{prefetch_layers}.log', 'w') as f:
        f.write(output)

    stats = parse_stats(output)
    stats['prefetch_layers'] = prefetch_layers
    stats['prefetch_count'] = PREFETCH_COUNT

    return stats


def print_summary(all_stats: list[dict]):
    """打印汇总对比表。"""
    print(f"\n\n{'='*90}")
    print(f"  Predictive Prefetch 基准测试汇总")
    print(f"  prefetch_count={PREFETCH_COUNT}, max_new_tokens={MAX_NEW_TOKENS}")
    print(f"{'='*90}")
    print(f"  {'Layers':>6s}  {'Hit%':>6s}  {'Hits':>6s}  {'Misses':>7s}  "
          f"{'t/s':>6s}  {'GPU ms':>7s}  {'CPU ms':>7s}  "
          f"{'ms/step':>8s}  {'Cache':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  "
          f"{'-'*6}  {'-'*7}  {'-'*7}  "
          f"{'-'*8}  {'-'*10}")

    for s in all_stats:
        hit_rate = s.get('hit_rate', 0)
        hits = s.get('hits', 0)
        misses = s.get('misses', 0)
        tps = s.get('tokens_per_sec', 0)
        gpu_ms = s.get('gpu_ffn_ms', 0)
        cpu_ms = s.get('cpu_ffn_ms', 0)
        ms_step = s.get('timeline_ms_per_step', 0)
        cache = f"{s.get('cache_size', '?')}/{s.get('cache_capacity', '?')}"
        layers = s.get('prefetch_layers', '?')

        print(f"  {layers:>6}  {hit_rate:>6.1f}  {hits:>6}  {misses:>7}  "
              f"{tps:>6.2f}  {gpu_ms:>7.2f}  {cpu_ms:>7.2f}  "
              f"{ms_step:>8.1f}  {cache:>10}")

    print(f"{'='*90}")

    # 时间线对比
    print(f"\n  时间线对比 (ms/step):")
    print(f"  {'Layers':>6s}  {'attn':>8s}  {'ffn':>8s}  {'gate':>8s}  {'expert':>8s}  {'shared':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for s in all_stats:
        tl = s.get('timeline', {})
        layers = s.get('prefetch_layers', '?')
        attn = tl.get('attn', {}).get('ms_per_step', 0)
        ffn = tl.get('ffn', {}).get('ms_per_step', 0)
        gate = tl.get('gate', {}).get('ms_per_step', 0)
        expert = tl.get('expert', {}).get('ms_per_step', 0)
        shared = tl.get('shared', {}).get('ms_per_step', 0)
        print(f"  {layers:>6}  {attn:>8.1f}  {ffn:>8.1f}  {gate:>8.1f}  {expert:>8.1f}  {shared:>8.1f}")
    print(f"{'='*90}")


def main():
    all_stats = []

    for pl in PREFETCH_LAYERS_RANGE:
        try:
            stats = run_benchmark(pl)
            all_stats.append(stats)
            # 中间结果
            print(f"  → hit_rate={stats.get('hit_rate', 0):.1f}%, "
                  f"t/s={stats.get('tokens_per_sec', 0):.2f}, "
                  f"ms/step={stats.get('timeline_ms_per_step', 0):.1f}")
        except subprocess.TimeoutExpired:
            print(f"  → TIMEOUT for prefetch_layers={pl}")
        except Exception as e:
            print(f"  → ERROR for prefetch_layers={pl}: {e}")

    if all_stats:
        print_summary(all_stats)
        # 保存 JSON 结果
        with open('/tmp/bench_prefetch_results.json', 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nResults saved to /tmp/bench_prefetch_results.json")


if __name__ == '__main__':
    main()
