"""
Plot inference-time comparison: v1 vs v2_cpu vs v2_gpu across 4 TIL datasets.

Single figure, 4 dataset groups x 3 grouped bars (predict wall seconds,
log y-axis), annotated with seconds and speedup vs v1.

Output: comparison/figs/inference_time_v1_vs_v2.{png,pdf}
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Nature Biotechnology style (matches plot_v1_vs_v2_comparison.py)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

COLORS = {
    'v1': '#888888',      # gray (baseline)
    'v2_cpu': '#4477AA',  # blue
    'v2_gpu': '#228833'   # green
}

DATASETS = ['steele', 'che', 'wang', 'bassez']


def load_predict_times():
    """Return {dataset: (v1_s, v2_cpu_s, v2_gpu_s, n_test)}."""
    out = {}
    for dataset in DATASETS:
        path = Path(f'comparison/result/pretraining/compare_{dataset}.json')
        with open(path) as f:
            data = json.load(f)
        t = data['time']
        out[dataset] = (t['v1']['predict_s'], t['v2_cpu']['predict_s'],
                        t['v2_gpu']['predict_s'], t['v1']['n_test'])
    return out


def plot():
    times = load_predict_times()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(DATASETS))
    width = 0.25

    v1 = [times[d][0] for d in DATASETS]
    vc = [times[d][1] for d in DATASETS]
    vg = [times[d][2] for d in DATASETS]

    b1 = ax.bar(x - width, v1, width, color=COLORS['v1'],
                edgecolor='black', linewidth=0.5, label='v1.0 (CPU)')
    b2 = ax.bar(x, vc, width, color=COLORS['v2_cpu'],
                edgecolor='black', linewidth=0.5, label='v2.0 (CPU)')
    b3 = ax.bar(x + width, vg, width, color=COLORS['v2_gpu'],
                edgecolor='black', linewidth=0.5, label='v2.0 (GPU)')

    ax.set_yscale('log')
    ax.set_ylabel('Inference wall time (s, log scale)', fontsize=11)
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{d.capitalize()}\n(n={times[d][3]:,})' for d in DATASETS])
    ax.set_title('Inference time: v1.0 vs Compocyte v2.0 (test split)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=False, fontsize=8)

    # Annotations: seconds + speedup vs v1, rotated 90 above each bar.
    for bars, vals, base in [(b1, v1, None), (b2, vc, v1), (b3, vg, v1)]:
        for bar, val, ref in zip(bars, vals, base if base else vals):
            if base is None:
                label = f'{val:.2f} s'
            else:
                label = f'{val:.2f} s\n{ref / val:.1f}\u00d7'
            ax.text(bar.get_x() + bar.get_width() / 2., val * 1.12, label,
                    ha='center', va='bottom', fontsize=7, rotation=90)

    # Headroom for annotations on log scale
    ax.set_ylim(top=max(v1) * 8)

    fig.tight_layout()

    out_dir = Path('comparison/figs')
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / 'inference_time_v1_vs_v2.png', dpi=300,
                bbox_inches='tight')
    fig.savefig(out_dir / 'inference_time_v1_vs_v2.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved comparison/figs/inference_time_v1_vs_v2.{png,pdf}')


if __name__ == '__main__':
    plot()
