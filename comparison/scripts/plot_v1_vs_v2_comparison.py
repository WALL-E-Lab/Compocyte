"""
Plot v1 vs v2.0 comparison figures for TIL and PBMC datasets.

Generates one figure per dataset, each with:
- Left panel: Speedup (Train + Total) for v1/v2_cpu/v2_gpu
- Right panel: Per-level accuracy AND coverage (grouped bars)

Output: comparison/figs/{dataset}_v1_vs_v2_comparison.{png,pdf}

Example:
    python comparison/scripts/plot_v1_vs_v2_comparison.py
    python comparison/scripts/plot_v1_vs_v2_comparison.py --domain pbmc --datasets oelen_e1
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Nature Biotechnology style
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

# Color scheme
COLORS = {
    'v1': '#888888',      # gray (baseline)
    'v2_cpu': '#4477AA',  # blue
    'v2_gpu': '#228833'   # green
}

DEFAULT_DATASETS = ['steele', 'che', 'wang', 'bassez']

RESULT_BASE = {
    'til': Path('comparison/result/pretraining'),
    'pbmc': Path('comparison/result/pretraining_pbmc'),
}

import re as _re


def load_dataset(dataset, domain='til'):
    """Load comparison JSON for a dataset."""
    path = RESULT_BASE[domain] / 'compare_{dataset}.json'.format(dataset=dataset)
    with open(path) as f:
        return json.load(f)


def dataset_levels(data):
    """Derive ordered prediction levels from the comparison payload."""
    levels = list(data['correctness_vs_truth']['v1']['per_level_vs_truth'].keys())

    def _key(col):
        m = _re.search(r'\d+', col)
        return int(m.group()) if m else 0

    return sorted(levels, key=_key)


def level_labels(levels):
    labels = []
    for lvl in levels:
        m = _re.search(r'(\d+)', lvl)
        labels.append('L{}'.format(m.group(1)) if m else lvl)
    return labels

def plot_speedup_panel(ax, data, dataset):
    """Left panel: speedup comparison (Train + Total)."""
    speedups = data['speedup_vs_v1']
    
    # Extract speedup values
    train_speedups = [1.0, speedups['v2_cpu']['train'], speedups['v2_gpu']['train']]
    total_speedups = [1.0, speedups['v2_cpu']['total'], speedups['v2_gpu']['total']]
    
    # Bar positions
    x = np.arange(2)
    width = 0.25
    
    # Plot train speedups (group 1)
    bars_train_v1 = ax.bar(x[0] - width, train_speedups[0], width, color=COLORS['v1'],
                           edgecolor='black', linewidth=0.5, label='v1.0 (CPU)')
    bars_train_v2cpu = ax.bar(x[0], train_speedups[1], width, color=COLORS['v2_cpu'],
                              edgecolor='black', linewidth=0.5, label='v2.0 (CPU)')
    bars_train_v2gpu = ax.bar(x[0] + width, train_speedups[2], width, color=COLORS['v2_gpu'],
                              edgecolor='black', linewidth=0.5, label='v2.0 (GPU)')
    
    # Plot total speedups (group 2)
    bars_total_v1 = ax.bar(x[1] - width, total_speedups[0], width, color=COLORS['v1'],
                           edgecolor='black', linewidth=0.5)
    bars_total_v2cpu = ax.bar(x[1], total_speedups[1], width, color=COLORS['v2_cpu'],
                              edgecolor='black', linewidth=0.5)
    bars_total_v2gpu = ax.bar(x[1] + width, total_speedups[2], width, color=COLORS['v2_gpu'],
                              edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars_train_v1, bars_train_v2cpu, bars_train_v2gpu,
                 bars_total_v1, bars_total_v2cpu, bars_total_v2gpu]:
        for bar in bars:
            height = bar.get_height()
            if height > 1.0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}×', ha='center', va='bottom', fontsize=8)
    
    # Reference line at 1.0×
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Phase', fontsize=10)
    ax.set_ylabel('Speedup (×v1.0)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Training', 'Total'])
    ax.set_ylim(0, max(total_speedups) * 1.2)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.set_title(f'Dataset: {dataset.capitalize()}', fontsize=11, fontweight='bold', pad=10)

def _pct(value):
    return value * 100 if value is not None else 0.0


def plot_accuracy_coverage_panel(ax_acc, ax_cov, data):
    """Right panel: per-level accuracy and coverage."""
    correctness = data['correctness_vs_truth']
    levels = dataset_levels(data)
    labels = level_labels(levels)

    # Extract accuracy and coverage for all 3 frameworks
    accuracy = {
        fw: [_pct(correctness[fw]['per_level_vs_truth'][lvl]['accuracy'])
             for lvl in levels]
        for fw in ('v1', 'v2_cpu', 'v2_gpu')
    }

    coverage = {
        fw: [correctness[fw]['per_level_vs_truth'][lvl]['coverage'] * 100
             for lvl in levels]
        for fw in ('v1', 'v2_cpu', 'v2_gpu')
    }

    x = np.arange(len(levels))
    width = 0.25
    
    # Accuracy subplot
    bars_acc_v1 = ax_acc.bar(x - width, accuracy['v1'], width, color=COLORS['v1'],
                             edgecolor='black', linewidth=0.5, label='v1.0')
    bars_acc_v2cpu = ax_acc.bar(x, accuracy['v2_cpu'], width, color=COLORS['v2_cpu'],
                                edgecolor='black', linewidth=0.5, label='v2.0 (CPU)')
    bars_acc_v2gpu = ax_acc.bar(x + width, accuracy['v2_gpu'], width, color=COLORS['v2_gpu'],
                                edgecolor='black', linewidth=0.5, label='v2.0 (GPU)')
    
    ax_acc.set_ylabel('Accuracy (%)', fontsize=10)
    ax_acc.set_ylim(0, 110)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, rotation=0)
    ax_acc.legend(loc='lower left', frameon=False, fontsize=7)
    ax_acc.set_title('Per-Level Accuracy', fontsize=10, pad=5)
    
    # Add value labels (only if accuracy < 95 to avoid clutter)
    for bars in [bars_acc_v1, bars_acc_v2cpu, bars_acc_v2gpu]:
        for bar in bars:
            height = bar.get_height()
            if height < 95:
                ax_acc.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.0f}', ha='center', va='bottom',
                           fontsize=6, rotation=90)
    
    # Coverage subplot
    bars_cov_v1 = ax_cov.bar(x - width, coverage['v1'], width, color=COLORS['v1'],
                             edgecolor='black', linewidth=0.5)
    bars_cov_v2cpu = ax_cov.bar(x, coverage['v2_cpu'], width, color=COLORS['v2_cpu'],
                                edgecolor='black', linewidth=0.5)
    bars_cov_v2gpu = ax_cov.bar(x + width, coverage['v2_gpu'], width, color=COLORS['v2_gpu'],
                                edgecolor='black', linewidth=0.5)
    
    ax_cov.set_ylabel('Coverage (%)', fontsize=10)
    ax_cov.set_xlabel('Hierarchy Level', fontsize=10)
    ax_cov.set_ylim(0, 110)
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(labels, rotation=0)
    ax_cov.set_title('Per-Level Coverage', fontsize=10, pad=5)
    
    # Add value labels
    for bars in [bars_cov_v1, bars_cov_v2cpu, bars_cov_v2gpu]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # only label if coverage > 5%
                ax_cov.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.0f}', ha='center', va='bottom',
                           fontsize=6, rotation=90)

def plot_dataset(dataset, domain='til'):
    """Generate figure for one dataset."""
    print(f'Plotting {dataset} ({domain})...')
    data = load_dataset(dataset, domain)
    
    # Create figure with 1x2 grid
    fig = plt.figure(figsize=(14, 5))
    
    # Grid spec: left panel (speedup) takes 1/3 width, right panel takes 2/3
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.4, wspace=0.3)
    
    # Left panel: speedup
    ax_speedup = fig.add_subplot(gs[:, 0])  # spans both rows
    plot_speedup_panel(ax_speedup, data, dataset)
    
    # Right panel: accuracy (top) and coverage (bottom)
    ax_accuracy = fig.add_subplot(gs[0, 1])
    ax_coverage = fig.add_subplot(gs[1, 1])
    plot_accuracy_coverage_panel(ax_accuracy, ax_coverage, data)
    
    # Add dataset metadata as annotation
    ref_fw = 'v1' if 'v1' in data['time'] else 'v2_cpu'
    n_train = data['time'][ref_fw]['n_train']
    n_test = data['time'][ref_fw]['n_test']
    metadata_text = f'n_train={n_train:,}  n_test={n_test:,}'
    fig.text(0.02, 0.02, metadata_text, fontsize=7, style='italic', color='gray')
    
    # Save
    out_dir = Path('comparison/figs')
    out_dir.mkdir(exist_ok=True)
    
    fig.savefig(out_dir / f'{dataset}_v1_vs_v2_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'{dataset}_v1_vs_v2_comparison.pdf', bbox_inches='tight')
    
    plt.close(fig)
    print(f'  ✓ Saved {dataset} figures')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', choices=['til', 'pbmc'], default='til')
    ap.add_argument('--datasets', nargs='+', default=None)
    args = ap.parse_args()
    datasets = args.datasets or (DEFAULT_DATASETS if args.domain == 'til' else [])
    if not datasets:
        raise SystemExit('No datasets requested: pass --datasets for --domain pbmc.')
    print('Generating v1 vs v2.0 comparison figures...\n')
    for dataset in datasets:
        plot_dataset(dataset, args.domain)
    print('\nAll figures saved to comparison/figs/')
