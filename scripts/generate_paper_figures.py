#!/usr/bin/env python3
"""
Generate paper-quality figures for MARS results.
All measured on A100 SXM4 40GB, D=768, K=10.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Professional style (IEEE-compatible) ─────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

OUTDIR = Path(__file__).parent.parent / 'paper' / 'figures'
OUTDIR.mkdir(exist_ok=True)

# ── Colors ───────────────────────────────────────────────
C_BLUE    = '#2C73D2'
C_GREEN   = '#0CA678'
C_ORANGE  = '#E8590C'
C_RED     = '#E03131'
C_PURPLE  = '#7048E8'
C_GRAY    = '#868E96'
C_LIGHT   = '#DEE2E6'
C_BLUE_L  = '#A5D8FF'


def save(fig, name):
    fig.savefig(OUTDIR / f'{name}.pdf')
    fig.savefig(OUTDIR / f'{name}.png')
    print(f'  {name}.pdf')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  FIG 1: Full scaling 1K -> 13M
# ═══════════════════════════════════════════════════════════
def fig_scaling():
    fig, ax = plt.subplots(figsize=(7.5, 4))

    labels = ['1K', '5K', '10K', '20K', '50K', '100K', '200K',
              '500K', '1M', '5M', '10M', '13M']
    p99 = [0.309, 0.429, 0.444, 0.543, 0.555, 0.738, 1.031,
           1.669, 2.668, 11.42, 22.27, 29.07]

    x = np.arange(len(labels))
    colors = [C_GREEN if v < 1.0 else C_BLUE if v < 5.0 else C_ORANGE
              for v in p99]

    bars = ax.bar(x, p99, 0.6, color=colors, edgecolor='white', linewidth=0.6)

    # Deadline reference lines
    ax.axhline(1.0, color=C_RED, ls='--', lw=1.5, alpha=0.7, label='1 ms (AV, 60 Hz)')
    ax.axhline(5.0, color=C_ORANGE, ls=':', lw=1.2, alpha=0.4, label='5 ms (AR/VR)')
    ax.axhline(20.0, color=C_PURPLE, ls=':', lw=1.2, alpha=0.4, label='20 ms (Voice)')

    # Value labels
    for bar, val in zip(bars, p99):
        txt = f'{val:.2f}' if val < 1 else f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4,
                txt, ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=bar.get_facecolor())

    ax.set_xlabel('Corpus Size')
    ax.set_ylabel('Wall-Clock p99 Latency (ms)')
    ax.set_title('MARS Retrieval Latency: 1K to 13M Memories\n'
                 'A100 SXM4 40 GB, D=768, K=10, cuBLAS+CUB')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_ylim(0, 34)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')

    fig.tight_layout()
    save(fig, 'fig_scaling')


# ═══════════════════════════════════════════════════════════
#  FIG 2: Before vs After optimization
# ═══════════════════════════════════════════════════════════
def fig_optimization_comparison():
    fig, ax = plt.subplots(figsize=(6.5, 4))

    labels = ['1K', '5K', '10K', '20K', '50K']
    before = [0.671, 0.739, 0.841, 1.250, 1.331]
    after  = [0.309, 0.429, 0.444, 0.543, 0.555]

    x = np.arange(len(labels))
    w = 0.32

    b1 = ax.bar(x - w/2, before, w, label='Before (custom kernels)',
                color=C_BLUE_L, edgecolor=C_BLUE, linewidth=1.2)
    b2 = ax.bar(x + w/2, after, w, label='Optimized (cuBLAS + CUB + fused)',
                color=C_GREEN, edgecolor='white', linewidth=0.6)

    ax.axhline(1.0, color=C_RED, ls='--', lw=2, alpha=0.7,
               label='1 ms AV deadline')

    # Speedup labels
    for i, (bef, aft) in enumerate(zip(before, after)):
        ax.text(i + w/2, aft + 0.02, f'{bef/aft:.1f}x',
                ha='center', fontsize=9, fontweight='bold', color='#1B7A41')

    # FAIL/PASS markers
    for i in [3, 4]:
        ax.text(i - w/2, before[i] + 0.02, 'FAIL',
                ha='center', fontsize=8, fontweight='bold', color=C_RED)
        ax.text(i + w/2, after[i] + 0.02 + 0.04, 'PASS',
                ha='center', fontsize=8, fontweight='bold', color='#1B7A41')

    ax.set_xlabel('Corpus Size')
    ax.set_ylabel('Wall-Clock p99 Latency (ms)')
    ax.set_title('Optimization Impact: cuBLAS + CUB + Fused Rerank\n'
                 'A100 SXM4, D=768, K=10, 60 Hz')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.55)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='none')

    fig.tight_layout()
    save(fig, 'fig_scaling_v04')


# ═══════════════════════════════════════════════════════════
#  FIG 3: Kernel backend comparison
# ═══════════════════════════════════════════════════════════
def fig_kernel_backends():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    configs = ['FP32\nCustom', 'FP16\nScalar', 'FP16\nWMMA', 'cuBLAS\n+CUB']
    c = [C_BLUE, C_ORANGE, C_RED, C_GREEN]

    # N=10K
    v10 = [0.973, 1.008, 0.897, 0.778]
    bars1 = ax1.bar(configs, v10, color=c, edgecolor='white', linewidth=0.6)
    ax1.axhline(1.0, color=C_RED, ls='--', lw=1.2, alpha=0.5)
    ax1.set_title('N = 10,000', fontweight='bold')
    ax1.set_ylabel('Wall-Clock p99 Latency (ms)')
    for bar, val in zip(bars1, v10):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

    # N=50K
    v50 = [1.530, 1.547, 1.138, 0.559]
    bars2 = ax2.bar(configs, v50, color=c, edgecolor='white', linewidth=0.6)
    ax2.axhline(1.0, color=C_RED, ls='--', lw=1.2, alpha=0.5)
    ax2.set_title('N = 50,000', fontweight='bold')
    for bar, val in zip(bars2, v50):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    # Speedup arrow
    ax2.annotate('2.7x faster', xy=(3, 0.559), xytext=(2.5, 1.2),
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=2),
                fontsize=10, fontweight='bold', color=C_GREEN, ha='center')

    fig.suptitle('Kernel Backend Comparison (A100 SXM4, D=768, K=10)',
                 fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.75)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'fig_optimization')


# ═══════════════════════════════════════════════════════════
#  FIG 4: Deadline compliance
# ═══════════════════════════════════════════════════════════
def fig_deadline():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # Five workloads — added MARS Episode-scoped at N=1M (the new winner).
    demos   = ['Voice Agent\n(30 Hz, N=10K)',
               'AR/VR Spatial\n(90 Hz, N=10K)',
               'Humanoid Robot\n(1 kHz, N=10K)',
               'AV Perception\n(60 Hz, N=10K)',
               'MARS Episode-scoped\n(any rate, N=1M)']
    p99     = [0.880, 1.560, 0.760, 0.870, 0.197]
    budgets = [20.0,  5.0,   1.0,   1.0,   1.0]
    colors  = [C_GREEN, C_GREEN, C_GREEN, C_GREEN, '#0CA678']

    y = np.arange(len(demos))
    h = 0.5

    bars = ax.barh(y, p99, h, color=colors, edgecolor='white', linewidth=0.6,
                   zorder=3)
    # Highlight the MARS Episode-scoped row with a darker edge
    bars[-1].set_edgecolor('#1B7A41')
    bars[-1].set_linewidth(1.4)

    for i, (m, b) in enumerate(zip(p99, budgets)):
        ax.plot(b, i, 'D', color=C_RED, markersize=10, zorder=5,
                markeredgecolor='white', markeredgewidth=1)
        ax.plot([m, b], [i, i], '--', color=C_LIGHT, lw=1, zorder=2)
        pct = (1 - m/b) * 100
        # Print the measured value just to the right of the bar tip.
        # For tight 1-ms budgets the bar-tip is near the diamond, so we
        # vertically offset above-the-row to keep them separated.
        if b >= 10:
            ax.text(m + 0.20, i, f'{m:.2f} ms',
                    va='center', ha='left', fontsize=9,
                    color='#1F6FB5', fontweight='bold')
        else:
            ax.text(m, i + 0.30, f'{m:.2f} ms',
                    va='bottom', ha='center', fontsize=8.5,
                    color='#1B7A41' if i == len(demos) - 1 else '#1F6FB5',
                    fontweight='bold')
        # Place the headroom label BELOW the row to keep the bar/diamond
        # area clean even when the budget is the same x as the diamond.
        if b >= 10:
            # Voice Agent: lots of room — keep label beyond the diamond
            ax.text(b + 0.4, i, f'{pct:.0f}% headroom',
                    va='center', ha='left', fontsize=9.5,
                    color=C_GRAY, fontweight='bold')
        else:
            # Tight 1-ms / 5-ms budgets: label sits just below the row
            ax.text(b + 0.05, i - 0.36, f'{pct:.0f}% headroom',
                    va='top', ha='left', fontsize=9,
                    color=C_GRAY, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(demos)
    ax.set_xlabel('Latency (ms)')
    ax.set_title('Deadline Compliance -- All Demos PASS (A100 SXM4 40 GB)\n'
                 'Episode-scoped delivers a perfect-recall multimodal answer with 80% headroom on the 1 ms AV deadline at N=1M',
                 fontweight='bold', fontsize=10.5)
    ax.set_xlim(0, 23)

    measured_p = mpatches.Patch(color=C_GREEN, label='Measured p99 (global path)')
    measured_e = mpatches.Patch(color='#0CA678', label='Measured p99 (episode-scoped)')
    budget_m = plt.Line2D([0], [0], marker='D', color='w',
                          markerfacecolor=C_RED, markersize=9,
                          label='Deadline budget', markeredgecolor='white')
    ax.legend(handles=[measured_p, measured_e, budget_m],
              loc='upper center', bbox_to_anchor=(0.5, -0.18),
              framealpha=0.95, edgecolor='none', ncol=3)

    fig.tight_layout()
    save(fig, 'fig_deadline')


# ═══════════════════════════════════════════════════════════
#  FIG 5: VRAM budget breakdown
# ═══════════════════════════════════════════════════════════
def fig_vram():
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = ['1K', '10K', '50K', '100K', '1M', '10M', '13M']
    Ns     = [1000, 10000, 50000, 100000, 1000000, 10000000, 13000000]
    D = 768; deg = 12

    emb  = [N * D * 4 / 1e6 for N in Ns]
    csr  = [(N+1+N*deg) * 4 / 1e6 for N in Ns]
    meta = [N * 16 / 1e6 for N in Ns]
    scr  = [N * 16 / 1e6 + 0.5 for N in Ns]

    x = np.arange(len(labels))
    w = 0.55

    ax.bar(x, emb, w, label='Embeddings (N x 768 x 4B)', color=C_BLUE)
    ax.bar(x, csr, w, bottom=emb, label='CSR graph', color=C_GREEN)
    b3 = [e+c for e, c in zip(emb, csr)]
    ax.bar(x, meta, w, bottom=b3, label='Metadata', color=C_ORANGE)
    b4 = [a+m for a, m in zip(b3, meta)]
    ax.bar(x, scr, w, bottom=b4, label='QueryContext scratch', color=C_PURPLE)

    totals = [e+c+m+s for e, c, m, s in zip(emb, csr, meta, scr)]
    for i, t in enumerate(totals):
        lbl = f'{t/1024:.1f} GB' if t > 500 else f'{t:.0f} MB'
        ax.text(i, t + max(totals)*0.02, lbl,
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Corpus Size')
    ax.set_ylabel('VRAM Usage (MB)')
    ax.set_title('MARS VRAM Budget Breakdown (D=768, avg degree=12)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(totals) * 1.12)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='none')

    fig.tight_layout()
    save(fig, 'fig_vram_budget')


if __name__ == '__main__':
    print('Generating figures...')
    fig_scaling()
    fig_optimization_comparison()
    fig_kernel_backends()
    fig_deadline()
    fig_vram()
    print('Done.')
