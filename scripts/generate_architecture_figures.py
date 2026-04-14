#!/usr/bin/env python3
"""
Generate GPU architecture diagrams for MARS paper.

Creates publication-quality figures showing:
  1. MARS GPU memory architecture (SM layout, HBM, data flow)
  2. MARS retrieval pipeline (kernel-level dataflow with timing)
  3. Traditional vs MARS memory model comparison
  4. Scaling results with updated cuBLAS+CUB numbers

Style inspired by IEEE Access GPU architecture diagrams.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent.parent / 'paper' / 'figures'
OUTDIR.mkdir(exist_ok=True)

# Color palette
C_SM_BG      = '#4a86c8'   # SM blocks
C_SM_CORE    = '#7ab648'   # CUDA cores inside SM
C_SM_TENSOR  = '#f5a623'   # Tensor cores
C_SM_SFU     = '#e8744f'   # SFU units
C_CPU_BG     = '#6baed6'   # CPU cores
C_CACHE      = '#bdd7ee'   # Cache
C_HBM        = '#c6efce'   # HBM / global memory
C_HBM_DATA   = '#70ad47'   # Data in HBM
C_HOST       = '#f2dcdb'   # Host memory
C_ARROW      = '#2e75b6'   # Flow arrows
C_DASHED     = '#7f7f7f'   # Dashed outlines
C_BG_LIGHT   = '#f8f9fa'   # Light background
C_TITLE      = '#1a3a5c'   # Title color
C_RED        = '#c0392b'
C_PURPLE     = '#8e44ad'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'DejaVu Sans'],
    'font.size': 9,
})


def draw_sm_block(ax, x, y, w, h, label, num_cores=16, num_tensor=2):
    """Draw a Streaming Multiprocessor block with CUDA cores and tensor cores."""
    # SM background
    sm = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                         facecolor=C_SM_BG, edgecolor='white', linewidth=1.5,
                         alpha=0.9)
    ax.add_patch(sm)
    ax.text(x + w/2, y + h - 0.12, label, ha='center', va='top',
            fontsize=7, fontweight='bold', color='white')

    # CUDA cores grid (4x4)
    core_size = min(w, h) * 0.1
    cols, rows = 4, 4
    cx_start = x + (w - cols * core_size * 1.3) / 2
    cy_start = y + 0.08

    for row in range(rows):
        for col in range(cols):
            cx = cx_start + col * core_size * 1.3
            cy = cy_start + row * core_size * 1.3
            if row < rows and col < cols:
                color = C_SM_CORE
                core = FancyBboxPatch((cx, cy), core_size, core_size,
                                      boxstyle='round,pad=0.005',
                                      facecolor=color, edgecolor='white',
                                      linewidth=0.5, alpha=0.85)
                ax.add_patch(core)

    # Tensor cores (bottom row, orange squares)
    for i in range(num_tensor):
        tx = cx_start + i * core_size * 1.8
        ty = y + 0.04
        tc = FancyBboxPatch((tx, ty), core_size * 1.2, core_size * 0.8,
                             boxstyle='round,pad=0.005',
                             facecolor=C_SM_TENSOR, edgecolor='white',
                             linewidth=0.5, alpha=0.85)
        ax.add_patch(tc)

    # SFU (small red)
    sx = cx_start + num_tensor * core_size * 1.8 + 0.05
    sfu = FancyBboxPatch((sx, y + 0.04), core_size * 0.8, core_size * 0.8,
                          boxstyle='round,pad=0.005',
                          facecolor=C_SM_SFU, edgecolor='white',
                          linewidth=0.5, alpha=0.85)
    ax.add_patch(sfu)


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 1: MARS GPU Memory Architecture
# ═══════════════════════════════════════════════════════════════════
def fig_gpu_architecture():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor('white')

    # ─── GPU die outline ────────────────────────────────────────
    gpu_outline = FancyBboxPatch((0.3, 2.8), 9.4, 4.8,
                                  boxstyle='round,pad=0.1',
                                  facecolor='none', edgecolor=C_DASHED,
                                  linewidth=2, linestyle='--')
    ax.add_patch(gpu_outline)
    ax.text(5, 7.4, 'NVIDIA A100 SXM4 (108 SMs, 80 GB HBM2e)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=C_TITLE)

    # ─── SMs (4x2 grid) ────────────────────────────────────────
    sm_w, sm_h = 1.0, 0.9
    sm_labels = ['SM 1', 'SM 2', 'SM 3', 'SM 4',
                 'SM 5', 'SM 6', 'SM ...', 'SM 108']
    positions = [
        (0.6, 6.3), (1.9, 6.3), (3.2, 6.3), (4.5, 6.3),
        (0.6, 5.1), (1.9, 5.1), (3.2, 5.1), (4.5, 5.1),
    ]
    for (x, y), label in zip(positions, sm_labels):
        draw_sm_block(ax, x, y, sm_w, sm_h, label)

    # SM legend
    ax.add_patch(FancyBboxPatch((0.6, 4.55), 0.12, 0.12,
                 boxstyle='round,pad=0.01', facecolor=C_SM_CORE, edgecolor='white'))
    ax.text(0.8, 4.6, 'CUDA Core', fontsize=6, va='center')
    ax.add_patch(FancyBboxPatch((2.0, 4.55), 0.12, 0.12,
                 boxstyle='round,pad=0.01', facecolor=C_SM_TENSOR, edgecolor='white'))
    ax.text(2.2, 4.6, 'Tensor Core', fontsize=6, va='center')
    ax.add_patch(FancyBboxPatch((3.5, 4.55), 0.12, 0.12,
                 boxstyle='round,pad=0.01', facecolor=C_SM_SFU, edgecolor='white'))
    ax.text(3.7, 4.6, 'SFU', fontsize=6, va='center')

    # ─── L2 Cache ───────────────────────────────────────────────
    l2 = FancyBboxPatch((0.5, 3.9), 5.2, 0.45,
                         boxstyle='round,pad=0.03',
                         facecolor=C_CACHE, edgecolor=C_SM_BG,
                         linewidth=1.5)
    ax.add_patch(l2)
    ax.text(3.1, 4.12, 'L2 Cache (40 MB)', ha='center', va='center',
            fontsize=8, fontweight='bold', color=C_TITLE)

    # ─── HBM (Global Memory) — right side with data layout ─────
    hbm = FancyBboxPatch((6.2, 3.0), 3.3, 4.5,
                          boxstyle='round,pad=0.05',
                          facecolor=C_HBM, edgecolor=C_HBM_DATA,
                          linewidth=2)
    ax.add_patch(hbm)
    ax.text(7.85, 7.2, 'GPU HBM (80 GB)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C_TITLE)

    # Data blocks inside HBM
    data_blocks = [
        (6.5, 6.3, 2.7, 0.6, 'Embeddings  N×768×FP32', '#4472c4'),
        (6.5, 5.5, 2.7, 0.55, 'CSR Graph  (row_offsets + col_indices)', '#548235'),
        (6.5, 4.8, 2.7, 0.5, 'Timestamps + Modalities + Importance', '#bf8f00'),
        (6.5, 4.1, 1.25, 0.45, 'QueryCtx\nScratch', C_PURPLE),
        (7.95, 4.1, 1.25, 0.45, 'FP16 Copy\n(optional)', C_SM_TENSOR),
        (6.5, 3.3, 2.7, 0.55, 'BFS Frontiers + Compact Results', C_RED),
    ]
    for x, y, w, h, label, color in data_blocks:
        block = FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.03',
                                facecolor=color, edgecolor='white',
                                linewidth=1, alpha=0.8)
        ax.add_patch(block)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold')

    # ─── Arrows: SMs <-> L2 <-> HBM ────────────────────────────
    ax.annotate('', xy=(3.1, 4.4), xytext=(3.1, 5.05),
                arrowprops=dict(arrowstyle='<->', color=C_ARROW, lw=2))
    ax.annotate('', xy=(5.9, 4.12), xytext=(6.15, 4.12),
                arrowprops=dict(arrowstyle='<->', color=C_ARROW, lw=2))
    ax.text(6.05, 4.35, '1.5 TB/s', fontsize=6, color=C_ARROW,
            fontweight='bold', ha='center', rotation=90)

    # ─── Host side ──────────────────────────────────────────────
    host = FancyBboxPatch((0.5, 0.3), 4.5, 2.0,
                           boxstyle='round,pad=0.05',
                           facecolor=C_HOST, edgecolor=C_DASHED,
                           linewidth=1.5, linestyle='--')
    ax.add_patch(host)
    ax.text(2.75, 2.05, 'Host CPU', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C_TITLE)

    # Host components
    components = [
        (0.8, 1.2, 1.5, 0.55, 'MemoryGraph\n(NSN Builder)', C_CPU_BG),
        (2.6, 1.2, 1.5, 0.55, 'Query Vector\n(pinned mem)', C_CPU_BG),
        (0.8, 0.5, 1.5, 0.5, 'Persistence\n(save/load)', '#a9d18e'),
        (2.6, 0.5, 1.5, 0.5, 'Streaming\nBuffer', '#a9d18e'),
    ]
    for x, y, w, h, label, color in components:
        block = FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.03',
                                facecolor=color, edgecolor='white',
                                linewidth=1)
        ax.add_patch(block)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color=C_TITLE)

    # Host <-> GPU arrow (PCIe)
    ax.annotate('', xy=(5, 3.3), xytext=(5, 2.3),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=2.5))
    ax.text(5.15, 2.8, 'PCIe\nH2D/D2H', fontsize=7, color=C_RED,
            fontweight='bold', ha='left')

    # ─── Key insight box ────────────────────────────────────────
    insight = FancyBboxPatch((5.8, 0.3), 3.8, 2.0,
                              boxstyle='round,pad=0.08',
                              facecolor='#fff3cd', edgecolor='#ffc107',
                              linewidth=1.5)
    ax.add_patch(insight)
    ax.text(7.7, 2.05, 'MARS Design Principle', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#856404')
    ax.text(7.7, 1.5, 'All data GPU-resident.\n'
            'Zero per-query allocation.\n'
            'Pre-allocated QueryContext.\n'
            'Only query vector crosses PCIe.',
            ha='center', va='center', fontsize=7.5, color='#856404',
            linespacing=1.4)

    fig.savefig(OUTDIR / 'fig_gpu_arch.pdf', facecolor='white')
    fig.savefig(OUTDIR / 'fig_gpu_arch.png', facecolor='white')
    print(f'  Saved {OUTDIR / "fig_gpu_arch.pdf"}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 2: Traditional Circular Buffer vs MARS (side-by-side)
# ═══════════════════════════════════════════════════════════════════
def fig_memory_model_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

    for ax in (ax1, ax2):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 6)
        ax.axis('off')

    # ─── (a) Traditional: separate per-modality circular buffers ──
    ax1.text(2.5, 5.7, '(a) Traditional: Per-Modality Buffers',
             ha='center', fontsize=10, fontweight='bold', color=C_TITLE)

    # GPU die outline
    ax1.add_patch(FancyBboxPatch((0.2, 0.3), 4.6, 5.2,
                  boxstyle='round,pad=0.05', facecolor='none',
                  edgecolor=C_DASHED, linewidth=1.5, linestyle='--'))
    ax1.text(2.5, 5.25, 'GPU HBM', ha='center', fontsize=8, color=C_DASHED)

    # Three separate circular buffers
    colors = ['#4472c4', '#ed7d31', '#70ad47']
    labels = ['Visual Buffer\n(ring, 2K)', 'Audio Buffer\n(ring, 1K)',
              'LiDAR Buffer\n(ring, 500)']
    for i, (color, label) in enumerate(zip(colors, labels)):
        y_pos = 4.0 - i * 1.4
        buf = FancyBboxPatch((0.5, y_pos), 1.8, 1.0,
                              boxstyle='round,pad=0.05',
                              facecolor=color, edgecolor='white',
                              linewidth=1.5, alpha=0.7)
        ax1.add_patch(buf)
        ax1.text(1.4, y_pos + 0.5, label, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

    # Separate query paths
    for i in range(3):
        y_pos = 4.5 - i * 1.4
        ax1.annotate('', xy=(2.5, y_pos), xytext=(3.2, y_pos),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))

    # Results (separate)
    for i, mod in enumerate(['Visual\nresults', 'Audio\nresults', 'LiDAR\nresults']):
        y_pos = 4.2 - i * 1.4
        ax1.text(3.8, y_pos, mod, ha='center', va='center',
                fontsize=6, color=colors[i],
                bbox=dict(boxstyle='round', facecolor='white',
                         edgecolor=colors[i], linewidth=1))

    # Problems
    ax1.text(2.5, 0.6, 'No cross-modal queries\n'
             'No temporal decay\nPer-buffer allocation overhead',
             ha='center', va='center', fontsize=7, color=C_RED,
             style='italic',
             bbox=dict(boxstyle='round', facecolor='#fce4ec',
                      edgecolor=C_RED, linewidth=1))

    # ─── (b) MARS: unified GPU-resident graph ──────────────────
    ax2.text(2.5, 5.7, '(b) MARS: Unified GPU-Resident Graph',
             ha='center', fontsize=10, fontweight='bold', color=C_TITLE)

    ax2.add_patch(FancyBboxPatch((0.2, 0.3), 4.6, 5.2,
                  boxstyle='round,pad=0.05', facecolor='none',
                  edgecolor=C_DASHED, linewidth=1.5, linestyle='--'))
    ax2.text(2.5, 5.25, 'GPU HBM', ha='center', fontsize=8, color=C_DASHED)

    # Unified memory graph
    graph = FancyBboxPatch((0.4, 2.5), 4.2, 2.5,
                            boxstyle='round,pad=0.05',
                            facecolor=C_HBM, edgecolor=C_HBM_DATA,
                            linewidth=2)
    ax2.add_patch(graph)
    ax2.text(2.5, 4.7, 'Multimodal NSN Graph', ha='center', fontsize=9,
            fontweight='bold', color=C_TITLE)

    # Nodes with cross-modal bridges
    node_positions = [
        (1.0, 3.8, '#4472c4', 'V'), (1.8, 4.0, '#4472c4', 'V'),
        (2.6, 3.9, '#ed7d31', 'A'), (3.4, 4.1, '#ed7d31', 'A'),
        (1.4, 3.2, '#70ad47', 'L'), (2.2, 3.0, '#70ad47', 'L'),
        (3.0, 3.3, '#4472c4', 'V'), (3.8, 3.5, '#ed7d31', 'A'),
        (4.0, 2.9, '#70ad47', 'L'), (1.0, 3.0, '#4472c4', 'V'),
    ]
    for x, y, color, mod in node_positions:
        circle = plt.Circle((x, y), 0.18, facecolor=color,
                            edgecolor='white', linewidth=1.5, alpha=0.85)
        ax2.add_patch(circle)
        ax2.text(x, y, mod, ha='center', va='center',
                fontsize=6, color='white', fontweight='bold')

    # Draw edges (including cross-modal bridges)
    edges = [
        (1.0, 3.8, 1.8, 4.0), (1.8, 4.0, 2.6, 3.9),  # V-V, V-A bridge
        (2.6, 3.9, 3.4, 4.1), (3.4, 4.1, 3.8, 3.5),  # A-A, A-A
        (1.4, 3.2, 2.2, 3.0), (2.2, 3.0, 3.0, 3.3),  # L-L, L-V bridge
        (1.0, 3.8, 1.4, 3.2), (2.6, 3.9, 2.2, 3.0),  # V-L, A-L bridges
        (3.0, 3.3, 3.8, 3.5), (1.0, 3.0, 1.4, 3.2),  # V-A, V-L
        (3.8, 3.5, 4.0, 2.9),  # A-L bridge
    ]
    for x1, y1, x2, y2 in edges:
        ax2.plot([x1, x2], [y1, y2], '-', color='#666', linewidth=0.8, alpha=0.5)

    # Unified query + result
    ax2.text(2.5, 2.7, 'Embeddings (768-D) + Timestamps + Modalities',
            ha='center', fontsize=6.5, color=C_TITLE, style='italic')

    # Pipeline box
    pipe = FancyBboxPatch((0.4, 1.4), 4.2, 0.8,
                           boxstyle='round,pad=0.05',
                           facecolor='#e8f0fe', edgecolor=C_ARROW,
                           linewidth=1.5)
    ax2.add_patch(pipe)
    ax2.text(2.5, 1.8, 'cuBLAS SGEMV -> Temporal Decay -> CUB Top-K -> BFS',
            ha='center', va='center', fontsize=7, fontweight='bold',
            color=C_ARROW)

    # Advantages
    ax2.text(2.5, 0.6, 'Cross-modal BFS in <0.04 ms\n'
             'Fused temporal decay\nZero per-query allocation',
             ha='center', va='center', fontsize=7, color=C_HBM_DATA,
             fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#e6f4ea',
                      edgecolor=C_HBM_DATA, linewidth=1))

    fig.suptitle('Memory Architecture: Traditional Robotics Stack vs MARS',
                 fontsize=12, fontweight='bold', color=C_TITLE, y=0.98)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_memory_model.pdf', facecolor='white')
    fig.savefig(OUTDIR / 'fig_memory_model.png', facecolor='white')
    print(f'  Saved {OUTDIR / "fig_memory_model.pdf"}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 3: Updated scaling with cuBLAS+CUB results
# ═══════════════════════════════════════════════════════════════════
def fig_scaling_updated():
    fig, ax = plt.subplots(figsize=(6, 4))

    N_labels = ['1K', '5K', '10K', '20K', '50K']

    # Before: custom kernels
    p99_before = [0.671, 0.739, 0.841, 1.250, 1.331]
    # After: cuBLAS+CUB default (fused rerank)
    p99_after  = [0.309, 0.429, 0.444, 0.543, 0.555]

    x = np.arange(len(N_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, p99_before, width,
                   label='Before (custom kernels)',
                   color='#bdd7ee', edgecolor='#4472c4', linewidth=1.2)
    bars2 = ax.bar(x + width/2, p99_after, width,
                   label='Optimized (cuBLAS + CUB + fused rerank)',
                   color='#70ad47', edgecolor='#548235', linewidth=1.2)

    # 1ms deadline
    ax.axhline(y=1.0, color=C_RED, linestyle='--', linewidth=2,
               label='1 ms AV perception deadline', alpha=0.8)

    # Speedup annotations
    for i, (before, after) in enumerate(zip(p99_before, p99_after)):
        speedup = before / after
        ax.text(i + width/2, after + 0.03, f'{speedup:.1f}x',
                ha='center', fontsize=8, fontweight='bold', color='#2d6a2e')

    # FAIL -> PASS annotations
    for i in [3, 4]:
        ax.annotate('FAIL', xy=(i - width/2, p99_before[i]),
                   xytext=(i - width/2, p99_before[i] + 0.08),
                   ha='center', fontsize=7, color=C_RED, fontweight='bold')
        ax.annotate('PASS', xy=(i + width/2, p99_after[i]),
                   xytext=(i + width/2, p99_after[i] + 0.03),
                   ha='center', fontsize=7, color='#2d6a2e', fontweight='bold')

    ax.set_xlabel('Corpus Size', fontsize=11)
    ax.set_ylabel('Wall-Clock p99 Latency (ms)', fontsize=11)
    ax.set_title('MARS: cuBLAS+CUB Achieves Sub-ms at All Corpus Sizes\n'
                 '(A100 SXM4 40GB, D=768, K=10, 60 Hz)',
                 fontsize=10, fontweight='bold', color=C_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(N_labels)
    ax.set_ylim(0, 1.6)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_scaling_v04.pdf', facecolor='white')
    fig.savefig(OUTDIR / 'fig_scaling_v04.png', facecolor='white')
    print(f'  Saved {OUTDIR / "fig_scaling_v04.pdf"}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 4: Kernel-level pipeline with per-stage timing
# ═══════════════════════════════════════════════════════════════════
def fig_kernel_pipeline():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Pipeline stages as waterfall
    stages = [
        (0.5, 'H2D Copy\n(query)', 0.02, '#95a5a6', 'pinned\nasync'),
        (2.0, 'cuBLAS SGEMV\n+ Fused Decay', 0.25, '#4472c4', 'N×D\ndot products'),
        (5.0, 'CUB Radix\nSort Top-K', 0.02, '#70ad47', 'O(N)\nparallel'),
        (7.5, 'BFS Graph\nExpansion', 0.04, '#ed7d31', '2-hop\nwarp-coop'),
        (9.8, 'D2H\nCompact', 0.01, '#95a5a6', 'O(visited)'),
    ]

    bar_h = 1.2
    bar_y = 2.0

    for x, label, time_ms, color, detail in stages:
        # Scale bar width proportionally (min width for visibility)
        w = max(time_ms * 8, 0.8)

        bar = FancyBboxPatch((x, bar_y), w, bar_h,
                              boxstyle='round,pad=0.05',
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.85)
        ax.add_patch(bar)

        # Stage label
        ax.text(x + w/2, bar_y + bar_h/2 + 0.15, label,
                ha='center', va='center', fontsize=8, color='white',
                fontweight='bold')

        # Timing
        ax.text(x + w/2, bar_y - 0.25, f'{time_ms:.2f} ms',
                ha='center', fontsize=8, fontweight='bold', color=color)

        # Detail below
        ax.text(x + w/2, bar_y - 0.65, detail,
                ha='center', fontsize=6.5, color='#666', style='italic')

        # Arrow to next
        if x < 9:
            ax.annotate('', xy=(x + w + 0.15, bar_y + bar_h/2),
                        xytext=(x + w + 0.02, bar_y + bar_h/2),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # Title
    ax.text(6, 3.7, 'MARS Retrieval Pipeline — Per-Stage Timing (N=10K, A100)',
            ha='center', fontsize=11, fontweight='bold', color=C_TITLE)

    # Total time box
    ax.text(6, 0.4, 'Total wall-clock: 0.44 ms  |  GPU kernel: 0.21 ms  |  '
            'Overhead: 0.23 ms (launch + sync)',
            ha='center', fontsize=8, fontweight='bold', color=C_TITLE,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd',
                     edgecolor='#ffc107', linewidth=1.5))

    fig.savefig(OUTDIR / 'fig_kernel_pipeline.pdf', facecolor='white')
    fig.savefig(OUTDIR / 'fig_kernel_pipeline.png', facecolor='white')
    print(f'  Saved {OUTDIR / "fig_kernel_pipeline.pdf"}')
    plt.close()


if __name__ == '__main__':
    print('Generating MARS architecture figures...')
    fig_gpu_architecture()
    fig_memory_model_comparison()
    fig_scaling_updated()
    fig_kernel_pipeline()
    print('Done. All figures in paper/figures/')
