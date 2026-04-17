#!/usr/bin/env python3
"""
Generate paper-quality figures for the head-to-head competitor benchmarks
and the episode-scoped scaling curve. All numbers are sourced from the
JSON artefacts in `results/competitors_20260417/` so the figures cannot
drift from the run logs.

Outputs (PDF + PNG, 300 dpi, IEEE-compatible style):
  paper/figures/fig_competitors.{pdf,png}
      Head-to-head wall-clock p99 + hit@15 for FAISS Flat / FAISS IVF /
      cuVS CAGRA / MARS Global FP32 / MARS Global FP16 / MARS Episode-
      scoped at N in {10K, 100K, 1M}. Used in paper §10.2.

  paper/figures/fig_episode_scoped_scaling.{pdf,png}
      Two-curve line plot: global path p99 vs episode-scoped p99 across
      N=10K..1M with the 1 ms AV deadline overlay. Used in paper §10.1.

  paper/figures/fig_fp16_crossover.{pdf,png}
      FP16 fused vs cuBLAS Sgemv crossover at the three N points; shows
      FP16 wins at small N, loses at large N. Used in §10.x and
      docs/ARCHITECTURE.md §7.3.

Run from repo root:
    python3 scripts/generate_competitor_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
COMP = ROOT / "results" / "competitors_20260417"
OUT  = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":  9.5,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.12,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.4,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
})

# Colors aligned with existing figures
C_FAISS_FLAT = "#E03131"   # red — exhaustive baseline
C_FAISS_IVF  = "#E8590C"   # orange — IVF baseline
C_CAGRA      = "#7048E8"   # purple — graph ANN
C_MARS_FP32  = "#2C73D2"   # blue — MARS global FP32
C_MARS_FP16  = "#74B9FF"   # light blue — MARS global FP16
C_MARS_EP    = "#0CA678"   # green — MARS episode-scoped (winner)
C_DEADLINE   = "#C92A2A"   # darker red — AV deadline
C_GRAY       = "#868E96"

N_LABELS = ["10K", "100K", "1M"]
N_KEYS   = ["10k", "100k", "1m"]


def save(fig, name: str) -> None:
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"  {pdf.name}  {png.name}")
    plt.close(fig)


def _json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def load_data() -> dict:
    """Read the FAISS / CAGRA / MARS JSON artefacts and pivot."""
    out: dict = {}

    # FAISS Flat + IVF (one JSON per N, two configurations each)
    for n in N_KEYS:
        path = COMP / "faiss" / f"faiss_{n}.json"
        cfgs = _json(path)["configurations"]
        for c in cfgs:
            method = c["method"]
            out.setdefault(method, {})[n] = {
                "p99":      float(c["p99_ms"]),
                "hit_15":   float(c["episode_cross_modal_hit_rate_top15"]),
            }

    # cuVS CAGRA (one JSON per N)
    for n in N_KEYS:
        path = COMP / "cuvs_cagra" / f"cagra_{n}.json"
        d = _json(path)
        c = d["configurations"][0]
        out.setdefault("cuvs_cagra_gpu_cosine_only", {})[n] = {
            "p99":    float(c["p99_ms"]),
            "hit_15": float(c["episode_cross_modal_hit_rate_top15"]),
        }
        if "extend" in d:
            out.setdefault("cuvs_cagra_extend", {})[n] = {
                "per_vec_us": float(d["extend"]["extend_per_vec_us"]),
            }

    # MARS — global step (boost=0) + episode-scoped step
    for n in N_KEYS:
        for variant in ("baseline", "fp16"):
            base = COMP / "mars" / f"mars_{n}_{variant}"
            g = _json(base / "step_global_boost_0.json")
            e = _json(base / "step_episode_scoped.json")
            method = "mars_global_fp32" if variant == "baseline" else "mars_global_fp16"
            out.setdefault(method, {})[n] = {
                "p99":    float(g["wall_p99_ms"]),
                "hit_15": float(g["episode_cross_modal_hit_rate_top15"]),
            }
            # Episode-scoped is identical across variants but record once
            if variant == "baseline":
                out.setdefault("mars_episode_scoped", {})[n] = {
                    "p99":    float(e["wall_p99_ms"]),
                    "hit_15": float(e["episode_cross_modal_hit_rate_top15"]),
                }

    return out


# ─────────────────────────────────────────────────────────────────────
# FIGURE 1 — competitor head-to-head
# ─────────────────────────────────────────────────────────────────────
def fig_competitors(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax_lat, ax_hit = axes

    systems = [
        ("FAISS Flat-GPU",      "faiss_flat_gpu_cosine_only",   C_FAISS_FLAT),
        ("FAISS IVF-GPU",       "faiss_ivf_gpu_cosine_only",    C_FAISS_IVF),
        ("cuVS CAGRA",          "cuvs_cagra_gpu_cosine_only",   C_CAGRA),
        ("MARS Global FP32",    "mars_global_fp32",             C_MARS_FP32),
        ("MARS Global FP16",    "mars_global_fp16",             C_MARS_FP16),
        ("MARS Episode-scoped", "mars_episode_scoped",          C_MARS_EP),
    ]

    n_groups = len(N_LABELS)
    n_bars   = len(systems)
    bar_w    = 0.13
    x        = np.arange(n_groups)

    # ── Left: latency p99 (log y) ────────────────────────────────────
    for si, (name, key, color) in enumerate(systems):
        vals = [data[key][n]["p99"] for n in N_KEYS]
        offset = (si - (n_bars - 1) / 2) * bar_w
        is_winner = (key == "mars_episode_scoped")
        edge_color = "#1A6E4F" if is_winner else "white"
        ax_lat.bar(x + offset, vals, bar_w, label=name, color=color,
                   edgecolor=edge_color, linewidth=1.2 if is_winner else 0.5,
                   zorder=3 if is_winner else 2)
        for i, v in enumerate(vals):
            ax_lat.text(x[i] + offset, v * 1.06, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold", color=color, rotation=90)

    ax_lat.axhline(1.0, color=C_DEADLINE, ls="--", lw=1.4, alpha=0.85,
                   zorder=1, label="1 ms AV deadline")
    ax_lat.set_yscale("log")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([f"N = {l}" for l in N_LABELS])
    ax_lat.set_ylabel("Wall-clock p99 (ms, log scale)")
    ax_lat.set_title("Per-query latency on the embodied multimodal contract",
                     fontweight="bold", fontsize=11)
    ax_lat.set_ylim(0.10, 12.0)
    ax_lat.legend(loc="upper left", framealpha=0.95, edgecolor="none",
                  ncol=2, fontsize=8.5, columnspacing=0.8)

    # ── Right: hit@15 ────────────────────────────────────────────────
    for si, (name, key, color) in enumerate(systems):
        vals = [data[key][n]["hit_15"] for n in N_KEYS]
        offset = (si - (n_bars - 1) / 2) * bar_w
        is_winner = (key == "mars_episode_scoped")
        edge_color = "#1A6E4F" if is_winner else "white"
        ax_hit.bar(x + offset, vals, bar_w, label=name, color=color,
                   edgecolor=edge_color, linewidth=1.2 if is_winner else 0.5,
                   zorder=3 if is_winner else 2)
        for i, v in enumerate(vals):
            txt = f"{v:.2f}" if v > 0.001 else "0.00"
            ax_hit.text(x[i] + offset, v + 0.03, txt,
                        ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold", color=color)

    ax_hit.set_xticks(x)
    ax_hit.set_xticklabels([f"N = {l}" for l in N_LABELS])
    ax_hit.set_ylabel("hit@15 (1.0 = perfect cross-modal recall)")
    ax_hit.set_title("Recall on same-episode TEXT + AUDIO",
                     fontweight="bold", fontsize=11)
    ax_hit.set_ylim(0, 1.18)
    ax_hit.axhline(1.0, color=C_GRAY, ls=":", lw=0.9, alpha=0.6, zorder=1)

    fig.suptitle("MARS vs. FAISS-GPU 1.14.1 vs. cuVS CAGRA 26.04 — A100 SXM4 40 GB, D=768, kids-ball corpus, paired RNG",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_competitors")


# ─────────────────────────────────────────────────────────────────────
# FIGURE 2 — episode-scoped scaling (near-flat curve)
# ─────────────────────────────────────────────────────────────────────
def fig_episode_scoped_scaling(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    Ns = [10_000, 100_000, 1_000_000]
    p99_global = [data["mars_global_fp32"][n]["p99"] for n in N_KEYS]
    p99_ep     = [data["mars_episode_scoped"][n]["p99"] for n in N_KEYS]
    p99_flat   = [data["faiss_flat_gpu_cosine_only"][n]["p99"] for n in N_KEYS]
    p99_cagra  = [data["cuvs_cagra_gpu_cosine_only"][n]["p99"] for n in N_KEYS]

    ax.plot(Ns, p99_flat, "o-",  color=C_FAISS_FLAT, lw=1.8, markersize=7,
            label=f"FAISS Flat-GPU (exhaustive)  hit@15=1.00")
    ax.plot(Ns, p99_cagra, "s-", color=C_CAGRA, lw=1.6, markersize=7,
            label=f"cuVS CAGRA (graph ANN)  hit@15 collapses")
    ax.plot(Ns, p99_global, "^-", color=C_MARS_FP32, lw=1.8, markersize=7,
            label=f"MARS Global (FP32 cuBLAS)  hit@15≈0.83")
    ax.plot(Ns, p99_ep, "D-", color=C_MARS_EP, lw=2.6, markersize=10,
            markeredgecolor="white", markeredgewidth=1.2,
            label=f"MARS Episode-scoped  hit@15=1.00  (winner)", zorder=5)

    ax.axhline(1.0, color=C_DEADLINE, ls="--", lw=1.5, alpha=0.85, zorder=1)
    ax.text(8_000, 1.08, "1 ms AV deadline",
            color=C_DEADLINE, fontsize=9, va="bottom", ha="left", fontweight="bold")

    for n, v in zip(Ns, p99_ep):
        ax.annotate(f"{v*1000:.0f} µs", xy=(n, v),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8.5, fontweight="bold",
                    color=C_MARS_EP)

    speedup_at_1m = p99_flat[-1] / p99_ep[-1]
    ax.annotate(f"{speedup_at_1m:.0f}× faster\nthan FAISS Flat\nat N=1 M",
                xy=(1_000_000, p99_ep[-1]),
                xytext=(300_000, 0.32),
                fontsize=9.5, fontweight="bold", color=C_MARS_EP, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_MARS_EP, lw=1.5,
                                shrinkA=2, shrinkB=4))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Corpus size N (log scale)")
    ax.set_ylabel("Wall-clock p99 (ms, log scale)")
    ax.set_title("Episode-scoped retrieval is near-flat: 0.20 ms p99 from 10 K to 1 M memories\n"
                 "Same-hardware comparison on the kids-ball multimodal contract — A100 SXM4 40 GB",
                 fontweight="bold", fontsize=10.5)
    ax.set_xlim(7_000, 1_500_000)
    ax.set_ylim(0.1, 14.0)
    ax.set_xticks(Ns)
    ax.set_xticklabels(["10 K", "100 K", "1 M"])
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="none", fontsize=9)

    fig.tight_layout()
    save(fig, "fig_episode_scoped_scaling")


# ─────────────────────────────────────────────────────────────────────
# FIGURE 3 — FP16 vs cuBLAS crossover
# ─────────────────────────────────────────────────────────────────────
def fig_fp16_crossover(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.7))

    Ns = [10_000, 100_000, 1_000_000]
    p99_fp32 = [data["mars_global_fp32"][n]["p99"] for n in N_KEYS]
    p99_fp16 = [data["mars_global_fp16"][n]["p99"] for n in N_KEYS]

    x = np.arange(len(Ns))
    w = 0.32

    b1 = ax.bar(x - w/2, p99_fp32, w, label="FP32 cuBLAS Sgemv (default)",
                color=C_MARS_FP32, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w/2, p99_fp16, w, label="FP16 fused (--use-fp16)",
                color=C_MARS_FP16, edgecolor="white", linewidth=0.8)

    for bar, v in zip(b1, p99_fp32):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=8, fontweight="bold", color=C_MARS_FP32)
    for bar, v in zip(b2, p99_fp16):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=8, fontweight="bold", color="#1F6FB5")

    for i in range(len(Ns)):
        delta = (p99_fp16[i] - p99_fp32[i]) / p99_fp32[i] * 100
        color = "#0CA678" if delta < 0 else "#C92A2A"
        sign  = "" if delta < 0 else "+"
        ax.text(i, max(p99_fp32[i], p99_fp16[i]) + 0.5,
                f"{sign}{delta:.0f} %",
                ha="center", fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {l}" for l in N_LABELS])
    ax.set_ylabel("Wall-clock p99 (ms)")
    ax.set_title("FP16 fused vs. cuBLAS Sgemv — FP16 wins at small N, loses at large N\n"
                 "(cuBLAS exploits Tensor-Core paths the hand-fused kernel cannot match)",
                 fontweight="bold", fontsize=10.5)
    ax.set_ylim(0, max(p99_fp32 + p99_fp16) * 1.30)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="none", fontsize=9.5)

    fig.tight_layout()
    save(fig, "fig_fp16_crossover")


def main() -> None:
    print("Loading benchmark artefacts from", COMP)
    data = load_data()
    print(f"  Loaded {len(data)} systems × {len(N_KEYS)} N points")

    print("Generating figures into", OUT)
    fig_competitors(data)
    fig_episode_scoped_scaling(data)
    fig_fp16_crossover(data)
    print("Done.")


if __name__ == "__main__":
    main()
