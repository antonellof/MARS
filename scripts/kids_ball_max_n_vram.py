#!/usr/bin/env python3
"""
Estimate maximum EmbodiedKidsBallCorpus (768-D) graph size N that fits in a VRAM budget
for the kids-ball CUDA path (upload_to_device + episode arrays + QueryContext).

Uses measured NSN degree from build_nsn_edges(6, 0.15) on this corpus (~11.02 edges per
node in CSR = sum of out-degrees). Does not run CUDA.

Host RAM: corpus construction holds full FP32 embeddings (~ N * D * 4 bytes) plus CSR
on CPU — you need comparable free DRAM before GPU upload.
"""
from __future__ import annotations

import argparse
import json
import sys

# From host probe: EmbodiedKidsBallCorpus + build_nsn_edges(6, 0.15), large N.
DEFAULT_AVG_CSR_DEG = 11.02
TILE_SIZE = 16384
GiB = 1024**3


def kids_ball_device_bytes(
    n: int,
    dim: int = 768,
    avg_csr_degree: float = DEFAULT_AVG_CSR_DEG,
    max_k: int = 64,
    cub_bytes_per_node: float = 24.0,
) -> dict:
    """Upper-ish model aligned to memory_cuda.cu + upload_to_device + episode upload."""
    if n < 10:
        raise ValueError("n must be >= 10")

    e = int(round(n * avg_csr_degree))

    # DeviceMemoryGraph (no importance / edge_hits / fp16 — kids sweep default)
    row_off = (n + 1) * 4
    col_idx = e * 4
    emb = n * dim * 4
    mod = n * 4
    ts = n * 4
    graph_core = row_off + col_idx + emb + mod + ts

    # upload_episode_ids + upload_episode_csr (each node once in members)
    num_eps = (n + 9) // 10  # KIDS_EP_LEN = 10
    ep_off = (num_eps + 1) * 4
    ep_mem = n * 4
    ep_ids = n * 4
    episode = ep_off + ep_mem + ep_ids

    # QueryContext (create_query_context): per-query scratch
    k = min(max_k, 64)
    num_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
    topk_buf = k + num_tiles * k
    topk_bytes = topk_buf * (4 + 4)

    scratch = (
        dim * 4  # d_query
        + n * 4 * 3  # d_sim, d_final, d_bfs_score
        + n * 4  # d_hop_count
        + topk_bytes
        + n * 4 * 2  # d_front_a, d_front_b
        + 8  # d_front_cnt + d_front_cnt_b
        + min(n, k * 500) * 16
        + 4  # d_compact_count
        + n * 4 * 4  # sort keys/vals in/out
        + int(cub_bytes_per_node * n)  # d_sort_temp (CUB radix sort; conservative)
        + 4096  # cuBLAS handle slack
        + 6 * 64  # events/stream
    )

    # Pinned host buffer in QueryContext (not VRAM — omit from GPU total)
    total_gpu = graph_core + episode + scratch
    return {
        "n": n,
        "dim": dim,
        "num_csr_edges_e": e,
        "avg_csr_degree": avg_csr_degree,
        "bytes_graph_core": graph_core,
        "bytes_episode": episode,
        "bytes_query_context": scratch,
        "bytes_total_gpu": total_gpu,
        "gib_total_gpu": total_gpu / GiB,
    }


def max_n_for_budget(
    budget_bytes: int,
    dim: int = 768,
    avg_csr_degree: float = DEFAULT_AVG_CSR_DEG,
    max_k: int = 64,
    cub_bytes_per_node: float = 24.0,
) -> int:
    """Binary search largest n with model total <= budget_bytes."""
    lo, hi = 10, 50_000_000
    best = 10
    while lo <= hi:
        mid = (lo + hi) // 2
        b = kids_ball_device_bytes(mid, dim, avg_csr_degree, max_k, cub_bytes_per_node)[
            "bytes_total_gpu"
        ]
        if b <= budget_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vram-gib", type=float, default=40.0, help="Target GPU memory (GiB)")
    p.add_argument(
        "--reserve-mib",
        type=float,
        default=768.0,
        help="Leave this many MiB headroom (driver/display/CUDA runtime)",
    )
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--max-k", type=int, default=64, help="QueryContext max_k (bench kids uses 64)")
    p.add_argument(
        "--avg-csr-degree",
        type=float,
        default=DEFAULT_AVG_CSR_DEG,
        help="CSR col_indices length / N after NSN build",
    )
    p.add_argument(
        "--cub-bytes-per-node",
        type=float,
        default=24.0,
        help="Conservative CUB radix-sort temp / N (tune if you see cudaMalloc failures)",
    )
    p.add_argument("--print-n", action="store_true", help="Print only recommended integer N")
    p.add_argument("--json", action="store_true", help="Emit full JSON on stdout")
    args = p.parse_args()

    budget = int(args.vram_gib * GiB - args.reserve_mib * 1024**2)
    if budget <= 0:
        print("error: non-positive budget after reserve", file=sys.stderr)
        return 1

    n_rec = max_n_for_budget(
        budget,
        dim=args.dim,
        avg_csr_degree=args.avg_csr_degree,
        max_k=args.max_k,
        cub_bytes_per_node=args.cub_bytes_per_node,
    )
    detail = kids_ball_device_bytes(
        n_rec,
        dim=args.dim,
        avg_csr_degree=args.avg_csr_degree,
        max_k=args.max_k,
        cub_bytes_per_node=args.cub_bytes_per_node,
    )
    host_emb_gib = n_rec * args.dim * 4 / GiB

    if args.print_n:
        print(n_rec)
        return 0
    if args.json:
        out = {
            "recommended_n": n_rec,
            "budget_bytes": budget,
            "detail": detail,
            "host_embeddings_gib_approx": round(host_emb_gib, 3),
            "note": "Host needs ~host_embeddings_gib_approx GiB free RAM for FP32 corpus build before upload.",
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Target budget: {budget / GiB:.3f} GiB (after {args.reserve_mib} MiB reserve)")
    print(f"Recommended N: {n_rec}")
    print(f"Model GPU total: {detail['gib_total_gpu']:.3f} GiB")
    print(f"Approx host FP32 embeddings for make(): {host_emb_gib:.2f} GiB")
    print("\nBreakdown (bytes):")
    for k in ("bytes_graph_core", "bytes_episode", "bytes_query_context", "bytes_total_gpu"):
        print(f"  {k}: {detail[k]} ({detail[k] / GiB:.4f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
