#!/usr/bin/env python3
"""
bench_kids_ball_cuvs_cagra.py — cuVS CAGRA baseline on the kids-ball corpus.

CAGRA is the closest direct competitor to MARS: GPU-resident graph-based ANN
with sub-millisecond single-query latency. It does NOT provide native temporal
decay, episode-scoped retrieval, or cross-modal bridges — but its raw cosine
nearest-neighbor latency is the right yardstick for the global Stage-1 path.

The corpus binary layout matches `demos/embodied_scene/bench_kids_sweep --dump-only`
(magic 'KIDS', uint32 LE 0x5344494b; version=1; N, D; embeddings; timestamps;
modalities; episode_ids).

Build/search defaults follow the cuVS quickstart for the small/medium-N regime
(graph_degree=64, intermediate_graph_degree=128, IVF-PQ build,
itopk_size=64, max_iterations=auto). We also report `extend()` performance
since it is the streaming-update analogue most often compared to MARS's
append-only CSR.

Usage:
  pip install --extra-index-url=https://pypi.nvidia.com 'cuvs-cu12==26.4.*'
  python3 scripts/bench_kids_ball_cuvs_cagra.py \
      --corpus results/competitors_20260417/corpus/kids_10k.bin \
      --queries 256 --search-k 96 --seed 2026

Output: JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time

import numpy as np

try:
    import cupy as cp
    from cuvs.neighbors import cagra
except ImportError as e:
    print(f"ERROR: cuVS / cupy not installed: {e}", file=sys.stderr)
    print("Run: pip install --extra-index-url=https://pypi.nvidia.com 'cuvs-cu12==26.4.*'", file=sys.stderr)
    sys.exit(1)

KIDS_MAGIC = 0x5344494B  # 'KIDS' little-endian


def load_corpus(path: str):
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != KIDS_MAGIC:
            raise ValueError(f"bad magic {magic:#x}, expected {KIDS_MAGIC:#x}")
        struct.unpack("<I", f.read(4))  # version
        N, D = struct.unpack("<ii", f.read(8))
        emb = np.fromfile(f, count=N * D, dtype=np.float32).reshape(N, D)
        ts = np.fromfile(f, count=N, dtype=np.float32)
        mod = np.fromfile(f, count=N, dtype=np.int32)
        epi = np.fromfile(f, count=N, dtype=np.int32)
    return emb, ts, mod, epi


def percentile_ms(lats: list, p: float) -> float:
    if not lats:
        return 0.0
    s = sorted(lats)
    idx = int(p / 100.0 * (len(s) - 1))
    return float(s[idx])


def episode_cross_modal_hit(mod: np.ndarray, epi: np.ndarray, result_ids: np.ndarray,
                            query_ep: int, top_k: int) -> bool:
    has_t = has_a = False
    for j in range(min(top_k, len(result_ids))):
        rid = int(result_ids[j])
        if rid < 0:
            continue
        if int(epi[rid]) != int(query_ep):
            continue
        m = int(mod[rid])
        if m == 0:
            has_t = True
        if m == 1:
            has_a = True
    return has_t and has_a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", type=int, default=256)
    ap.add_argument("--search-k", type=int, default=96,
                    help="neighbors retrieved per query (CAGRA requires itopk_size >= search_k)")
    ap.add_argument("--itopk-size", type=int, default=128,
                    help="CAGRA's intermediate top-K queue; must be >= search_k")
    ap.add_argument("--algo", default="single_cta",
                    help="CAGRA search algo: 'single_cta' (best for batch=1) or 'multi_cta' (batch>>1)")
    ap.add_argument("--graph-degree", type=int, default=64)
    ap.add_argument("--intermediate-graph-degree", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--measure-extend", action="store_true",
                    help="Also measure CAGRA extend() throughput on a 10%% suffix")
    args = ap.parse_args()

    emb, ts, mod, epi = load_corpus(args.corpus)
    N, D = emb.shape

    img_idx = np.where(mod == 2)[0].astype(np.int64)  # IMAGE = 2
    if len(img_idx) == 0:
        print("ERROR: no IMAGE nodes in corpus", file=sys.stderr)
        return 2

    rng = np.random.RandomState(args.seed)
    qix = rng.choice(img_idx, size=args.queries, replace=True)
    queries_h = emb[qix].astype(np.float32)

    # Move base + queries to device
    base_d    = cp.asarray(emb, dtype=cp.float32)
    queries_d = cp.asarray(queries_h, dtype=cp.float32)

    # ── Build ──────────────────────────────────────────────────────────
    build_params = cagra.IndexParams(
        metric="inner_product",
        intermediate_graph_degree=args.intermediate_graph_degree,
        graph_degree=args.graph_degree,
    )
    cp.cuda.Stream.null.synchronize()
    t_b0 = time.perf_counter()
    index = cagra.build(build_params, base_d)
    cp.cuda.Stream.null.synchronize()
    build_ms = (time.perf_counter() - t_b0) * 1000.0

    # ── Warmup ─────────────────────────────────────────────────────────
    if args.itopk_size < args.search_k:
        args.itopk_size = args.search_k
    search_params = cagra.SearchParams(itopk_size=args.itopk_size, algo=args.algo)
    cagra.search(search_params, index, queries_d[: min(10, args.queries)], args.search_k)
    cp.cuda.Stream.null.synchronize()

    # ── Per-query timed search ─────────────────────────────────────────
    lats: list = []
    hits_sk = 0
    hits_15 = 0
    all_ids: list = []
    for i in range(args.queries):
        q = queries_d[i : i + 1]
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        distances, neighbors = cagra.search(search_params, index, q, args.search_k)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
        ids_h = cp.asnumpy(neighbors).reshape(-1)
        all_ids.append(ids_h)
        ep = int(epi[qix[i]])
        if episode_cross_modal_hit(mod, epi, ids_h, ep, args.search_k):
            hits_sk += 1
        if episode_cross_modal_hit(mod, epi, ids_h, ep, 15):
            hits_15 += 1

    # ── Recall vs FAISS-style ground truth (skip — no oracle here; CAGRA
    #    is the only graph-ANN, recall is implicitly captured by hit@15) ──

    cfg = {
        "method": "cuvs_cagra_gpu_cosine_only",
        "N": int(N),
        "D": int(D),
        "search_k": int(args.search_k),
        "itopk_size": int(args.itopk_size),
        "graph_degree": int(args.graph_degree),
        "intermediate_graph_degree": int(args.intermediate_graph_degree),
        "num_queries": int(args.queries),
        "build_ms": float(build_ms),
        "p50_ms": percentile_ms(lats, 50),
        "p99_ms": percentile_ms(lats, 99),
        "mean_ms": float(np.mean(lats)),
        "max_ms": float(np.max(lats)),
        "episode_cross_modal_hit_rate_search_k": hits_sk / max(1, args.queries),
        "episode_cross_modal_hit_rate_top15": hits_15 / max(1, args.queries),
    }

    extend_block = None
    if args.measure_extend and N >= 1000:
        n_extra = max(100, N // 10)
        rng2 = np.random.RandomState(args.seed + 7)
        extra_emb = rng2.randn(n_extra, D).astype(np.float32)
        extra_emb /= (np.linalg.norm(extra_emb, axis=1, keepdims=True) + 1e-9)
        extra_d = cp.asarray(extra_emb, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        t_e0 = time.perf_counter()
        extend_params = cagra.ExtendParams()
        index = cagra.extend(extend_params, index, extra_d)
        cp.cuda.Stream.null.synchronize()
        ext_ms = (time.perf_counter() - t_e0) * 1000.0
        extend_block = {
            "extend_n_added":     int(n_extra),
            "extend_total_ms":    float(ext_ms),
            "extend_per_vec_us":  float(ext_ms * 1000.0 / n_extra),
            "note": "Random extra vectors (NOT same distribution as base); CAGRA recall guidance: stable up to ~20% additions (see cuVS docs).",
        }

    out = {
        "tool": "bench_kids_ball_cuvs_cagra",
        "corpus_path": args.corpus,
        "note": ("cuVS CAGRA cosine inner-product graph search; no temporal "
                 "decay, no cross-modal bridges, no episode scope. Compare "
                 "p99 latency to MARS global path; compare hit@15 to FAISS Flat."),
        "configurations": [cfg],
    }
    if extend_block is not None:
        out["extend"] = extend_block

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
