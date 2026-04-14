#!/usr/bin/env python3
"""
bench_kids_ball_faiss.py — FAISS GPU baselines on the kids-ball corpus binary.

Expects the same layout as demos/embodied_scene/bench_kids_sweep --dump-only:
  magic 'KIDS' (uint32 LE 0x5344494b), version=1, N, D, embeddings, timestamps,
  modalities, episode_ids.

IMPORTANT (thesis guardrail): FAISS returns cosine inner-product neighbors only.
It does not run MARS temporal rerank or NSN BFS. episode_cross_modal_hit_rate
here measures whether *those* top-K neighbors span TEXT+AUDIO in the *same
episode* as the query image node — a multimodal co-retrieval diagnostic, not
equivalent to the full MARS pipeline.

Usage:
  pip install faiss-gpu numpy
  ./demos/embodied_scene/bench_kids_sweep --dump-only results/kids_corpus_10k.bin 10000
  python3 scripts/bench_kids_ball_faiss.py --corpus results/kids_corpus_10k.bin

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
    import faiss
except ImportError:
    print("ERROR: faiss-gpu not installed. Run: pip install faiss-gpu numpy",
          file=sys.stderr)
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


def run_flat(emb: np.ndarray, mod: np.ndarray, epi: np.ndarray, img_idx: np.ndarray,
             num_queries: int, top_k: int, seed: int) -> dict:
    N, D = emb.shape
    rng = np.random.RandomState(seed)
    qix = rng.choice(img_idx, size=num_queries, replace=True)
    queries = emb[qix].astype(np.float32)

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(D)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(emb)

    gpu_index.search(queries[: min(10, len(queries))], top_k)

    lats: list = []
    hits = 0
    for i in range(len(queries)):
        q = queries[i : i + 1]
        t0 = time.perf_counter()
        _, I = gpu_index.search(q, top_k)
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
        ep = int(epi[qix[i]])
        if episode_cross_modal_hit(mod, epi, I[0], ep, top_k):
            hits += 1

    return {
        "method": "faiss_flat_gpu_cosine_only",
        "N": int(N),
        "D": int(D),
        "K": top_k,
        "num_queries": num_queries,
        "p50_ms": percentile_ms(lats, 50),
        "p99_ms": percentile_ms(lats, 99),
        "mean_ms": float(np.mean(lats)),
        "max_ms": float(np.max(lats)),
        "episode_cross_modal_hit_rate": hits / max(1, num_queries),
    }


def run_ivf(emb: np.ndarray, mod: np.ndarray, epi: np.ndarray, img_idx: np.ndarray,
            num_queries: int, top_k: int, seed: int) -> dict | None:
    N, D = emb.shape
    if N < 1000:
        return None

    nlist = max(1, min(int(np.sqrt(N)), 1024))
    nprobe = max(1, min(nlist // 4, 64))

    rng = np.random.RandomState(seed)
    qix = rng.choice(img_idx, size=num_queries, replace=True)
    queries = emb[qix].astype(np.float32)

    res_gt = faiss.StandardGpuResources()
    gt = faiss.IndexFlatIP(D)
    gt_gpu = faiss.index_cpu_to_gpu(res_gt, 0, gt)
    gt_gpu.add(emb)
    _, gt_ids = gt_gpu.search(queries, top_k)
    del gt_gpu, gt, res_gt

    res = faiss.StandardGpuResources()
    quantizer = faiss.IndexFlatIP(D)
    index_ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_ivf)
    gpu_index.train(emb)
    gpu_index.add(emb)
    faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", nprobe)

    gpu_index.search(queries[: min(10, len(queries))], top_k)

    lats: list = []
    hits = 0
    recalls = []
    for i in range(len(queries)):
        q = queries[i : i + 1]
        t0 = time.perf_counter()
        _, I = gpu_index.search(q, top_k)
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
        gt_set = set(gt_ids[i].tolist())
        recalls.append(len(gt_set & set(I[0].tolist())) / max(1, len(gt_set)))
        ep = int(epi[qix[i]])
        if episode_cross_modal_hit(mod, epi, I[0], ep, top_k):
            hits += 1

    return {
        "method": "faiss_ivf_gpu_cosine_only",
        "N": int(N),
        "D": int(D),
        "K": top_k,
        "nlist": nlist,
        "nprobe": nprobe,
        "num_queries": num_queries,
        "p50_ms": percentile_ms(lats, 50),
        "p99_ms": percentile_ms(lats, 99),
        "mean_ms": float(np.mean(lats)),
        "max_ms": float(np.max(lats)),
        "recall_at_k_vs_flat": float(np.mean(recalls)),
        "episode_cross_modal_hit_rate": hits / max(1, num_queries),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to kids corpus .bin")
    ap.add_argument("--queries", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=914)
    args = ap.parse_args()

    if faiss.get_num_gpus() < 1:
        print("ERROR: No FAISS GPU", file=sys.stderr)
        sys.exit(1)

    emb, _ts, mod, epi = load_corpus(args.corpus)
    img_idx = np.where(mod == 2)[0]  # MOD_IMAGE
    if len(img_idx) == 0:
        print("ERROR: no IMAGE nodes", file=sys.stderr)
        sys.exit(1)

    out = {
        "tool": "bench_kids_ball_faiss",
        "corpus_path": args.corpus,
        "note": "Cosine-only vector search; compare latency to MARS sweep JSON; "
                "episode_hit is diagnostic on flat neighbors, not BFS-expanded.",
        "configurations": [],
    }
    out["configurations"].append(
        run_flat(emb, mod, epi, img_idx, args.queries, args.top_k, args.seed))
    ivf = run_ivf(emb, mod, epi, img_idx, args.queries, args.top_k, args.seed)
    if ivf:
        out["configurations"].append(ivf)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
