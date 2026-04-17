# Next Steps

## Current status

The project has a working CUDA retrieval engine with six pipeline
versions across six optimization rounds, validated on four GPUs, with
a same-hardware FAISS/cuVS-CAGRA comparison and an arXiv paper now
framed around **episode-scoped retrieval** as the headline
contribution.

### What's done

- **CUDA engine** — 11 kernels across 6 pipeline versions, all validated
- **Episode-scoped fast path** (`RetrievalScope::EpisodeScoped`) —
  near-flat ~200 µs p99 from N=10K to N=1M on A100 SXM4
- **FP16 fused cosine kernel** (opt-in `--use-fp16`) with documented
  small-N / large-N crossover vs cuBLAS
- **Hardware validation** — A100X 80GB, A100 PCIE 40GB, A100 SXM4
  40GB, RTX 5060 Ti 16GB
- **Six optimization rounds**, each addressing specific bottlenecks
- **Same-hardware FAISS-Flat / FAISS-IVF / cuVS-CAGRA comparison** —
  paired RNG seed=2026, kids-ball multimodal contract; full SUMMARY
  in `results/competitors_20260417/`
- **arXiv paper revision (2026-04-17)** — reframed around
  episode-scoped retrieval, demoted trivial theorems to propositions,
  added scope-crossover proposition, statistical caveats, fairer
  baseline acknowledgements, evaluation hardening track
- **Engine server** — JSON-line REPL for Python adapters

### What's remaining (next)

| Track | Items | Cost | Time |
|-------|-------|------|------|
| **1. Evaluation hardening** | Items 1–5 below | ~$15 | 1–2 days |
| 2. arXiv submission | Upload finalized PDF | $0 | 10 min |
| 3. Python bindings (pybind11) | LangChain / LlamaIndex integration | $0 | 1–2 days |
| 4. Multi-GPU sharding | NVLink cross-shard BFS | $10+ | 1 week |

---

## 1. Evaluation hardening (highest priority, addresses paper §10.1)

These were explicitly called out by the internal review as required
to make the headline numbers reviewer-proof. Each is queued in §10.1
of the paper as the first track of *Future Work*.

### 1a. Locked-clock re-runs with ≥10K queries

Current p99 numbers are over 128–256 paired probes on non-locked-clock
vast.ai instances. At sub-millisecond scale, run-to-run jitter is
±10 % and the 12th-largest of 128 samples has wide CIs.

```bash
# Pin SM + memory clocks before running:
nvidia-smi --lock-gpu-clocks=1410,1410        # A100 SXM4 base clock
nvidia-smi --lock-memory-clocks=1215,1215
make bench-sweep PROBES=10000                 # 10K queries / N
```

Re-publish the head-to-head table (`results/competitors_20260417/`)
with locked clocks and ≥10K probes. Confidence intervals (5th–95th
percentile of bootstrap p99) on every cell.

### 1b. Real-encoder benchmark (CLIP / CLAP / E5)

The kids-ball corpus uses Gaussian-perturbed cluster centroids in
768-D. Real-encoder embeddings have different distance distributions
and broader clusters; the exact crossover N at which cosine ANN
loses recall on real data may shift.

- Pull a CLIP-ViT-L/14 (image), CLAP-HTSAT (audio), and
  multilingual-E5-base (text) corpus of ~1M vectors with episode
  groupings (object tracks, conversation turns).
- Re-run the head-to-head table (Table 13) on real embeddings.
- Report Recall@10 *and* `episode_cross_modal_hit_rate_top15`.

### 1c. FAISS-Flat-GPU + IDSelectorBatch baseline

The fairest baseline for episode-scoped MARS is FAISS-Flat-GPU with
`IDSelectorBatch` / `IDSelectorRange` set to the episode member ids.
Add this row to Table 13.

```python
import faiss, numpy as np
sel = faiss.IDSelectorRange(ep_lo, ep_hi)
params = faiss.SearchParametersIVF(sel=sel)  # or SearchParameters for Flat
D, I = index.search(q[None], k=15, params=params)
```

### 1d. Standard temporal-IR metrics

Augment the AV temporal-relevance experiment (§5.1) with
**TS-Recall@10** and **time-NDCG@10** alongside TP@10, so reviewers
can compare against the temporal-IR literature.

### 1e. HNSW CPU baseline row

Add an HNSW CPU baseline (e.g., `nmslib` / `hnswlib`) to Table 4 to
make the "GPU-resident matters" claim explicit. Expected: HNSW p99
in the 1–10 ms range on the same N, depending on `efSearch`.

### 1f. Per-stage Nsight Compute breakdowns

Per-stage timings via `ncu --set full` for the four global-path
kernels at N ∈ {10K, 100K, 1M}: cuBLAS SGEMV, decay epilogue, CUB
radix top-K, BFS. This lets reviewers reason about Roofline limits
and validates the bandwidth-bound model in §4.

---

## 2. arXiv submission

```bash
cd paper && tectonic --reruns 3 main.tex
```

Upload to [arxiv.org/submit](https://arxiv.org/submit) under `cs.DC`
with `cs.IR` cross-listing.

## 3. Python bindings

pybind11 wrapper exposing `add()`, `query()`, `delete()`, `stats()`,
plus `query_episode()` for the episode-scoped fast path. Integration
target: LangChain, LlamaIndex, custom Python pipelines.

## 4. Multi-GPU sharding

NVLink-based partitioning for corpora exceeding single-GPU HBM
(>13M on A100 40GB). Per-shard episode CSR + cross-shard BFS over
NVLink (600 GB/s).

---

## Cost summary

| Phase | Cost | Time |
|-------|------|------|
| 1a Locked-clock re-runs | ~$5 | 4 hours |
| 1b Real-encoder benchmark | ~$5 | 1 day |
| 1c FAISS+IDSelector row | ~$2 | 2 hours |
| 1d Temporal-IR metrics | $0 | 1 hour |
| 1e HNSW CPU row | $0 | 2 hours |
| 1f Nsight Compute breakdowns | ~$3 | 4 hours |
| arXiv submission | $0 | 10 min |
| **Total to v2 submission** | **~$15** | **~2 days** |
