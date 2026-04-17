# Benchmarks

Detailed performance data for MARS. All numbers are measured on real
hardware via vast.ai.

> **Statistical caveat (read first).** Reported p99s come from
> 128–256 paired probes on **non-locked-clock** vast.ai instances
> with shared multi-tenant hosts. At the sub-millisecond scale,
> run-to-run jitter is on the order of ±10 % and the p99 of a
> 128-probe sample has wide confidence intervals (the 12th-largest
> value is sensitive to a single outlier). Conclusions that depend
> on differences smaller than ~10 % should be re-measured with locked
> SM/memory clocks (`nvidia-smi --lock-gpu-clocks=...`) and
> ≥10K queries. See §6.2 of the paper and the *Evaluation Hardening*
> track in §10.1 for the queued re-runs.

---

## Hardware index

| Tag | GPU | TDP | HBM | Tenancy | Used in |
|-----|-----|-----|-----|---------|---------|
| **A100-SXM4-40GB** | NVIDIA A100 SXM4 | 400 W | 40 GB | shared (vast.ai) | episode-scoped sweep, FAISS/CAGRA head-to-head, large-corpus scaling |
| **A100-PCIE-40GB** | NVIDIA A100 PCIE | 250 W | 40 GB | shared (vast.ai) | demonstrator p99 sweep, FP32 scaling |
| **A100X-80GB**     | NVIDIA A100 SXM4 | 400 W | 80 GB | shared (vast.ai) | early same-hardware FAISS comparison |
| **RTX-5060-Ti-16GB** | NVIDIA Blackwell | 180 W | 16 GB | local                                   | consumer-GPU validation |

The **RTX 5060 Ti vs A100 PCIE** consumer-GPU finding (RTX faster on
the demonstrator workloads) reflects the A100 PCIE's lower 250 W TDP
and shared-tenant overhead on vast.ai, *not* a generational win for
Blackwell over Ampere. Re-running on a dedicated locked-clock A100
SXM4 reverses the order on the global path.

---

## Episode-scoped retrieval (headline)

When the application supplies `query_episode_id` (embodied loops,
AV per-track memory, voice/AR session), Stage 1 is restricted to the
episode CSR member range and BFS is skipped (depth = 0). The work
collapses from Θ(N·D) to Θ(|episode|·D). On the kids-ball multimodal
contract (10-node episodes, paired RNG seed=2026), A100 SXM4 40 GB:

| N | Global p99 | **Episode-scoped p99** | Speedup | Hit@15 (global → scoped) |
|---|-----------:|-----------------------:|:-------:|------------------------:|
| 10K  | 0.349 ms | **0.165 ms** | 2.1×    | 0.83 → 1.00 |
| 50K  | 0.462 ms | **0.172 ms** | 2.7×    | 0.88 → 1.00 |
| 100K | 0.551 ms | **0.174 ms** | 3.2×    | 0.79 → 1.00 |
| 1M   | 2.561 ms | **0.197 ms** | **13×** | 0.84 → 1.00 |

Episode-scoped is the only configuration in our sweep that meets the
**1 ms AV deadline at N=10⁶**.

**Caveats:**
- Correct only when the right episode is known *before* the query.
  Cross-episode retrieval still requires the global path with NSN
  bridges + temporal decay.
- Synthetic kids-ball corpus uses Gaussian-perturbed cluster
  centroids; real-encoder (CLIP/CLAP/E5) clusters may be broader and
  the absolute hit@15 advantage may shrink. Re-measurement on real
  embeddings is queued (§10.1 of the paper).
- A FAISS-Flat-GPU + `IDSelectorBatch` baseline doing the same work
  has not yet been measured; that is the queued fairer baseline.

---

## Same-hardware competitor head-to-head (2026-04-17)

A100 SXM4 40 GB, paired RNG (seed = 2026), kids-ball corpus,
metric = `episode_cross_modal_hit_rate_top15`:

| Method | N=10K p99 / hit@15 | N=100K p99 / hit@15 | N=1M p99 / hit@15 |
|--------|-------------------:|--------------------:|------------------:|
| FAISS Flat-GPU (exhaustive cosine)  | 0.18 ms / **1.00** | 0.78 ms / **1.00** | 6.64 ms / **1.00** |
| FAISS IVF-GPU (nlist=√N, nprobe=64) | 0.45 ms / 0.52     | 0.60 ms / 0.37     | 1.42 ms / **0.00** |
| cuVS CAGRA (graph_degree=64, IP)    | 3.32 ms / **1.00** | 3.23 ms / 0.01     | 3.25 ms / **0.00** |
| MARS Global, FP32 cuBLAS            | 0.47 ms / 0.83     | 0.67 ms / 0.79     | **2.51 ms** / 0.84 |
| MARS Global, FP16 fused             | **0.28 ms** / 0.83 | 0.66 ms / 0.79     | 3.73 ms / 0.84    |
| **MARS Episode-scoped**             | **0.19 ms / 1.00** | **0.20 ms / 1.00** | **0.20 ms / 1.00** |

Two findings:

1. **MARS Episode-scoped is Pareto-optimal** on the latency–recall
   frontier at every N. At N=1M it is 33× faster than FAISS-Flat
   (the only baseline matching its hit@15) and 16× faster than CAGRA.
2. **Cosine ANN baselines lose recall on the multimodal+episode
   metric at scale.** With 10-node episodes and 999 990 distractors
   at N=1M, FAISS-IVF and CAGRA cannot recover the per-episode
   TEXT/AUDIO neighbors regardless of `nprobe` or `search_k`. Only
   MARS-ep (episode CSR) and FAISS-Flat (exhaustive) recover them.

Full reproduction in
[`results/competitors_20260417/SUMMARY.md`](../results/competitors_20260417/SUMMARY.md).

---

## Same-hardware comparison (earlier sweep, A100 SXM4 80GB, D=768, K=10)

All systems measured on the same machine with the same data:

| System | N=2.4K | N=10K | N=20K | N=50K | Cross-modal | Temporal decay |
|--------|--------|-------|-------|-------|-------------|----------------|
| FAISS GPU Flat | 0.10 ms | 0.12 ms | 0.18 ms | 0.35 ms | No | No |
| FAISS GPU IVF | 0.13 ms | 0.15 ms | 0.30 ms | 0.28 ms | No | No |
| cuVS CAGRA | 2.60 ms | 2.29 ms | 2.29 ms | 2.47 ms | No | No |
| **MARS Global** | **0.26 ms** | **0.34 ms** | **0.36 ms** | **0.44 ms** | **Yes** | **Yes** |

> **What this table does and doesn't say.** MARS Global pays a fixed
> ~0.15 ms overhead for the temporal-decay kernel + the 4-kernel
> orchestration vs FAISS-Flat's single SGEMV; that overhead is what
> closes at large N (the ~0.44 vs 0.35 ms gap at N=50K, then crossing
> over by N=100K — see the head-to-head table above). For pure cosine
> top-K, FAISS-Flat at small N is faster. The MARS overhead buys
> kernel-level temporal decay, cross-modal BFS, and streaming inserts.

### GPU kernel time (excluding host overhead)

| System | N=2.4K | N=10K | N=50K |
|--------|--------|-------|-------|
| FAISS Flat | 0.09 ms | 0.12 ms | 0.35 ms |
| **MARS** | **0.10 ms** | **0.17 ms** | **0.29 ms** |

---

## Demonstrator results

### A100 PCIE 40GB (all pass)

| Bench | Rate | Corpus | Budget | Wall p99 | GPU p99 | Miss rate | Verdict |
|-------|------|--------|--------|----------|---------|-----------|---------|
| AV | 60 Hz | 2,400 | 1 ms | 0.87 ms | 0.68 ms | 0% | **pass** |
| Robot | 1 kHz | 6,000 | 1 ms | 0.76 ms | 0.69 ms | 0% | **pass** |
| AR/VR | 90 Hz | 20,000 | 5 ms | 1.56 ms | 1.37 ms | 0% | **pass** |
| Voice | 30 Hz | 3,000 | 20 ms | 0.88 ms | 0.68 ms | 0% | **pass** |

### RTX 5060 Ti 16GB (all pass, 2.3–2.8× faster)

| Bench | Rate | Corpus | Budget | Wall p99 | GPU p99 | Verdict |
|-------|------|--------|--------|----------|---------|---------|
| AV | 60 Hz | 2,400 | 1 ms | **0.31 ms** | 0.23 ms | **pass** |
| Robot | 1 kHz | 6,000 | 1 ms | **0.29 ms** | 0.23 ms | **pass** |
| AR/VR | 90 Hz | 20,000 | 5 ms | **0.67 ms** | 0.58 ms | **pass** |
| Voice | 30 Hz | 3,000 | 20 ms | **0.32 ms** | 0.23 ms | **pass** |

---

## Sustained benchmarks (30s real-world duration)

### A100 PCIE (all pass)

| Test | Rate | Duration | Corpus | Budget | p99 wall | Misses | Verdict |
|------|------|----------|--------|--------|----------|--------|---------|
| AV 30s | 60 Hz | 30s | 5K | 1 ms | 0.91 ms | 0 | **pass** |
| Robot 15s | 1 kHz | 15s | 10K | 1 ms | 0.66 ms | 0 | **pass** |
| AR 30s | 90 Hz | 30s | 50K | 5 ms | 1.70 ms | 0 | **pass** |
| Voice 30s | 30 Hz | 30s | 10K | 20 ms | 0.96 ms | 0 | **pass** |

### RTX 5060 Ti (all pass)

| Test | Rate | Duration | Corpus | Budget | p99 wall | Misses | Verdict |
|------|------|----------|--------|--------|----------|--------|---------|
| AV 30s | 60 Hz | 30s | 5K | 1 ms | 0.32 ms | 0 | **pass** |
| Robot 15s | 1 kHz | 15s | 10K | 1 ms | 0.32 ms | 0 | **pass** |
| AR 30s | 90 Hz | 30s | 50K | 5 ms | 0.91 ms | 0 | **pass** |
| Voice 30s | 30 Hz | 30s | 10K | 20 ms | 0.36 ms | 0 | **pass** |

---

## Scaling sweeps

### FP32 (60 Hz × 15s, 1 ms budget)

**A100 PCIE:**

| N | Wall p99 | GPU p99 | Verdict |
|---|---------|---------|---------|
| 1K | 0.84 ms | 0.67 ms | **pass** |
| 5K | 0.91 ms | 0.70 ms | **pass** |
| 10K | 0.96 ms | 0.75 ms | **pass** |
| 20K | 1.59 ms | 1.38 ms | fail |
| 50K | 1.69 ms | 1.51 ms | fail |

**RTX 5060 Ti** (all pass):

| N | Wall p99 | GPU p99 | Verdict |
|---|---------|---------|---------|
| 1K | 0.29 ms | 0.22 ms | **pass** |
| 5K | 0.32 ms | 0.24 ms | **pass** |
| 10K | 0.35 ms | 0.27 ms | **pass** |
| 20K | 0.66 ms | 0.58 ms | **pass** |
| 50K | 0.90 ms | 0.82 ms | **pass** |

### FP16 + CUDA Graph (large corpus, 60 Hz)

| N | RTX 5060 Ti | A100 SXM4 |
|---|------------|-----------|
| 50K | 0.89 ms | 1.22 ms |
| 100K | 1.18 ms | 1.38 ms |
| 200K | 1.75 ms | 1.67 ms |
| 500K | 3.53 ms | 2.51 ms |
| 1M | 6.51 ms | 7.70 ms |

> **Statistical caveat on the 500K → 1M flip.** The order between
> RTX 5060 Ti (3.53 → 6.51 ms) and A100 SXM4 (2.51 → 7.70 ms) at
> N=500K vs N=1M is reported faithfully but should not be over-read.
> The 1M row was measured on a non-locked-clock vast.ai A100 SXM4
> with shared tenancy; a locked-clock re-run with ≥10K queries is
> queued in §10.1 of the paper. Treat the rows as showing
> "FP16 fused stays roughly competitive on consumer hardware up to
> 1M but is not a substitute for cuBLAS at large N" rather than as a
> definitive ranking.

### When to enable `--use-fp16`

| N | FP32 cuBLAS | FP16 fused | Δ |
|---|------------:|----------:|---:|
| 10K  | 0.474 ms | **0.280 ms** | **−41 %** |
| 100K | 0.670 ms | 0.655 ms | flat |
| 1M   | **2.507 ms** | 3.731 ms | **+49 %** |

cuBLAS `Sgemv` exploits Tensor Core paths and per-arch tile
heuristics that the hand-fused kernel cannot match once N exceeds
the L2 working set. Use `--use-fp16` for N < 100K only; default to
cuBLAS for production deployments at large N.

---

## Optimization history

Six pipeline versions, each addressing a specific bottleneck:

| Version | Key change | Impact |
|---------|-----------|--------|
| Baseline | Per-query cudaMalloc, sync D2H | AV=1.41ms, Robot=1.12ms (both fail) |
| Targeted fixes | Pre-allocated buffers, GPU BFS init | AV=0.96ms (pass), Robot=1.21ms (fail) |
| Device-driven | Device-driven BFS, result compaction, pinned memory | AV=0.87ms, Robot=0.76ms (both pass) |
| FP16+Graph | FP16 similarity, CUDA Graph capture | 1M memories at 6.5ms |
| CMNG | CAGRA-style graph ANN + cross-modal bridges | Nearly flat latency 50K→200K |
| **cuBLAS+CUB (current)** | cuBLAS SGEMV similarity, CUB radix top-K | **GPU kernel matches FAISS Flat** |

### Current vs baseline improvement

| N | Baseline p99 | MARS p99 | Speedup |
|---|------------|------------|---------|
| 2.4K | 0.87 ms | **0.26 ms** | 3.3× |
| 10K | 0.96 ms | **0.34 ms** | 2.8× |
| 20K | 1.59 ms | **0.36 ms** | 4.4× |
| 50K | 1.69 ms | **0.44 ms** | 3.8× |

---

## Running benchmarks

```bash
# MARS pipeline (recommended)
make bench-mars                    # cuBLAS + CUB at N=2.4K, 10K, 20K, 50K

# Additional pipelines
make bench-av                    # 60 Hz, 2.4K memories
make bench-robot                 # 1 kHz, 6K memories
make bench-ar                    # 90 Hz, 20K memories
make bench-voice                 # 30 Hz, 3K memories
make bench-sustained             # all four 15-30s tests
make bench-scale                 # N = 1K through 50K
make bench-large                 # N = 50K through 1M (FP16+Graph)
make bench-fp16                  # FP32 vs FP16 A/B comparison

# FAISS GPU comparison (requires pip install faiss-gpu-cu12 numpy)
python scripts/bench_faiss_comparison.py

# CMNG graph ANN (experimental)
make bench-cmng                  # N = 10K through 200K
```

---

## GPUs tested

| GPU | VRAM | Architecture | CUDA | Results folder |
|-----|------|-------------|------|---------------|
| A100X 80GB | 80 GB | Ampere sm_80 | 12.0 | `results/a100x-80gb-v2/` |
| A100 PCIE 40GB | 40 GB | Ampere sm_80 | 12.8 | `results/a100-pcie-40gb-v3/` |
| A100 SXM4 80GB | 80 GB | Ampere sm_80 | 12.6–12.8 | `results/a100-sxm4-*` |
| RTX 5060 Ti 16GB | 16 GB | Blackwell sm_120 | 13.0 | `results/rtx5060ti*/` |
