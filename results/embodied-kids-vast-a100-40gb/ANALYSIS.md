# Embodied kids-ball benchmark — vast.ai A100 40GB (local snapshot)

## Provenance

| Field | Value |
|--------|--------|
| Host | vast.ai `ssh8.vast.ai`, port `17155` |
| GPU | NVIDIA **A100-SXM4-40GB** (CUDA **12.8**, driver/runtime **12080**) |
| Branch | `feature/embodied-kids-ball-benchmark` (commit **8046b9b** area) |
| Corpus | `EmbodiedKidsBallCorpus::make(..., seed=2026)`; 10k dump: `kids_corpus_10k.bin` |

Files in this directory:

| File | Description |
|------|-------------|
| `mars_sweep.json` | MARS N-sweep: wall/kernel p99 + episode hit rates |
| `faiss_baseline.json` | FAISS GPU flat + IVF on same 10k corpus (`--search-k 96`) |
| `demo_run.json` | One `demos/embodied_scene/demo` stdout JSON (450 queries) |
| `kids_corpus_10k.bin` | Binary corpus for FAISS replay (**~30 MB**, not committed) |

## Metrics definitions

- **Seeds:** `top_k = 15` (unchanged on GPU).
- **MARS “compact” window:** host returns up to **`max_results_returned = 96`** sorted compact results (`RetrievalConfig::max_results_returned`). **episode_cross_modal_hit_rate_compact** = IMAGE probe has ≥1 same-episode **TEXT** and ≥1 same-episode **AUDIO** in that window.
- **Top-15 slice:** same predicate but only the **first 15** ranked rows (UI-style / strict head).
- **FAISS:** cosine **Inner Product** on L2-normalized vectors; **`search_k = 96`** for latency + episode checks; **no** temporal decay or BFS.

## Results summary

### MARS sweep (`mars_sweep.json`)

256 random IMAGE probes per N; full pipeline (temporal + cuBLAS/CUB + BFS + compaction).

| N | Build (ms) | Wall p99 (ms) | Kernel p99 (ms) | Ep. hit compact | Ep. hit top15 |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 17.9 | 0.18 | 0.09 | **0.99** | **0.80** |
| 5,000 | 88.1 | 0.28 | 0.16 | **0.99** | **0.81** |
| 10,000 | 176.1 | 0.30 | 0.18 | **0.98** | **0.82** |
| 20,000 | 347.5 | 0.33 | 0.22 | **0.98** | **0.84** |
| 50,000 | 1031.4 | 0.40 | 0.29 | **0.99** | **0.85** |

**Latency:** Sub-millisecond **wall p99** through **N = 50k** on this workload; build time scales roughly linearly with N (NSN construction).

**Quality:** **Compact** same-episode cross-modal hit stays **~98–99%** across the sweep. **Top-15** is lower (**~80–85%**) because the strict head is dominated by highest-scoring neighbors (often more images), even though TEXT+AUDIO from the same episode usually appear slightly lower in the full ranked compact list.

### FAISS baseline on N = 10,000 (`faiss_baseline.json`)

| Method | p99 (ms) | Episode @96 | Episode @15 | Notes |
|--------|----------|----------------|---------------|--------|
| **Flat IP** | ~0.17 | 1.00 | 1.00 | Exact cosine; trivial for this clustered synthetic corpus |
| **IVF** (nlist 100, nprobe 25) | ~0.45 | ~0.50 | ~0.50 | **~0.92** recall@96 vs flat; episode hit drops with approximate search |

### Embodied demo (`demo_run.json`)

One run: **p99 ~0.10 ms** vs **20 ms** budget; **compact ~0.98**, **top15 ~0.84** (450 queries). Large **max** latency (~60 ms) is consistent with an occasional cold-path / driver spike, not the steady-state p99.

## Interpretation

1. **MARS vs FAISS flat (latency):** At N=10k, **FAISS flat** is still **~1.7× faster** on single-query p99 than **MARS wall** (~0.17 ms vs ~0.30 ms) — expected: MARS runs rerank + BFS + compaction that flat IP skips.

2. **MARS vs FAISS (episode metric):** With **`search_k = 96`**, flat search achieves **100%** episode hit on this synthetic geometry (same-episode nodes stay near in embedding space). **MARS compact** at **~98%** is in the same ballpark, showing the pipeline **does not throw away** same-episode cross-modal signal when evaluated over a comparable **result depth** (96), not only the top-15 head.

3. **IVF:** Lower recall vs flat directly lowers **episode hit** (~50% here); a production baseline would tune **nlist/nprobe** or use flat for small **N** where IVF is known to be weak (consistent with the main paper’s small-N IVF discussion).

4. **Head vs window:** Reporting **both** compact and top15 rates separates **“in the retrieved set”** from **“in the first screenful of ranks”** — useful for demos and ablations.

## Regenerating

On a GPU box with the repo:

```bash
./demos/embodied_scene/bench_kids_sweep --dump-only results/embodied-kids-vast-a100-40gb/kids_corpus_10k.bin 10000 768
./demos/embodied_scene/bench_kids_sweep results/embodied-kids-vast-a100-40gb/mars_sweep.json
python3 scripts/bench_kids_ball_faiss.py --corpus results/embodied-kids-vast-a100-40gb/kids_corpus_10k.bin | tee results/embodied-kids-vast-a100-40gb/faiss_baseline.json
./demos/embodied_scene/demo | sed -n '/^{/,/^}/p' > results/embodied-kids-vast-a100-40gb/demo_run.json
```
