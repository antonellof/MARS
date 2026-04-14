# vast.ai validation matrix (embodied kids-ball)

Use this checklist when running **hardware ablations** on a rented GPU. Copy JSON outputs into this folder (or `results/embodied-kids-vast-a100-40gb/`) and record `RUN_META.json` (git SHA, GPU name, driver, CUDA, clock lock if any).

## Core sweep (MARS)

```bash
make clean && make
make tests
./bench_kids_sweep results/vast-validation-matrix/mars_kids_sweep_$(date +%Y%m%d).json
```

`bench_kids_sweep` (v1.2+) emits **variants**:

| Variant | Meaning |
|---------|---------|
| `global_baseline` | Full `N` similarity + temporal + cuBLAS/CUB + BFS (default). |
| `global_episode_boost_0.35` | Same + `episode_same_boost=0.35` for nodes in the probe episode. |
| `episode_scoped_no_bfs` | Similarity + decay **only** over members of the probe episode (`O(M·D)`); **BFS hops = 0**. |

## Optional rows (extend matrix)

| Experiment | Command / note |
|------------|----------------|
| FAISS flat baseline | `python3 scripts/bench_kids_ball_faiss.py --corpus …/kids_corpus_10k.bin` |
| FP16 embeddings | `upload_fp16_embeddings(dg)` + `ctx.use_fp16 = true` (custom harness or extend bench) |
| CUDA graph | `ctx.use_cuda_graph = true` (global retrieval only; scoped disables graph) |
| NSN ablation | Rebuild graph with `MemoryGraph::NSNConfig` phase flags off |

## Artifacts

- `validation_matrix.template.json` — empty slots for measured numbers + file paths.
- Measured runs: name files `mars_kids_sweep_<date>.json`, `faiss_<date>.json`, etc.

See also: [docs/MEMORY_LAYOUT_EMBODIED.md](../../docs/MEMORY_LAYOUT_EMBODIED.md), [TEMPORAL_HIERARCHY.md](../../docs/TEMPORAL_HIERARCHY.md), [MULTIMODAL_ROUTING.md](../../docs/MULTIMODAL_ROUTING.md).
