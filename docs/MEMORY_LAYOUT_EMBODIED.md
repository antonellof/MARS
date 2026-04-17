# Memory layout: embodied scale and GPU traffic

This note complements [ARCHITECTURE.md](ARCHITECTURE.md) §1 with **layout and precision choices** relevant to **fixed-session embodied corpora** (e.g. kids-ball: ~10 nodes per episode, thousands of episodes) and **kids-scale N** (1K–50K) used in `bench_kids_sweep`.

## Row-major AoS (current default)

Embeddings are stored **row-major**: node `i` occupies `embeddings[i*D .. i*D+D-1]`.

- **Pros:** Simple host build, one `cudaMemcpy` to device, natural cuBLAS `SGEMV` layout (`lda = D`, column-major query as in MARS).
- **Cons:** For a block assigned to **one dimension tile** of many nodes, accesses can be **stride-D** across the matrix; depending on kernel mapping, this may leave HBM bandwidth below peak compared to blocked layouts.

## Structure-of-arrays (SoA) / blocked tiles

**Idea:** Reorder storage so that a CTA reads **contiguous** bytes for the dimensions it is responsible for across a **tile of nodes** (or store `[tile][d][lane]`).

- **Benefit:** Better coalescing when the inner loop is over **nodes** at fixed `d`, or when using tensor-core tiles that expect **16×16** fragments.
- **Cost:** Host reorder on insert/rebuild; index translation `logical_id → (tile, slot)`; NSN CSR still indexes **logical node ids**.

**When it matters most:** Very large **D** (e.g. 1024–1536) and wide warps reading many nodes per CTA—not the dominant term at **D=768** on A100 for **N≈10K**, but worth revisiting before **N≥1M** tuning.

## FP16 / BF16 primary store

| Layout | Bytes per embedding (D=768) | N=10K VRAM (embeddings only) | Bandwidth vs FP32 |
|--------|----------------------------|------------------------------|-------------------|
| FP32   | 3072                     | ~30 MB                       | 1×                |
| FP16   | 1536                     | ~15 MB                       | ~2× less moved    |
| BF16   | 1536                     | ~15 MB                       | Similar to FP16 |

MARS already supports an **FP16 device copy** (`d_embeddings_fp16`) and optional **FP16 fused similarity** (`QueryContext::use_fp16`). **BF16** would follow the same pattern with `__nv_bfloat16` where hardware supports it.

**Tradeoff:** Half precision reduces **dynamic range**; for **L2-normalized** unit vectors the main risk is accumulation error in long dot products—typically acceptable at D≤1024 with FP32 accumulate (current scalar FP16 path widens to float).

## VRAM sketch (kids N = 10 000, D = 768, avg degree ≈12)

Roughly (same order as ARCHITECTURE §1 table, scaled):

| Component | FP32 | + FP16 copy |
|-----------|------|-------------|
| `embeddings` | ~30 MB | +~15 MB |
| CSR + modalities + timestamps | ~1 MB | — |
| Episode sidecar (`episode_ids` + CSR lists) | &lt; 1 MB | — |

Episode metadata is **negligible** vs embeddings at these N.

## Recommendation order

1. **Measure** with Nsight Compute whether Stage 1 is **DRAM-limited** vs **math-limited** for your GPU and N.
2. If DRAM-limited: **FP16 primary** (already scaffolded) before a full SoA rewrite.
3. If episode-local queries are common: **episode-scoped subset scan** (see [memory_cuda.cuh](../include/memory_cuda.cuh) `RetrievalScope`) reduces **effective** embedding traffic to **O(M·D)** with **M = episode size** (e.g. 10), independent of global N.
