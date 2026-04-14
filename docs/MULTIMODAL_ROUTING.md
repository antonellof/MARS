# Multimodal routing: seeds, merges, and NSN bridges

NSN **Phase 5 cross-modal bridges** ([ARCHITECTURE.md](ARCHITECTURE.md) §2) guarantee **graph reachability**: an IMAGE probe can **reach** TEXT/AUDIO nodes in bounded hops. They do **not** guarantee that TEXT/AUDIO appear in the **first K** ranked results after global similarity + top-K + BFS compaction—see embodied metrics (`episode_cross_modal_hit_rate_compact` vs `_top15`) in `results/embodied-kids-vast-a100-40gb/ANALYSIS.md`.

This document specifies **routing strategies** that improve **head-of-list** multimodal quality without silently changing the default **global O(N)** contract.

## 1. Modality-aware seeds

**Idea:** Form seeds from **more than one modality view** of the query.

**Variants:**

- **A — Filtered then merge:** Run similarity with `modality_filter = query_modality` to get **K₁** seeds, then run **unfiltered** (or other modality) for **K₂**, **merge/dedupe** before BFS.
- **B — Parallel K per modality:** Three small top-K passes (TEXT/AUDIO/IMAGE) each with its own **K_m**, merge by score, cap total seeds ≤ `ctx.max_k`.

**Invariant:** Seeds are still **node ids** into the **same** CSR; BFS and compaction unchanged.

**Cost:** Extra GEMV launches or a wider effective K at the top—tune vs latency budget.

## 2. Per-modality centroid graph (spec)

**Idea:** Supernodes represent **modality clusters**; BFS can jump **cluster → cluster** in one hop.

**Mechanism:** Offline or periodic clustering → add **sparse long edges** from each node to its centroid and centroid-to-centroid bridges.

**Tradeoff:** Higher degree on centroids; needs **degree caps** to keep BFS frontier bounded.

## 3. Residual / learned cross-modal score (spec)

**Idea:** At ranking time, apply a **small additive or multiplicative correction** when promoting cross-modal neighbors (e.g. boost TEXT when query is IMAGE) using a **fixed table** or **learned gate**.

**Contract:** Document as **rerank policy** on top of geometric similarity—not a replacement for normalized cosine unless explicitly benchmarked.

## 4. Episode sidecar + `max_results_returned`

**Implemented path:** `RetrievalConfig` supports **episode-scoped** subset scoring and optional **same-episode score boost** (see [memory_cuda.cuh](../include/memory_cuda.cuh)). Together with **`max_results_returned`**, the host can score **cross-modal hits** over a **wider window** than strict top-15 without inflating BFS seed count.

## 5. NSN bridge invariants (unchanged)

- Undirected CSR, sorted `col_indices`, no self-loops.
- Phase 5 ensures **each node has at least one edge** toward **other modalities** when enabled.

Routing policies above **must not** break these invariants; they only change **which nodes become seeds** or **how scores are adjusted** before/after top-K.

## References

- [memory_graph.cpp](../src/memory_graph.cpp) — `build_nsn_edges` phases.
- [demos/embodied_scene/demo.cu](../demos/embodied_scene/demo.cu) — episode cross-modal metrics.
