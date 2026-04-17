# Fair baseline classes (retrieval comparisons)

When comparing MARS to **FAISS**, **cuVS**, or other vector search, label the baseline **by contract**, not by library name alone:

| Class | What is measured | Typical latency driver |
|-------|-------------------|-------------------------|
| **Flat exact** | Full linear scan over **N** vectors (GPU or CPU), same distance as MARS Stage 1. | Memory bandwidth + GEMV width |
| **Multi-pass host** | Search + **host-side** temporal rerank / second pass over candidates. | Adds D2H + CPU sort — compare **wall time** to `query_memory_fast` wall time if claiming pipeline parity. |
| **ANN (IVF / graph / PQ)** | Approximate candidate generation + optional refine. | Recall–latency tradeoff; not the same contract as global exact scan. |
| **Episode-local / scoped** | Restrict candidates to **session or episode** (e.g. `RetrievalScope::EpisodeScoped`). | **O(M·D)** with **M ≪ N** — faster by design; label as a **different retrieval contract**, not a silent replacement for global memory. |

**Integrity:** Avoid implying an order-of-magnitude speedup unless **work units match** (same N, D, single-pass vs multi-pass, global vs scoped). Use measured JSON from `bench_kids_sweep` / `validate` / FAISS scripts with explicit scope in the filename and in the paper table caption.
