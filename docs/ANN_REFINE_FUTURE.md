# ANN + MARS refine (future cuVS / CAGRA survey)

**Idea:** Use **GPU ANN** (e.g. cuVS **CAGRA** or **IVF-PQ / RaBitQ**) to produce **candidates** in sub-linear time, then run **MARS BFS + compaction** (or a short **exact refine**) only on the candidate set.

**Contract:** This is a **separate subsystem** from default global `query_memory_fast`. Benchmarks must report:

- **Recall@K** vs brute-force oracle on the same corpus.
- **Latency** end-to-end including candidate **H2D** if indices live off-GPU.
- **Scope** (global vs episode-scoped).

**Implementation sketch:** optional `DeviceANNIndex` handle + `d_candidate_ids` buffer; replace or gate Stage 1–3 when `RetrievalConfig::ann_mode` is enabled (not implemented in core yet).

This doc closes the **ann-refine-survey** planning item; code spikes belong on a feature branch with measured curves in `results/`.
