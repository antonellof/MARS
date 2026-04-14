# Temporal hierarchy: hot tier, warm NSN, and buckets

The default pipeline applies **per-query exponential decay** over **all N** memories (`temporal_rerank_kernel` or fused similarity+decay). That matches **exact global semantics** but does not scale down work when **most mass is recent**.

This document specifies a **two-tier (and optional bucket)** model aligned with the architecture plan—implementation can be incremental behind config flags.

## Goals

- Keep **exact semantics** available as **`GLOBAL` / full-N`** mode.
- Add a **fast path** where **most queries** only touch a **small active set** `A(t)` (seconds of sensor data).
- Preserve **cross-session** retrieval via a **warm** layer (NSN + occasional full refresh).

## Tier A — Hot ring (exact, small)

- **Structure:** Circular buffer of the last **T_hot** seconds (or last **N_hot** inserts), stored contiguously in GPU memory.
- **Query:** Compute `score = exp(-λ·age) · sim(q, x)` **only** for `i ∈ A(t)` (vectorized or small GEMV).
- **Latency:** `O(|A|·D)` with `|A| ≪ N` (e.g. humanoid 1 kHz: keep |A| in the low thousands).
- **Consistency:** On each insert, push to ring; optionally **mirror** into Tier B asynchronously.

## Tier B — Warm NSN (approximate reach)

- **Structure:** Existing CSR NSN over **all** nodes (or over **non-hot** nodes plus hub links from hot).
- **Query:** Start from **hot top-K** seeds + **bridge** edges into warm regions; run **bounded-hop BFS** (current MARS Stage 4) with **score decay per hop**.
- **Role:** Retrieve **older** but **graph-reachable** memories without a full-N similarity pass every frame.

## Tier C — Full refresh (rare)

- **Trigger:** Periodic (e.g. every Δ seconds), on **multimodal session change**, or when **recall@K** monitor drops.
- **Action:** Full-N similarity (or ANN candidate generation + refine) to **re-seed** hot/warm structures.

## Piecewise λ and time buckets

**Problem:** `exp(-λ·age)` is cheap per element but still **touches every element** in full-N mode.

**Mitigation (spec):**

1. **Quantize** timestamps to **buckets** (e.g. 100 ms walls).
2. Precompute **bucket decay multipliers** `w[b]` so per-node work becomes a **table lookup** + multiply (still linear in N, but can replace `expf` with **shared-memory broadcast** of a few weights).
3. **Piecewise λ:** Use `λ₁` for age &lt; τ and `λ₂` after (two exponentials or linear blend)—documented policy only; same linear scan unless combined with **hot tier**.

## Math sketch (hot + warm)

Let `A(t)` be hot indices, `B` the full graph.

- **Default query:** compute scores on `A(t)` only; take top-K₀.
- **Expand:** BFS from those seeds for **h** hops on `B` with hop decay (already in MARS).
- **Optional:** Additive **background score** from a stale **centroid** or **last full-scan** summary (future work).

## Interaction with streaming insertion

The **streaming-insertion** feature branch (ring + incremental NSN) is the natural host for **Tier A** physical eviction. This doc is the **semantic contract**; the ring buffer is the **mechanism**.

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) — fused similarity + decay, BFS stages.
- [paper/main.tex](../paper/main.tex) — limitations, streaming narrative.
