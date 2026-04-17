# MARS — Zenodo description (April-17 update)

**MARS (Memory for Autonomous Real-Time Systems)** is a GPU-resident
memory store whose retrieval API exposes three orthogonal ranking
primitives as first-class kernel-level controls — *scope*, *time*,
and *cross-modal connectivity* — for real-time embodied AI:
autonomous vehicles, humanoid robots, AR/VR headsets, and voice
agents.

Existing GPU vector search libraries (FAISS-GPU, cuVS CAGRA) expose
a single contract: rank by cosine similarity over the entire indexed
corpus. Embodied workloads typically know more than that at query
time — an active track id, a dialogue session id, a current sub-task,
or a recency window over which a memory is still operative. Encoding
this knowledge as host-side filtering over the output of a global ANN
sweep wastes GPU work and, as we show, breaks recall at scale on tiny
multimodal clusters. MARS treats those three signals as kernel-level
parameters rather than as host-side post-processing.

## Three ranking primitives, one GPU kernel pipeline

1. **Episode-scoped retrieval** (`RetrievalScope::EpisodeScoped`) —
   the central contribution of this revision. Restricts Stage 1 to
   the CSR member range of a named episode and skips cross-modal BFS
   (members are already cross-modal by construction), producing a
   **near-flat ~200 µs p99 from N=10K to N=1M** at perfect cross-modal
   `hit@15`. On the kids-ball multimodal-episode metric this is
   **33× faster than FAISS-Flat-GPU** and **16× faster than cuVS
   CAGRA** at N=1M, and the only configuration in our sweep that
   meets the 1 ms AV deadline at N=10⁶.
2. **Native temporal decay** folded into the score *before* top-K
   selection (`score × exp(-λ·age)`). Matches a two-stage FAISS +
   post-hoc filter pipeline at the same TP@10 = 0.910 in AV
   perception (0.26 ms vs 0.25 ms p99); the contribution is
   API consolidation, not raw speedup.
3. **Cross-modal bridges in the graph topology** — a warp-cooperative
   BFS over a Neural Shortcut Network (NSN) that guarantees, by
   construction, *structural* reachability of every modality from any
   seed within bounded hops. Whether the reached neighbors are
   semantically relevant depends on the shared embedding space; we
   make no claim beyond structural reachability.
4. **Hand-fused FP16 cosine kernel** (opt-in via `--use-fp16`) wins
   by 41 % at N=10K but loses by 49 % at N=1M against cuBLAS Sgemv
   (which exploits Tensor-Core paths the hand-fused kernel cannot
   match), so the build keeps cuBLAS as the default and FP16 as a
   documented small-N opt-in.

The retrieval pipeline consists of four GPU-resident kernels on the
default global path: cuBLAS SGEMV for cosine similarity with a
lightweight temporal decay kernel, CUB radix sort for top-K selection,
and warp-cooperative BFS over the multimodal NSN with cross-modal
bridges across text, audio, image, and sensor embeddings in a shared
768-dimensional space. The **episode-scoped fast path** is one
restricted-Stage-1 kernel followed by top-K — BFS is skipped because
episode members are already cross-modal by construction
(`episode_csr` member list). All data is GPU-resident with zero
per-query allocation.

## Headline measurements (A100 SXM4 40 GB, D=768, K=10)

- **Episode-scoped path:** **~200 µs p99 across N=10K..1M (near-flat)**
  at hit@15 = 1.00 on the embodied multimodal contract.
- **Global path:** sub-millisecond p99 at all corpus sizes up to
  N=50K, 2.67 ms at N=1M, scaling to 13M memories (40 GB VRAM limit)
  at 29 ms.
- **Same-hardware paired-RNG comparison** against FAISS-GPU 1.14.1
  (Flat + IVF) and cuVS CAGRA 26.04 documented in
  `results/competitors_20260417/SUMMARY.md`.

Five workloads pass empirical p99 deadlines: AV perception (60 Hz,
1 ms), humanoid robot (1 kHz, 1 ms), AR/VR spatial (90 Hz, 5 ms),
voice agent (30 Hz, 20 ms), and **MARS Episode-scoped (any rate,
N=1M, 1 ms — 80 % headroom)**. A raw cuBLAS-only ring-buffer
baseline achieves 0.12 ms — 3.2× faster than the global path at
N=2,400 — but provides no temporal decay, no cross-modal retrieval,
and no streaming insertion.

## Honest scope and caveats

- **Synthetic-corpus disclaimer.** The kids-ball benchmark uses
  Gaussian-perturbed cluster centroids in 768-D with a known, dense
  small-cluster structure. Real-encoder embeddings (CLIP, CLAP, E5)
  have different distance distributions and broader clusters; the
  exact crossover N at which cosine ANN loses recall on real data
  may shift. The qualitative point — that exhaustive cosine + episode
  CSR is the right primitive when episodes are known and small —
  should generalise; the absolute hit@15 numbers should not be quoted
  without re-measurement on the target encoder.
- **Statistical caveats.** Reported p99s are over 128–256 paired
  probes on non-locked-clock vast.ai instances. Run-to-run jitter at
  the sub-100 µs scale is on the order of ±10 %, which is large
  relative to the 0.01 ms gap between MARS and FAISS+filter on the
  temporal-decay experiment. Conclusions that depend on differences
  smaller than this should be re-measured with locked clocks and
  ≥10K queries.
- **Fairer FAISS baseline (queued).** A FAISS-Flat-GPU sweep with
  `IDSelectorBatch` / `IDSelectorRange` set to the episode member ids
  would do roughly the same work as MARS Episode-scoped and is the
  next baseline to add. It is not yet measured in this revision and
  is the first item in the *Evaluation Hardening* track of §10.

Source code, reproducible benchmarks (FAISS-GPU and cuVS CAGRA
harnesses included), refreshed architecture diagrams, and the LaTeX
paper — now framed around episode-scoped retrieval as the headline
contribution — are available at <https://github.com/antonellof/MARS>
under the MIT license. The implementation is CUDA C++17, includes
17 passing host-only unit tests, and reproduces on cloud GPUs with
one command. One known issue is documented for transparency: the
optional CUDA Graph capture path (`--use-cuda-graph`) currently
corrupts results because counters and the episode-scoped reset are
not re-initialised between graph replays
(`memory_cuda.cu:1197`); the global and episode-scoped fast paths
land cleanly without it.
