# Iteration steps (episode boost coarse search)

This folder holds **per-step JSON** plus **SUMMARY.{json,md}** from a single `bench_kids_sweep` invocation.

## Command

```bash
mkdir -p results/iteration_steps/run_local
./demos/embodied_scene/bench_kids_sweep \
  --boost-grid 0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5 \
  --iterate-n 10000 \
  --iterate-probes 256 \
  --iterate-steps-dir results/iteration_steps/run_local
```

**Latency-aware ranking** (maximize `composite - penalty * wall_p99_ms`):

```bash
./demos/embodied_scene/bench_kids_sweep \
  --boost-grid 0.20,0.22,0.24,0.26,0.28,0.30 \
  --iterate-n 10000 --iterate-probes 256 \
  --iterate-wall-ms-penalty 1.0 \
  --iterate-steps-dir results/iteration_steps/run_latency
```

Convenience: `make bench-kids-iterate`, `make bench-kids-iterate-refine`, `make bench-kids-iterate-refine-latency`, `make bench-kids-iterate-refine-micro`.

**Larger corpora:** the default **0.24** was chosen at **N=10k**. Optimal `episode_same_boost` can shift as the NSN and candidate pool grow. Re-run the same micro-grid at larger **N**:

```bash
make bench-kids-iterate-refine-micro-20k
make bench-kids-iterate-refine-micro-50k
make bench-kids-iterate-refine-micro-100k
make bench-kids-iterate-refine-micro-250k
make bench-kids-iterate-refine-micro-500k
make bench-kids-iterate-refine-micro-1m
```

**Up to N=1M:** `EmbodiedKidsBallCorpus` and the CUDA path use `int32_t` counts; **1M×768 FP32** embeddings are **~3 GiB** on device (plus CSR, episode arrays, and `QueryContext` scratch — still typically fine on **A100 40GB+**). Expect **minutes** for NSN host build and **long** iterate wall time unless you lower `--iterate-probes`. The `bench-kids-iterate-refine-micro-1m` target uses **128** probes per boost for that reason.

Or any N: `./demos/embodied_scene/bench_kids_sweep --boost-grid … --iterate-n 1000000 --iterate-probes 64 --iterate-steps-dir results/iteration_steps/my_1m_run`. `SUMMARY.json` records `iterate_n`, `iterate_probes`, and `corpus_seed`.

**Host RAM:** peak includes a full host copy of embeddings during `make()` and upload (~3 GiB at 1M); use a machine with sufficient **CPU RAM** (≥16 GiB free recommended for 1M).

### Fill available VRAM (e.g. A100 40 GB)

`scripts/kids_ball_max_n_vram.py` estimates the largest kids-ball **N** (768-D, NSN `k=6`, `p=0.15`) that fits a GPU budget including `QueryContext`, using the measured CSR degree **≈11.02**. Example for **40 GiB** card leaving **768 MiB** headroom:

```bash
python3 scripts/kids_ball_max_n_vram.py --vram-gib 40 --reserve-mib 768
```

One-shot smoke (default **0.24** boost, **4** probes — long **host** NSN build at ~13M nodes):

```bash
make bench-kids-vram-max-smoke
# Other cards: VRAM_GIB=80 make bench-kids-vram-max-smoke
```

**Host DRAM:** at max **N** the FP32 embedding matrix alone is **~N×768×4** bytes (≈**38 GiB** at 40 GB-class **N**). The machine needs that headroom **before** `cudaMemcpy`.

**Runtime:** NSN construction is CPU-heavy; very large **N** can take many minutes.

- **`step_global_boost_<milli>.json`** — one file per boost value (milli = round(boost × 1000)).
- **`step_episode_scoped.json`** — reference run (different retrieval contract).
- **`SUMMARY.json` / `SUMMARY.md`** — best global boost under criterion  
  `composite_score = 2 * hit_top15 + hit_compact`.

## Applying the winner

1. Open `SUMMARY.md` and note `best_global_episode_same_boost`.
2. Set the same value in [`demos/embodied_scene/demo.cu`](../../demos/embodied_scene/demo.cu) as `cfg.episode_same_boost`.

Re-run the full multi-N sweep when you change defaults:

```bash
./demos/embodied_scene/bench_kids_sweep results/kids_ball_mars_sweep.json
```

## Saved hardware runs

| Run | GPU | Notes |
|-----|-----|--------|
| [vast_a100_20260414](vast_a100_20260414/) | A100-SXM4-40GB | Coarse grid: best **0.25** at N=10k (see `SUMMARY.md`). |
| [vast_a100_refine2](vast_a100_refine2/) | A100-SXM4-40GB | Grid 0.20–0.30 (step 0.02): best **0.24** (0.25 not sampled). |
| [vast_a100_refine2_latency](vast_a100_refine2_latency/) | A100-SXM4-40GB | Same grid, `--iterate-wall-ms-penalty 1.0`: best still **0.24**. |
| [vast_a100_refine3](vast_a100_refine3/) | A100-SXM4-40GB | Micro 0.23–0.26 incl. **0.25**: **0.24** wins composite; demo default **0.24**. |
