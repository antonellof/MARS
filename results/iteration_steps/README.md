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

Convenience: `make bench-kids-iterate`, `make bench-kids-iterate-refine`, `make bench-kids-iterate-refine-latency`.

- **`step_global_boost_<milli>.json`** — one file per boost value (milli = round(boost × 1000)).
- **`step_episode_scoped.json`** — reference run (different retrieval contract).
- **`SUMMARY.json` / `SUMMARY.md`** — best global boost under criterion  
  `composite_score = 2 * hit_top15 + hit_compact`.

## Applying the winner

1. Open `SUMMARY.md` and note `best_global_episode_same_boost`.
2. Set the same value in [`demos/embodied_scene/demo.cu`](../../demos/embodied_scene/demo.cu) as `cfg.episode_same_boost` (or keep 0.35 if within noise of optimum).

Re-run the full multi-N sweep when you change defaults:

```bash
./demos/embodied_scene/bench_kids_sweep results/kids_ball_mars_sweep.json
```

## Saved hardware runs

| Run | GPU | Notes |
|-----|-----|--------|
| [vast_a100_20260414](vast_a100_20260414/) | A100-SXM4-40GB | Best **global** boost **0.25** at N=10k (`SUMMARY.md`); demo default updated to match. |
