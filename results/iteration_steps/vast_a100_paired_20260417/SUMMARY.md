# Paired-probes re-tune of `episode_same_boost` (A100 SXM4 40GB)

**Hardware:** NVIDIA A100-SXM4-40GB on `ssh5.vast.ai:19293`
(driver 565.57.01, CUDA 12.6, Ubuntu 24.04.4).
**Branch base:** `feature/embodied-kids-ball-benchmark` @ `3b04e1c`.
**Patch under test:** paired-probe RNG fix
(`bench_kids_sweep.cu` v1.3) — same probe sequence across all variants
and all boost values at a given N. Previously the RNG seed mixed in
`vi` / `boost*10000`, so each row of the boost grid measured a
different probe sample, hiding the boost effect under sampling noise.

## TL;DR

1. The `episode_same_boost` parameter has **no measurable effect** on
   the kids-ball metric (IMAGE probe → same-episode TEXT+AUDIO in the
   compacted rank window) for any value in `[0.0, 2.0]` at
   N ∈ {10K, 50K, 100K, 1M}. Hit rates change by at most a single
   probe out of the sample size, and the boost only adds a small
   constant wall-clock cost. Recommended default: **`0.0`**.
2. **Episode-scoped retrieval** (`RetrievalScope::EpisodeScoped`,
   used when `query_episode_id` is known) Pareto-dominates every
   global variant: lower latency *and* perfect hit rate.

## Boost grid is flat (probe-equivalent)

Each row is a paired comparison — same probes, same corpus, same
NSN graph; only `cfg.episode_same_boost` changes.

### N = 10 000, 512 probes

| boost | hit_top15 | hit_compact | composite | wall p99 (ms) |
|------:|----------:|------------:|----------:|--------------:|
| 0.00  | 0.826     | 0.979       | 2.6309    | 0.349 |
| 0.05  | 0.826     | 0.980       | 2.6328    | 0.333 |
| 0.10  | 0.826     | 0.979       | 2.6309    | 0.295 |
| 0.15  | 0.826     | 0.980       | 2.6328    | 0.282 |
| 0.20  | 0.826     | 0.980       | 2.6328    | 0.282 |
| 0.25  | 0.826     | 0.979       | 2.6309    | 0.281 |
| 0.30  | 0.826     | 0.980       | 2.6328    | 0.291 |
| 0.40  | 0.826     | 0.980       | 2.6328    | 0.292 |
| 0.50  | 0.826     | 0.979       | 2.6309    | 0.285 |
| 0.75  | 0.826     | 0.980       | 2.6328    | 0.287 |
| 1.00  | 0.826     | 0.980       | 2.6328    | 0.281 |
| 2.00  | 0.826     | 0.979       | 2.6309    | 0.282 |
| **scoped** | **1.000** | **1.000** | **3.0000** | **0.165** |

`hit_top15` is identical across every global boost (it is a 0/1
indicator that flips for the same probes regardless of the boost).
`hit_compact` flips one probe out of 512 between values — the boost
is below the measurement noise floor.

### N = 50 000, 256 probes

| boost | hit_top15 | hit_compact | composite | wall p99 (ms) |
|------:|----------:|------------:|----------:|--------------:|
| 0.0   | 0.883     | 0.992       | 2.7578    | 0.462 |
| 0.1   | 0.883     | 0.992       | 2.7578    | 0.437 |
| 0.2   | 0.883     | 0.992       | 2.7578    | 0.430 |
| 0.3   | 0.883     | 0.992       | 2.7578    | 0.403 |
| 0.5   | 0.879     | 0.992       | 2.7500    | 0.390 |
| 1.0   | 0.879     | 0.992       | 2.7500    | 0.388 |
| **scoped** | **1.000** | **1.000** | **3.0000** | **0.172** |

### N = 100 000, 256 probes

| boost | hit_top15 | hit_compact | composite | wall p99 (ms) |
|------:|----------:|------------:|----------:|--------------:|
| 0.0   | 0.793     | 0.980       | 2.5664    | 0.551 |
| 0.1   | 0.793     | 0.977       | 2.5625    | 0.583 |
| 0.2   | 0.793     | 0.977       | 2.5625    | 0.504 |
| 0.5   | 0.793     | 0.977       | 2.5625    | 0.497 |
| **scoped** | **1.000** | **1.000** | **3.0000** | **0.174** |

### N = 1 000 000, 128 probes

| boost | hit_top15 | hit_compact | composite | wall p99 (ms) |
|------:|----------:|------------:|----------:|--------------:|
| 0.0   | 0.844     | 0.977       | 2.6641    | 2.561 |
| 0.1   | 0.844     | 0.977       | 2.6641    | 2.468 |
| 0.2   | 0.844     | 0.977       | 2.6641    | 2.463 |
| 0.3   | 0.844     | 0.977       | 2.6641    | 2.467 |
| 0.5   | 0.844     | 0.977       | 2.6641    | 2.469 |
| 1.0   | 0.844     | 0.977       | 2.6641    | 2.471 |
| 2.0   | 0.844     | 0.977       | 2.6641    | 2.469 |
| **scoped** | **1.000** | **1.000** | **3.0000** | **0.197** |

## Episode-scoped scaling (Global vs Scoped, paired probes)

| N        | global p99 | scoped p99 | scoped speedup | global hit_t15 | scoped hit_t15 |
|---------:|-----------:|-----------:|---------------:|---------------:|---------------:|
| 10 000   | 0.349 ms   | 0.165 ms   | 2.1×           | 0.826          | 1.000          |
| 50 000   | 0.462 ms   | 0.172 ms   | 2.7×           | 0.883          | 1.000          |
| 100 000  | 0.551 ms   | 0.174 ms   | 3.2×           | 0.793          | 1.000          |
| 1 000 000| 2.561 ms   | 0.197 ms   | **13.0×**      | 0.844          | 1.000          |

When `query_episode_id` is known, episode-scoped retrieval scales
nearly flat with N (only ~10 episode members per query) and returns
the strict superset of useful results for this metric. The speedup
grows with N because Stage 1 work for the global path is Θ(N·D).

## Multi-N MARS sweep (paired)

`kids_ball_mars_sweep_paired.json` (v1.3) — three retrieval variants
per N, now using the same probe seed across variants:

| N      | baseline hit_t15 | boost-0.24 hit_t15 | scoped hit_t15 | baseline p99 | scoped p99 |
|-------:|-----------------:|-------------------:|---------------:|-------------:|-----------:|
| 1 000  | 0.801            | 0.801              | 1.000          | 0.177 ms     | 0.115 ms   |
| 5 000  | 0.809            | 0.809              | 1.000          | 0.269 ms     | 0.187 ms   |
| 10 000 | 0.816            | 0.816              | 1.000          | 0.255 ms     | 0.159 ms   |
| 20 000 | 0.836            | 0.836              | 1.000          | 0.283 ms     | 0.159 ms   |
| 50 000 | 0.852            | 0.852              | 1.000          | 0.353 ms     | 0.158 ms   |

`baseline` and `boost_0.24` now produce identical hit rates at every N
— direct evidence that the previous "boost helps" / "0.24 wins"
conclusion was probe-sampling noise from the unpaired RNG.

## What changed in code

* `demos/embodied_scene/bench_kids_sweep.cu` — paired-probes RNG (v1.3
  on both `mars-kids-ball-sweep` and `mars-kids-boost-iterate` JSONs).
* `demos/embodied_scene/demo.cu` — `episode_same_boost` default
  changed from `0.24f` to `0.0f`; new `--scope=episode` /
  `--episode-scoped` flag toggles `RetrievalScope::EpisodeScoped` and
  populates `query_episode_member_begin/_count`.

## Demo verification (N = 3 600 default)

Same machine, same binary, two CLI options:

| mode                | p99       | hit_top15 | hit_compact |
|---------------------|----------:|----------:|------------:|
| `demo` (global)     | 0.098 ms  | 0.836     | 0.984       |
| `demo --scope=episode` | **0.069 ms** | **1.000** | **1.000** |

Episode-scoped is 30 % faster *and* +16.4 pp on top-15 hits at the
demo's default corpus size.
