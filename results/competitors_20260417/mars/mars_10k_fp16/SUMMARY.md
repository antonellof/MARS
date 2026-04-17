# Kids-ball iteration summary (N=10000)

**Criterion:** maximize `2 * hit_top15 + hit_compact` over **global** episode boosts.

## Best global boost

- **episode_same_boost:** 0.0000
- **composite_score:** 2.6250
- **ranking_score:** 2.6250
- **wall p99 (ms):** 0.2796
- **Artifact:** `results/competitors_20260417/mars/mars_10k_fp16/step_global_boost_0.json`

## Episode-scoped reference (different contract)

- **composite_score:** 3.0000 (often ~3.0 when both hits saturate)
- **wall p99 (ms):** 0.1909
- **Artifact:** `results/competitors_20260417/mars/mars_10k_fp16/step_episode_scoped.json`

**Latency:** episode-scoped is materially faster at this N (subset similarity).

## Recommendation

For **open-world** global memory, set `RetrievalConfig::episode_same_boost` to **0.000** if optimizing the stated composite at N=10000.
For **session-local** UX metrics, prefer `RetrievalScope::EpisodeScoped` and label benchmarks accordingly.
