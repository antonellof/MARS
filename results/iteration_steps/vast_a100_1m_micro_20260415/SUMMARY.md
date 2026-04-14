# Kids-ball iteration summary (N=1000000)

**Criterion:** maximize `2 * hit_top15 + hit_compact` over **global** episode boosts.

## Best global boost

- **episode_same_boost:** 0.2450
- **composite_score:** 2.7578
- **ranking_score:** 2.7578
- **wall p99 (ms):** 2.5561
- **Artifact:** `results/iteration_steps/make_refine_micro_1m/step_global_boost_245.json`

## Episode-scoped reference (different contract)

- **composite_score:** 3.0000 (often ~3.0 when both hits saturate)
- **wall p99 (ms):** 0.2116
- **Artifact:** `results/iteration_steps/make_refine_micro_1m/step_episode_scoped.json`

**Latency:** episode-scoped is materially faster at this N (subset similarity).

## Recommendation

For **open-world** global memory, set `RetrievalConfig::episode_same_boost` to **0.245** if optimizing the stated composite at N=1000000.
For **session-local** UX metrics, prefer `RetrievalScope::EpisodeScoped` and label benchmarks accordingly.
