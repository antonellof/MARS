# Kids-ball iteration summary (N=10000)

**Criterion:** maximize `ranking_score = 2*hit_top15 + hit_compact - 1.0000 * wall_p99_ms` over **global** episode boosts.

## Best global boost

- **episode_same_boost:** 0.2400
- **composite_score:** 2.7266
- **ranking_score:** 2.4814
- **wall p99 (ms):** 0.2452
- **Artifact:** `results/iteration_steps/vast_a100_refine2_latency/step_global_boost_240.json`

## Episode-scoped reference (different contract)

- **composite_score:** 3.0000 (often ~3.0 when both hits saturate)
- **wall p99 (ms):** 0.1557
- **Artifact:** `results/iteration_steps/vast_a100_refine2_latency/step_episode_scoped.json`

**Latency:** episode-scoped is materially faster at this N (subset similarity).

## Recommendation

For **open-world** global memory, set `RetrievalConfig::episode_same_boost` to **0.240** if optimizing the **ranking** criterion (with wall-ms penalty) at N=10000.
For **session-local** UX metrics, prefer `RetrievalScope::EpisodeScoped` and label benchmarks accordingly.
