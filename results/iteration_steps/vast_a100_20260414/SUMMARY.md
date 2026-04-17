# Kids-ball iteration summary (N=10000)

**Criterion:** maximize `2 * hit_top15 + hit_compact` over **global** episode boosts.

## Best global boost

- **episode_same_boost:** 0.2500
- **composite_score:** 2.6641
- **wall p99 (ms):** 0.2409
- **Artifact:** `results/iteration_steps/vast_a100_20260414/step_global_boost_250.json`

## Episode-scoped reference (different contract)

- **composite_score:** 3.0000 (often ~3.0 when both hits saturate)
- **wall p99 (ms):** 0.1543
- **Artifact:** `results/iteration_steps/vast_a100_20260414/step_episode_scoped.json`

**Latency:** episode-scoped is materially faster at this N (subset similarity).

## Recommendation

For **open-world** global memory, set `RetrievalConfig::episode_same_boost` to **0.250** if optimizing the stated composite at N=10000 **on this coarse grid only**.
For **session-local** UX metrics, prefer `RetrievalScope::EpisodeScoped` and label benchmarks accordingly.

**Note:** A finer grid that includes **0.24** and **0.25** (`../vast_a100_refine3/SUMMARY.md`) prefers **0.24** for the same composite at the same N/probes; `demos/embodied_scene/demo.cu` follows that refine result.
