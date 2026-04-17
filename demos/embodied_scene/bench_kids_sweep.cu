// ============================================================================
//  bench_kids_sweep.cu — MARS scaling benchmark on embodied kids-ball corpus.
//
//  Sweeps corpus size N, measures wall + kernel p99, build_ms, and
//  same-episode cross-modal hit rate on the compacted rank window (see
//  RetrievalConfig::max_results_returned) plus strict top-15 for comparison.
//
//  Emits multiple retrieval **variants** per run (global baseline, global +
//  episode_same_boost, episode_scoped subset) for vast.ai validation matrix.
//
//  Usage:
//    ./bench_kids_sweep [out.json]
//    ./bench_kids_sweep --dump-corpus path.bin [--dump-n 5000] [out.json]
//
//  Iteration / coarse grid (saves one JSON per boost + SUMMARY):
//    ./bench_kids_sweep --boost-grid 0,0.15,0.25,0.35,0.5 --iterate-n 10000 \
//        --iterate-steps-dir results/iteration_steps/my_run --iterate-probes 256
//  Optional latency-aware ranking (pick best by composite - penalty * wall_p99_ms):
//    ... --iterate-wall-ms-penalty 1.0
//
//  Path flags (iterate mode only — the global Stage 1 path is what they tune):
//    --use-fp16        Use FP16 fused similarity (halves Stage 1 bandwidth, FP32 accumulate).
//                      Disables cuBLAS SGEMV in this run and uploads a half-precision copy
//                      of the embedding matrix.
//    --use-cuda-graph  Capture+replay the global retrieval pipeline as a CUDA Graph after
//                      the first warm query. Episode-scoped reference always runs without
//                      capture (different scope = recapture).
//
//  Binary corpus format (little-endian) for FAISS baseline scripts:
//    uint32 magic 'KIDS' (0x5344494b)
//    uint32 version (=1)
//    int32  N, D
//    float  embeddings[N*D]
//    float  timestamps[N]
//    int32  modalities[N]
//    int32  episode_ids[N]
// ============================================================================
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "memory_graph.h"
#include "memory_cuda.cuh"

namespace {

constexpr uint32_t kCorpusMagic   = 0x5344494bu; // 'SDIK' LE -> bytes KIDS
constexpr uint32_t kCorpusVersion = 1u;

double percentile_p99(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(0.99 * double(v.size() - 1));
    return v[idx];
}

bool episode_cross_modal_hit(const std::vector<QueryResult>& results,
                             int32_t query_episode,
                             const std::vector<int32_t>& episode_ids,
                             int32_t top_check) {
    bool has_text  = false;
    bool has_audio = false;
    const int32_t n = std::min(top_check, static_cast<int32_t>(results.size()));
    for (int32_t i = 0; i < n; ++i) {
        int32_t nid = results[i].node_id;
        if (nid < 0 || nid >= static_cast<int32_t>(episode_ids.size()))
            continue;
        if (episode_ids[nid] != query_episode)
            continue;
        if (results[i].modality == MOD_TEXT)  has_text  = true;
        if (results[i].modality == MOD_AUDIO) has_audio = true;
    }
    return has_text && has_audio;
}

void write_corpus_bin(const char* path, const MemoryGraph& g,
                       const std::vector<int32_t>& episode_ids) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) {
        std::perror(path);
        std::exit(1);
    }
    const int32_t N = g.num_nodes;
    const int32_t D = g.embedding_dim;
    std::fwrite(&kCorpusMagic, sizeof(kCorpusMagic), 1, fp);
    std::fwrite(&kCorpusVersion, sizeof(kCorpusVersion), 1, fp);
    std::fwrite(&N, sizeof(N), 1, fp);
    std::fwrite(&D, sizeof(D), 1, fp);
    std::fwrite(g.embeddings.data(), sizeof(float), size_t(N) * size_t(D), fp);
    std::fwrite(g.timestamps.data(), sizeof(float), size_t(N), fp);
    std::fwrite(g.modalities.data(), sizeof(int32_t), size_t(N), fp);
    std::fwrite(episode_ids.data(), sizeof(int32_t), size_t(N), fp);
    std::fclose(fp);
}

struct SweepRow {
    int32_t N = 0;
    double  build_ms = 0;
    double  wall_p99_ms = 0;
    double  kernel_p99_ms = 0;
    double  episode_cross_modal_hit_rate_compact = 0;
    double  episode_cross_modal_hit_rate_top15   = 0;
};

struct BenchVariant {
    const char*       name = nullptr;
    RetrievalScope    scope = RetrievalScope::Global;
    float             episode_same_boost = 0.0f;
};

std::vector<float> parse_boost_list(const char* s) {
    std::vector<float> v;
    if (!s || !*s) return v;
    const char* p = s;
    while (*p) {
        char* endp = nullptr;
        float x = std::strtof(p, &endp);
        if (endp == p) break;
        v.push_back(x);
        if (*endp == ',') {
            ++endp;
        } else if (*endp == '\0') {
            break;
        } else {
            break;
        }
        p = endp;
    }
    return v;
}

// Returns 0 on success. Writes step_boost_<milli>.json per boost and SUMMARY.{json,md}.
// If wall_ms_penalty > 0, best global step maximizes (composite - wall_ms_penalty * wall_p99_ms).
//
// Optional context flags:
//   use_fp16:        opt into FP16 fused similarity (Stage 1 bandwidth halved). Falls back
//                    to cuBLAS or scalar FP32 if disabled. Toggle via --use-fp16.
//   use_cuda_graph:  capture+replay the global retrieval pipeline as a CUDA Graph (skips
//                    per-launch overhead). Currently only valid for the global path; the
//                    episode-scoped reference always runs without graph capture. Toggle
//                    via --use-cuda-graph.
int run_boost_iterate_mode(const std::vector<float>& boosts, int32_t grid_n, int32_t grid_probes,
                           const char* steps_dir, int32_t dim, int32_t top_k, int32_t max_results_host,
                           int32_t query_ctx_max_k, uint32_t corpus_seed, double wall_ms_penalty,
                           bool use_fp16 = false, bool use_cuda_graph = false) {
    if (boosts.empty()) {
        std::fprintf(stderr, "--boost-grid: empty list\n");
        return 1;
    }
    {
        std::string cmd = std::string("mkdir -p \"") + steps_dir + "\"";
        if (std::system(cmd.c_str()) != 0) {
            std::fprintf(stderr, "failed: %s\n", cmd.c_str());
            return 1;
        }
    }

    auto t_build0 = std::chrono::high_resolution_clock::now();
    EmbodiedKidsBallCorpus corp = EmbodiedKidsBallCorpus::make(grid_n, dim, corpus_seed);
    MemoryGraph& g = corp.graph;
    g.build_nsn_edges(6, 0.15);
    auto t_build1 = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();

    std::vector<int32_t> image_nodes;
    for (int32_t i = 0; i < g.num_nodes; ++i)
        if (g.modalities[i] == MOD_IMAGE)
            image_nodes.push_back(i);
    if (image_nodes.empty()) {
        std::fprintf(stderr, "No IMAGE nodes\n");
        return 1;
    }

    HostEpisodeCSR ep_csr = build_episode_csr(corp.episode_ids, g.num_nodes);
    DeviceMemoryGraph dg = upload_to_device(g);
    upload_episode_ids(dg, corp.episode_ids.data(), g.num_nodes);
    upload_episode_csr(dg, ep_csr);
    if (use_fp16) {
        upload_fp16_embeddings(dg);
    }
    QueryContext ctx = create_query_context(g.num_nodes, dim, query_ctx_max_k);
    if (use_fp16) {
        ctx.use_fp16   = true;
        ctx.use_cublas = false;  // FP16 fused path is the alternative to cuBLAS SGEMV
    }
    ctx.use_cuda_graph = use_cuda_graph;

    struct StepRec {
        float   boost = 0;
        std::string file;
        double  wall_p99 = 0;
        double  kern_p99 = 0;
        double  hit_c = 0;
        double  hit_t15 = 0;
        double  score = 0;        // 2*hit_top15 + hit_compact
        double  ranking_score = 0; // score - wall_ms_penalty * wall_p99 (used for best if penalty>0)
    };
    std::vector<StepRec> steps;
    steps.reserve(boosts.size());

    for (float boost : boosts) {
        RetrievalConfig cfg;
        cfg.top_k                = top_k;
        cfg.max_results_returned = max_results_host;
        cfg.bfs_max_hops         = 2;
        cfg.time_decay_lambda    = 8e-6f;
        cfg.bfs_score_decay      = 0.55f;
        cfg.modality_filter      = -1;
        cfg.retrieval_scope      = RetrievalScope::Global;
        cfg.episode_same_boost   = boost;
        cfg.query_episode_member_begin = 0;
        cfg.query_episode_member_count = 0;

        // SAME probe sample across boosts for paired comparison at a given N.
        std::mt19937 rng(uint32_t(grid_n) ^ 0xB00Bu);
        std::uniform_int_distribution<size_t> pick(0, image_nodes.size() - 1);
        std::vector<double> wall_ms, kern_ms;
        wall_ms.reserve(size_t(grid_probes));
        kern_ms.reserve(size_t(grid_probes));
        int32_t hits_compact = 0;
        int32_t hits_top15   = 0;

        for (int32_t p = 0; p < grid_probes; ++p) {
            const int32_t node = image_nodes[pick(rng)];
            const float* qemb  = &g.embeddings[size_t(node) * dim];
            float qts = g.timestamps[node] + float(p) * 0.001f;
            const int32_t ep = corp.episode_ids[node];
            cfg.query_episode_id = ep;

            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            RetrievalStats stats;
            auto res = query_memory_fast(dg, ctx, qemb, qts, cfg, stats);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            kern_ms.push_back(double(stats.gpu_ms_total));
            const int32_t wfull = static_cast<int32_t>(res.size());
            if (episode_cross_modal_hit(res, ep, corp.episode_ids, wfull))
                ++hits_compact;
            if (episode_cross_modal_hit(res, ep, corp.episode_ids, top_k))
                ++hits_top15;
        }

        StepRec rec;
        rec.boost    = boost;
        rec.wall_p99 = percentile_p99(wall_ms);
        rec.kern_p99 = percentile_p99(kern_ms);
        rec.hit_c    = grid_probes > 0 ? double(hits_compact) / double(grid_probes) : 0.0;
        rec.hit_t15  = grid_probes > 0 ? double(hits_top15) / double(grid_probes) : 0.0;
        rec.score    = 2.0 * rec.hit_t15 + rec.hit_c;
        rec.ranking_score = rec.score - wall_ms_penalty * rec.wall_p99;

        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/step_global_boost_%d.json",
                      steps_dir, (int)std::lround(double(boost) * 1000.0));
        rec.file = fname;
        FILE* fo = std::fopen(fname, "w");
        if (!fo) {
            std::perror(fname);
            destroy_query_context(ctx);
            free_device(dg);
            return 1;
        }
        std::fprintf(fo, "{\n");
        std::fprintf(fo, "  \"step\": \"global_episode_boost\",\n");
        std::fprintf(fo, "  \"episode_same_boost\": %.6f,\n", double(boost));
        std::fprintf(fo, "  \"N\": %d,\n", grid_n);
        std::fprintf(fo, "  \"num_probes\": %d,\n", grid_probes);
        std::fprintf(fo, "  \"build_ms\": %.3f,\n", build_ms);
        std::fprintf(fo, "  \"wall_p99_ms\": %.4f,\n", rec.wall_p99);
        std::fprintf(fo, "  \"kernel_p99_ms\": %.4f,\n", rec.kern_p99);
        std::fprintf(fo, "  \"episode_cross_modal_hit_rate_compact\": %.6f,\n", rec.hit_c);
        std::fprintf(fo, "  \"episode_cross_modal_hit_rate_top15\": %.6f,\n", rec.hit_t15);
        std::fprintf(fo, "  \"composite_score\": %.6f,\n", rec.score);
        std::fprintf(fo, "  \"ranking_score\": %.6f,\n", rec.ranking_score);
        std::fprintf(fo, "  \"iterate_wall_ms_penalty\": %.6f,\n", wall_ms_penalty);
        std::fprintf(fo, "  \"use_fp16\": %s,\n", use_fp16 ? "true" : "false");
        std::fprintf(fo, "  \"use_cuda_graph\": %s\n", use_cuda_graph ? "true" : "false");
        std::fprintf(fo, "}\n");
        std::fclose(fo);
        steps.push_back(rec);
        std::fprintf(stderr, "[iterate] boost=%.3f fp16=%d graph=%d -> comp=%.4f rank=%.4f "
                             "wall_p99=%.3f kern_p99=%.3f hit_c=%.3f hit_t15=%.3f -> %s\n",
                     double(boost), int(use_fp16), int(use_cuda_graph),
                     rec.score, rec.ranking_score, rec.wall_p99, rec.kern_p99, rec.hit_c,
                     rec.hit_t15, fname);
    }

    // Episode-scoped reference (same corpus / ctx)
    {
        RetrievalConfig cfg;
        cfg.top_k                = top_k;
        cfg.max_results_returned = max_results_host;
        cfg.bfs_max_hops         = 2;
        cfg.time_decay_lambda    = 8e-6f;
        cfg.bfs_score_decay      = 0.55f;
        cfg.modality_filter      = -1;
        cfg.retrieval_scope      = RetrievalScope::EpisodeScoped;
        cfg.episode_same_boost   = 0.0f;

        // SAME probe sample as the global boost loop so episode-scoped is paired.
        std::mt19937 rng(uint32_t(grid_n) ^ 0xB00Bu);
        std::uniform_int_distribution<size_t> pick(0, image_nodes.size() - 1);
        std::vector<double> wall_ms, kern_ms;
        wall_ms.reserve(size_t(grid_probes));
        kern_ms.reserve(size_t(grid_probes));
        int32_t hits_compact = 0;
        int32_t hits_top15   = 0;

        for (int32_t p = 0; p < grid_probes; ++p) {
            const int32_t node = image_nodes[pick(rng)];
            const float* qemb  = &g.embeddings[size_t(node) * dim];
            float qts = g.timestamps[node] + float(p) * 0.001f;
            const int32_t ep = corp.episode_ids[node];
            cfg.query_episode_id = ep;
            cfg.query_episode_member_begin = ep_csr.ep_csr_offsets[static_cast<size_t>(ep)];
            cfg.query_episode_member_count =
                ep_csr.ep_csr_offsets[static_cast<size_t>(ep) + 1u] -
                ep_csr.ep_csr_offsets[static_cast<size_t>(ep)];

            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            RetrievalStats stats;
            auto res = query_memory_fast(dg, ctx, qemb, qts, cfg, stats);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            kern_ms.push_back(double(stats.gpu_ms_total));
            const int32_t wfull = static_cast<int32_t>(res.size());
            if (episode_cross_modal_hit(res, ep, corp.episode_ids, wfull))
                ++hits_compact;
            if (episode_cross_modal_hit(res, ep, corp.episode_ids, top_k))
                ++hits_top15;
        }

        StepRec rec;
        rec.boost    = -1.0f;
        rec.wall_p99 = percentile_p99(wall_ms);
        rec.kern_p99 = percentile_p99(kern_ms);
        rec.hit_c    = grid_probes > 0 ? double(hits_compact) / double(grid_probes) : 0.0;
        rec.hit_t15  = grid_probes > 0 ? double(hits_top15) / double(grid_probes) : 0.0;
        rec.score    = 2.0 * rec.hit_t15 + rec.hit_c;
        rec.ranking_score = rec.score - wall_ms_penalty * rec.wall_p99;

        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/step_episode_scoped.json", steps_dir);
        rec.file = fname;
        FILE* fo = std::fopen(fname, "w");
        if (!fo) {
            std::perror(fname);
            destroy_query_context(ctx);
            free_device(dg);
            return 1;
        }
        std::fprintf(fo, "{\n");
        std::fprintf(fo, "  \"step\": \"episode_scoped_no_bfs\",\n");
        std::fprintf(fo, "  \"N\": %d,\n", grid_n);
        std::fprintf(fo, "  \"num_probes\": %d,\n", grid_probes);
        std::fprintf(fo, "  \"build_ms\": %.3f,\n", build_ms);
        std::fprintf(fo, "  \"wall_p99_ms\": %.4f,\n", rec.wall_p99);
        std::fprintf(fo, "  \"kernel_p99_ms\": %.4f,\n", rec.kern_p99);
        std::fprintf(fo, "  \"episode_cross_modal_hit_rate_compact\": %.6f,\n", rec.hit_c);
        std::fprintf(fo, "  \"episode_cross_modal_hit_rate_top15\": %.6f,\n", rec.hit_t15);
        std::fprintf(fo, "  \"composite_score\": %.6f,\n", rec.score);
        std::fprintf(fo, "  \"ranking_score\": %.6f,\n", rec.ranking_score);
        std::fprintf(fo, "  \"iterate_wall_ms_penalty\": %.6f,\n", wall_ms_penalty);
        std::fprintf(fo, "  \"use_fp16\": %s,\n", use_fp16 ? "true" : "false");
        std::fprintf(fo, "  \"use_cuda_graph\": %s,\n", use_cuda_graph ? "true" : "false");
        std::fprintf(fo, "  \"note\": \"Different retrieval contract: rank within episode members only.\"\n");
        std::fprintf(fo, "}\n");
        std::fclose(fo);
        steps.push_back(rec);
        std::fprintf(stderr, "[iterate] episode_scoped -> comp=%.4f rank=%.4f wall_p99=%.3f -> %s\n",
                     rec.score, rec.ranking_score, rec.wall_p99, fname);
    }

    destroy_query_context(ctx);
    free_device(dg);

    size_t best_i = 0;
    double best_key = -1.0e300;
    for (size_t i = 0; i < steps.size(); ++i) {
        if (steps[i].boost < 0.f) continue;
        const double key = steps[i].ranking_score;
        if (key > best_key) {
            best_key = key;
            best_i   = i;
        }
    }

    const StepRec* scoped = nullptr;
    for (size_t i = 0; i < steps.size(); ++i) {
        if (steps[i].boost < 0.f) {
            scoped = &steps[i];
            break;
        }
    }

    char sum_path[640];
    std::snprintf(sum_path, sizeof(sum_path), "%s/SUMMARY.json", steps_dir);
    FILE* fs = std::fopen(sum_path, "w");
    if (!fs) {
        std::perror(sum_path);
        return 1;
    }
    std::fprintf(fs, "{\n");
    std::fprintf(fs, "  \"tool\": \"mars-kids-boost-iterate\",\n");
    std::fprintf(fs, "  \"version\": \"1.4\",\n");
    std::fprintf(fs, "  \"iterate_n\": %d,\n", grid_n);
    std::fprintf(fs, "  \"iterate_probes\": %d,\n", grid_probes);
    std::fprintf(fs, "  \"corpus_seed\": %u,\n", corpus_seed);
    std::fprintf(fs, "  \"use_fp16\": %s,\n", use_fp16 ? "true" : "false");
    std::fprintf(fs, "  \"use_cuda_graph\": %s,\n", use_cuda_graph ? "true" : "false");
    if (wall_ms_penalty > 0.0) {
        std::fprintf(fs,
            "  \"criterion\": \"maximize ranking_score = 2*hit_top15 + hit_compact - %.6f * wall_p99_ms "
            "(global boosts only)\",\n",
            wall_ms_penalty);
    } else {
        std::fprintf(fs, "  \"criterion\": \"maximize composite_score = 2*hit_top15 + hit_compact (global "
                         "boosts only)\",\n");
    }
    std::fprintf(fs, "  \"iterate_wall_ms_penalty\": %.6f,\n", wall_ms_penalty);
    std::fprintf(fs, "  \"best_global_episode_same_boost\": %.6f,\n", double(steps[best_i].boost));
    std::fprintf(fs, "  \"best_global_composite_score\": %.6f,\n", steps[best_i].score);
    std::fprintf(fs, "  \"best_global_ranking_score\": %.6f,\n", steps[best_i].ranking_score);
    std::fprintf(fs, "  \"best_global_step_file\": \"%s\",\n", steps[best_i].file.c_str());
    if (scoped) {
        std::fprintf(fs, "  \"episode_scoped_composite_score\": %.6f,\n", scoped->score);
        std::fprintf(fs, "  \"episode_scoped_wall_p99_ms\": %.4f,\n", scoped->wall_p99);
        std::fprintf(fs, "  \"episode_scoped_step_file\": \"%s\",\n", scoped->file.c_str());
    }
    std::fprintf(fs, "  \"steps\": [\n");
    for (size_t i = 0; i < steps.size(); ++i) {
        std::fprintf(fs,
            "    {\"boost\":%.6f,\"file\":\"%s\",\"composite_score\":%.6f,\"ranking_score\":%.6f,"
            "\"wall_p99_ms\":%.4f,\"hit_top15\":%.6f,\"hit_compact\":%.6f}%s\n",
            double(steps[i].boost), steps[i].file.c_str(), steps[i].score, steps[i].ranking_score,
            steps[i].wall_p99, steps[i].hit_t15, steps[i].hit_c,
            (i + 1 < steps.size()) ? "," : "");
    }
    std::fprintf(fs, "  ]\n}\n");
    std::fclose(fs);

    std::snprintf(sum_path, sizeof(sum_path), "%s/SUMMARY.md", steps_dir);
    fs = std::fopen(sum_path, "w");
    if (fs) {
        std::fprintf(fs, "# Kids-ball iteration summary (N=%d)\n\n", grid_n);
        if (wall_ms_penalty > 0.0) {
            std::fprintf(fs, "**Criterion:** maximize `ranking_score = 2*hit_top15 + hit_compact - %.4f "
                             "* wall_p99_ms` over **global** episode boosts.\n\n",
                         wall_ms_penalty);
        } else {
            std::fprintf(fs, "**Criterion:** maximize `2 * hit_top15 + hit_compact` over **global** episode "
                             "boosts.\n\n");
        }
        std::fprintf(fs, "## Best global boost\n\n");
        std::fprintf(fs, "- **episode_same_boost:** %.4f\n", double(steps[best_i].boost));
        std::fprintf(fs, "- **composite_score:** %.4f\n", steps[best_i].score);
        std::fprintf(fs, "- **ranking_score:** %.4f\n", steps[best_i].ranking_score);
        std::fprintf(fs, "- **wall p99 (ms):** %.4f\n", steps[best_i].wall_p99);
        std::fprintf(fs, "- **Artifact:** `%s`\n\n", steps[best_i].file.c_str());
        if (scoped) {
            std::fprintf(fs, "## Episode-scoped reference (different contract)\n\n");
            std::fprintf(fs, "- **composite_score:** %.4f (often ~3.0 when both hits saturate)\n", scoped->score);
            std::fprintf(fs, "- **wall p99 (ms):** %.4f\n", scoped->wall_p99);
            std::fprintf(fs, "- **Artifact:** `%s`\n\n", scoped->file.c_str());
            if (scoped->wall_p99 < steps[best_i].wall_p99 * 0.85)
                std::fprintf(fs, "**Latency:** episode-scoped is materially faster at this N (subset similarity).\n\n");
        }
        std::fprintf(fs, "## Recommendation\n\n");
        if (wall_ms_penalty > 0.0) {
            std::fprintf(fs, "For **open-world** global memory, set `RetrievalConfig::episode_same_boost` to "
                             "**%.3f** if optimizing the **ranking** criterion (with wall-ms penalty) at N=%d.\n",
                         double(steps[best_i].boost), grid_n);
        } else {
            std::fprintf(fs, "For **open-world** global memory, set `RetrievalConfig::episode_same_boost` to "
                             "**%.3f** if optimizing the stated composite at N=%d.\n",
                         double(steps[best_i].boost), grid_n);
        }
        std::fprintf(fs, "For **session-local** UX metrics, prefer `RetrievalScope::EpisodeScoped` and label "
                         "benchmarks accordingly.\n");
        std::fclose(fs);
    }

    std::fprintf(stderr, "Wrote SUMMARY under %s\n", steps_dir);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const int32_t DIM = 768;
    const int32_t TOP_K = 15;
    const int32_t MAX_RESULTS_HOST = 96;
    const int32_t QUERY_CTX_MAX_K  = 64;
    const int32_t NUM_PROBES = 256;
    const uint32_t CORPUS_SEED = 2026u;

    std::vector<int32_t> n_list = {1000, 5000, 10000, 20000, 50000};
    const char* out_path = "results/kids_ball_mars_sweep.json";
    const char* dump_path = nullptr;
    int32_t      dump_n    = -1;
    const char* boost_grid       = nullptr;
    const char* iterate_steps_dir = nullptr;
    int32_t      iterate_n       = 10000;
    int32_t      iterate_probes  = NUM_PROBES;
    double       iterate_wall_ms_penalty = 0.0;
    bool         iterate_use_fp16       = false;
    bool         iterate_use_cuda_graph = false;

    const BenchVariant variants[] = {
        {"global_baseline", RetrievalScope::Global, 0.0f},
        {"global_episode_boost_0.24", RetrievalScope::Global, 0.24f},
        {"episode_scoped_no_bfs", RetrievalScope::EpisodeScoped, 0.0f},
    };

    // Fast path: write corpus binary only (no GPU) for FAISS baseline scripts.
    if (argc >= 4 && std::strcmp(argv[1], "--dump-only") == 0) {
        const char* bin_path = argv[2];
        const int32_t N      = std::max(10, std::atoi(argv[3]));
        const int32_t dim    = (argc >= 5) ? std::max(2, std::atoi(argv[4])) : DIM;
        EmbodiedKidsBallCorpus corp = EmbodiedKidsBallCorpus::make(N, dim, CORPUS_SEED);
        std::fprintf(stderr, "Kids-ball corpus dump-only: N=%d D=%d -> %s\n",
                     N, dim, bin_path);
        write_corpus_bin(bin_path, corp.graph, corp.episode_ids);
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-corpus") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-n") == 0 && i + 1 < argc) {
            dump_n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--boost-grid") == 0 && i + 1 < argc) {
            boost_grid = argv[++i];
        } else if (std::strcmp(argv[i], "--iterate-steps-dir") == 0 && i + 1 < argc) {
            iterate_steps_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--iterate-n") == 0 && i + 1 < argc) {
            iterate_n = std::max(10, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--iterate-probes") == 0 && i + 1 < argc) {
            iterate_probes = std::max(16, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--iterate-wall-ms-penalty") == 0 && i + 1 < argc) {
            iterate_wall_ms_penalty = std::strtod(argv[++i], nullptr);
            if (iterate_wall_ms_penalty < 0.0) iterate_wall_ms_penalty = 0.0;
        } else if (std::strcmp(argv[i], "--use-fp16") == 0) {
            iterate_use_fp16 = true;
        } else if (std::strcmp(argv[i], "--use-cuda-graph") == 0) {
            iterate_use_cuda_graph = true;
        } else if (argv[i][0] != '-') {
            out_path = argv[i];
        }
    }
    if (dump_path && dump_n < 0)
        dump_n = n_list.back();

    if (boost_grid) {
        std::vector<float> bl = parse_boost_list(boost_grid);
        if (bl.empty()) {
            std::fprintf(stderr,
                         "bench_kids_sweep: --boost-grid needs comma-separated floats, e.g. "
                         "0,0.15,0.25,0.35\n");
            return 1;
        }
        const char* sd =
            iterate_steps_dir ? iterate_steps_dir : "results/iteration_steps/latest";
        return run_boost_iterate_mode(bl, iterate_n, iterate_probes, sd, DIM, TOP_K,
                                      MAX_RESULTS_HOST, QUERY_CTX_MAX_K, CORPUS_SEED,
                                      iterate_wall_ms_penalty,
                                      iterate_use_fp16, iterate_use_cuda_graph);
    }

    std::vector<std::vector<SweepRow>> rows_by_variant;
    rows_by_variant.resize(sizeof(variants) / sizeof(variants[0]));

    for (int32_t N : n_list) {
        if (N < 10) continue;

        auto t_build0 = std::chrono::high_resolution_clock::now();
        EmbodiedKidsBallCorpus corp = EmbodiedKidsBallCorpus::make(N, DIM, CORPUS_SEED);
        MemoryGraph& g = corp.graph;
        g.build_nsn_edges(6, 0.15);
        auto t_build1 = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();

        if (dump_path && N == dump_n) {
            std::fprintf(stderr, "Writing corpus dump N=%d -> %s\n", N, dump_path);
            write_corpus_bin(dump_path, g, corp.episode_ids);
        }

        std::vector<int32_t> image_nodes;
        for (int32_t i = 0; i < g.num_nodes; ++i)
            if (g.modalities[i] == MOD_IMAGE)
                image_nodes.push_back(i);
        if (image_nodes.empty()) {
            std::fprintf(stderr, "No IMAGE nodes at N=%d\n", N);
            continue;
        }

        HostEpisodeCSR ep_csr = build_episode_csr(corp.episode_ids, g.num_nodes);

        DeviceMemoryGraph dg = upload_to_device(g);
        upload_episode_ids(dg, corp.episode_ids.data(), g.num_nodes);
        upload_episode_csr(dg, ep_csr);

        // One context for all variants — avoids redundant cuBLAS/CUB allocation per variant.
        QueryContext ctx = create_query_context(g.num_nodes, DIM, QUERY_CTX_MAX_K);

        for (size_t vi = 0; vi < sizeof(variants) / sizeof(variants[0]); ++vi) {
            const BenchVariant& bv = variants[vi];

            RetrievalConfig cfg;
            cfg.top_k                = TOP_K;
            cfg.max_results_returned = MAX_RESULTS_HOST;
            cfg.bfs_max_hops         = 2;
            cfg.time_decay_lambda    = 8e-6f;
            cfg.bfs_score_decay      = 0.55f;
            cfg.modality_filter      = -1;
            cfg.retrieval_scope      = bv.scope;
            cfg.episode_same_boost   = bv.episode_same_boost;

            // SAME probe sample across variants for paired comparison at a given N.
            std::mt19937 rng(uint32_t(N) ^ 0xA5A5A5A5u);
            std::uniform_int_distribution<size_t> pick(0, image_nodes.size() - 1);

            std::vector<double> wall_ms;
            std::vector<double> kern_ms;
            wall_ms.reserve(NUM_PROBES);
            kern_ms.reserve(NUM_PROBES);
            int32_t hits_compact = 0;
            int32_t hits_top15   = 0;

            for (int32_t p = 0; p < NUM_PROBES; ++p) {
                const int32_t node = image_nodes[pick(rng)];
                const float* qemb  = &g.embeddings[size_t(node) * DIM];
                float qts = g.timestamps[node] + float(p) * 0.001f;

                const int32_t ep = corp.episode_ids[node];
                cfg.query_episode_id = ep;
                if (bv.scope == RetrievalScope::EpisodeScoped) {
                    cfg.query_episode_member_begin = ep_csr.ep_csr_offsets[static_cast<size_t>(ep)];
                    cfg.query_episode_member_count =
                        ep_csr.ep_csr_offsets[static_cast<size_t>(ep) + 1u] -
                        ep_csr.ep_csr_offsets[static_cast<size_t>(ep)];
                } else {
                    cfg.query_episode_member_begin = 0;
                    cfg.query_episode_member_count = 0;
                }

                cudaDeviceSynchronize();
                auto t0 = std::chrono::high_resolution_clock::now();
                RetrievalStats stats;
                auto res = query_memory_fast(dg, ctx, qemb, qts, cfg, stats);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();

                wall_ms.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
                kern_ms.push_back(double(stats.gpu_ms_total));

                const int32_t wfull = static_cast<int32_t>(res.size());
                if (episode_cross_modal_hit(res, ep, corp.episode_ids, wfull))
                    ++hits_compact;
                if (episode_cross_modal_hit(res, ep, corp.episode_ids, TOP_K))
                    ++hits_top15;
            }

            SweepRow row;
            row.N = N;
            row.build_ms = build_ms;
            row.wall_p99_ms = percentile_p99(wall_ms);
            row.kernel_p99_ms = percentile_p99(kern_ms);
            row.episode_cross_modal_hit_rate_compact =
                NUM_PROBES > 0 ? double(hits_compact) / double(NUM_PROBES) : 0.0;
            row.episode_cross_modal_hit_rate_top15 =
                NUM_PROBES > 0 ? double(hits_top15) / double(NUM_PROBES) : 0.0;
            rows_by_variant[vi].push_back(row);

            std::fprintf(stderr,
                         "[%s] N=%7d  build=%.1f ms  wall_p99=%.3f  kern_p99=%.3f  "
                         "ep_hit_compact=%.3f ep_hit_top15=%.3f\n",
                         bv.name, N, build_ms, row.wall_p99_ms, row.kernel_p99_ms,
                         row.episode_cross_modal_hit_rate_compact,
                         row.episode_cross_modal_hit_rate_top15);
        }

        destroy_query_context(ctx);
        free_device(dg);
    }

    FILE* fo = std::fopen(out_path, "w");
    if (!fo) {
        std::perror(out_path);
        return 1;
    }
    std::fprintf(fo, "{\n");
    std::fprintf(fo, "  \"tool\": \"mars-kids-ball-sweep\",\n");
    std::fprintf(fo, "  \"version\": \"1.3\",\n");
    std::fprintf(fo, "  \"note\": \"Multiple retrieval variants per N for vast.ai matrix. "
                     "episode_scoped = similarity+decay over episode members only, BFS hops 0. "
                     "global_episode_boost = same-episode score multiplier before top-K. "
                     "v1.3: probe sample is the same across variants at a given N (paired comparison). "
                     "See docs/MEMORY_LAYOUT_EMBODIED.md, TEMPORAL_HIERARCHY.md, MULTIMODAL_ROUTING.md.\",\n");
    std::fprintf(fo, "  \"embedding_dim\": %d,\n", DIM);
    std::fprintf(fo, "  \"top_k\": %d,\n", TOP_K);
    std::fprintf(fo, "  \"max_results_returned\": %d,\n", MAX_RESULTS_HOST);
    std::fprintf(fo, "  \"num_probes_per_n\": %d,\n", NUM_PROBES);
    std::fprintf(fo, "  \"corpus_seed\": %u,\n", CORPUS_SEED);
    std::fprintf(fo, "  \"variants\": [\n");
    for (size_t vi = 0; vi < sizeof(variants) / sizeof(variants[0]); ++vi) {
        const BenchVariant& bv = variants[vi];
        const char* scope_str =
            (bv.scope == RetrievalScope::EpisodeScoped) ? "episode_scoped" : "global";
        std::fprintf(fo, "    {\n");
        std::fprintf(fo, "      \"name\": \"%s\",\n", bv.name);
        std::fprintf(fo, "      \"retrieval_scope\": \"%s\",\n", scope_str);
        std::fprintf(fo, "      \"episode_same_boost\": %.4f,\n", double(bv.episode_same_boost));
        std::fprintf(fo, "      \"configurations\": [\n");
        for (size_t i = 0; i < rows_by_variant[vi].size(); ++i) {
            const SweepRow& r = rows_by_variant[vi][i];
            std::fprintf(fo, "        {\n");
            std::fprintf(fo, "          \"N\": %d,\n", r.N);
            std::fprintf(fo, "          \"build_ms\": %.3f,\n", r.build_ms);
            std::fprintf(fo, "          \"wall_p99_ms\": %.4f,\n", r.wall_p99_ms);
            std::fprintf(fo, "          \"kernel_p99_ms\": %.4f,\n", r.kernel_p99_ms);
            std::fprintf(fo, "          \"episode_cross_modal_hit_rate_compact\": %.4f,\n",
                         r.episode_cross_modal_hit_rate_compact);
            std::fprintf(fo, "          \"episode_cross_modal_hit_rate_top15\": %.4f\n",
                         r.episode_cross_modal_hit_rate_top15);
            std::fprintf(fo, "        }%s\n", (i + 1 < rows_by_variant[vi].size()) ? "," : "");
        }
        std::fprintf(fo, "      ]\n");
        std::fprintf(fo, "    }%s\n", (vi + 1 < sizeof(variants) / sizeof(variants[0])) ? "," : "");
    }
    std::fprintf(fo, "  ]\n}\n");
    std::fclose(fo);
    std::fprintf(stderr, "Wrote %s\n", out_path);
    return 0;
}
