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

    const BenchVariant variants[] = {
        {"global_baseline", RetrievalScope::Global, 0.0f},
        {"global_episode_boost_0.35", RetrievalScope::Global, 0.35f},
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
        } else if (argv[i][0] != '-') {
            out_path = argv[i];
        }
    }
    if (dump_path && dump_n < 0)
        dump_n = n_list.back();

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

        for (size_t vi = 0; vi < sizeof(variants) / sizeof(variants[0]); ++vi) {
            const BenchVariant& bv = variants[vi];
            QueryContext ctx = create_query_context(g.num_nodes, DIM, QUERY_CTX_MAX_K);

            RetrievalConfig cfg;
            cfg.top_k                = TOP_K;
            cfg.max_results_returned = MAX_RESULTS_HOST;
            cfg.bfs_max_hops         = 2;
            cfg.time_decay_lambda    = 8e-6f;
            cfg.bfs_score_decay      = 0.55f;
            cfg.modality_filter      = -1;
            cfg.retrieval_scope      = bv.scope;
            cfg.episode_same_boost   = bv.episode_same_boost;

            std::mt19937 rng(uint32_t(N) ^ uint32_t(vi) ^ 0xA5A5A5A5u);
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

            destroy_query_context(ctx);

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

        free_device(dg);
    }

    FILE* fo = std::fopen(out_path, "w");
    if (!fo) {
        std::perror(out_path);
        return 1;
    }
    std::fprintf(fo, "{\n");
    std::fprintf(fo, "  \"tool\": \"mars-kids-ball-sweep\",\n");
    std::fprintf(fo, "  \"version\": \"1.2\",\n");
    std::fprintf(fo, "  \"note\": \"Multiple retrieval variants per N for vast.ai matrix. "
                     "episode_scoped = similarity+decay over episode members only, BFS hops 0. "
                     "global_episode_boost = same-episode score multiplier before top-K. "
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
