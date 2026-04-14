// ============================================================================
//  bench_kids_sweep.cu — MARS scaling benchmark on embodied kids-ball corpus.
//
//  Sweeps corpus size N, measures wall + kernel p99, build_ms, and
//  episode_cross_modal_hit_rate (IMAGE query → same-episode TEXT+AUDIO in top-K).
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
    double  episode_cross_modal_hit_rate = 0;
};

} // namespace

int main(int argc, char** argv) {
    const int32_t DIM = 768;
    const int32_t TOP_K = 15;
    const int32_t NUM_PROBES = 256;
    const uint32_t CORPUS_SEED = 2026u;

    std::vector<int32_t> n_list = {1000, 5000, 10000, 20000, 50000};
    const char* out_path = "results/kids_ball_mars_sweep.json";
    const char* dump_path = nullptr;
    int32_t      dump_n    = -1;

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

    std::vector<SweepRow> rows;
    rows.reserve(n_list.size());

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

        DeviceMemoryGraph dg = upload_to_device(g);
        QueryContext ctx = create_query_context(g.num_nodes, DIM, TOP_K);

        RetrievalConfig cfg;
        cfg.top_k             = TOP_K;
        cfg.bfs_max_hops      = 2;
        cfg.time_decay_lambda = 8e-6f;
        cfg.bfs_score_decay   = 0.55f;
        cfg.modality_filter   = -1;

        std::mt19937 rng(uint32_t(N) ^ 0xA5A5A5A5u);
        std::uniform_int_distribution<size_t> pick(0, image_nodes.size() - 1);

        std::vector<double> wall_ms;
        std::vector<double> kern_ms;
        wall_ms.reserve(NUM_PROBES);
        kern_ms.reserve(NUM_PROBES);
        int32_t hits = 0;

        for (int32_t p = 0; p < NUM_PROBES; ++p) {
            const int32_t node = image_nodes[pick(rng)];
            const float* qemb  = &g.embeddings[size_t(node) * DIM];
            float qts = g.timestamps[node] + float(p) * 0.001f;

            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            RetrievalStats stats;
            auto res = query_memory_fast(dg, ctx, qemb, qts, cfg, stats);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();

            wall_ms.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
            kern_ms.push_back(double(stats.gpu_ms_total));

            if (episode_cross_modal_hit(res, corp.episode_ids[node],
                                        corp.episode_ids, TOP_K))
                ++hits;
        }

        destroy_query_context(ctx);
        free_device(dg);

        SweepRow row;
        row.N = N;
        row.build_ms = build_ms;
        row.wall_p99_ms = percentile_p99(wall_ms);
        row.kernel_p99_ms = percentile_p99(kern_ms);
        row.episode_cross_modal_hit_rate =
            NUM_PROBES > 0 ? double(hits) / double(NUM_PROBES) : 0.0;
        rows.push_back(row);

        std::fprintf(stderr, "N=%7d  build=%.1f ms  wall_p99=%.3f  kern_p99=%.3f  ep_hit=%.3f\n",
                     N, build_ms, row.wall_p99_ms, row.kernel_p99_ms,
                     row.episode_cross_modal_hit_rate);
    }

    FILE* fo = std::fopen(out_path, "w");
    if (!fo) {
        std::perror(out_path);
        return 1;
    }
    std::fprintf(fo, "{\n");
    std::fprintf(fo, "  \"tool\": \"mars-kids-ball-sweep\",\n");
    std::fprintf(fo, "  \"version\": \"1.0\",\n");
    std::fprintf(fo, "  \"note\": \"MARS full pipeline (temporal + cuBLAS/CUB + BFS). "
                     "FAISS baselines use scripts/bench_kids_ball_faiss.py on --dump-corpus output.\",\n");
    std::fprintf(fo, "  \"embedding_dim\": %d,\n", DIM);
    std::fprintf(fo, "  \"top_k\": %d,\n", TOP_K);
    std::fprintf(fo, "  \"num_probes_per_n\": %d,\n", NUM_PROBES);
    std::fprintf(fo, "  \"corpus_seed\": %u,\n", CORPUS_SEED);
    std::fprintf(fo, "  \"configurations\": [\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const SweepRow& r = rows[i];
        std::fprintf(fo, "    {\n");
        std::fprintf(fo, "      \"N\": %d,\n", r.N);
        std::fprintf(fo, "      \"build_ms\": %.3f,\n", r.build_ms);
        std::fprintf(fo, "      \"wall_p99_ms\": %.4f,\n", r.wall_p99_ms);
        std::fprintf(fo, "      \"kernel_p99_ms\": %.4f,\n", r.kernel_p99_ms);
        std::fprintf(fo, "      \"episode_cross_modal_hit_rate\": %.4f\n",
                     r.episode_cross_modal_hit_rate);
        std::fprintf(fo, "    }%s\n", (i + 1 < rows.size()) ? "," : "");
    }
    std::fprintf(fo, "  ]\n}\n");
    std::fclose(fo);
    std::fprintf(stderr, "Wrote %s\n", out_path);
    return 0;
}
