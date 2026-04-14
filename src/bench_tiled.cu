// ============================================================================
//  bench_tiled.cu — Benchmark for out-of-core tiled retrieval.
//  Tests corpus sizes from 1M to 100M using host-resident embeddings
//  streamed through GPU VRAM in tiles.
//
//  Usage: ./bench_tiled [N] [D] [tile_size] [num_queries]
//  Default: ./bench_tiled 100000000 768 5000000 5
// ============================================================================
#include "tiled_query.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    int32_t N          = argc > 1 ? std::atoi(argv[1]) : 100000000;  // 100M default
    int32_t D          = argc > 2 ? std::atoi(argv[2]) : 768;
    int32_t tile_size  = argc > 3 ? std::atoi(argv[3]) : 5000000;
    int32_t num_queries = argc > 4 ? std::atoi(argv[4]) : 5;

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "══════════════════════════════════════════════════════\n");
    std::fprintf(stderr, "  MARS Tiled Out-of-Core Benchmark\n");
    std::fprintf(stderr, "    corpus    : %d (%.1fM)\n", N, N / 1e6);
    std::fprintf(stderr, "    dimension : %d\n", D);
    std::fprintf(stderr, "    tile_size : %d (%.1fM)\n", tile_size, tile_size / 1e6);
    std::fprintf(stderr, "    queries   : %d\n", num_queries);
    std::fprintf(stderr, "    host RAM  : %.1f GB (embeddings)\n",
                 double(N) * D * sizeof(float) / 1e9);
    std::fprintf(stderr, "══════════════════════════════════════════════════════\n\n");

    // Allocate host-resident corpus
    std::fprintf(stderr, "  Allocating %.1f GB host memory for %dM embeddings...\n",
                 double(N) * D * sizeof(float) / 1e9, int(N / 1000000));

    auto t_alloc = std::chrono::high_resolution_clock::now();

    // Use pinned memory for faster H2D transfers
    float* h_embeddings;
    cudaMallocHost(&h_embeddings, size_t(N) * D * sizeof(float));
    if (!h_embeddings) {
        std::fprintf(stderr, "  ERROR: Failed to allocate %.1f GB pinned memory.\n",
                     double(N) * D * sizeof(float) / 1e9);
        std::fprintf(stderr, "  Try reducing N or using regular malloc.\n");
        // Fallback to regular malloc
        h_embeddings = static_cast<float*>(std::malloc(size_t(N) * D * sizeof(float)));
        if (!h_embeddings) {
            std::fprintf(stderr, "  FATAL: Cannot allocate host memory.\n");
            return 1;
        }
        std::fprintf(stderr, "  Using pageable memory (slower transfers).\n");
    }

    auto* h_modalities = static_cast<int32_t*>(std::malloc(size_t(N) * sizeof(int32_t)));
    auto* h_timestamps = static_cast<float*>(std::malloc(size_t(N) * sizeof(float)));

    if (!h_modalities || !h_timestamps) {
        std::fprintf(stderr, "  FATAL: Cannot allocate metadata.\n");
        return 1;
    }

    auto t_alloc_end = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "  Allocated in %.1f s\n",
                 std::chrono::duration<double>(t_alloc_end - t_alloc).count());

    // Generate synthetic data (random L2-normalized embeddings)
    std::fprintf(stderr, "  Generating synthetic embeddings...\n");
    auto t_gen = std::chrono::high_resolution_clock::now();

    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Generate in chunks to avoid RNG bottleneck
    #pragma omp parallel for schedule(static) if(N > 1000000)
    for (int64_t i = 0; i < N; ++i) {
        std::mt19937 local_rng(42 + i);
        std::normal_distribution<float> local_gauss(0.0f, 1.0f);
        float* vec = h_embeddings + i * D;
        float norm = 0.0f;
        for (int32_t d = 0; d < D; ++d) {
            vec[d] = local_gauss(local_rng);
            norm += vec[d] * vec[d];
        }
        norm = std::sqrt(norm);
        for (int32_t d = 0; d < D; ++d) vec[d] /= norm;

        h_modalities[i] = i % 3;  // round-robin modalities
        h_timestamps[i] = float(i) / float(N);  // 0.0 to 1.0
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "  Generated in %.1f s\n",
                 std::chrono::duration<double>(t_gen_end - t_gen).count());

    // Create tiled context
    auto ctx = create_tiled_context(tile_size, D, 10);

    TiledQueryConfig cfg;
    cfg.tile_size = tile_size;
    cfg.top_k = 10;
    cfg.time_decay_lambda = 1e-8f;

    // Generate query
    std::vector<float> query(D);
    {
        float norm = 0.0f;
        for (int32_t d = 0; d < D; ++d) {
            query[d] = gauss(rng);
            norm += query[d] * query[d];
        }
        norm = std::sqrt(norm);
        for (int32_t d = 0; d < D; ++d) query[d] /= norm;
    }

    // Warmup
    std::fprintf(stderr, "  Warming up...\n");
    {
        TiledQueryStats warmup_stats;
        auto results = tiled_query(ctx, h_embeddings, h_modalities, h_timestamps,
                                    std::min(N, tile_size), D,
                                    query.data(), 1.0f, cfg, warmup_stats);
        std::fprintf(stderr, "  Warmup: %.1f ms (%d results)\n",
                     warmup_stats.total_ms, (int)results.size());
    }

    // Run queries
    std::fprintf(stderr, "\n  Running %d queries over %dM corpus...\n", num_queries, int(N/1000000));
    std::vector<float> query_times;
    query_times.reserve(num_queries);

    for (int32_t q = 0; q < num_queries; ++q) {
        TiledQueryStats stats;
        auto results = tiled_query(ctx, h_embeddings, h_modalities, h_timestamps,
                                    N, D, query.data(), 1.0f, cfg, stats);

        query_times.push_back(stats.total_ms);
        std::fprintf(stderr, "  Query %d: %.1f ms (transfer: %.1f ms, compute: %.1f ms, "
                     "merge: %.3f ms, tiles: %d, results: %d)\n",
                     q + 1, stats.total_ms, stats.transfer_ms, stats.compute_ms,
                     stats.merge_ms, stats.num_tiles, (int)results.size());

        if (q == 0) {
            std::fprintf(stderr, "  Top-5 results:\n");
            for (int32_t k = 0; k < std::min(5, (int)results.size()); ++k) {
                std::fprintf(stderr, "    [%d] node=%d score=%.6f mod=%d\n",
                             k, results[k].node_id, results[k].score, results[k].modality);
            }
        }
    }

    // Statistics
    std::sort(query_times.begin(), query_times.end());
    float p50 = query_times[query_times.size() / 2];
    float p99 = query_times[std::min((int)query_times.size() - 1,
                                      (int)(query_times.size() * 0.99))];
    float mean = 0;
    for (float t : query_times) mean += t;
    mean /= query_times.size();

    // JSON output
    std::printf("{\n");
    std::printf("  \"tool\": \"mars-tiled-bench\",\n");
    std::printf("  \"corpus_size\": %d,\n", N);
    std::printf("  \"corpus_millions\": %.1f,\n", N / 1e6);
    std::printf("  \"embedding_dim\": %d,\n", D);
    std::printf("  \"tile_size\": %d,\n", tile_size);
    std::printf("  \"num_tiles\": %d,\n", (N + tile_size - 1) / tile_size);
    std::printf("  \"num_queries\": %d,\n", num_queries);
    std::printf("  \"latency_ms\": {\n");
    std::printf("    \"p50\": %.1f,\n", p50);
    std::printf("    \"p99\": %.1f,\n", p99);
    std::printf("    \"mean\": %.1f,\n", mean);
    std::printf("    \"min\": %.1f,\n", query_times.front());
    std::printf("    \"max\": %.1f\n", query_times.back());
    std::printf("  },\n");
    std::printf("  \"host_ram_gb\": %.1f,\n", double(N) * D * sizeof(float) / 1e9);
    std::printf("  \"gpu_vram_per_tile_gb\": %.1f\n", double(tile_size) * D * sizeof(float) / 1e9);
    std::printf("}\n");

    // Cleanup
    destroy_tiled_context(ctx);
    cudaFreeHost(h_embeddings);  // or free() if fallback
    std::free(h_modalities);
    std::free(h_timestamps);

    std::fprintf(stderr, "\n══════════════════════════════════════════════════════\n");
    std::fprintf(stderr, "  %dM corpus: p50=%.1f ms, mean=%.1f ms, p99=%.1f ms\n",
                 int(N / 1000000), p50, mean, p99);
    std::fprintf(stderr, "══════════════════════════════════════════════════════\n\n");

    return 0;
}
