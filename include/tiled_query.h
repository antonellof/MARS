// ============================================================================
//  tiled_query.h
//  Out-of-core tiled retrieval for corpus sizes exceeding GPU VRAM.
//
//  Strategy: embeddings live in host pinned memory. For each query, we
//  stream tiles of embeddings to GPU, compute cosine similarities via
//  cuBLAS SGEMV, apply temporal decay, select per-tile top-K, then
//  merge across tiles. This trades bandwidth for capacity — a 100M
//  corpus at D=768 FP16 needs ~150 GB, streamed in tiles through
//  40 GB of HBM.
//
//  Performance model:
//    T_per_tile = H2D_transfer + cuBLAS_SGEMV + temporal_rerank + top_K
//    T_total    = num_tiles × T_per_tile + merge
//
//  At PCIe Gen4 ~25 GB/s, a 5M-vector FP16 tile (7.5 GB) transfers
//  in ~300 ms. The compute per tile is ~5 ms. So the approach is
//  bandwidth-bound: ~300 ms per tile, ~6 seconds for 100M.
//  With PCIe Gen5 (64 GB/s) this drops to ~2.3 seconds.
//
//  This is NOT sub-millisecond — it's a capacity extension for offline
//  or batch queries against very large corpora. Real-time workloads
//  should keep the working set in GPU VRAM (up to 25M FP16 on A100).
// ============================================================================
#ifndef TILED_QUERY_H
#define TILED_QUERY_H

#include "memory_graph.h"
#include "memory_cuda.cuh"
#include <cstdint>
#include <vector>

struct TiledQueryConfig {
    int32_t tile_size       = 5000000;  // vectors per tile (5M default)
    int32_t top_k           = 10;
    float   time_decay_lambda = 1e-8f;
    int32_t modality_filter = -1;
    bool    use_fp16        = true;     // FP16 for 2x bandwidth savings
};

struct TiledQueryResult {
    int32_t node_id;
    float   score;
    int32_t modality;
};

struct TiledQueryStats {
    int32_t total_nodes     = 0;
    int32_t num_tiles       = 0;
    float   total_ms        = 0.0f;
    float   transfer_ms     = 0.0f;    // cumulative H2D time
    float   compute_ms      = 0.0f;    // cumulative GPU kernel time
    float   merge_ms        = 0.0f;    // final merge time
};

// Pre-allocated context for tiled queries. Create once, reuse.
struct TiledQueryContext {
    int32_t max_tile_size   = 0;
    int32_t embedding_dim   = 0;
    int32_t max_k           = 0;

    // GPU-side tile buffer (re-used per tile)
    float*   d_tile_embeddings  = nullptr;  // tile_size × D
    float*   d_tile_sim         = nullptr;  // tile_size
    float*   d_query            = nullptr;  // D
    int32_t* d_tile_modalities  = nullptr;  // tile_size
    float*   d_tile_timestamps  = nullptr;  // tile_size

    // Pinned host staging
    float*   h_query_pinned     = nullptr;  // D

    // cuBLAS
    void*    cublas_handle      = nullptr;

    cudaStream_t stream         = nullptr;
    bool valid                  = false;
};

// Create/destroy tiled query context
TiledQueryContext create_tiled_context(int32_t max_tile_size,
                                       int32_t embedding_dim,
                                       int32_t max_k = 64);
void destroy_tiled_context(TiledQueryContext& ctx);

// Execute a tiled query over a host-resident corpus.
// embeddings: host memory, N × D, row-major (FP32)
// modalities: host memory, N
// timestamps: host memory, N
std::vector<TiledQueryResult>
tiled_query(TiledQueryContext& ctx,
            const float* h_embeddings,     // N × D, host memory
            const int32_t* h_modalities,   // N, host memory
            const float* h_timestamps,     // N, host memory
            int32_t N,
            int32_t D,
            const float* h_query,          // D, host memory
            float query_timestamp,
            const TiledQueryConfig& cfg,
            TiledQueryStats& stats);

#endif // TILED_QUERY_H
