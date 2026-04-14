// ============================================================================
//  tiled_query.cu
//  Out-of-core tiled retrieval for corpus sizes exceeding GPU VRAM.
// ============================================================================
#include "tiled_query.h"
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d — %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st__ = (call); \
    if (st__ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %s:%d — status %d\n", \
                     __FILE__, __LINE__, (int)st__); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// ── Temporal decay kernel (applied per-tile after SGEMV) ────────────
__global__ void tiled_temporal_rerank_kernel(
    float*         __restrict__ scores,
    const float*   __restrict__ timestamps,
    const int32_t* __restrict__ modalities,
    int32_t N_tile,
    float query_ts,
    float lambda,
    int32_t modality_filter)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_tile) return;

    if (modality_filter >= 0 && modalities[tid] != modality_filter) {
        scores[tid] = -FLT_MAX;
        return;
    }

    float age = fmaxf(0.0f, query_ts - timestamps[tid]);
    scores[tid] *= __expf(-lambda * age);
}

// ── Per-tile top-K extraction (simple single-block kernel) ──────────
__global__ void tiled_topk_kernel(
    const float* __restrict__ scores,
    int32_t* __restrict__ top_idx,
    float*   __restrict__ top_val,
    int32_t N_tile,
    int32_t K,
    int32_t global_offset)  // add to indices to get global node IDs
{
    // Single block, each thread maintains a local min-heap of size K
    // Thread 0 does final merge (simple for K<=64)
    extern __shared__ float shared[];

    float*   s_val = shared;
    int32_t* s_idx = reinterpret_cast<int32_t*>(shared + K * blockDim.x);

    int32_t tid = threadIdx.x;
    int32_t stride = blockDim.x;

    // Initialize local top-K
    for (int32_t k = 0; k < K; ++k) {
        s_val[tid * K + k] = -FLT_MAX;
        s_idx[tid * K + k] = -1;
    }

    // Scan scores, maintain per-thread top-K
    for (int32_t i = tid; i < N_tile; i += stride) {
        float v = scores[i];
        // Find min in local top-K
        int32_t min_k = 0;
        float min_v = s_val[tid * K];
        for (int32_t k = 1; k < K; ++k) {
            if (s_val[tid * K + k] < min_v) {
                min_v = s_val[tid * K + k];
                min_k = k;
            }
        }
        if (v > min_v) {
            s_val[tid * K + min_k] = v;
            s_idx[tid * K + min_k] = i + global_offset;
        }
    }
    __syncthreads();

    // Thread 0 merges all per-thread results
    if (tid == 0) {
        for (int32_t t = 1; t < blockDim.x && t < stride; ++t) {
            for (int32_t k = 0; k < K; ++k) {
                float v = s_val[t * K + k];
                int32_t idx = s_idx[t * K + k];
                // Insert into thread-0's top-K if better
                int32_t min_k = 0;
                float min_v = s_val[0 * K];
                for (int32_t kk = 1; kk < K; ++kk) {
                    if (s_val[0 * K + kk] < min_v) {
                        min_v = s_val[0 * K + kk];
                        min_k = kk;
                    }
                }
                if (v > min_v) {
                    s_val[0 * K + min_k] = v;
                    s_idx[0 * K + min_k] = idx;
                }
            }
        }
        // Write final top-K
        for (int32_t k = 0; k < K; ++k) {
            top_val[k] = s_val[k];
            top_idx[k] = s_idx[k];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Context management
// ═══════════════════════════════════════════════════════════════════

TiledQueryContext create_tiled_context(int32_t max_tile_size,
                                       int32_t embedding_dim,
                                       int32_t max_k) {
    TiledQueryContext ctx;
    ctx.max_tile_size = max_tile_size;
    ctx.embedding_dim = embedding_dim;
    ctx.max_k = max_k;

    size_t tile_bytes = size_t(max_tile_size) * embedding_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&ctx.d_tile_embeddings, tile_bytes));
    CUDA_CHECK(cudaMalloc(&ctx.d_tile_sim, max_tile_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_query, embedding_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_tile_modalities, max_tile_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_tile_timestamps, max_tile_size * sizeof(float)));

    CUDA_CHECK(cudaMallocHost(&ctx.h_query_pinned, embedding_dim * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    ctx.cublas_handle = handle;

    CUDA_CHECK(cudaStreamCreate(&ctx.stream));
    ctx.valid = true;
    return ctx;
}

void destroy_tiled_context(TiledQueryContext& ctx) {
    if (!ctx.valid) return;
    cudaFree(ctx.d_tile_embeddings);
    cudaFree(ctx.d_tile_sim);
    cudaFree(ctx.d_query);
    cudaFree(ctx.d_tile_modalities);
    cudaFree(ctx.d_tile_timestamps);
    cudaFreeHost(ctx.h_query_pinned);
    if (ctx.cublas_handle)
        cublasDestroy(static_cast<cublasHandle_t>(ctx.cublas_handle));
    cudaStreamDestroy(ctx.stream);
    ctx.valid = false;
}

// ═══════════════════════════════════════════════════════════════════
//  Tiled query execution
// ═══════════════════════════════════════════════════════════════════

std::vector<TiledQueryResult>
tiled_query(TiledQueryContext& ctx,
            const float* h_embeddings,
            const int32_t* h_modalities,
            const float* h_timestamps,
            int32_t N,
            int32_t D,
            const float* h_query,
            float query_timestamp,
            const TiledQueryConfig& cfg,
            TiledQueryStats& stats)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    const int32_t K = cfg.top_k;
    const int32_t tile_size = std::min(cfg.tile_size, N);
    const int32_t num_tiles = (N + tile_size - 1) / tile_size;

    stats.total_nodes = N;
    stats.num_tiles = num_tiles;

    // Upload query (once)
    std::memcpy(ctx.h_query_pinned, h_query, D * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_query, ctx.h_query_pinned,
                                D * sizeof(float),
                                cudaMemcpyHostToDevice, ctx.stream));

    // Collect per-tile top-K candidates
    struct Candidate { int32_t id; float score; int32_t modality; };
    std::vector<Candidate> all_candidates;
    all_candidates.reserve(num_tiles * K);

    // GPU buffers for per-tile top-K
    int32_t* d_topk_idx;
    float*   d_topk_val;
    CUDA_CHECK(cudaMalloc(&d_topk_idx, K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_topk_val, K * sizeof(float)));

    cublasHandle_t handle = static_cast<cublasHandle_t>(ctx.cublas_handle);
    cublasSetStream(handle, ctx.stream);

    float transfer_ms = 0, compute_ms = 0;

    for (int32_t tile = 0; tile < num_tiles; ++tile) {
        int32_t offset = tile * tile_size;
        int32_t this_tile = std::min(tile_size, N - offset);

        auto t_h2d_start = std::chrono::high_resolution_clock::now();

        // H2D: embeddings tile
        size_t emb_bytes = size_t(this_tile) * D * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_tile_embeddings,
                                    h_embeddings + size_t(offset) * D,
                                    emb_bytes,
                                    cudaMemcpyHostToDevice, ctx.stream));

        // H2D: modalities + timestamps
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_tile_modalities,
                                    h_modalities + offset,
                                    this_tile * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_tile_timestamps,
                                    h_timestamps + offset,
                                    this_tile * sizeof(float),
                                    cudaMemcpyHostToDevice, ctx.stream));

        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        auto t_h2d_end = std::chrono::high_resolution_clock::now();
        transfer_ms += std::chrono::duration<float, std::milli>(t_h2d_end - t_h2d_start).count();

        auto t_compute_start = std::chrono::high_resolution_clock::now();

        // cuBLAS SGEMV: similarities = E^T * q
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                                  D, this_tile, &alpha,
                                  ctx.d_tile_embeddings, D,
                                  ctx.d_query, 1,
                                  &beta, ctx.d_tile_sim, 1));

        // Temporal rerank
        int32_t blocks = (this_tile + 255) / 256;
        tiled_temporal_rerank_kernel<<<blocks, 256, 0, ctx.stream>>>(
            ctx.d_tile_sim, ctx.d_tile_timestamps, ctx.d_tile_modalities,
            this_tile, query_timestamp, cfg.time_decay_lambda,
            cfg.modality_filter);

        // Per-tile top-K
        int32_t topk_threads = std::min(256, this_tile);
        size_t smem = topk_threads * K * (sizeof(float) + sizeof(int32_t));
        tiled_topk_kernel<<<1, topk_threads, smem, ctx.stream>>>(
            ctx.d_tile_sim, d_topk_idx, d_topk_val,
            this_tile, K, offset);

        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        auto t_compute_end = std::chrono::high_resolution_clock::now();
        compute_ms += std::chrono::duration<float, std::milli>(t_compute_end - t_compute_start).count();

        // D2H: read per-tile top-K
        std::vector<int32_t> h_idx(K);
        std::vector<float>   h_val(K);
        CUDA_CHECK(cudaMemcpy(h_idx.data(), d_topk_idx, K * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_val.data(), d_topk_val, K * sizeof(float),
                               cudaMemcpyDeviceToHost));

        for (int32_t k = 0; k < K; ++k) {
            if (h_idx[k] >= 0 && h_val[k] > -FLT_MAX * 0.5f) {
                int32_t gid = h_idx[k];
                all_candidates.push_back({gid, h_val[k], h_modalities[gid]});
            }
        }
    }

    cudaFree(d_topk_idx);
    cudaFree(d_topk_val);

    // Merge: sort all candidates, take top-K
    auto t_merge_start = std::chrono::high_resolution_clock::now();
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.score > b.score;
              });
    if (static_cast<int32_t>(all_candidates.size()) > K)
        all_candidates.resize(K);

    std::vector<TiledQueryResult> results;
    results.reserve(K);
    for (const auto& c : all_candidates) {
        results.push_back({c.id, c.score, c.modality});
    }
    auto t_merge_end = std::chrono::high_resolution_clock::now();

    auto t_end = std::chrono::high_resolution_clock::now();
    stats.transfer_ms = transfer_ms;
    stats.compute_ms  = compute_ms;
    stats.merge_ms    = std::chrono::duration<float, std::milli>(t_merge_end - t_merge_start).count();
    stats.total_ms    = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    return results;
}
