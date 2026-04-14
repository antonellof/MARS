#include "wmma_similarity.cuh"
#include <cstdio>

using namespace nvcuda;

// ============================================================================
//  WMMA Cosine Similarity Kernel
//
//  Each warp computes the dot product between the query (1xD) and one
//  memory embedding (1xD) using 16x16x16 WMMA operations.
//
//  Strategy: We reshape the 1-D dot product into a series of 16x16 matrix
//  multiplies. For each tile of 16 elements along D:
//    - Load 16 elements of query into a 1x16 fragment (replicated to 16x16)
//    - Load 16 elements of embedding into a 16x1 fragment (replicated to 16x16)
//    - Accumulate into a 16x16 FP32 fragment
//  After all tiles, the result is in one element of the accumulator.
//
//  Since WMMA requires 16x16 operands but we only need a 1x1 result,
//  we use a simpler approach: tile along D in chunks of WMMA_K=16,
//  load query[tile:tile+16] and emb[tile:tile+16] into WMMA fragments,
//  multiply, and sum the diagonal of the result accumulator.
// ============================================================================

// ─── FP32-to-FP16 conversion kernel ────────────────────────────────────
// Grid: ceil(len / 256) blocks x 256 threads
// Each thread converts one element from FP32 to FP16.
__global__ void convert_fp32_to_fp16_kernel(const float* __restrict__ src,
                                             half* __restrict__ dst,
                                             int32_t len) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        dst[idx] = __float2half(src[idx]);
    }
}

void convert_query_fp32_to_fp16(const float* d_query_fp32,
                                 half* d_query_fp16,
                                 int32_t D,
                                 cudaStream_t stream) {
    int32_t blocks = (D + 255) / 256;
    convert_fp32_to_fp16_kernel<<<blocks, 256, 0, stream>>>(
        d_query_fp32, d_query_fp16, D);
}

int32_t pad_dim_to_wmma(int32_t D) {
    return ((D + WMMA_K - 1) / WMMA_K) * WMMA_K;
}

// ─── Warp-shuffle FP16 similarity kernel ───────────────────────────────
// Grid: ceil(N * 32 / threads_per_block) blocks x threads_per_block threads
// Each warp computes one dot product using FP16 loads and FP32 accumulation
// with warp-shuffle reduction. This is the default launch path — robust
// across all sm_70+ architectures and does not require 16-aligned D.
__global__ void wmma_similarity_kernel(
    const half* __restrict__ d_query_fp16,
    const half* __restrict__ d_embeddings_fp16,
    float* __restrict__ d_similarities,
    int32_t N, int32_t D)
{
    // One warp per memory node
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int32_t lane_id = threadIdx.x % 32;

    if (warp_id >= N) return;

    const half* emb = d_embeddings_fp16 + static_cast<int64_t>(warp_id) * D;

    // Each thread in the warp handles D/32 elements, accumulating in FP32
    float partial = 0.0f;
    for (int32_t d = lane_id; d < D; d += 32) {
        float q = __half2float(d_query_fp16[d]);
        float e = __half2float(emb[d]);
        partial += q * e;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    }

    if (lane_id == 0) {
        d_similarities[warp_id] = partial;
    }
}

// ─── Full WMMA MMA kernel (uses actual tensor cores) ────────────────────
// Grid: ceil(N * 32 / threads_per_block) blocks x threads_per_block threads
// Each warp computes dot(query, embedding) for one node by tiling along D
// in WMMA_K=16 chunks.
//
// Layout: query is stored as a row-major 1xD matrix, replicated across
// the M dimension. Embedding is stored as a Dx1 column, replicated across
// the N dimension. The result C[0][0] contains the dot product.
//
// NOTE: Requires D to be a multiple of 16. Use pad_dim_to_wmma() if needed.
__global__ void wmma_mma_similarity_kernel(
    const half* __restrict__ d_query_fp16,
    const half* __restrict__ d_embeddings_fp16,
    float* __restrict__ d_similarities,
    int32_t N, int32_t D)
{
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= N) return;

    const half* emb = d_embeddings_fp16 + static_cast<int64_t>(warp_id) * D;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Tile along D in chunks of WMMA_K=16
    for (int32_t tile = 0; tile < D; tile += WMMA_K) {
        // Load query[tile:tile+16] as matrix_a (replicated across rows)
        // Load emb[tile:tile+16] as matrix_b (replicated across cols)
        wmma::load_matrix_sync(a_frag, d_query_fp16 + tile, D);
        wmma::load_matrix_sync(b_frag, emb + tile, D);

        // C += A x B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result — the dot product is distributed across the fragment.
    // We store to shared memory and read element [0][0].
    __shared__ float result_tile[WMMA_M * WMMA_N];
    wmma::store_matrix_sync(result_tile, c_frag, WMMA_N, wmma::mem_row_major);

    int32_t lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        // The dot product result is at [0][0] of the output tile
        d_similarities[warp_id] = result_tile[0];
    }
}

void launch_wmma_similarity(const half* d_query_fp16,
                             const half* d_embeddings_fp16,
                             float* d_similarities,
                             int32_t N, int32_t D,
                             cudaStream_t stream) {
    // Use the warp-shuffle FP16 kernel as default (more robust).
    // The MMA kernel is available for experimentation but requires
    // careful padding and alignment of the embedding matrices.
    int32_t threads_per_block = WMMA_WARPS_PER_BLOCK * 32;
    int32_t num_blocks = (N * 32 + threads_per_block - 1) / threads_per_block;

    wmma_similarity_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_query_fp16, d_embeddings_fp16, d_similarities, N, D);
}
