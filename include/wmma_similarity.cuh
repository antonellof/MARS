#ifndef WMMA_SIMILARITY_CUH
#define WMMA_SIMILARITY_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>  // WMMA intrinsics
#include <cstdint>

// ============================================================================
//  Tensor-Core Accelerated Cosine Similarity
//
//  Uses WMMA (Warp Matrix Multiply-Accumulate) to compute the dot product
//  between a query embedding and all N memory embeddings using 16x16x16
//  matrix multiply operations on tensor cores.
//
//  Requirements:
//    - Compute capability >= 7.0 (V100+)
//    - Embedding dimension D must be a multiple of 16 (pad if needed)
//    - FP16 embeddings must be uploaded via upload_fp16_embeddings()
//
//  The kernel treats the dot product as a 1xD x Dx1 matrix multiply:
//    query (1xD in FP16) x embedding^T (Dx1 in FP16) -> similarity (FP32)
//
//  In practice, we tile the operation: for each warp, we process a 16x16
//  chunk of the D dimension and accumulate the partial products.
//
//  Grid: (N / WARPS_PER_BLOCK) blocks x (WARPS_PER_BLOCK x 32) threads
//  Each warp computes the similarity for one memory node.
// ============================================================================

// WMMA tile dimensions (fixed by hardware)
constexpr int32_t WMMA_M = 16;
constexpr int32_t WMMA_N = 16;
constexpr int32_t WMMA_K = 16;

// Number of warps per block for the WMMA kernel
constexpr int32_t WMMA_WARPS_PER_BLOCK = 4;

// Launch the WMMA similarity kernel.
// d_query_fp16: query embedding in FP16 (length D, must be 16-aligned)
// d_embeddings_fp16: all N embeddings in FP16 (N x D, row-major, 16-aligned D)
// d_similarities: output FP32 similarities (length N)
// N: number of nodes
// D: embedding dimension (must be multiple of 16)
void launch_wmma_similarity(const half* d_query_fp16,
                             const half* d_embeddings_fp16,
                             float* d_similarities,
                             int32_t N, int32_t D,
                             cudaStream_t stream = nullptr);

// Pad embedding dimension to next multiple of 16.
// Returns the padded dimension.
int32_t pad_dim_to_wmma(int32_t D);

// Convert FP32 query to FP16 on device (utility kernel).
void convert_query_fp32_to_fp16(const float* d_query_fp32,
                                 half* d_query_fp16,
                                 int32_t D,
                                 cudaStream_t stream = nullptr);

#endif // WMMA_SIMILARITY_CUH
