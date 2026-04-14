#ifndef MEMORY_BUDGET_H
#define MEMORY_BUDGET_H

#include <cstdint>
#include <cstddef>

// ============================================================================
//  VRAM Budget Calculator
//
//  Computes the deterministic worst-case GPU memory footprint for a MARS
//  deployment. All allocations are made at init time; no dynamic GPU memory
//  allocation occurs during retrieval queries.
//
//  This is critical for safety-critical systems that must prove their
//  memory usage is bounded at certification time.
// ============================================================================

struct MemoryBudget {
    // Input parameters
    int32_t num_nodes;
    int32_t num_edges;        // total CSR edges (undirected, counted twice)
    int32_t embedding_dim;
    int32_t max_k;            // top-K for retrieval
    int32_t max_compact;      // max BFS results to compact

    // Itemized allocations (bytes)
    size_t csr_row_offsets;   // (N+1) × sizeof(int32_t)
    size_t csr_col_indices;   // E × sizeof(int32_t)
    size_t embeddings_fp32;   // N × D × sizeof(float)
    size_t embeddings_fp16;   // N × D × sizeof(half)  (0 if FP16 disabled)
    size_t modalities;        // N × sizeof(int32_t)
    size_t timestamps;        // N × sizeof(float)
    size_t importance;        // N × sizeof(float)
    size_t edge_hits;         // E × sizeof(int32_t)

    // QueryContext scratch buffers
    size_t scratch_query;     // D × sizeof(float)
    size_t scratch_sim;       // N × sizeof(float)
    size_t scratch_final;     // N × sizeof(float)
    size_t scratch_bfs;       // N × sizeof(float) + N × sizeof(int32_t)
    size_t scratch_topk;      // K × (sizeof(int32_t) + sizeof(float))
    size_t scratch_frontiers; // 2 × N × sizeof(int32_t) + 2 × sizeof(int32_t)
    size_t scratch_compact;   // max_compact × sizeof(CompactResult) + sizeof(int32_t)
    size_t scratch_cublas;    // cuBLAS handle + CUB temp (estimated)
    size_t scratch_events;    // 5 × cudaEvent + 1 × cudaStream (fixed overhead)

    // Totals
    size_t total_graph;       // sum of graph arrays
    size_t total_scratch;     // sum of QueryContext buffers
    size_t total_bytes;       // total_graph + total_scratch
    double total_mb;          // total_bytes / (1024*1024)
    double total_gb;          // total_bytes / (1024*1024*1024)
};

// Compute the worst-case VRAM budget for a given deployment configuration.
// Parameters:
//   num_nodes     — maximum number of memories in the graph
//   embedding_dim — dimensionality of embeddings (typically 768)
//   avg_degree    — expected average node degree (default 12 from NSN k=6)
//   max_k         — maximum top-K for retrieval (default 64)
//   max_compact   — max BFS results returned (default: min(N, K*500))
//   include_fp16  — include FP16 embedding copy in budget
MemoryBudget compute_memory_budget(int32_t num_nodes,
                                    int32_t embedding_dim,
                                    int32_t avg_degree = 12,
                                    int32_t max_k = 64,
                                    int32_t max_compact = 0,
                                    bool include_fp16 = false);

// Print a human-readable breakdown of the budget to stdout.
void print_budget(const MemoryBudget& budget);

// Check if a budget fits within available VRAM (in bytes).
// Returns true if total_bytes <= available_vram.
bool budget_fits(const MemoryBudget& budget, size_t available_vram);

#endif // MEMORY_BUDGET_H
