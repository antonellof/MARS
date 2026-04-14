#include "memory_budget.h"
#include <cstdio>
#include <algorithm>

MemoryBudget compute_memory_budget(int32_t num_nodes,
                                    int32_t embedding_dim,
                                    int32_t avg_degree,
                                    int32_t max_k,
                                    int32_t max_compact,
                                    bool include_fp16) {
    MemoryBudget b{};
    b.num_nodes = num_nodes;
    int32_t N = num_nodes;
    int32_t D = embedding_dim;
    int32_t E = N * avg_degree;  // edges counted twice (undirected)
    b.num_edges = E;
    b.embedding_dim = D;
    b.max_k = max_k;
    b.max_compact = max_compact > 0 ? max_compact : std::min(N, max_k * 500);

    // Graph arrays
    b.csr_row_offsets  = size_t(N + 1) * sizeof(int32_t);
    b.csr_col_indices  = size_t(E) * sizeof(int32_t);
    b.embeddings_fp32  = size_t(N) * D * sizeof(float);
    b.embeddings_fp16  = include_fp16 ? size_t(N) * D * sizeof(uint16_t) : 0;
    b.modalities       = size_t(N) * sizeof(int32_t);
    b.timestamps       = size_t(N) * sizeof(float);
    b.importance       = size_t(N) * sizeof(float);
    b.edge_hits        = size_t(E) * sizeof(int32_t);

    // QueryContext scratch
    b.scratch_query     = size_t(D) * sizeof(float);
    b.scratch_sim       = size_t(N) * sizeof(float);
    b.scratch_final     = size_t(N) * sizeof(float);
    b.scratch_bfs       = size_t(N) * sizeof(float) + size_t(N) * sizeof(int32_t);
    b.scratch_topk      = size_t(max_k) * (sizeof(int32_t) + sizeof(float));
    b.scratch_frontiers = 2 * size_t(N) * sizeof(int32_t) + 2 * sizeof(int32_t);
    // CompactResult: node_id(4) + score(4) + modality(4) + hops(4) = 16 bytes
    b.scratch_compact   = size_t(b.max_compact) * 16 + sizeof(int32_t);
    // cuBLAS handle (~4KB) + CUB temp storage (~N*8 for radix sort)
    b.scratch_cublas    = 4096 + size_t(N) * 8;
    // CUDA events/streams: ~5 events × ~64 bytes + 1 stream × ~64 bytes
    b.scratch_events    = 6 * 64;

    // Totals
    b.total_graph = b.csr_row_offsets + b.csr_col_indices +
                    b.embeddings_fp32 + b.embeddings_fp16 +
                    b.modalities + b.timestamps +
                    b.importance + b.edge_hits;

    b.total_scratch = b.scratch_query + b.scratch_sim + b.scratch_final +
                      b.scratch_bfs + b.scratch_topk + b.scratch_frontiers +
                      b.scratch_compact + b.scratch_cublas + b.scratch_events;

    b.total_bytes = b.total_graph + b.total_scratch;
    b.total_mb = double(b.total_bytes) / (1024.0 * 1024.0);
    b.total_gb = double(b.total_bytes) / (1024.0 * 1024.0 * 1024.0);

    return b;
}

void print_budget(const MemoryBudget& budget) {
    std::printf("\n");
    std::printf("═══════════════════════════════════════════════════════════\n");
    std::printf("  MARS VRAM Budget — %d nodes × %d-D\n",
                budget.num_nodes, budget.embedding_dim);
    std::printf("═══════════════════════════════════════════════════════════\n");
    std::printf("\n  Graph arrays:\n");
    std::printf("    CSR row_offsets:     %12zu bytes\n", budget.csr_row_offsets);
    std::printf("    CSR col_indices:     %12zu bytes\n", budget.csr_col_indices);
    std::printf("    Embeddings (FP32):   %12zu bytes\n", budget.embeddings_fp32);
    if (budget.embeddings_fp16 > 0)
        std::printf("    Embeddings (FP16):   %12zu bytes\n", budget.embeddings_fp16);
    std::printf("    Modalities:          %12zu bytes\n", budget.modalities);
    std::printf("    Timestamps:          %12zu bytes\n", budget.timestamps);
    std::printf("    Importance:          %12zu bytes\n", budget.importance);
    std::printf("    Edge hits:           %12zu bytes\n", budget.edge_hits);
    std::printf("    ─────────────────────────────────\n");
    std::printf("    Subtotal (graph):    %12zu bytes  (%.2f MB)\n",
                budget.total_graph, double(budget.total_graph) / (1024.0 * 1024.0));

    std::printf("\n  QueryContext scratch:\n");
    std::printf("    Query vector:        %12zu bytes\n", budget.scratch_query);
    std::printf("    Similarity scores:   %12zu bytes\n", budget.scratch_sim);
    std::printf("    Final scores:        %12zu bytes\n", budget.scratch_final);
    std::printf("    BFS state:           %12zu bytes\n", budget.scratch_bfs);
    std::printf("    Top-K buffers:       %12zu bytes\n", budget.scratch_topk);
    std::printf("    BFS frontiers:       %12zu bytes\n", budget.scratch_frontiers);
    std::printf("    Result compaction:   %12zu bytes\n", budget.scratch_compact);
    std::printf("    cuBLAS/CUB:          %12zu bytes\n", budget.scratch_cublas);
    std::printf("    Events/streams:      %12zu bytes\n", budget.scratch_events);
    std::printf("    ─────────────────────────────────\n");
    std::printf("    Subtotal (scratch):  %12zu bytes  (%.2f MB)\n",
                budget.total_scratch, double(budget.total_scratch) / (1024.0 * 1024.0));

    std::printf("\n  ═════════════════════════════════════\n");
    std::printf("    TOTAL:               %12zu bytes\n", budget.total_bytes);
    std::printf("                         %12.2f MB\n", budget.total_mb);
    std::printf("                         %12.4f GB\n", budget.total_gb);
    std::printf("  ═════════════════════════════════════\n\n");
}

bool budget_fits(const MemoryBudget& budget, size_t available_vram) {
    return budget.total_bytes <= available_vram;
}
