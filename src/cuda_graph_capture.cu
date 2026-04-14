// ============================================================================
//  cuda_graph_capture.cu
//  CUDA Graph capture/replay management for the MARS retrieval pipeline.
//
//  The 4-kernel pipeline (similarity, temporal rerank, top-K, BFS) is
//  captured into a CUDA graph on the first query and replayed for all
//  subsequent queries with matching parameters. This eliminates per-query
//  kernel launch overhead (~5-15 us per launch x 4 kernels = 20-60 us
//  saved per query).
//
//  The graph is automatically invalidated and re-captured when any
//  grid-dimension-affecting parameter changes (N, K, bfs_max_hops,
//  modality_filter, or FP16 mode).
//
//  Thread safety: none. Each QueryContext is assumed to be used by a
//  single thread (or externally synchronized).
// ============================================================================

#include "cuda_graph_capture.h"
#include <cstdio>

// ── Parameter change detection ──────────────────────────────────────
bool graph_needs_recapture(const QueryContext& ctx,
                            const DeviceMemoryGraph& dg,
                            const RetrievalConfig& cfg) {
    if (!ctx.graph_captured) return true;
    if (ctx.graph_N != dg.num_nodes) return true;
    if (ctx.graph_K != cfg.top_k) return true;
    if (ctx.graph_bfs_hops != cfg.bfs_max_hops) return true;
    if (ctx.graph_mod_filter != cfg.modality_filter) return true;
    if (ctx.graph_fp16 != ctx.use_fp16) return true;
    return false;
}

// ── Graph invalidation ──────────────────────────────────────────────
void invalidate_graph(QueryContext& ctx) {
    if (ctx.graph_exec) {
        cudaGraphExecDestroy(ctx.graph_exec);
        ctx.graph_exec = nullptr;
    }
    if (ctx.captured_graph) {
        cudaGraphDestroy(ctx.captured_graph);
        ctx.captured_graph = nullptr;
    }
    ctx.graph_captured   = false;
    ctx.graph_N          = 0;
    ctx.graph_K          = 0;
    ctx.graph_bfs_hops   = 0;
    ctx.graph_mod_filter = -1;
    ctx.graph_fp16       = false;
}

// ── Begin stream capture ────────────────────────────────────────────
//
// Starts CUDA stream capture in global mode. The caller must launch
// the pipeline kernels into ctx.stream after this call, then call
// finalize_graph_capture() to end capture.
//
// Global capture mode is used because all kernels launch on a single
// stream. Thread-local mode would be needed for multi-stream graphs.
bool begin_graph_capture(const DeviceMemoryGraph& dg,
                          QueryContext& ctx,
                          const RetrievalConfig& cfg) {
    // Tear down any existing graph first
    invalidate_graph(ctx);

    cudaError_t err = cudaStreamBeginCapture(ctx.stream,
                                              cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA graph capture begin failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    // Record the parameters that were in effect at capture time.
    // graph_needs_recapture() compares these against current values.
    ctx.graph_N          = dg.num_nodes;
    ctx.graph_K          = cfg.top_k;
    ctx.graph_bfs_hops   = cfg.bfs_max_hops;
    ctx.graph_mod_filter = cfg.modality_filter;
    ctx.graph_fp16       = ctx.use_fp16;

    return true;
}

// ── Finalize capture and instantiate executable graph ───────────────
//
// Ends stream capture, extracts the graph, and instantiates an
// executable graph object that can be launched with cudaGraphLaunch().
//
// The instantiation step performs GPU-side optimization of the graph
// (kernel fusion, memory coalescing) which is why replay is faster
// than individual launches.
bool finalize_graph_capture(QueryContext& ctx) {
    cudaError_t err;

    err = cudaStreamEndCapture(ctx.stream, &ctx.captured_graph);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA graph capture end failed: %s\n",
                     cudaGetErrorString(err));
        // Reset state since capture failed
        ctx.captured_graph = nullptr;
        ctx.graph_N = 0;
        ctx.graph_K = 0;
        ctx.graph_bfs_hops = 0;
        ctx.graph_mod_filter = -1;
        return false;
    }

    // Instantiate the executable graph. The nullptr arguments for
    // error node and log buffer mean errors go to stderr only.
    err = cudaGraphInstantiate(&ctx.graph_exec, ctx.captured_graph,
                                nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA graph instantiation failed: %s\n",
                     cudaGetErrorString(err));
        cudaGraphDestroy(ctx.captured_graph);
        ctx.captured_graph = nullptr;
        return false;
    }

    ctx.graph_captured = true;
    return true;
}

// ── Graph replay ────────────────────────────────────────────────────
//
// Launches the previously captured graph on ctx.stream. The query
// vector must already reside in ctx.d_query (copied before this call).
//
// The graph replays all 4 kernels with the exact same grid/block
// dimensions, shared memory sizes, and kernel arguments that were
// recorded during capture. Only the data pointed to by the device
// pointers changes between replays (the query embedding and
// timestamps).
bool replay_retrieval_graph(QueryContext& ctx) {
    if (!ctx.graph_captured || !ctx.graph_exec) {
        std::fprintf(stderr, "No captured graph to replay\n");
        return false;
    }

    cudaError_t err = cudaGraphLaunch(ctx.graph_exec, ctx.stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA graph launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    return true;
}
