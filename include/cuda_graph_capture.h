#ifndef CUDA_GRAPH_CAPTURE_H
#define CUDA_GRAPH_CAPTURE_H

#include "memory_cuda.cuh"

// ============================================================================
//  CUDA Graph Capture for MARS Retrieval Pipeline
//
//  Captures the 4-kernel retrieval pipeline as a CUDA Graph for replay.
//  This eliminates per-query kernel launch overhead (~5-15 us per launch)
//  by recording the full pipeline once and replaying it for subsequent
//  queries with the same grid dimensions.
//
//  Constraints:
//    - N, K, bfs_max_hops, and modality_filter must be constant between
//      capture and replay. If any change, the graph must be re-captured.
//    - The query vector is updated via cudaMemcpy before each replay.
//    - FP16 mode must match between capture and replay.
//
//  Usage:
//    QueryContext ctx = create_query_context(N, D, K);
//    ctx.use_cuda_graph = true;
//
//    // First query: captures the graph
//    auto results = query_memory_fast(dg, ctx, query, ts, cfg, stats);
//
//    // Subsequent queries: replays the captured graph
//    auto results2 = query_memory_fast(dg, ctx, query2, ts, cfg, stats);
// ============================================================================

// Check if the graph needs re-capture (parameters changed since last capture).
// Returns true if no graph is captured or if any capture-time parameter
// (N, K, bfs_max_hops, modality_filter, fp16 mode) has changed.
bool graph_needs_recapture(const QueryContext& ctx,
                            const DeviceMemoryGraph& dg,
                            const RetrievalConfig& cfg);

// Invalidate the captured graph and free associated resources.
// Safe to call even if no graph is captured.
void invalidate_graph(QueryContext& ctx);

// Begin CUDA stream capture on ctx.stream. After this call, the caller
// should launch the pipeline kernels into the stream, then call
// finalize_graph_capture() to end capture and instantiate the executable.
//
// Records the current parameters (N, K, etc.) into ctx so that
// graph_needs_recapture() can detect changes.
//
// Returns true if stream capture began successfully.
bool begin_graph_capture(const DeviceMemoryGraph& dg,
                          QueryContext& ctx,
                          const RetrievalConfig& cfg);

// End CUDA stream capture and instantiate the executable graph.
// Must be called after begin_graph_capture() and after all pipeline
// kernels have been launched into the capturing stream.
//
// On success, sets ctx.graph_captured = true and populates
// ctx.captured_graph and ctx.graph_exec.
//
// Returns true if capture and instantiation succeeded.
bool finalize_graph_capture(QueryContext& ctx);

// Replay the previously captured graph on ctx.stream.
// The query vector must already be copied to ctx.d_query before calling.
// Returns true if replay succeeded.
bool replay_retrieval_graph(QueryContext& ctx);

#endif // CUDA_GRAPH_CAPTURE_H
