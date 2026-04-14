#ifndef STREAMING_H
#define STREAMING_H

#include "memory_graph.h"
#include <cstdint>
#include <vector>
#include <mutex>

// ============================================================================
//  StreamingBuffer — Pre-allocated ring buffer for online node insertion.
//
//  Design: Nodes accumulate in a fixed-capacity staging area on the host.
//  When the buffer is full or a manual flush is triggered, all staged nodes
//  are batch-committed to the MemoryGraph and NSN edges are incrementally
//  added. This avoids per-node cudaMalloc and full graph rebuilds.
//
//  Thread safety: All public methods are mutex-protected. Multiple producer
//  threads can call insert() concurrently; a single consumer calls flush().
// ============================================================================

struct StagedNode {
    int32_t              id;          // unique node ID
    int32_t              modality;    // MOD_TEXT, MOD_AUDIO, MOD_IMAGE
    float                timestamp;
    std::vector<float>   embedding;   // length = embedding_dim
};

struct StreamingConfig {
    int32_t capacity        = 1024;   // max nodes before auto-flush
    int32_t embedding_dim   = 768;
    bool    auto_flush      = true;   // flush when capacity reached
    int32_t nsn_k           = 6;      // ring lattice degree for new edges
    double  nsn_p           = 0.15;   // small-world rewiring probability
};

struct FlushStats {
    int32_t nodes_committed = 0;
    int32_t edges_added     = 0;
    float   flush_ms        = 0.0f;   // wall-clock time for flush
};

class StreamingBuffer {
public:
    explicit StreamingBuffer(const StreamingConfig& cfg);
    ~StreamingBuffer() = default;

    // Insert a single node into the staging buffer.
    // Returns true if accepted, false if buffer is full and auto_flush is off.
    // If auto_flush is on and buffer is full, triggers flush() first.
    bool insert(int32_t id, int32_t modality, float timestamp,
                const float* embedding);

    // Insert from vectors (convenience).
    bool insert(const StagedNode& node);

    // Commit all staged nodes to the target graph.
    // Incrementally adds NSN edges for the new nodes.
    // Returns stats about the flush operation.
    FlushStats flush(MemoryGraph& target);

    // Number of nodes currently staged (not yet flushed).
    int32_t staged_count() const;

    // Number of nodes flushed so far (lifetime total).
    int64_t total_flushed() const;

    // Current capacity.
    int32_t capacity() const { return cfg_.capacity; }

    // Reset buffer (discard staged nodes, keep config).
    void reset();

private:
    StreamingConfig         cfg_;
    std::vector<StagedNode> buffer_;
    int64_t                 total_flushed_ = 0;
    mutable std::mutex      mu_;

    // Incrementally add NSN edges for newly appended nodes.
    void add_incremental_edges_(MemoryGraph& graph, int32_t old_n, int32_t new_n);
};

#endif // STREAMING_H
