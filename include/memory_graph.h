// ============================================================================
//  memory_graph.h
//  GPU-resident multimodal memory graph. Each node represents a "memory"
//  — a text snippet, an audio clip, or an image — with:
//    • a dense embedding vector (D dimensions, typically 768 or 1024)
//    • a modality tag  (TEXT | AUDIO | IMAGE)
//    • a timestamp     (for temporal decay / "forgetting")
//    • CSR-format edges to other memories in the Neural Shortcut Network
// ============================================================================
#ifndef MEMORY_GRAPH_H
#define MEMORY_GRAPH_H

#include <cstdint>
#include <string>
#include <vector>

enum Modality : int32_t {
    MOD_TEXT  = 0,
    MOD_AUDIO = 1,
    MOD_IMAGE = 2,
    MOD_COUNT = 3
};

// Host-side representation of the multimodal memory graph
struct MemoryGraph {
    int32_t num_nodes      = 0;
    int32_t num_edges      = 0;        // edges counted twice (undirected)
    int32_t embedding_dim  = 0;        // D

    // CSR topology
    std::vector<int32_t> row_offsets;  // (N+1)
    std::vector<int32_t> col_indices;  // num_edges

    // Per-node payload
    std::vector<float>   embeddings;   // N * D, row-major
    std::vector<int32_t> modalities;   // N
    std::vector<float>   timestamps;   // N  (seconds since epoch, float ok here)

    // Build a small synthetic multimodal corpus for testing
    static MemoryGraph synthetic(int32_t n_text,
                                 int32_t n_audio,
                                 int32_t n_image,
                                 int32_t dim,
                                 uint32_t seed = 42);

    // Build Neural Shortcut Network edges over the existing nodes
    void build_nsn_edges(int32_t k = 6, double p = 0.15);

    // Phase-disable flags for ablation studies
    struct NSNConfig {
        int32_t k              = 6;
        double  p              = 0.15;
        bool    enable_ring    = true;   // Phase 1: ring lattice
        bool    enable_skips   = true;   // Phase 2: hierarchical skip connections
        bool    enable_hubs    = true;   // Phase 3: hub supernodes at sqrt(N)
        bool    enable_rewire  = true;   // Phase 4: small-world rewiring
        bool    enable_bridges = true;   // Phase 5: cross-modal bridges
    };

    // Build NSN with configurable phase enables (for ablation experiments)
    void build_nsn_edges_configurable(const NSNConfig& cfg);

    void print_summary() const;
    size_t device_bytes() const;
};

// ─── Episode grouping (host CSR over node ids) ──────────────────────
// Built from per-node episode_ids[i]. Used for optional episode-scoped GPU
// retrieval (see RetrievalScope in memory_cuda.cuh).
struct HostEpisodeCSR {
    int32_t               num_episodes = 0;
    std::vector<int32_t>  ep_csr_offsets;  // length num_episodes + 1
    std::vector<int32_t>  ep_csr_members;  // concatenated node indices per episode
};

HostEpisodeCSR build_episode_csr(const std::vector<int32_t>& episode_ids,
                                 int32_t num_nodes);

// Global: full-graph similarity + BFS. EpisodeScoped: similarity + decay only
// over nodes in query_episode_id (O(M·D), M = episode size); BFS hops forced 0.
enum class RetrievalScope : int32_t { Global = 0, EpisodeScoped = 1 };

// ─── Embodied “kids playing with ball” multimodal corpus ─────────────
// Fixed episode template (10 nodes): repeating cycle scales to arbitrary N.
// Modalities follow the embodied mapping (RGB→IMAGE, mic/IMU→AUDIO, ASR/state→TEXT).
// episode_ids[i] is the episode index for node i (same episode shares one latent prototype).
struct EmbodiedKidsBallCorpus {
    MemoryGraph           graph;
    std::vector<int32_t>  episode_ids;  // size == graph.num_nodes after make

    // n_nodes >= 10 recommended so at least one full episode exists.
    static EmbodiedKidsBallCorpus make(int32_t n_nodes, int32_t dim, uint32_t seed = 42);
};

#endif // MEMORY_GRAPH_H
