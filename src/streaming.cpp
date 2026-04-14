#include "streaming.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <set>

StreamingBuffer::StreamingBuffer(const StreamingConfig& cfg) : cfg_(cfg) {
    buffer_.reserve(cfg.capacity);
}

bool StreamingBuffer::insert(int32_t id, int32_t modality, float timestamp,
                             const float* embedding) {
    std::lock_guard<std::mutex> lock(mu_);
    if (static_cast<int32_t>(buffer_.size()) >= cfg_.capacity) {
        if (!cfg_.auto_flush) return false;
        // Auto-flush requires caller to provide graph — can't do it here.
        // Signal that buffer is full.
        return false;
    }
    StagedNode node;
    node.id = id;
    node.modality = modality;
    node.timestamp = timestamp;
    node.embedding.assign(embedding, embedding + cfg_.embedding_dim);
    buffer_.push_back(std::move(node));
    return true;
}

bool StreamingBuffer::insert(const StagedNode& node) {
    return insert(node.id, node.modality, node.timestamp, node.embedding.data());
}

FlushStats StreamingBuffer::flush(MemoryGraph& target) {
    std::lock_guard<std::mutex> lock(mu_);
    FlushStats stats;
    if (buffer_.empty()) return stats;

    auto t0 = std::chrono::high_resolution_clock::now();

    int32_t old_n = target.num_nodes;
    int32_t batch_n = static_cast<int32_t>(buffer_.size());
    int32_t new_n = old_n + batch_n;

    // Resize target arrays
    target.embeddings.resize(size_t(new_n) * target.embedding_dim);
    target.modalities.resize(new_n);
    target.timestamps.resize(new_n);

    // Copy staged nodes into target
    for (int32_t i = 0; i < batch_n; ++i) {
        const auto& node = buffer_[i];
        int32_t idx = old_n + i;
        std::memcpy(&target.embeddings[size_t(idx) * target.embedding_dim],
                     node.embedding.data(),
                     target.embedding_dim * sizeof(float));
        target.modalities[idx] = node.modality;
        target.timestamps[idx] = node.timestamp;
    }
    target.num_nodes = new_n;

    // Set embedding_dim if not yet set
    if (target.embedding_dim == 0 && cfg_.embedding_dim > 0) {
        target.embedding_dim = cfg_.embedding_dim;
    }

    // Incrementally add NSN edges for new nodes
    int32_t old_edges = target.num_edges;
    add_incremental_edges_(target, old_n, new_n);
    stats.edges_added = target.num_edges - old_edges;

    stats.nodes_committed = batch_n;
    total_flushed_ += batch_n;
    buffer_.clear();

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.flush_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return stats;
}

int32_t StreamingBuffer::staged_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int32_t>(buffer_.size());
}

int64_t StreamingBuffer::total_flushed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_flushed_;
}

void StreamingBuffer::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    buffer_.clear();
}

void StreamingBuffer::add_incremental_edges_(MemoryGraph& graph,
                                              int32_t old_n, int32_t new_n) {
    // Build adjacency list from existing CSR
    std::vector<std::set<int32_t>> adj(new_n);
    for (int32_t i = 0; i < old_n; ++i) {
        for (int32_t j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j) {
            adj[i].insert(graph.col_indices[j]);
        }
    }

    std::mt19937 rng(42 + old_n);  // deterministic but varies with graph size

    int32_t k = cfg_.nsn_k;

    for (int32_t i = old_n; i < new_n; ++i) {
        // Phase 1: Ring lattice — connect to k/2 nearest neighbors by index
        for (int32_t offset = 1; offset <= k / 2; ++offset) {
            int32_t left  = ((i - offset) % new_n + new_n) % new_n;
            int32_t right = (i + offset) % new_n;
            if (left != i)  { adj[i].insert(left);  adj[left].insert(i); }
            if (right != i) { adj[i].insert(right); adj[right].insert(i); }
        }

        // Phase 5: Cross-modal bridges — one edge to each other modality
        int32_t my_mod = graph.modalities[i];
        for (int32_t m = 0; m < MOD_COUNT; ++m) {
            if (m == my_mod) continue;
            // Find closest node of modality m by embedding similarity
            int32_t best = -1;
            float best_sim = -2.0f;
            // Sample up to 100 random nodes of target modality
            std::vector<int32_t> candidates;
            for (int32_t c = 0; c < new_n; ++c) {
                if (graph.modalities[c] == m) candidates.push_back(c);
            }
            // Subsample if too many
            if (candidates.size() > 100) {
                std::shuffle(candidates.begin(), candidates.end(), rng);
                candidates.resize(100);
            }
            for (int32_t c : candidates) {
                float sim = 0.0f;
                for (int32_t d = 0; d < graph.embedding_dim; ++d) {
                    sim += graph.embeddings[size_t(i) * graph.embedding_dim + d] *
                           graph.embeddings[size_t(c) * graph.embedding_dim + d];
                }
                if (sim > best_sim) { best_sim = sim; best = c; }
            }
            if (best >= 0 && best != i) {
                adj[i].insert(best);
                adj[best].insert(i);
            }
        }
    }

    // Rebuild CSR from adjacency lists
    graph.row_offsets.resize(new_n + 1);
    graph.row_offsets[0] = 0;
    graph.col_indices.clear();
    for (int32_t i = 0; i < new_n; ++i) {
        for (int32_t n : adj[i]) {
            graph.col_indices.push_back(n);
        }
        graph.row_offsets[i + 1] = static_cast<int32_t>(graph.col_indices.size());
    }
    graph.num_edges = static_cast<int32_t>(graph.col_indices.size());
}
