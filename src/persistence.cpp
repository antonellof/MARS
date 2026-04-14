// ============================================================================
//  persistence.cpp
//  Binary checkpoint/restore for MARS MemoryGraph.
//
//  Uses std::fstream in binary mode. All multi-byte values are written in
//  native endianness (the file is not portable across architectures, which
//  is fine for GPU-resident working-set checkpoints on the same machine).
//
//  FNV-1a 64-bit checksum is appended after all data. On load, the checksum
//  is recomputed over all bytes preceding it and compared.
// ============================================================================

#include "persistence.h"

#include <cstring>
#include <fstream>
#include <vector>

// ─── FNV-1a constants ───────────────────────────────────────────────
static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
static constexpr uint64_t FNV_PRIME        = 1099511628211ULL;

uint64_t fnv1a_checksum(const void* data, size_t len) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= FNV_PRIME;
    }
    return hash;
}

// ─── Incremental FNV-1a (for computing checksum while writing) ──────
static uint64_t fnv1a_update(uint64_t hash, const void* data, size_t len) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= FNV_PRIME;
    }
    return hash;
}

// ─── save_graph ─────────────────────────────────────────────────────
bool save_graph(const MemoryGraph& graph, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;

    uint64_t hash = FNV_OFFSET_BASIS;

    // Helper: write raw bytes and update checksum
    auto emit = [&](const void* ptr, size_t bytes) -> bool {
        out.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
        if (!out.good()) return false;
        hash = fnv1a_update(hash, ptr, bytes);
        return true;
    };

    // Magic
    const char magic[4] = {'M', 'A', 'R', 'S'};
    if (!emit(magic, 4)) return false;

    // Version
    uint32_t version = MARS_FILE_VERSION;
    if (!emit(&version, sizeof(version))) return false;

    // Header fields
    if (!emit(&graph.num_nodes,     sizeof(graph.num_nodes)))     return false;
    if (!emit(&graph.num_edges,     sizeof(graph.num_edges)))     return false;
    if (!emit(&graph.embedding_dim, sizeof(graph.embedding_dim))) return false;

    // CSR arrays
    if (!graph.row_offsets.empty()) {
        if (!emit(graph.row_offsets.data(),
                  graph.row_offsets.size() * sizeof(int32_t))) return false;
    }
    if (!graph.col_indices.empty()) {
        if (!emit(graph.col_indices.data(),
                  graph.col_indices.size() * sizeof(int32_t))) return false;
    }

    // Embeddings
    if (!graph.embeddings.empty()) {
        if (!emit(graph.embeddings.data(),
                  graph.embeddings.size() * sizeof(float))) return false;
    }

    // Modalities
    if (!graph.modalities.empty()) {
        if (!emit(graph.modalities.data(),
                  graph.modalities.size() * sizeof(int32_t))) return false;
    }

    // Timestamps
    if (!graph.timestamps.empty()) {
        if (!emit(graph.timestamps.data(),
                  graph.timestamps.size() * sizeof(float))) return false;
    }

    // Checksum (not included in the hash itself)
    out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    if (!out.good()) return false;

    out.close();
    return out.good();
}

// ─── load_graph ─────────────────────────────────────────────────────
bool load_graph(MemoryGraph& graph, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    // Read entire file into memory for checksum verification
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size < 0) return false;
    size_t total = static_cast<size_t>(file_size);

    // Minimum: magic(4) + version(4) + 3 header fields(12) + checksum(8) = 28
    if (total < 28) return false;

    std::vector<uint8_t> buf(total);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(total));
    if (!in.good()) return false;
    in.close();

    // Verify checksum: hash everything except the last 8 bytes
    size_t data_len = total - sizeof(uint64_t);
    uint64_t computed = fnv1a_checksum(buf.data(), data_len);

    uint64_t stored = 0;
    std::memcpy(&stored, buf.data() + data_len, sizeof(uint64_t));

    if (computed != stored) return false;

    // Parse header
    size_t offset = 0;

    // Magic
    if (buf[0] != 'M' || buf[1] != 'A' || buf[2] != 'R' || buf[3] != 'S')
        return false;
    offset += 4;

    // Version
    uint32_t version = 0;
    std::memcpy(&version, buf.data() + offset, sizeof(version));
    offset += sizeof(version);
    if (version != MARS_FILE_VERSION) return false;

    // Header fields
    int32_t num_nodes = 0, num_edges = 0, embedding_dim = 0;
    std::memcpy(&num_nodes,     buf.data() + offset, sizeof(int32_t)); offset += sizeof(int32_t);
    std::memcpy(&num_edges,     buf.data() + offset, sizeof(int32_t)); offset += sizeof(int32_t);
    std::memcpy(&embedding_dim, buf.data() + offset, sizeof(int32_t)); offset += sizeof(int32_t);

    // Sanity checks on sizes
    if (num_nodes < 0 || num_edges < 0 || embedding_dim < 0) return false;

    // Compute expected data size
    size_t expected = offset
        + static_cast<size_t>(num_nodes + 1) * sizeof(int32_t)  // row_offsets
        + static_cast<size_t>(num_edges)     * sizeof(int32_t)  // col_indices
        + static_cast<size_t>(num_nodes)     * embedding_dim * sizeof(float) // embeddings
        + static_cast<size_t>(num_nodes)     * sizeof(int32_t)  // modalities
        + static_cast<size_t>(num_nodes)     * sizeof(float)    // timestamps
        + sizeof(uint64_t);                                      // checksum

    if (total != expected) return false;

    // Populate graph
    graph.num_nodes     = num_nodes;
    graph.num_edges     = num_edges;
    graph.embedding_dim = embedding_dim;

    // row_offsets
    graph.row_offsets.resize(num_nodes + 1);
    std::memcpy(graph.row_offsets.data(), buf.data() + offset,
                (num_nodes + 1) * sizeof(int32_t));
    offset += (num_nodes + 1) * sizeof(int32_t);

    // col_indices
    graph.col_indices.resize(num_edges);
    if (num_edges > 0) {
        std::memcpy(graph.col_indices.data(), buf.data() + offset,
                    num_edges * sizeof(int32_t));
    }
    offset += num_edges * sizeof(int32_t);

    // embeddings
    size_t emb_count = static_cast<size_t>(num_nodes) * embedding_dim;
    graph.embeddings.resize(emb_count);
    if (emb_count > 0) {
        std::memcpy(graph.embeddings.data(), buf.data() + offset,
                    emb_count * sizeof(float));
    }
    offset += emb_count * sizeof(float);

    // modalities
    graph.modalities.resize(num_nodes);
    if (num_nodes > 0) {
        std::memcpy(graph.modalities.data(), buf.data() + offset,
                    num_nodes * sizeof(int32_t));
    }
    offset += num_nodes * sizeof(int32_t);

    // timestamps
    graph.timestamps.resize(num_nodes);
    if (num_nodes > 0) {
        std::memcpy(graph.timestamps.data(), buf.data() + offset,
                    num_nodes * sizeof(float));
    }

    return true;
}
