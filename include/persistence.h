// ============================================================================
//  persistence.h
//  Binary persistence layer for MARS memory graphs. Supports checkpoint
//  and restore of the full MemoryGraph (CSR topology + embeddings +
//  modalities + timestamps) to/from a compact binary file with FNV-1a
//  integrity checking.
//
//  File format (version 1):
//    [magic: 4 bytes "MARS"]
//    [version: uint32_t]
//    [num_nodes: int32_t] [num_edges: int32_t] [embedding_dim: int32_t]
//    [row_offsets: (num_nodes+1) x int32_t]
//    [col_indices: num_edges x int32_t]
//    [embeddings: num_nodes x embedding_dim x float]
//    [modalities: num_nodes x int32_t]
//    [timestamps: num_nodes x float]
//    [checksum: uint64_t]  (FNV-1a of all preceding bytes)
// ============================================================================
#ifndef PERSISTENCE_H
#define PERSISTENCE_H

#include "memory_graph.h"
#include <string>
#include <cstdint>

constexpr uint32_t MARS_FILE_VERSION = 1;

// Save a MemoryGraph to a binary file. Returns true on success.
bool save_graph(const MemoryGraph& graph, const std::string& path);

// Load a MemoryGraph from a binary file. Returns true on success.
// On failure, `graph` is left in an unspecified state.
bool load_graph(MemoryGraph& graph, const std::string& path);

// Compute FNV-1a checksum of raw bytes.
uint64_t fnv1a_checksum(const void* data, size_t len);

#endif // PERSISTENCE_H
