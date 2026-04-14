// ============================================================================
//  tests/test_memory_graph.cpp
//
//  Host-only unit tests for the multimodal memory graph. No CUDA needed.
//  Verifies the structural invariants that the retrieval pipeline depends
//  on, so that a broken NSN construction is caught before it reaches the
//  GPU.
//
//  Tests:
//    T1  Synthetic corpus has correct per-modality counts
//    T2  Embeddings are approximately L2-normalized
//    T3  CSR format is internally consistent (row_offsets monotone,
//        col_indices in range, each row sorted)
//    T4  NSN is undirected (every edge a->b has a reciprocal b->a)
//    T5  No self-loops
//    T6  Cross-modal bridges: every node has at least one neighbor in
//        every other modality
//    T7  Average degree is within the expected bound
//    T8–T17 persistence, budget, streaming
//    T18 Embodied kids-ball corpus: CSR valid after NSN build
//
//  Build:  g++ -std=c++17 -Iinclude -o tests/run_tests \
//              src/memory_graph.cpp tests/test_memory_graph.cpp
//  Run:    ./tests/run_tests
// ============================================================================

#include "memory_graph.h"
#include "persistence.h"
#include "memory_budget.h"
#include "streaming.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// ─── Tiny assertion framework ───────────────────────────────────────
struct TestResult {
    std::string name;
    bool        passed;
    std::string message;
};

static std::vector<TestResult> g_results;

#define CHECK(expr, msg) do {                                                  \
    if (!(expr)) {                                                             \
        g_results.push_back({ __func__, false,                                 \
            std::string(#expr) + " failed: " + (msg) });                       \
        return;                                                                \
    }                                                                          \
} while (0)

#define PASS() do {                                                            \
    g_results.push_back({ __func__, true, "" });                               \
} while (0)

// ─── Tests ──────────────────────────────────────────────────────────

void T1_synthetic_counts() {
    auto g = MemoryGraph::synthetic(100, 50, 50, 768);
    CHECK(g.num_nodes == 200, "expected 200 total nodes");
    int counts[MOD_COUNT] = {0};
    for (int32_t m : g.modalities) {
        CHECK(m >= 0 && m < MOD_COUNT, "modality out of range");
        counts[m]++;
    }
    CHECK(counts[MOD_TEXT]  == 100, "wrong text count");
    CHECK(counts[MOD_AUDIO] == 50,  "wrong audio count");
    CHECK(counts[MOD_IMAGE] == 50,  "wrong image count");
    PASS();
}

void T2_embeddings_normalized() {
    auto g = MemoryGraph::synthetic(50, 25, 25, 256);
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        double norm = 0.0;
        for (int32_t d = 0; d < g.embedding_dim; ++d) {
            double v = g.embeddings[size_t(i) * g.embedding_dim + d];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        // L2 norms should be very close to 1.0 after normalization
        CHECK(std::fabs(norm - 1.0) < 1e-4, "embedding not L2-normalized");
    }
    PASS();
}

void T3_csr_format_valid() {
    auto g = MemoryGraph::synthetic(80, 40, 40, 128);
    g.build_nsn_edges();

    // Monotonicity
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        CHECK(g.row_offsets[i] <= g.row_offsets[i + 1],
              "row_offsets not monotone");
    }
    CHECK(g.row_offsets[g.num_nodes] == g.num_edges,
          "row_offsets[N] != num_edges");

    // In-range col_indices + sorted per row
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        int32_t prev = -1;
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j) {
            int32_t n = g.col_indices[j];
            CHECK(n >= 0 && n < g.num_nodes, "neighbor out of range");
            CHECK(n > prev, "neighbors not strictly sorted");
            prev = n;
        }
    }
    PASS();
}

void T4_edges_symmetric() {
    auto g = MemoryGraph::synthetic(60, 30, 30, 64);
    g.build_nsn_edges();

    // Build adjacency set for O(1) lookup
    std::vector<std::set<int32_t>> adj(g.num_nodes);
    for (int32_t i = 0; i < g.num_nodes; ++i)
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j)
            adj[i].insert(g.col_indices[j]);

    for (int32_t i = 0; i < g.num_nodes; ++i) {
        for (int32_t n : adj[i]) {
            CHECK(adj[n].count(i) == 1, "edge not reciprocated");
        }
    }
    PASS();
}

void T5_no_self_loops() {
    auto g = MemoryGraph::synthetic(50, 25, 25, 64);
    g.build_nsn_edges();
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j) {
            CHECK(g.col_indices[j] != i, "self-loop detected");
        }
    }
    PASS();
}

void T6_cross_modal_bridges() {
    auto g = MemoryGraph::synthetic(100, 50, 50, 64);
    g.build_nsn_edges();

    // For every node, check it has at least one neighbor in every OTHER
    // modality. This is the core guarantee the paper depends on.
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        int32_t my_mod = g.modalities[i];
        bool seen[MOD_COUNT] = { false };
        seen[my_mod] = true;  // trivially satisfied for own modality
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j) {
            int32_t nb = g.col_indices[j];
            seen[g.modalities[nb]] = true;
        }
        for (int32_t m = 0; m < MOD_COUNT; ++m) {
            if (!seen[m]) {
                std::fprintf(stderr,
                    "  node %d (mod %d) has no neighbor of modality %d\n",
                    i, my_mod, m);
                CHECK(false, "missing cross-modal bridge");
            }
        }
    }
    PASS();
}

void T7_degree_bounded() {
    // Small enough that we can count: degree should be bounded by a
    // constant + log(N) from the NSN construction.
    auto g = MemoryGraph::synthetic(400, 200, 200, 64);
    g.build_nsn_edges();

    int32_t max_deg = 0;
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        int32_t d = g.row_offsets[i + 1] - g.row_offsets[i];
        if (d > max_deg) max_deg = d;
    }
    double avg_deg = double(g.num_edges) / g.num_nodes;

    // Hub nodes get extra edges, so the max is higher, but the average
    // should be well under 30 for the NSN parameters we use.
    CHECK(avg_deg < 30.0, "average degree unexpectedly high");
    CHECK(max_deg < 200,  "max degree unexpectedly high");
    PASS();
}

// ─── Persistence tests ──────────────────────────────────────────────

void T8_save_load_roundtrip() {
    auto g = MemoryGraph::synthetic(100, 50, 50, 128);
    g.build_nsn_edges();

    const std::string path = "/tmp/mars_test.bin";
    CHECK(save_graph(g, path), "save_graph failed");

    MemoryGraph loaded;
    CHECK(load_graph(loaded, path), "load_graph failed");

    CHECK(loaded.num_nodes     == g.num_nodes,     "num_nodes mismatch");
    CHECK(loaded.num_edges     == g.num_edges,     "num_edges mismatch");
    CHECK(loaded.embedding_dim == g.embedding_dim, "embedding_dim mismatch");

    // row_offsets
    CHECK(loaded.row_offsets.size() == g.row_offsets.size(),
          "row_offsets size mismatch");
    for (size_t i = 0; i < g.row_offsets.size(); ++i) {
        CHECK(loaded.row_offsets[i] == g.row_offsets[i],
              "row_offsets value mismatch");
    }

    // col_indices
    CHECK(loaded.col_indices.size() == g.col_indices.size(),
          "col_indices size mismatch");
    for (size_t i = 0; i < g.col_indices.size(); ++i) {
        CHECK(loaded.col_indices[i] == g.col_indices[i],
              "col_indices value mismatch");
    }

    // embeddings
    CHECK(loaded.embeddings.size() == g.embeddings.size(),
          "embeddings size mismatch");
    for (size_t i = 0; i < g.embeddings.size(); ++i) {
        CHECK(loaded.embeddings[i] == g.embeddings[i],
              "embeddings value mismatch");
    }

    // modalities
    CHECK(loaded.modalities.size() == g.modalities.size(),
          "modalities size mismatch");
    for (size_t i = 0; i < g.modalities.size(); ++i) {
        CHECK(loaded.modalities[i] == g.modalities[i],
              "modalities value mismatch");
    }

    // timestamps
    CHECK(loaded.timestamps.size() == g.timestamps.size(),
          "timestamps size mismatch");
    for (size_t i = 0; i < g.timestamps.size(); ++i) {
        CHECK(loaded.timestamps[i] == g.timestamps[i],
              "timestamps value mismatch");
    }

    // Clean up
    std::remove(path.c_str());
    PASS();
}

void T9_load_corrupted_file() {
    auto g = MemoryGraph::synthetic(50, 25, 25, 64);
    g.build_nsn_edges();

    const std::string path = "/tmp/mars_test_corrupt.bin";
    CHECK(save_graph(g, path), "save_graph failed");

    // Read the file, flip one byte in the middle, write it back
    std::ifstream in(path, std::ios::binary);
    CHECK(in.is_open(), "failed to open file for corruption");
    in.seekg(0, std::ios::end);
    size_t sz = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    in.close();

    // Corrupt a byte in the data region (past the header)
    size_t corrupt_pos = sz / 2;
    buf[corrupt_pos] ^= 0xFF;

    std::ofstream out(path, std::ios::binary);
    CHECK(out.is_open(), "failed to open file for writing corruption");
    out.write(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    out.close();

    MemoryGraph loaded;
    CHECK(!load_graph(loaded, path), "load_graph should fail on corrupted file");

    std::remove(path.c_str());
    PASS();
}

void T10_load_nonexistent_file() {
    MemoryGraph loaded;
    CHECK(!load_graph(loaded, "/tmp/mars_nonexistent_file_12345.bin"),
          "load_graph should fail on missing file");
    PASS();
}

// ─── Budget tests ─────────────────────────────────────────────────

void T11_budget_basic() {
    auto b = compute_memory_budget(1000, 768);
    CHECK(b.num_nodes == 1000, "wrong num_nodes");
    CHECK(b.embedding_dim == 768, "wrong embedding_dim");
    CHECK(b.total_bytes > 0, "total should be positive");
    CHECK(b.total_graph > b.total_scratch, "graph should dominate at N=1000");
    CHECK(b.embeddings_fp32 == 1000ULL * 768 * 4, "wrong FP32 embedding size");
    CHECK(b.total_bytes >= b.embeddings_fp32, "total must include embeddings");
    PASS();
}

void T12_budget_fp16() {
    auto b32 = compute_memory_budget(5000, 768, 12, 64, 0, false);
    auto b16 = compute_memory_budget(5000, 768, 12, 64, 0, true);
    CHECK(b16.total_bytes > b32.total_bytes, "FP16 should add to total");
    CHECK(b16.embeddings_fp16 == 5000ULL * 768 * 2, "wrong FP16 size");
    CHECK(b32.embeddings_fp16 == 0, "FP32-only should have no FP16");
    PASS();
}

void T13_budget_scaling() {
    auto b1 = compute_memory_budget(1000, 768);
    auto b10 = compute_memory_budget(10000, 768);
    auto b100 = compute_memory_budget(100000, 768);
    CHECK(b10.total_bytes > b1.total_bytes, "should scale with N");
    CHECK(b100.total_bytes > b10.total_bytes, "should scale with N");
    auto b1m = compute_memory_budget(1000000, 768);
    CHECK(b1m.total_gb > 2.9, "1M×768 should be > 2.9 GB");
    CHECK(b1m.total_gb < 10.0, "1M×768 should be < 10 GB");
    PASS();
}

void T14_budget_fits() {
    auto b = compute_memory_budget(1000, 768);
    size_t vram_80gb = 80ULL * 1024 * 1024 * 1024;
    CHECK(budget_fits(b, vram_80gb), "1K nodes should fit in 80 GB");
    CHECK(!budget_fits(b, 100), "1K nodes should NOT fit in 100 bytes");
    PASS();
}

// ─── Streaming tests ──────────────────────────────────────────────

void T15_streaming_insert_basic() {
    StreamingConfig cfg;
    cfg.capacity = 64;
    cfg.embedding_dim = 64;
    StreamingBuffer buf(cfg);

    auto g = MemoryGraph::synthetic(30, 15, 15, 64);
    g.build_nsn_edges();
    int32_t orig_nodes = g.num_nodes;

    std::vector<float> emb(64, 0.1f);
    for (int32_t i = 0; i < 10; ++i) {
        bool ok = buf.insert(orig_nodes + i, MOD_TEXT, 1.0f, emb.data());
        CHECK(ok, "insert should succeed");
    }
    CHECK(buf.staged_count() == 10, "should have 10 staged");

    auto stats = buf.flush(g);
    CHECK(stats.nodes_committed == 10, "should commit 10");
    CHECK(g.num_nodes == orig_nodes + 10, "graph should grow by 10");
    CHECK(stats.edges_added > 0, "should add edges");
    CHECK(buf.staged_count() == 0, "buffer should be empty after flush");
    PASS();
}

void T16_streaming_capacity_limit() {
    StreamingConfig cfg;
    cfg.capacity = 5;
    cfg.embedding_dim = 32;
    cfg.auto_flush = false;
    StreamingBuffer buf(cfg);

    std::vector<float> emb(32, 0.5f);
    for (int32_t i = 0; i < 5; ++i) {
        CHECK(buf.insert(i, MOD_AUDIO, 1.0f, emb.data()), "insert should succeed");
    }
    CHECK(!buf.insert(5, MOD_AUDIO, 1.0f, emb.data()), "insert should fail at capacity");
    PASS();
}

void T17_streaming_flush_preserves_bridges() {
    StreamingConfig cfg;
    cfg.capacity = 128;
    cfg.embedding_dim = 64;
    StreamingBuffer buf(cfg);

    auto g = MemoryGraph::synthetic(60, 30, 30, 64);
    g.build_nsn_edges();

    std::vector<float> emb(64, 0.1f);
    for (int32_t i = 0; i < 30; ++i) {
        int32_t mod = i % MOD_COUNT;
        buf.insert(g.num_nodes + i, mod, 2.0f, emb.data());
    }
    buf.flush(g);

    int32_t old_n = 120;
    for (int32_t i = old_n; i < g.num_nodes; ++i) {
        int32_t my_mod = g.modalities[i];
        bool has_other_mod = false;
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j) {
            if (g.modalities[g.col_indices[j]] != my_mod) {
                has_other_mod = true;
                break;
            }
        }
        CHECK(has_other_mod, "new node should have cross-modal bridge");
    }
    PASS();
}

void T18_embodied_kids_ball_csr() {
    EmbodiedKidsBallCorpus c = EmbodiedKidsBallCorpus::make(220, 96, 2026u);
    CHECK(c.graph.num_nodes == 220, "wrong N");
    CHECK(int32_t(c.episode_ids.size()) == 220, "episode_ids size");
    CHECK(c.graph.modalities.size() == 220u, "modalities size");
    c.graph.build_nsn_edges(6, 0.15);
    for (int32_t i = 0; i < c.graph.num_nodes; ++i) {
        CHECK(c.graph.row_offsets[i] <= c.graph.row_offsets[i + 1],
              "row_offsets not monotone");
    }
    CHECK(c.graph.row_offsets[c.graph.num_nodes] == c.graph.num_edges,
          "row_offsets[N] != num_edges");
    for (int32_t i = 0; i < c.graph.num_nodes; ++i) {
        int32_t prev = -1;
        for (int32_t j = c.graph.row_offsets[i]; j < c.graph.row_offsets[i + 1]; ++j) {
            int32_t n = c.graph.col_indices[j];
            CHECK(n >= 0 && n < c.graph.num_nodes, "neighbor out of range");
            CHECK(n > prev, "neighbors not strictly sorted");
            prev = n;
        }
    }
    PASS();
}

// ─── Test runner ────────────────────────────────────────────────────

int main() {
    std::printf("\n═══════════════════════════════════════════════════\n");
    std::printf("  cuda-multimodal-memory unit tests\n");
    std::printf("═══════════════════════════════════════════════════\n\n");

    T1_synthetic_counts();
    T2_embeddings_normalized();
    T3_csr_format_valid();
    T4_edges_symmetric();
    T5_no_self_loops();
    T6_cross_modal_bridges();
    T7_degree_bounded();
    T8_save_load_roundtrip();
    T9_load_corrupted_file();
    T10_load_nonexistent_file();
    T11_budget_basic();
    T12_budget_fp16();
    T13_budget_scaling();
    T14_budget_fits();
    T15_streaming_insert_basic();
    T16_streaming_capacity_limit();
    T17_streaming_flush_preserves_bridges();
    T18_embodied_kids_ball_csr();

    int32_t passed = 0, failed = 0;
    for (const auto& r : g_results) {
        if (r.passed) {
            std::printf("  ✓ %s\n", r.name.c_str());
            ++passed;
        } else {
            std::printf("  ✗ %s\n    %s\n", r.name.c_str(), r.message.c_str());
            ++failed;
        }
    }
    std::printf("\n  %d passed, %d failed\n\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
