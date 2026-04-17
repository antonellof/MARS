// ============================================================================
//  demos/embodied_scene/demo.cu
//
//  Embodied multimodal scenario: children playing with a ball (synthetic).
//  Corpus from EmbodiedKidsBallCorpus::make(); queries use IMAGE-node
//  embeddings; success = same-episode TEXT + AUDIO in ranked results (BFS +
//  bridges). Primary metric scans full host-return window (max_results_returned);
//  secondary reports strict top-15 slice for UI-style comparison.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "memory_graph.h"
#include "memory_cuda.cuh"
#include "frame_timer.h"

using namespace demo;

static bool episode_cross_modal_hit(const std::vector<QueryResult>& results,
                                    int32_t query_episode,
                                    const std::vector<int32_t>& episode_ids,
                                    int32_t top_check) {
    bool has_text  = false;
    bool has_audio = false;
    const int32_t n = std::min(top_check, static_cast<int32_t>(results.size()));
    for (int32_t i = 0; i < n; ++i) {
        int32_t nid = results[i].node_id;
        if (nid < 0 || nid >= static_cast<int32_t>(episode_ids.size()))
            continue;
        if (episode_ids[nid] != query_episode)
            continue;
        if (results[i].modality == MOD_TEXT)  has_text  = true;
        if (results[i].modality == MOD_AUDIO) has_audio = true;
    }
    return has_text && has_audio;
}

int main(int argc, char** argv) {
    constexpr int32_t FRAME_RATE_HZ   = 15;
    constexpr int32_t NUM_FRAMES      = 450;   // 30 s at 15 Hz
    constexpr int32_t CORPUS_SIZE     = 3600;
    constexpr int32_t EMBEDDING_DIM = 768;
    constexpr double  DEADLINE_P99_MS = 20.0;
    constexpr int32_t TOP_K              = 15;
    constexpr int32_t MAX_RESULTS_HOST   = 96;  // compacted list for episode metric
    constexpr int32_t QUERY_CTX_MAX_K    = 64;  // device buffers (>= TOP_K)

    int32_t corpus_size = CORPUS_SIZE;
    bool use_episode_scope = false;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--scope=episode") == 0 || std::strcmp(a, "--episode-scoped") == 0) {
            use_episode_scope = true;
        } else if (std::strcmp(a, "--scope=global") == 0) {
            use_episode_scope = false;
        } else if (a[0] != '-') {
            corpus_size = std::atoi(a);
        }
    }
    if (corpus_size < 10) corpus_size = 10;

    DemoReport report;
    report.demo_name       = "embodied_scene";
    report.scenario        =
        "Kids + ball: IMAGE probes; same-episode TEXT+AUDIO in compacted rank window";
    report.frame_rate_hz   = FRAME_RATE_HZ;
    report.deadline_p99_ms = DEADLINE_P99_MS;
    report.corpus_size     = corpus_size;
    report.embedding_dim   = EMBEDDING_DIM;
    report.num_frames      = NUM_FRAMES;

    EmbodiedKidsBallCorpus corp = EmbodiedKidsBallCorpus::make(
        corpus_size, EMBEDDING_DIM, /*seed=*/2026u);
    MemoryGraph& graph = corp.graph;
    graph.build_nsn_edges(6, 0.15);
    HostEpisodeCSR ep_csr = build_episode_csr(corp.episode_ids, graph.num_nodes);
    std::fprintf(stderr, "Embodied kids-ball graph: %d nodes, %d undirected edges\n",
                 graph.num_nodes, graph.num_edges / 2);

    std::vector<int32_t> image_nodes;
    image_nodes.reserve(size_t(graph.num_nodes) / 3u);
    for (int32_t i = 0; i < graph.num_nodes; ++i)
        if (graph.modalities[i] == MOD_IMAGE)
            image_nodes.push_back(i);
    if (image_nodes.empty()) {
        std::fprintf(stderr, "No IMAGE nodes in corpus\n");
        return 1;
    }

    DeviceMemoryGraph dg = upload_to_device(graph);
    upload_episode_ids(dg, corp.episode_ids.data(), graph.num_nodes);
    upload_episode_csr(dg, ep_csr);
    QueryContext ctx = create_query_context(graph.num_nodes, EMBEDDING_DIM,
                                            QUERY_CTX_MAX_K);

    RetrievalConfig cfg;
    cfg.top_k                   = TOP_K;
    cfg.max_results_returned    = MAX_RESULTS_HOST;
    cfg.bfs_max_hops            = 2;
    cfg.time_decay_lambda       = 8e-6f;
    cfg.bfs_score_decay         = 0.55f;
    cfg.modality_filter         = -1;
    cfg.retrieval_scope         = use_episode_scope ? RetrievalScope::EpisodeScoped
                                                     : RetrievalScope::Global;
    // Paired-probes A100 sweeps (results/iteration_steps/paired_wide_10k and
    // .../paired_wide_1m) show no measurable hit-rate change for any boost in
    // [0, 2.0] at this metric (kids-ball IMAGE -> same-episode TEXT+AUDIO),
    // while the boost adds a small constant cost. Default to 0.0; keep the
    // hook for downstream workloads where same-episode emphasis matters.
    cfg.episode_same_boost      = 0.0f;

    std::mt19937 rng(914u);
    std::uniform_int_distribution<size_t> pick_img(0, image_nodes.size() - 1);

    int32_t hits_compact = 0;
    int32_t hits_top15   = 0;
    int32_t total = 0;

    for (int32_t frame = 0; frame < NUM_FRAMES; ++frame) {
        const int32_t node = image_nodes[pick_img(rng)];
        const float* qemb  = &graph.embeddings[size_t(node) * EMBEDDING_DIM];
        float qts = graph.timestamps[node] + float(frame) * 0.001f;

        const int32_t ep = corp.episode_ids[node];
        cfg.query_episode_id = ep;
        if (use_episode_scope) {
            cfg.query_episode_member_begin =
                ep_csr.ep_csr_offsets[static_cast<size_t>(ep)];
            cfg.query_episode_member_count =
                ep_csr.ep_csr_offsets[static_cast<size_t>(ep) + 1u] -
                ep_csr.ep_csr_offsets[static_cast<size_t>(ep)];
        } else {
            cfg.query_episode_member_begin = 0;
            cfg.query_episode_member_count = 0;
        }

        RetrievalStats stats;
        auto results = query_memory_fast(dg, ctx, qemb, qts, cfg, stats);
        report.latencies.add(double(stats.gpu_ms_total));
        ++total;

        const int32_t wfull = static_cast<int32_t>(results.size());
        if (episode_cross_modal_hit(results, ep, corp.episode_ids, wfull))
            ++hits_compact;
        if (episode_cross_modal_hit(results, ep, corp.episode_ids, TOP_K))
            ++hits_top15;
    }

    report.add_metric("episode_cross_modal_hit_rate_compact",
                      total > 0 ? double(hits_compact) / double(total) : 0.0);
    report.add_metric("episode_cross_modal_hit_rate_top15",
                      total > 0 ? double(hits_top15) / double(total) : 0.0);
    report.add_metric("total_queries", double(total));

    report.print_summary();
    report.print_json();

    destroy_query_context(ctx);
    free_device(dg);
    return report.passed() ? 0 : 1;
}
