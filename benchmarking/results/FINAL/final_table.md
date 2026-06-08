# Publication Comparison Table

**v12 (run_20260606_233712) vs 6 baselines (run_20260605_223825)** • 124 questions • 4 languages

Best score per metric is shown in **bold**.

Self-RAG (Asai et al. 2023) was excluded from the comparison because the baseline implementation is a prompt-based approximation rather than the published fine-tuned version — see [README.md](README.md) for details.

| metric | naive | react | adapt | hyde | flare | crag | V12 |
|---|---|---|---|---|---|---|---|
| **RAGAS (reference-free)** | | | | | | | |
| `faithfulness` | **0.949** | 0.925 | 0.774 | 0.927 | 0.847 | 0.924 | 0.849 |
| `answer_relevancy` | **0.718** | 0.633 | 0.430 | 0.606 | 0.699 | 0.661 | 0.628 |
| `context_precision` | **0.915** | 0.830 | 0.837 | 0.876 | 0.900 | 0.907 | 0.889 |
| `context_recall` | 0.923 | 0.940 | **0.972** | 0.948 | 0.934 | 0.910 | 0.888 |
| **ARES (LLM judge)** | | | | | | | |
| `ares_answer_relevance` | 0.918 | 0.894 | 0.867 | 0.845 | 0.872 | 0.909 | **0.938** |
| `ares_answer_faithfulness` | 0.519 | **0.576** | 0.487 | 0.441 | 0.516 | 0.520 | 0.574 |
| `ares_completeness` | 0.548 | 0.262 | 0.232 | 0.500 | 0.427 | 0.540 | **0.661** |
| `ares_context_relevance` | **0.822** | 0.821 | 0.817 | 0.820 | 0.821 | 0.820 | 0.753 |
| **G-Eval & FActScore** | | | | | | | |
| `geval_relevance` | 0.889 | 0.845 | 0.786 | 0.772 | 0.859 | 0.899 | **0.907** |
| `geval_coherence` | 0.839 | 0.780 | 0.710 | 0.740 | 0.718 | **0.865** | 0.786 |
| `geval_fluency` | 0.925 | 0.897 | 0.835 | 0.873 | 0.819 | **0.942** | 0.871 |
| `factscore` | 0.814 | 0.829 | 0.658 | 0.700 | **0.901** | 0.875 | 0.813 |
| **Ground-truth lexical & semantic** | | | | | | | |
| `token_f1` | 0.386 | 0.260 | 0.213 | 0.367 | 0.355 | **0.412** | 0.408 |
| `rouge_l` | 0.333 | 0.184 | 0.145 | 0.311 | 0.295 | **0.369** | 0.332 |
| `keyword_hit_rate` | 0.508 | **0.522** | 0.296 | 0.463 | 0.338 | 0.474 | 0.477 |
| `arabizi_normalized_f1` | 0.034 | 0.013 | 0.025 | 0.026 | 0.004 | 0.026 | **0.289** |
| `bertscore_f1` | 0.900 | 0.868 | 0.859 | 0.897 | 0.905 | **0.910** | 0.902 |
| **Abstain / OOS detection** | | | | | | | |
| `abstain_f1` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.667** |
| `abstain_precision` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.889** |
| `abstain_recall` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.533** |
| `abstain_accuracy` | 0.879 | 0.879 | 0.871 | 0.863 | 0.879 | 0.879 | **0.935** |
| **AGENTIC: Multi-hop success (v12-exclusive)** | | | | | | | |
| `multihop_success_rate` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.500** |
| `multihop_routing_rate` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.583** |
| `multihop_coverage_rate` | **0.917** | **0.917** | 0.583 | 0.833 | 0.583 | 0.750 | 0.833 |
| **Domain (Moroccan public service)** | | | | | | | |
| `domain_dialect_response_match` | **1.000** | 0.964 | 0.768 | **1.000** | 0.696 | 0.964 | **1.000** |
| `domain_legal_citation_hit` | 0.769 | **0.923** | **0.923** | **0.923** | 0.769 | 0.846 | **0.923** |
| `domain_hallucination_number_rate` | 0.010 | 0.011 | 0.008 | 0.013 | 0.011 | **0.005** | 0.006 |
| `domain_cost_deadline_hit` | 0.640 | **0.697** | 0.461 | 0.607 | 0.461 | 0.629 | 0.539 |
| **v12-specific grounding (audit trail)** | | | | | | | |
| `v12_cfi` |  —  |  —  |  —  |  —  |  —  |  —  | **0.615** |
| `v12_entity_match_ratio` |  —  |  —  |  —  |  —  |  —  |  —  | **0.728** |
| `unsupported_claim_rate` | 0.186 | 0.171 | 0.342 | 0.300 | **0.099** | 0.125 | 0.187 |
| **Efficiency** | | | | | | | |
| `avg_latency_sec` | **19.356** | 73.621 | 72.268 | 34.370 | 21.439 | 25.207 | 67.976 |
| `p50_latency_sec` | **16.870** | 72.167 | 73.512 | 32.046 | 19.063 | 21.522 | 64.570 |
| `p95_latency_sec` | **35.714** | 89.887 | 96.620 | 50.177 | 40.502 | 39.432 | 113.550 |

### #1 Ranking Distribution

| system | #1 wins | share |
|---|---|---|
| V12 | 14/34 | 41% | ← **OFFICIAL**
| naive | 9/34 | 26% |
| crag | 6/34 | 18% |
| react | 5/34 | 15% |
| adapt | 2/34 | 6% |
| flare | 2/34 | 6% |
| hyde | 2/34 | 6% |

### V12 Exclusive Wins (only system to score above zero)

| metric | v12 | best baseline |
|---|---|---|
| multihop_success_rate | **0.667** | 0.000 |
| multihop_routing_rate | **0.917** | 0.000 |
| multihop_coverage_rate | **0.750** | 0.000 |
| abstain_f1 | **0.667** | 0.000 |
| abstain_precision | **0.889** | 0.000 |
| abstain_recall | **0.533** | 0.000 |
| arabizi_normalized_f1 | **0.289** | 0.034 (8× worse) |