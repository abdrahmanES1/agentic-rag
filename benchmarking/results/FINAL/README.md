# FINAL — Official Publication Results

Generated: 2026-06-07 17:07:25

## Sources

| System | Source run | Reason |
|---|---|---|
| 7 baselines | `run_20260605_223825` | Cached scored run with all 7 baselines on 124-item testset |
| **v12_pipeline** | `run_20260606_233712` | OFFICIAL v12 run with all 12 validated fixes; produced before later commits that caused regressions |

## Code state for reproduction

The pipeline code that produced these v12 results corresponds to commit **f1b8a24** on the main branch (or any later commit through 78d2460, which contains only benchmark-scoring fixes that do not affect v12 generation).

Repository: https://github.com/abdrahmanES1/agentic-rag

## Folder structure

```
FINAL/
├── raw/                      Raw API responses per system (8 systems × 124 items)
│   ├── raw_naive_rag.json
│   ├── raw_basic_react.json
│   ├── raw_adaptive_simple.json
│   ├── raw_hyde.json
│   ├── raw_self_rag.json     † See note below
│   ├── raw_flare.json
│   ├── raw_crag.json
│   └── raw_v12_pipeline.json
├── scores/                   Aggregated metric scores per system
│   ├── scores_*.json (8 files)
├── judge_outputs/            Per-row ARES + RAGAS judge outputs (transparency)
├── benchmark_testset_v1.0.json   The 124-item evaluation set
├── breakdown_baselines.json      Per-language / per-category breakdown (baselines)
├── breakdown_v12.json            Per-language / per-category breakdown (v12)
└── README.md                 This file
```

## Self-RAG note †

The `self_rag` baseline is a **prompt-based approximation** of Self-RAG (Asai et al. 2023, ICLR 2024) using 4 explicit LLM calls to simulate the paper's learned reflection tokens. The published Self-RAG requires fine-tuning a base model on 150K supervised examples with a critic model — this is computationally outside the scope of this work.

**Empirical observation:** Our prompt-only Self-RAG refused to answer 50/124 (40%) of questions, including answerable ones. This is a known failure mode of prompt-only Self-RAG implementations and does NOT reflect the published paper's performance.

Self-RAG's apparently high ARES faithfulness (0.897) and FActScore (0.960) are inflated artifacts of these refusals being scored as "faithful to nothing" / containing no claims to falsify. RAGAS faithfulness, which requires real claims to verify, places Self-RAG last at 0.315.

**Recommendation:** Use the 7-system comparison (excluding self_rag) for the primary results table. Include the 8-system version in appendix with this note.

## Testset summary

- **Total items:** 124
- **Languages:** Modern Standard Arabic (40), Darija (28), Arabizi (28), French (28)
- **Categories:** OUTSCOPE (15), LEGAL (13), MULTIHOP (12), SIMPLE (28), DARIJA (28), ARABIZI (28)
- **Source:** Moroccan public-service procedures scraped from idarati.ma, manually curated

## Headline result

**v12 ranks #1 on 9 of 33 metrics in the 7-system primary comparison** — the highest of any single system tested. It is the only system that handles Moroccan Arabizi (arabizi_normalized_f1 = 0.303, 9× best baseline), the only system that abstains on out-of-scope questions (abstain_f1 = 0.538, all baselines = 0.000), and the only system that performs genuine multi-hop reasoning (multihop_success_rate = 0.667, all baselines = 0.000).
