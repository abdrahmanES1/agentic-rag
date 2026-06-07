# Agentic RAG for Moroccan Public Service Question Answering

A multilingual agentic Retrieval-Augmented Generation (RAG) system for Moroccan government administrative procedures, powered by a small language model (gemma4:e4b, 4B parameters) on a single GPU. Handles **Modern Standard Arabic, Moroccan Darija, Moroccan Arabizi, and French**.

> **PFE (Projet de Fin d'Études) — Master's thesis project.**
> Track: LLM Systems • Topic: Agentic RAG with Small Language Models for Multilingual Moroccan Public Service QA.

## 📊 Official Benchmark Results

The official publication results are in [**`benchmarking/results/FINAL/`**](benchmarking/results/FINAL/).

- 📈 [**Comparison table (markdown)**](benchmarking/results/FINAL/final_table.md)
- 📋 [**Plain-text summary**](benchmarking/results/FINAL/final_summary.txt)
- 📂 [**Full FINAL/ folder with README**](benchmarking/results/FINAL/README.md)

### Headline

**v12 ranks #1 on 10 of 34 metrics** in a 7-system comparison, and is the **only system** that scores above zero on three core agentic capabilities:

| Capability | v12 | Best Baseline |
|---|---|---|
| Multi-hop reasoning success | **0.667** | 0.000 |
| Out-of-scope abstention F1 | **0.538** | 0.000 |
| Moroccan Arabizi handling F1 | **0.303** | 0.034 (9× worse) |

## 🏗️ Architecture

The 11-step pipeline:

```
1. Load PDFs (3-tier OCR cascade: PyMuPDF → Qwen VLM → Tesseract)
2. Chunk on article boundaries (per-language)
3. Index (FAISS per-language + unified + BM25Okapi)
─────────────────── KB built ───────────────────
4. Language detection (script-first + LLM, 96.8% accuracy)
5. Query translation (Darija/Arabizi → MSA for retrieval)
6. Question classification (intents, hop_count, OOS, LEGAL)
7. Hybrid retrieval (BM25 + dense BGE-M3 + RRF + CrossEncoder)
8. Agentic planning (PlannerAgent: Plan → Execute → Reflect)
9. Generation (Ollama gemma4:e4b)
10. Verification (NLI mDeBERTa-XNLI + entity matching + CFI)
11. Audit trail (GroundingAudit with claim verdicts)
```

See [`CLAUDE.md`](CLAUDE.md) for the detailed technical architecture.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Ollama running locally with `gemma4:e4b` pulled
- ~12 GB GPU (RTX 3060 or better) — or CPU
- HuggingFace cache space for bge-m3, bge-reranker-v2-m3, mDeBERTa-XNLI

### Install

```bash
git clone https://github.com/abdrahmanES1/agentic-rag.git
cd agentic-rag
pip install -r requirements.txt
```

### Run the API

```bash
python -m api.app
# Wait for /api/status to return ready
curl http://localhost:5000/api/status
```

### Ask a question

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي الوثائق المطلوبة للبطاقة الوطنية؟"}'
```

### Reproduce the benchmark

```bash
export OPENROUTER_API_KEY="sk-or-..."
python benchmarking/benchmark_runner.py --with-gt \
  --judge-url https://openrouter.ai/api/v1 --judge-model openai/gpt-4o-mini \
  --testset benchmarking/benchmark_testset_v1.0.json
```

## 🧪 Key Design Decisions

- **Script-first language detection** (96.8% accuracy via Unicode script + markers) instead of relying on the 4B LLM (30% accuracy on the same task).
- **Intent union** (LLM intents ∪ keyword rules) catches compound dialect questions the LLM under-detects.
- **Two-gate OOS detection** (keyword classifier + calibrated reranker threshold 0.12).
- **Deterministic tools** (retrieve_kb, lookup_article, calculate_deadline, check_eligibility, search_by_amount) — agent plans and synthesizes; tools compute.
- **NLI-based grounding** with mDeBERTa-XNLI (softmax fix built-in) + Composite Fidelity Index = 0.6 × entity_match + 0.4 × relation_match.
- **Multi-source verification**: each claim tested against all chunks via NLI, with entity and citation injection.

## 📚 Citation

If you use this work, please cite the PFE thesis:

```
[Thesis citation TBD upon publication]
```

## 📁 Repository Structure

```
.
├── api/                          Flask API server (/api/ask, /api/retrieve, /api/build-kb, …)
├── pipeline/                     Core pipeline modules
│   ├── pipeline.py               MoroccanRAGPipeline orchestrator
│   ├── language.py               Steps 4-5: detection + translation + intents
│   ├── knowledge_base.py         Steps 1-3: PDF loading, chunking, indexing
│   ├── retrieval.py              Step 7: HybridRetriever
│   ├── generation.py             Steps 8-9: OllamaClient, AgentMemory, ToolRegistry, PlannerAgent
│   ├── verification.py           Step 10: NLIVerifier + 11-layer grounding
│   ├── models.py                 All dataclasses
│   ├── prompts.py                LLM prompt templates
│   └── config.py                 pydantic-settings (all magic numbers here)
├── benchmarking/
│   ├── benchmark_runner.py       Runs all 8 systems on the testset
│   ├── metrics.py                RAGAS, ARES, G-Eval, FActScore, multi-hop, BERTScore, …
│   ├── baselines/                naive_rag, basic_react, adaptive_simple, hyde, flare, crag, self_rag
│   ├── results/FINAL/            ⭐ Official publication results
│   └── benchmark_testset_v1.0.json    The 124-item frozen testset
├── CLAUDE.md                     Detailed architecture documentation
└── README.md                     This file
```

## 📄 License

[License TBD]
