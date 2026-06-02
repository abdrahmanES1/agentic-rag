# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Moroccan government-document RAG system (Arabic + French). Handles Darija (Moroccan colloquial Arabic) and Arabizi (romanized Darija). The pipeline is agentic: simple questions get single-hop retrieval; complex questions get multi-step planning with tool calls and self-reflection.

## Running the System

**Start the API server:**
```
python -m api.app
```
Flask starts immediately; models load in a background thread. Poll `GET /api/status` to know when it's ready. Default: `http://localhost:5000`.

**Build or rebuild the knowledge base only:**
```python
pipe = MoroccanRAGPipeline()
pipe.setup()
pipe.build_knowledge_base(force_rebuild=True)   # force_rebuild=False loads from cache
```
PDFs are read from `settings.pdf_dir` (default `./docs`). Indexes are saved to `settings.index_dir` (default `./indexes`).

**Run benchmarks:**
```
# API must be running first
python benchmarking/benchmark_runner.py
python benchmarking/benchmark_runner.py --quick          # first 10 questions only
python benchmarking/benchmark_runner.py --resume         # continue after a crash
python benchmarking/benchmark_runner.py --baselines naive_rag,hyde
python benchmarking/benchmark_runner.py --no-ragas --no-ares   # skip LLM-judge metrics
python benchmarking/benchmark_runner.py --with-gt        # enable ground-truth metrics
```
Can be run from `benchmarking/` or from the repo root — the script inserts the repo root into `sys.path` automatically.

**Debug a query with full trace:**
```python
from debug_pipeline import PipelineDebugger
debugger = PipelineDebugger(pipeline)
report = debugger.debug("ما هي الوثائق المطلوبة؟")
```

## Module Layout

```
api/
  app.py         — Flask factory; sets KMP_DUPLICATE_LIB_OK before any torch/faiss import
  routes.py      — all HTTP endpoints (/api/ask, /api/retrieve, /api/build-kb, etc.)
  schemas.py     — dict serializers for PipelineResult and RetrievalResult

pipeline/
  config.py      — pydantic-settings Settings; every magic number is here; reads .env
  models.py      — all dataclasses (Chunk, ScoredChunk, RetrievalResult, PipelineResult,
                   ExecutionTrace, ToolCall, AgentState, GroundingAudit, ...)
  knowledge_base.py  — Steps 1-3: PDF loading, chunking, BM25/FAISS indexing
  language.py    — Steps 4-5: 5-signal language detection + MSA query translation
  retrieval.py   — Steps 7-8: HybridRetriever (BM25 + dense RRF + CrossEncoder rerank)
  generation.py  — Steps 8-9: OllamaClient, AgentMemory, ToolRegistry, PlannerAgent
  verification.py — Step 10: NLIVerifier (mDeBERTa-XNLI), entity extraction, CFI score
  pipeline.py    — Step orchestrator: MoroccanRAGPipeline wires all modules together
  prompts.py     — LLM prompt templates

benchmarking/
  benchmark_runner.py — compares 8 pipelines; CheckpointManager for crash-safe runs
  metrics.py          — RAGAS, ARES-style LLM judge, BERTScore, ROUGE-L, CFI, etc.
  shared.py           — OllamaClient + api_retrieve() / api_ask() helpers for baselines
  api_v2.py           — legacy standalone baseline used for comparison

debug_pipeline.py  — PipelineDebugger + three-tier SOTA OCR loader (pymupdf4llm → marker-pdf → fitz+Tesseract)
```

## 11-Step Pipeline Architecture

**Offline (KB build — Steps 1-3):**
1. **Load PDFs** — three-tier OCR cascade: PyMuPDF digital extraction → Qwen VLM OCR → Tesseract fallback. Each page picks the tier with sufficient confidence.
2. **Chunk** — language-aware splitting on article boundaries (`المادة/الفصل` for Arabic, `Article/Chapitre` for French); 400-token chunks with 50-token overlap.
3. **Index** — per-language FAISS flat indexes (Arabic, French) plus a unified cross-language index; parallel BM25Okapi indexes with language-specific tokenizers (diacritic stripping, hamza normalization for Arabic; accent folding, stopword removal for French).

**Online (query — Steps 4-11):**
4. **Language detection** — 5-signal ensemble: Darija markers → Arabizi markers → script ratio → domain keyword lists → ML detectors (Lingua + langdetect + langid). Falls through to the next signal when confidence is low.
5. **Question classification** — flags: SIMPLE / MULTIHOP / LEGAL / OUTSCOPE; detects intents (DOCUMENTS, PROCEDURE, COST, DEADLINE, ELIGIBILITY, LEGAL, COMPARISON, OUT_OF_SCOPE); estimates hop_count.
6. **Query translation** — Darija/Arabizi queries are translated to MSA before retrieval; original query is preserved in `PipelineResult.original_query`.
7. **Hybrid retrieval** — BM25 + dense (BGE-M3) fused with RRF (k=60), then CrossEncoder (BGE-reranker-v2-m3) reranking. Returns a `RetrievalResult` with per-chunk BM25/dense/RRF/reranker scores — no context is consumed silently.
8. **Agentic planning** — MULTIHOP questions: PlannerAgent generates a JSON plan of sub-questions mapped to tools; each ToolCall records its retrieved contexts into `ExecutionTrace` so benchmarks can see ALL contexts across hops.
9. **Generation** — OllamaClient (OpenAI-SDK pointing at Ollama). Context window is 8192 tokens (avoid Ollama's default 2048 silent truncation). Gemma4 thinking mode is disabled (`think=False`) to prevent `content=None` responses.
10. **Verification** — NLIVerifier (mDeBERTa-XNLI) applies softmax to raw logits before reading the entailment index. Computes per-claim grounding, entity extraction/matching, and a Composite Fidelity Index (CFI = 0.6 × entity_match + 0.4 × relation_match).
11. **Audit trail** — `GroundingAudit` is written to `./audit_logs/` with timestamps, claim-level verdicts, entity matches, and warnings.

`PipelineResult.to_ragas_contexts()` returns the union of `retrieval.contexts` and `execution_trace.all_contexts` — the correct contexts list for RAGAS faithfulness/precision on multi-hop queries.

## Key Design Decisions

**OpenMP / FAISS import order on Windows:** `faiss` must be imported before `sentence_transformers`/`torch`. Both ship `libiomp5md.dll`; importing FAISS first lets it claim the OpenMP slot so torch's copy is silently skipped. `pipeline/pipeline.py` has a `import faiss  # noqa: F401` at the top for this reason.

**`KMP_DUPLICATE_LIB_OK=TRUE`:** Set in `api/app.py` before any other import as a secondary safeguard against the OpenMP duplicate-library segfault on Windows. Any script that imports pipeline modules directly (e.g., standalone benchmarks, one-off scripts) must set this env var before importing torch or faiss.

**Device auto-detection:** `settings.device = "auto"` (default) calls `torch.cuda.is_available()` at startup and selects CUDA or CPU automatically. Override by setting `DEVICE=cpu` or `DEVICE=cuda` in `.env`.

**Two-tier benchmark metrics:**
- *Always-on (reference-free):* RAGAS faithfulness / answer_relevancy / context_precision; ARES-style LLM judge (via Ollama, no Stanford install); latency percentiles (avg/p50/p95/p99); v12-specific CFI and claim_grounded_ratio.
- *Needs ground truth (`--with-gt`):* exact_match, token_F1, ROUGE-L, BERTScore, context_recall, answer_correctness, keyword_hit_rate, abstain accuracy/precision/recall/F1, per-category and per-language breakdowns, win-rate matrix.

**Benchmark checkpoint/resume:** `CheckpointManager` writes results atomically after each baseline finishes (`raw_<baseline>.json`) and after each individual v12 question (`raw_v12_pipeline.json`). A `manifest.json` tracks which baselines have results and scores. `--resume` skips completed baselines and restores partial v12 progress. Results land in `benchmarking/results/run_<timestamp>/`.

**RetrievalResult design:** The original monolith consumed retrieval context inside `_generate()` and never exposed it to callers — making RAGAS evaluation impossible. `RetrievalResult` was introduced to surface chunks, scores, formatted context, and retrieval metadata to every caller (API, benchmarks, audit). Similarly, `ExecutionTrace` collects every `ToolCall`'s contexts so multi-hop RAGAS evaluation sees all retrieved evidence, not just the first hop.

## Configuration

All settings live in `pipeline/config.py` as a `pydantic-settings` `Settings` object loaded from `.env` in the repo root. Key overrides:

| Env var | Default | Notes |
|---|---|---|
| `GENERATOR_MODEL` | `gemma4:e4b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | LM Studio / Ollama endpoint |
| `OLLAMA_NUM_CTX` | `8192` | Ollama context window (avoid default 2048) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace model ID |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace model ID |
| `NLI_MODEL` | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | For verification |
| `PDF_DIR` | `./docs` | Source PDFs |
| `INDEX_DIR` | `./indexes` | FAISS + BM25 + pickle cache |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `ENABLE_RERANKER` | `true` | Set to `false` to skip CrossEncoder |

GPU note: install torch with the correct CUDA wheel separately (see `requirements.txt` header); `faiss-cpu` is the default — replace with `faiss-gpu` on Linux for GPU indexing.
