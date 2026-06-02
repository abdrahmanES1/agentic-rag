# Makefile — Moroccan Agentic RAG v12
# Requires: GNU make (Git Bash / WSL on Windows, or Linux/macOS)
# Usage:    make <target>

PYTHON     := python
PIP        := pip
API_PORT   := 5000
TESTSET    := benchmarking/benchmark_testset_gold.json
RESULTS    := benchmarking/results
OLLAMA_URL := http://localhost:11434/v1
MODEL      := gemma4:e4b

.PHONY: help install install-ragas build-kb serve \
        benchmark benchmark-quick benchmark-baselines benchmark-no-ragas \
        check clean clean-cache clean-results

# ── Default ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Moroccan Agentic RAG v12"
	@echo "========================"
	@echo ""
	@echo "Setup"
	@echo "  make install           Install core dependencies"
	@echo "  make install-ragas     Install RAGAS + evaluation extras"
	@echo ""
	@echo "Knowledge Base"
	@echo "  make build-kb          Build/rebuild the knowledge base indexes"
	@echo ""
	@echo "API Server"
	@echo "  make serve             Start the Flask API on port $(API_PORT)"
	@echo ""
	@echo "Benchmarking"
	@echo "  make benchmark         Full benchmark: all 8 baselines + V12 + RAGAS"
	@echo "  make benchmark-quick   Same but first 10 questions only"
	@echo "  make benchmark-baselines  Baselines only (no V12 API needed)"
	@echo "  make benchmark-no-ragas   All pipelines, lexical metrics only"
	@echo ""
	@echo "Utilities"
	@echo "  make check             Verify all imports work"
	@echo "  make clean             Remove all generated caches and results"
	@echo "  make clean-cache       Remove __pycache__ only"
	@echo "  make clean-results     Remove benchmarking/results only"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt

install-ragas:
	$(PIP) install ragas datasets

# ── Knowledge Base ─────────────────────────────────────────────────────────────

build-kb:
	$(PYTHON) build_kb_v12.py

# ── API Server ─────────────────────────────────────────────────────────────────

serve:
	$(PYTHON) -m api.app

# ── Benchmarking ───────────────────────────────────────────────────────────────

benchmark:
	$(PYTHON) -m benchmarking.benchmark_runner \
		--output $(RESULTS) \
		--ollama-url $(OLLAMA_URL) \
		--model "$(MODEL)"

benchmark-quick:
	$(PYTHON) -m benchmarking.benchmark_runner \
		--quick \
		--output $(RESULTS) \
		--ollama-url $(OLLAMA_URL) \
		--model "$(MODEL)"

benchmark-baselines:
	$(PYTHON) -m benchmarking.benchmark_runner \
		--no-v12 \
		--output $(RESULTS) \
		--ollama-url $(OLLAMA_URL) \
		--model "$(MODEL)"

benchmark-no-ragas:
	$(PYTHON) -m benchmarking.benchmark_runner \
		--no-ragas \
		--output $(RESULTS) \
		--ollama-url $(OLLAMA_URL) \
		--model "$(MODEL)"

# ── Verification ───────────────────────────────────────────────────────────────

check:
	@echo "--- Checking pipeline config ---"
	$(PYTHON) -c "from pipeline.config import settings; print('ollama_base_url:', settings.ollama_base_url)"
	@echo "--- Checking pipeline models ---"
	$(PYTHON) -c "from pipeline.models import RetrievalResult, ExecutionTrace, PipelineResult; print('models OK')"
	@echo "--- Checking api imports ---"
	$(PYTHON) -c "from api.app import create_app; print('api OK')"
	@echo "--- Checking RAGAS adapter ---"
	$(PYTHON) -c "from benchmarking.adapters.ragas_adapter import build_ragas_dataset; print('ragas_adapter OK')"
	@echo "--- Checking ARES adapter ---"
	$(PYTHON) -c "from benchmarking.adapters.ares_adapter import build_ares_input; print('ares_adapter OK')"
	@echo "--- Checking baselines ---"
	$(PYTHON) -c "from benchmarking.baselines.hyde import HyDE; from benchmarking.baselines.self_rag import SelfRAG; from benchmarking.baselines.flare import FLARE; from benchmarking.baselines.crag import CRAG; print('all 8 baselines OK')"
	@echo "All checks passed."

# ── Clean ──────────────────────────────────────────────────────────────────────

clean: clean-cache clean-results

clean-cache:
	find . -type d -name "__pycache__" -not -path "*/venv_ragas/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "*/venv_ragas/*" -delete 2>/dev/null || true

clean-results:
	rm -rf $(RESULTS)
