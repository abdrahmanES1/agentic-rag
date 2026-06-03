# -*- coding: utf-8 -*-
"""
Pipeline configuration — loaded from .env file or environment variables.
All magic numbers live here; the rest of the code uses `settings.*`.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file's parent (project root)
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    pdf_dir: str = Field(default="./docs")
    index_dir: str = Field(default="./indexes")
    audit_log_dir: str = Field(default="./audit_logs")
    doc_registry_file: str = Field(default="./indexes/doc_registry.json")

    # ── Ollama / LLM ─────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434/v1")
    ollama_api_key: str = Field(default="ollama")
    generator_model: str = Field(default="gemma4:e4b")
    api_timeout: int = Field(default=60)
    api_max_retries: int = Field(default=3)
    api_retry_delay: float = Field(default=2.0)
    temperature: float = Field(default=0.1)
    max_new_tokens: int = Field(default=2000)
    # CRITICAL: Ollama default 2048 causes silent truncation for long RAG contexts
    ollama_num_ctx: int = Field(default=8192)
    # think=False prevents Gemma4 thinking mode which returns content=None
    ollama_think: bool = Field(default=False)
    ollama_repeat_penalty: float = Field(default=1.05)
    ollama_keep_alive: str = Field(default="10m")

    # ── Static models (local) ─────────────────────────────────────────────────
    embedding_model: str = Field(default="BAAI/bge-m3")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    enable_reranker: bool = Field(default=True)
    nli_model: str = Field(
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    # "auto" → use CUDA if available, else CPU. Override with "cpu" or "cuda".
    device: str = Field(default="auto")
    # Load the embedding/reranker/NLI models in fp16 on CUDA (~halves their VRAM,
    # ~6 GB → ~3 GB) so a co-resident local LLM (Ollama) gets more GPU. fp16
    # inference is numerically negligible for ranking/NLI. Ignored on CPU.
    model_fp16: bool = Field(default=False)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=400)
    chunk_overlap: int = Field(default=50)
    retrieve_top_k: int = Field(default=10)
    reranker_top_n: int = Field(default=20)
    compress_top_n: int = Field(default=5)
    bm25_weight: float = Field(default=0.4)
    dense_weight: float = Field(default=0.6)
    outscope_score_threshold: float = Field(default=0.002)
    # CrossEncoder reranker score below which retrieval is treated as out-of-scope
    # (no chunk actually answers the query). Better signal than the RRF rank for
    # topically-related-but-unanswerable questions. Calibrate against the abstain set.
    outscope_reranker_threshold: float = Field(default=0.0)
    dense_min_score_unified: float = Field(default=0.20)
    dense_min_score_per_lang: float = Field(default=0.25)

    # ── Language detection ────────────────────────────────────────────────────
    arabic_script_min: float = Field(default=0.55)
    french_latin_min: float = Field(default=0.65)
    lang_confidence_min: float = Field(default=0.60)
    darija_marker_min: int = Field(default=2)
    arabizi_marker_min: int = Field(default=2)

    # ── Grounding thresholds (government-grade strict) ────────────────────────
    nli_grounding_threshold: float = Field(default=0.75)
    grounding_threshold: float = Field(default=0.30)
    claim_grounded_ratio: float = Field(default=0.80)
    entity_exact_match: bool = Field(default=True)
    ambiguous_nli_threshold: float = Field(default=0.65)
    # Composite Fidelity Index weights (must sum to 1.0). Claim grounding is the
    # primary faithfulness signal; entity + relation are structural modulators.
    # The entity term is renormalized away when an answer has no extractable
    # entities, so CFI no longer defaults vacuously to ~1.0.
    cfi_weight_claim: float = Field(default=0.50)
    cfi_weight_entity: float = Field(default=0.30)
    cfi_weight_relation: float = Field(default=0.20)

    # ── Agent & Memory ────────────────────────────────────────────────────────
    max_agent_steps: int = Field(default=4)
    memory_max_chunks: int = Field(default=12)
    memory_min_score: float = Field(default=0.001)
    memory_max_per_src: int = Field(default=5)

    # ── Feature flags ─────────────────────────────────────────────────────────
    enable_darija: bool = Field(default=True)
    enable_arabizi: bool = Field(default=True)
    enable_query_translation: bool = Field(default=True)
    enable_llm_judge: bool = Field(default=False)
    enable_audit_trail: bool = Field(default=True)
    enable_entity_verification: bool = Field(default=True)
    enable_chain_verification: bool = Field(default=True)
    enable_contextual_retrieval: bool = Field(default=False)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    min_digital_chars: int = Field(default=100)
    min_page_words: int = Field(default=20)
    min_chunk_words: int = Field(default=10)
    tesseract_lang: str = Field(default="ara+fra")
    ocr_scale_factor: float = Field(default=2.0)
    use_docling: bool = Field(default=False)

    # ── Flask API ─────────────────────────────────────────────────────────────
    flask_debug: bool = Field(default=False)
    flask_port: int = Field(default=5000)
    flask_host: str = Field(default="0.0.0.0")


settings = Settings()

# Ensure required directories exist on import
for _d in [settings.pdf_dir, settings.index_dir, settings.audit_log_dir]:
    Path(_d).mkdir(parents=True, exist_ok=True)
