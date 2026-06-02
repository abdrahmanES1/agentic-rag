# -*- coding: utf-8 -*-
"""
Step orchestrator — MoroccanRAGPipeline.

Thin coordinator that wires together the focused pipeline modules:
  4. Language detection & query translation  (pipeline.language)
  5. Question classification                 (pipeline.language)
  6. Hybrid retrieval                        (pipeline.retrieval)
  7. Agentic generation                      (pipeline.generation)
  8. 11-layer verification                   (pipeline.verification)

Returns PipelineResult with retrieval: RetrievalResult and
execution_trace: ExecutionTrace so callers (API, benchmarks) can access
ALL contexts used across the full pipeline for RAGAS/ARES evaluation.
"""

import logging
import time
from typing import Optional

# faiss MUST be imported before sentence_transformers/torch on Windows.
# Both ship libiomp5md; importing faiss first lets it claim the OpenMP slot
# so torch's copy is silently ignored (KMP_DUPLICATE_LIB_OK=TRUE in app.py
# prevents the crash; this order is an additional safeguard).
import faiss  # noqa: F401 — side-effect import: sets OpenMP affinity first

from sentence_transformers import CrossEncoder, SentenceTransformer

from pipeline.config import settings
from pipeline.generation import AgentMemory, OllamaClient, PlannerAgent
from pipeline.knowledge_base import KnowledgeBase
from pipeline.language import classify_question, detect_language, translate_to_msa
from pipeline.models import PipelineResult, RetrievalResult, ScoredChunk
from pipeline.retrieval import HybridRetriever
from pipeline.verification import NLIVerifier, verify_output

log = logging.getLogger("MoroccanRAG")


def _resolve_device() -> str:
    """Return 'cuda' if a GPU is available and settings.device is 'auto', else 'cpu'."""
    if settings.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                log.info(f"  GPU detected: {gpu} — using CUDA")
                return "cuda"
        except Exception:
            pass
        log.info("  No GPU detected — using CPU")
        return "cpu"
    return settings.device


class MoroccanRAGPipeline:
    """
    End-to-end Moroccan RAG pipeline.

    Usage:
        pipe = MoroccanRAGPipeline()
        pipe.setup()
        pipe.build_knowledge_base()
        result = pipe.ask("ما هي الوثائق المطلوبة للبطاقة الوطنية؟")
        # result.retrieval.contexts → List[str] for RAGAS
        # result.execution_trace.all_contexts → ALL contexts for RAGAS multi-hop
    """

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self.ollama: Optional[OllamaClient] = None
        self.kb: Optional[KnowledgeBase] = None
        self._retriever: Optional[HybridRetriever] = None
        self._nli: Optional[NLIVerifier] = None
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        log.info("  Initializing MoroccanRAGPipeline...")
        device = _resolve_device()

        self.embedding_model = SentenceTransformer(settings.embedding_model, device=device)
        log.info(f"  Embedding model loaded: {settings.embedding_model} ({device})")

        if settings.enable_reranker:
            try:
                self.reranker = CrossEncoder(settings.reranker_model, device=device, max_length=512)
                log.info(f"  Reranker loaded: {settings.reranker_model} ({device})")
            except Exception as exc:
                log.warning(f"  Reranker not loaded: {exc}")
                self.reranker = None
        else:
            log.info("  Reranker disabled (ENABLE_RERANKER=false)")

        self.ollama = OllamaClient()
        log.info(f"  Ollama client: {settings.ollama_base_url} | model={settings.generator_model}")

        self._nli = NLIVerifier()
        self._nli._load(device=device)

        self._ready = True
        log.info("  Pipeline ready.")

    # ── Knowledge base ────────────────────────────────────────────────────────

    def build_knowledge_base(
        self,
        pdf_dir: str = None,
        force_rebuild: bool = False,
        contextual_retrieval: bool = None,
    ) -> None:
        """
        Build or load the knowledge base from disk.

        contextual_retrieval: if True, enrich chunks with LLM context at build
            time (FIX 69). Defaults to settings.enable_contextual_retrieval.
        """
        from pipeline.knowledge_base import KnowledgeBase, chunk_documents, load_documents

        self.kb = KnowledgeBase(self.embedding_model)
        if not force_rebuild and self.kb.load(settings.index_dir):
            log.info("  Knowledge base loaded from disk cache.")
            self._wire_retriever()
            return

        log.info("  Building knowledge base from PDFs...")
        pages = load_documents(pdf_dir or settings.pdf_dir)
        ar_chunks, fr_chunks = chunk_documents(pages)

        use_ctx = contextual_retrieval if contextual_retrieval is not None else settings.enable_contextual_retrieval
        ollama_for_enrich = self.ollama if use_ctx else None
        if use_ctx:
            log.info("  Contextual Retrieval enabled (FIX 69)")
        self.kb.build(ar_chunks, fr_chunks, ollama=ollama_for_enrich)
        self.kb.save(settings.index_dir)
        self._wire_retriever()

    def _wire_retriever(self) -> None:
        self._retriever = HybridRetriever(self.kb, self.embedding_model, self.reranker)

    # ── KB status helper ──────────────────────────────────────────────────────

    def kb_status(self) -> dict:
        if self.kb is None:
            log.info("  KB: NOT INITIALIZED")
            return {}
        arabic_ok = self.kb.arabic_faiss is not None and len(self.kb.arabic_chunks) > 0
        french_ok = self.kb.french_faiss is not None and len(self.kb.french_chunks) > 0
        unified_ok = self.kb.unified_faiss is not None and len(self.kb.all_chunks) > 0
        ar_count = len(self.kb.arabic_chunks)
        fr_count = len(self.kb.french_chunks)
        all_count = len(self.kb.all_chunks)
        status = {
            # UI-expected keys (index.html reads kb.arabic / kb.french / kb.unified
            # and kb.arabic_chunks / kb.french_chunks / kb.total_chunks)
            "arabic": arabic_ok,
            "french": french_ok,
            "unified": unified_ok,
            "arabic_chunks": ar_count,
            "french_chunks": fr_count,
            "total_chunks": all_count,
            # Legacy aliases kept for backward compatibility
            "all_chunks": all_count,
            "arabic_ready": arabic_ok,
            "french_ready": french_ok,
            "unified_ready": unified_ok,
            "dense_mode": "unified cross-lingual" if unified_ok else "per-language fallback",
        }
        log.info(f"  KB status: {status}")
        return status

    # ── Main ask() ────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        skip_verify: bool = False,
    ) -> PipelineResult:
        """
        Run the full pipeline and return a PipelineResult.

        result.retrieval          — initial retrieval (chunks, scores, context)
        result.execution_trace    — every tool call in the agentic loop + all contexts
        result.to_ragas_contexts() — merged contexts for RAGAS faithfulness/precision

        skip_verify=True skips Step 10 (verification) for speed.
        """
        if not self._ready or self.kb is None:
            raise RuntimeError("Call setup() and build_knowledge_base() first.")
        t0 = time.time()

        # ── Step 4: Language detection ────────────────────────────────────────
        language, confidence = detect_language(question)
        log.info(f"  Language: {language} (conf={confidence:.2f})")

        # ── Step 5: Query translation (Darija/Arabizi → MSA) ─────────────────
        retrieval_query = question
        translation_used = False
        if language in ("Darija", "Arabizi") and settings.enable_query_translation:
            msa_query = translate_to_msa(question, language, self.ollama)
            if msa_query:
                retrieval_query = msa_query
                translation_used = True
                log.info(f"  Query translated: {question[:50]} → {msa_query[:50]}")

        # ── Step 6: Classify question ─────────────────────────────────────────
        flags = classify_question(retrieval_query, language, confidence, self.ollama)
        log.info(f"  Flags: {flags.summary()}")

        if flags.OUTSCOPE:
            agent = PlannerAgent(self.ollama, self._retriever)
            return PipelineResult(
                question=question,
                answer=agent._get_refusal(language),
                language=language,
                lang_confidence=confidence,
                flags=flags,
                is_grounded=False,
                is_abstained=True,
                latency_sec=time.time() - t0,
                translation_used=translation_used,
                original_query=question if translation_used else None,
            )

        # ── Step 7: Hybrid retrieval (BM25 + dense + RRF + rerank) ───────────
        retrieval: RetrievalResult = self._retriever.retrieve(
            question=question,
            flags=flags,
            query_translated=translation_used,
            retrieval_query=retrieval_query,
        )

        if self._retriever.is_out_of_scope(retrieval):
            log.info("  Out of scope (low retrieval score)")
            agent = PlannerAgent(self.ollama, self._retriever)
            return PipelineResult(
                question=question,
                answer=agent._get_abstain(language),
                language=language,
                lang_confidence=confidence,
                flags=flags,
                retrieval=retrieval,
                is_grounded=False,
                is_abstained=True,
                latency_sec=time.time() - t0,
                translation_used=translation_used,
                original_query=question if translation_used else None,
            )

        # ── Step 8-9: Generate (PlannerAgent: Plan→Execute→Reflect→Synthesise) ─
        agent = PlannerAgent(self.ollama, self._retriever)
        raw_answer, steps, intermediate_answers, agent_state = agent.run(
            question, flags, retrieval.chunks
        )

        if skip_verify:
            return PipelineResult(
                question=question,
                answer=raw_answer,
                language=language,
                lang_confidence=confidence,
                flags=flags,
                sources=list(agent.memory.source_counts().keys()),
                retrieval=retrieval,
                execution_trace=agent_state.execution_trace,
                is_grounded=False,
                is_abstained=False,
                latency_sec=time.time() - t0,
                agent_steps=steps,
                memory_stats=_memory_stats(agent.memory),
                translation_used=translation_used,
                original_query=question if translation_used else None,
                agent_state=agent_state,
            )

        # ── Step 10: Verify (11 layers) ───────────────────────────────────────
        t_verify = time.time()
        all_chunks = agent.memory.get_top_chunks(settings.memory_max_chunks)
        # retrieval.chunks is List[ScoredChunk] — extract the inner Chunk objects
        generation_chunks = [sc.chunk for sc in retrieval.chunks]
        verified, is_grounded, is_abstained, audit = verify_output(
            answer=raw_answer,
            all_chunks=all_chunks,
            flags=flags,
            ollama=self.ollama,
            question=question,
            generation_chunks=generation_chunks,
            intermediate_answers=intermediate_answers,
            agent_state=agent_state,
        )

        agent_state.execution_trace.verification_latency_ms = round((time.time() - t_verify) * 1000)

        result = PipelineResult(
            question=question,
            answer=verified,
            language=language,
            lang_confidence=confidence,
            flags=flags,
            sources=list(agent.memory.source_counts().keys()),
            retrieval=retrieval,
            execution_trace=agent_state.execution_trace,
            is_grounded=is_grounded,
            is_abstained=is_abstained,
            latency_sec=time.time() - t0,
            agent_steps=steps,
            memory_stats=_memory_stats(agent.memory),
            audit_trail=audit,
            translation_used=translation_used,
            original_query=question if translation_used else None,
            agent_state=agent_state,
        )
        log.info(
            f"  Done: {result.language} | grounded={result.is_grounded} | "
            f"latency={result.latency_sec:.1f}s | steps={result.agent_steps}"
        )
        return result

    # ── Convenience for direct retrieval-only calls ───────────────────────────

    def retrieve(
        self,
        query: str,
        flags=None,
        query_translated: bool = False,
        retrieval_query: Optional[str] = None,
    ) -> RetrievalResult:
        """Public retrieve entry point — used by /api/retrieve and benchmarking."""
        if not self._ready or self.kb is None:
            raise RuntimeError("Call setup() and build_knowledge_base() first.")
        if flags is None:
            from pipeline.models import QuestionFlags
            flags = QuestionFlags(SIMPLE=True, language="unknown")
        return self._retriever.retrieve(
            question=query,
            flags=flags,
            query_translated=query_translated,
            retrieval_query=retrieval_query,
        )

    def is_out_of_scope(self, retrieval: RetrievalResult) -> bool:
        """Delegate OOS check to the retriever."""
        return self._retriever.is_out_of_scope(retrieval)

    def plan_preview(self, question: str):
        """Return the AgentPlan for a question without running generation."""
        if not self._ready or self.kb is None:
            raise RuntimeError("Call setup() and build_knowledge_base() first.")
        from pipeline.language import classify_question, detect_language
        from pipeline.models import AgentState
        language, confidence = detect_language(question)
        flags = classify_question(question, language, confidence, self.ollama)
        agent = PlannerAgent(self.ollama, self._retriever)
        state = AgentState(language=flags.language)
        return flags, language, agent._plan(question, flags, state)


def _memory_stats(memory: AgentMemory) -> dict:
    return {
        "size": memory.size(),
        "added": memory.stats["total_added"],
        "evicted": memory.stats["evicted"],
        "rejected_dup": memory.stats["rejected_duplicate"],
    }
