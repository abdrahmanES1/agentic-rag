# -*- coding: utf-8 -*-
"""
Request / response schemas for the Moroccan RAG API.

These are plain dicts (not Pydantic models) to avoid an extra dependency.
Used by routes.py to serialize PipelineResult → JSON.
"""

from typing import Optional

from pipeline.models import PipelineResult


def pipeline_result_to_dict(result: PipelineResult) -> dict:
    """
    Full serialization of PipelineResult including:
      - retrieval.contexts     → List[str]  for RAGAS faithfulness / precision
      - execution_trace        → all agentic tool calls + contexts
      - audit_trail            → CFI, claim ratio, entity ratio
      - flags                  → question classification

    This is the key fix: the old api_v2.py never included 'retrieval',
    making RAGAS/ARES evaluation impossible. Now every response carries the
    full retrieval context.
    """
    return {
        "question": result.question,
        "answer": result.answer,
        "language": result.language,
        "lang_confidence": round(result.lang_confidence, 3),
        "flags": result.flags.to_dict() if result.flags else None,
        "sources": result.sources,
        "is_grounded": result.is_grounded,
        "is_abstained": result.is_abstained,
        "latency_sec": round(result.latency_sec, 2),
        "agent_steps": result.agent_steps,
        "memory_stats": result.memory_stats,
        "translation_used": result.translation_used,
        "original_query": result.original_query,
        # ── Retrieval context (RAGAS / ARES) ─────────────────────────────
        "retrieval": result.retrieval.to_dict() if result.retrieval else None,
        # ── Execution trace (multi-hop contexts) ──────────────────────────
        "execution_trace": result.execution_trace.to_dict() if result.execution_trace else None,
        # ── Convenience: merged RAGAS contexts ───────────────────────────
        "ragas_contexts": result.to_ragas_contexts(),
        # ── Audit trail ───────────────────────────────────────────────────
        "audit_trail": result.audit_trail.to_dict() if result.audit_trail else None,
        # ── Agent state trace ─────────────────────────────────────────────
        "agent_state": result.agent_state.to_audit_dict() if result.agent_state else None,
    }


def retrieval_result_to_dict(result, language: str, flags, is_outscope: bool, top_k: int) -> dict:
    """Serialize a retrieve-only response."""
    chunks_out = [
        {
            "text": sc.chunk.text,
            "source": sc.chunk.source,
            "page": sc.chunk.page,
            "language": sc.chunk.language,
            "article_number": sc.chunk.article_number,
            "rrf_score": round(sc.rrf_score, 4),
            "bm25_score": round(sc.bm25_score, 4),
            "dense_score": round(sc.dense_score, 4),
        }
        for sc in result.chunks[:top_k]
    ]
    return {
        "language": language,
        "flags": flags.to_dict() if flags else None,
        "is_outscope": is_outscope,
        "chunks": chunks_out,
        "context": result.context,
        "contexts": result.to_ragas_contexts()[:top_k],
        "metadata": {
            "strategy": result.retriever_strategy,
            "top_k_retrieved": result.top_k_retrieved,
            "top_n_reranked": result.top_n_reranked,
            "reranker_applied": result.reranker_applied,
            "query_translated": result.query_translated,
            "retrieval_query": result.retrieval_query,
        },
    }
