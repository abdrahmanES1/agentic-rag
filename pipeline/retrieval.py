# -*- coding: utf-8 -*-
"""
Steps 7-8 — Hybrid retrieval (BM25 + dense + RRF) and reranking.

Key change vs the monolith: `HybridRetriever.retrieve()` returns a
`RetrievalResult` that includes chunks, scores, formatted context, and
metadata — everything RAGAS / ARES need. Nothing is consumed silently.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from pipeline.config import settings
from pipeline.knowledge_base import KnowledgeBase, tokenize_for_bm25
from pipeline.models import (
    Chunk,
    QuestionFlags,
    RetrievalResult,
    ScoredChunk,
)

log = logging.getLogger("MoroccanRAG")

_RRF_K = 60
_RERANKER_MAX_CHARS = 900
_GARBAGE_MIN_WORDS = 5


class HybridRetriever:
    """
    Hybrid BM25 + dense retrieval with RRF fusion and optional reranking.

    Returns `RetrievalResult` — the complete retrieval output including scores,
    metadata, and the formatted context string passed to the LLM.
    """

    def __init__(self, kb: KnowledgeBase, embedding_model: SentenceTransformer, reranker: Optional[CrossEncoder] = None):
        self.kb = kb
        self.emb = embedding_model
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        flags: QuestionFlags,
        query_translated: bool = False,
        retrieval_query: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Run hybrid retrieval and return a RetrievalResult.

        Parameters
        ----------
        question : str
            Original user question (used for display / citation).
        flags : QuestionFlags
            Classification output (used for MULTIHOP reranking).
        query_translated : bool
            Whether the query was translated from Darija/Arabizi.
        retrieval_query : str, optional
            The query actually sent to the retriever (post-translation).
            Falls back to `question` if not provided.
        """
        rq = retrieval_query or question
        kb = self.kb

        arabic_ok = kb.arabic_faiss is not None and len(kb.arabic_chunks) > 0
        french_ok = kb.french_faiss is not None and len(kb.french_chunks) > 0
        unified_ok = kb.unified_faiss is not None and len(kb.all_chunks) > 0

        if not arabic_ok and not french_ok:
            raise RuntimeError("FAISS indexes empty. Build the knowledge base first.")

        # Encode query once
        q_emb = self.emb.encode([rq], normalize_embeddings=True).astype("float32")
        rrf_map: Dict[str, ScoredChunk] = {}

        # ── Dense retrieval ────────────────────────────────────────────────────
        if unified_ok:
            _dense_unified(q_emb, kb, rrf_map)
        else:
            if arabic_ok:
                _dense_per_lang(q_emb, kb.arabic_chunks, kb.arabic_faiss, rrf_map)
            if french_ok:
                _dense_per_lang(q_emb, kb.french_chunks, kb.french_faiss, rrf_map)

        # ── Sparse retrieval ───────────────────────────────────────────────────
        if arabic_ok:
            _bm25_search(rq, kb.arabic_chunks, kb.arabic_bm25, "arabic_msa", rrf_map)
        if french_ok:
            _bm25_search(rq, kb.french_chunks, kb.french_bm25, "french", rrf_map)

        if not rrf_map:
            return RetrievalResult(
                chunks=[],
                context="",
                retriever_strategy="hybrid",
                top_k_retrieved=0,
                top_n_reranked=0,
                query_translated=query_translated,
                retrieval_query=rq,
            )

        sorted_results = sorted(rrf_map.values(), key=lambda r: r.rrf_score, reverse=True)
        top_k = sorted_results[: settings.retrieve_top_k]
        top_k_count = len(top_k)

        # ── Reranking ──────────────────────────────────────────────────────────
        candidates = [sc for sc in top_k if not _is_garbage(sc.chunk.text)]
        if not candidates:
            candidates = top_k
        candidates = candidates[: settings.reranker_top_n]

        reranker_scores: List[float] = []
        reranker_applied = False

        if self.reranker is not None and rq and len(candidates) > 1:
            try:
                pairs = _build_reranker_pairs(rq, candidates, flags)
                raw_scores = self.reranker.predict(pairs, show_progress_bar=False)
                ranked = sorted(zip(raw_scores, candidates), key=lambda x: float(x[0]), reverse=True)
                candidates = [sc for _, sc in ranked]
                reranker_scores = [float(s) for s, _ in ranked]
                for sc, rs in zip(candidates, reranker_scores):
                    sc.reranker_score = rs
                reranker_applied = True
                log.info(f"  Reranker applied — top score: {reranker_scores[0]:.3f}")
            except Exception as e:
                log.warning(f"  Reranker failed: {e} — using RRF order")

        final_chunks = candidates[: settings.compress_top_n]

        # ── Build context string ───────────────────────────────────────────────
        context = _build_context(final_chunks)

        # ── Collect per-chunk scores ───────────────────────────────────────────
        bm25_scores = [sc.bm25_score for sc in final_chunks]
        dense_scores = [sc.dense_score for sc in final_chunks]
        rrf_scores = [sc.rrf_score for sc in final_chunks]
        final_reranker = [sc.reranker_score for sc in final_chunks]

        log.info(f"  Retrieved: {len(final_chunks)} chunks | unified={unified_ok} | reranked={reranker_applied}")

        return RetrievalResult(
            chunks=final_chunks,
            context=context,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            rrf_scores=rrf_scores,
            reranker_scores=final_reranker,
            retriever_strategy="hybrid",
            top_k_retrieved=top_k_count,
            top_n_reranked=len(final_chunks),
            reranker_applied=reranker_applied,
            query_translated=query_translated,
            retrieval_query=rq,
        )

    def is_out_of_scope(self, retrieval: RetrievalResult) -> bool:
        """Return True if top retrieval score is below the out-of-scope threshold."""
        if not retrieval.rrf_scores:
            return True
        return retrieval.rrf_scores[0] < settings.outscope_score_threshold


# ── Search helpers ────────────────────────────────────────────────────────────


def _dense_unified(q_emb: np.ndarray, kb: KnowledgeBase, rrf_map: Dict[str, ScoredChunk]) -> None:
    k = min(settings.retrieve_top_k, len(kb.all_chunks))
    scores, idxs = kb.unified_faiss.search(q_emb, k)
    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0])):
        if idx < 0 or float(score) < settings.dense_min_score_unified:
            continue
        chunk = kb.all_chunks[idx]
        if chunk.chunk_id not in rrf_map:
            rrf_map[chunk.chunk_id] = ScoredChunk(chunk=chunk)
        rrf_map[chunk.chunk_id].dense_score = float(score)
        rrf_map[chunk.chunk_id].rrf_score += settings.dense_weight / (_RRF_K + rank + 1)


def _dense_per_lang(q_emb: np.ndarray, chunks: List[Chunk], faiss_index, rrf_map: Dict[str, ScoredChunk]) -> None:
    k = min(settings.retrieve_top_k, len(chunks))
    scores, idxs = faiss_index.search(q_emb, k)
    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0])):
        if idx < 0 or float(score) < settings.dense_min_score_per_lang:
            continue
        chunk = chunks[idx]
        if chunk.chunk_id not in rrf_map:
            rrf_map[chunk.chunk_id] = ScoredChunk(chunk=chunk)
        rrf_map[chunk.chunk_id].dense_score = float(score)
        rrf_map[chunk.chunk_id].rrf_score += settings.dense_weight / (_RRF_K + rank + 1)


def _bm25_search(question: str, chunks: List[Chunk], bm25, lang: str, rrf_map: Dict[str, ScoredChunk]) -> None:
    k = min(settings.retrieve_top_k, len(chunks))
    tokenized_q = tokenize_for_bm25(question, lang)
    bm25_scores = bm25.get_scores(tokenized_q)
    if float(max(bm25_scores)) <= 0.0:
        return
    top_idx = np.argsort(bm25_scores)[::-1][:k]
    for rank, idx in enumerate(top_idx):
        chunk = chunks[idx]
        if chunk.chunk_id not in rrf_map:
            rrf_map[chunk.chunk_id] = ScoredChunk(chunk=chunk)
        rrf_map[chunk.chunk_id].bm25_score = float(bm25_scores[idx])
        rrf_map[chunk.chunk_id].rrf_score += settings.bm25_weight / (_RRF_K + rank + 1)


def _is_garbage(text: str) -> bool:
    if not text or len(text.split()) < _GARBAGE_MIN_WORDS:
        return True
    ar = sum(1 for c in text if "؀" <= c <= "ۿ")
    lat = sum(1 for c in text if c.isalpha() and c.isascii() and c.islower())
    junk = sum(1 for c in text if c in "0123456789/\\(),;:")
    n = max(len(text), 1)
    return (ar / n < 0.10) and (lat / n < 0.15) and (junk / n > 0.15)


def _truncate_for_reranker(text: str, max_chars: int = _RERANKER_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ["؟", "!", ".", "،", "\n"]:
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[: idx + 1]
    return cut


def _build_reranker_pairs(query: str, candidates: List[ScoredChunk], flags: QuestionFlags) -> List[Tuple[str, str]]:
    if flags.MULTIHOP and len(flags.intents) > 1:
        intent_queries = {
            "DOCUMENTS": ("الوثائق المطلوبة", "documents requis"),
            "COST": ("التكلفة والرسوم", "frais et coût"),
            "DEADLINE": ("مدة الإنجاز", "délai de traitement"),
            "PROCEDURE": ("خطوات الإجراءات", "étapes procédure"),
            "ELIGIBILITY": ("شروط الأهلية", "conditions éligibilité"),
            "LEGAL": ("العقوبات", "sanctions pénalités"),
        }
        pairs = []
        for sc in candidates:
            best_query = query
            for intent in flags.intents:
                if intent in intent_queries:
                    ar_q, fr_q = intent_queries[intent]
                    q = ar_q if sc.chunk.language in ("arabic_msa", "mixed") else fr_q
                    best_query = q + " " + query
                    break
            pairs.append((best_query, _truncate_for_reranker(sc.chunk.text)))
        return pairs
    return [(query, _truncate_for_reranker(sc.chunk.text)) for sc in candidates]


def _build_context(chunks: List[ScoredChunk]) -> str:
    parts = [f"[Source: {sc.chunk.source} | Page: {sc.chunk.page}]\n{sc.chunk.text}" for sc in chunks]
    return "\n\n".join(parts)


# ── Legacy function for backward compatibility ────────────────────────────────


def hybrid_retrieve(
    question: str,
    flags: QuestionFlags,
    kb: KnowledgeBase,
    embedding_model: SentenceTransformer,
    reranker: Optional[CrossEncoder] = None,
) -> Tuple[RetrievalResult, bool]:
    """
    Backward-compatible wrapper returning (RetrievalResult, is_out_of_scope).
    New code should use HybridRetriever directly.
    """
    retriever = HybridRetriever(kb, embedding_model, reranker)
    result = retriever.retrieve(question, flags)
    is_out_of_scope = retriever.is_out_of_scope(result)
    return result, is_out_of_scope
