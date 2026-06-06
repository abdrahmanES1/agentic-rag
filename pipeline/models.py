# -*- coding: utf-8 -*-
"""
All dataclasses for the Moroccan RAG pipeline.

Key additions vs the original monolith:
  - RetrievalResult: surfaces all retrieval data (chunks, scores, metadata)
    to callers — critical for RAGAS/ARES benchmarking.
  - ToolCall / ExecutionTrace: captures every tool invocation during the
    agentic multi-hop loop so benchmarking can see ALL contexts used, not
    just the initial retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def short_source(source: str) -> str:
    """
    Compact a chunk source for citation display. Scraped sources are full URLs
    (e.g. https://idarati.ma/informationnel/ar/thematique/<uuid>/<uuid>) whose
    long UUID/path fragments bloat answers and pollute lexical/number metrics.
    Return just the domain (a substring of the full URL, so existing citation
    validation still matches). Non-URL sources (filenames) are returned as-is.
    """
    s = (source or "").strip()
    if s.lower().startswith("http"):
        s = re.sub(r"^https?://", "", s).rstrip("/")
        return s.split("/", 1)[0] or s
    return s


# ── Knowledge Base primitives ─────────────────────────────────────────────────


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    language: str
    chunk_id: str
    is_ocr: bool = False
    article_number: str = ""
    law_name: str = ""


@dataclass
class ScoredChunk:
    chunk: Chunk
    rrf_score: float = 0.0
    bm25_score: float = 0.0
    dense_score: float = 0.0
    reranker_score: float = 0.0
    step_found: int = 0


# ── Document registry ─────────────────────────────────────────────────────────


@dataclass
class DocumentRecord:
    filename: str
    source_url: str = ""
    domain: str = "unknown"
    language: str = "unknown"
    doc_type: str = "unknown"
    law_ref: str = ""
    pages: int = 0
    chunks: int = 0
    file_hash: str = ""
    ingested_at: str = ""
    ocr_used: bool = False
    quality_score: float = 1.0


# ── Question classification ───────────────────────────────────────────────────

VALID_INTENTS = {
    "DOCUMENTS",
    "PROCEDURE",
    "COST",
    "DEADLINE",
    "ELIGIBILITY",
    "LEGAL",
    "COMPARISON",
    "OUT_OF_SCOPE",
}


@dataclass
class QuestionFlags:
    SIMPLE: bool = False
    MULTIHOP: bool = False
    LEGAL: bool = False
    OUTSCOPE: bool = False
    language: str = "unknown"
    confidence: float = 0.0
    hop_count: int = 1
    intents: List[str] = field(default_factory=list)

    def summary(self) -> str:
        active = [
            k
            for k, v in {
                "SIMPLE": self.SIMPLE,
                "MULTIHOP": self.MULTIHOP,
                "LEGAL": self.LEGAL,
                "OUTSCOPE": self.OUTSCOPE,
            }.items()
            if v
        ]
        intent_str = f" intents={self.intents}" if self.intents else ""
        return f"[{', '.join(active)}] | lang={self.language} (conf={self.confidence:.2f}){intent_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "SIMPLE": self.SIMPLE,
            "MULTIHOP": self.MULTIHOP,
            "LEGAL": self.LEGAL,
            "OUTSCOPE": self.OUTSCOPE,
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "hop_count": self.hop_count,
            "intents": self.intents,
        }


# ── Retrieval result — the key new type ──────────────────────────────────────


@dataclass
class RetrievalResult:
    """
    Complete output of one retrieval call, including all data needed for
    RAGAS / ARES evaluation.

    This is the fix for the benchmarking problem: previously retrieval context
    was consumed internally by _generate() and never surfaced to callers.
    Now every caller (pipeline, API, benchmarks) can read exactly what was
    retrieved and with what scores.
    """

    # ── What was retrieved ────────────────────────────────────────────────────
    chunks: List[ScoredChunk]
    # The exact context string that was passed to the LLM
    context: str

    # ── Per-chunk scores (for RAGAS context_precision / context_recall) ───────
    bm25_scores: List[float] = field(default_factory=list)
    dense_scores: List[float] = field(default_factory=list)
    rrf_scores: List[float] = field(default_factory=list)
    reranker_scores: List[float] = field(default_factory=list)

    # ── Retriever metadata (for ARES attribution) ─────────────────────────────
    retriever_strategy: str = "hybrid"
    top_k_retrieved: int = 0
    top_n_reranked: int = 0
    reranker_applied: bool = False
    query_translated: bool = False
    retrieval_query: str = ""

    def to_ragas_contexts(self) -> List[str]:
        """Return chunk texts as a list — the 'contexts' field RAGAS expects."""
        return [sc.chunk.text for sc in self.chunks]

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable form for the API response and audit logs."""
        return {
            "contexts": self.to_ragas_contexts(),
            "context_formatted": self.context,
            "chunk_sources": [
                {
                    "source": sc.chunk.source,
                    "page": sc.chunk.page,
                    "language": sc.chunk.language,
                    "article_number": sc.chunk.article_number,
                    "chunk_id": sc.chunk.chunk_id,
                }
                for sc in self.chunks
            ],
            "scores": {
                "bm25": self.bm25_scores,
                "dense": self.dense_scores,
                "rrf": self.rrf_scores,
                "reranker": self.reranker_scores,
            },
            "metadata": {
                "strategy": self.retriever_strategy,
                "top_k_retrieved": self.top_k_retrieved,
                "top_n_reranked": self.top_n_reranked,
                "reranker_applied": self.reranker_applied,
                "query_translated": self.query_translated,
                "retrieval_query": self.retrieval_query,
            },
        }


# ── Agentic loop tracing ──────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """One tool invocation inside the agentic planning loop."""

    step_index: int
    tool_name: str
    query: str
    intent: str
    contexts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    chunks_returned: int = 0
    latency_ms: float = 0.0
    intermediate_answer: str = ""
    reflection: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "query": self.query,
            "intent": self.intent,
            "contexts": self.contexts,
            "scores": self.scores,
            "chunks_returned": self.chunks_returned,
            "latency_ms": self.latency_ms,
            "intermediate_answer": self.intermediate_answer,
            "reflection": self.reflection,
        }


@dataclass
class ExecutionTrace:
    """
    Complete trace of the agentic multi-hop loop.

    For RAGAS/ARES evaluation we need ALL contexts retrieved across the
    entire execution, not just the initial retrieval. This trace captures
    every tool call and its retrieved chunks in order.
    """

    plan: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    # UNION of all retrieved contexts across the entire execution
    all_contexts: List[str] = field(default_factory=list)
    # The context string fed to the final synthesis call
    synthesis_context: str = ""
    plan_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    # "simple" | "multihop" — which branch run() took
    agent_path: str = "unknown"

    def add_tool_call(self, tc: ToolCall) -> None:
        self.tool_calls.append(tc)
        # Accumulate contexts, deduplicating by content
        for ctx in tc.contexts:
            if ctx not in self.all_contexts:
                self.all_contexts.append(ctx)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_path": self.agent_path,
            "plan": self.plan,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "all_contexts": self.all_contexts,
            "synthesis_context": self.synthesis_context,
            "steps_executed": len(self.tool_calls),
            "plan_latency_ms": self.plan_latency_ms,
        }


# ── Planning dataclasses ──────────────────────────────────────────────────────


@dataclass
class PlannedStep:
    step_id: int
    intent: str
    sub_question: str
    tool: str
    tool_args: Dict[str, Any]
    rationale: str = ""
    depends_on: int = 0


@dataclass
class AgentPlan:
    steps: List[PlannedStep]
    raw_plan_text: str = ""
    plan_source: str = "llm"


@dataclass
class AgentState:
    question: str
    language: str
    flags: QuestionFlags
    plan: Optional[AgentPlan] = None
    step: int = 0
    facts: Dict[str, str] = field(default_factory=dict)
    evidence: Dict[str, List[Chunk]] = field(default_factory=dict)
    reflections: List[str] = field(default_factory=list)
    scratchpad: List[str] = field(default_factory=list)
    execution_trace: ExecutionTrace = field(default_factory=ExecutionTrace)
    is_complete: bool = False
    final_answer: Optional[str] = None

    def log(self, text: str) -> None:
        self.scratchpad.append(text)

    def to_audit_dict(self) -> Dict[str, Any]:
        plan_data = None
        if self.plan:
            plan_data = {
                "source": self.plan.plan_source,
                "steps": [
                    {
                        "id": s.step_id,
                        "intent": s.intent,
                        "tool": s.tool,
                        "rationale": s.rationale,
                        "sub_question": s.sub_question,
                    }
                    for s in self.plan.steps
                ],
            }
        return {
            "plan": plan_data,
            "facts": self.facts,
            "reflections": self.reflections,
            "scratchpad": self.scratchpad,
            "steps_run": self.step,
            "is_complete": self.is_complete,
            "execution_trace": self.execution_trace.to_dict(),
        }


# ── Grounding / verification ──────────────────────────────────────────────────


@dataclass
class ClaimVerification:
    claim: str
    grounded: bool
    nli_score: float
    supporting_chunk: Optional[Chunk]
    evidence_text: str
    confidence: float
    method: str  # "nli" | "nli_softmax" | "llm_judge" | "word_overlap"


@dataclass
class EntityVerification:
    entity: str
    entity_type: str  # "DATE" | "AMOUNT" | "LOCATION" | "DURATION" | "LEGAL_REF"
    found_in_context: bool
    exact_match: bool
    source_chunk: Optional[Chunk] = None


@dataclass
class GroundingAudit:
    timestamp: str
    question: str
    answer: str
    claims: List[ClaimVerification]
    entities: List[EntityVerification]
    overall_grounded: bool
    claim_grounded_ratio: float
    entity_match_ratio: float
    composite_fidelity_index: float
    warnings: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_grounded": self.overall_grounded,
            "claim_grounded_ratio": round(self.claim_grounded_ratio, 3),
            "entity_match_ratio": round(self.entity_match_ratio, 3),
            "composite_fidelity_index": round(self.composite_fidelity_index, 3),
            "warnings": self.warnings,
            "metadata": self.metadata,
            "claims": [
                {
                    "claim": c.claim,
                    "grounded": c.grounded,
                    "nli_score": round(c.nli_score, 3),
                    "confidence": round(c.confidence, 3),
                    "method": c.method,
                    "evidence": c.evidence_text[:200],
                }
                for c in self.claims
            ],
            "entities": [
                {
                    "entity": e.entity,
                    "entity_type": e.entity_type,
                    "found": e.found_in_context,
                    "exact_match": e.exact_match,
                }
                for e in self.entities
            ],
        }


# ── Pipeline output ───────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    question: str
    answer: str
    language: str
    lang_confidence: float = 0.0
    flags: Optional[QuestionFlags] = None
    sources: List[str] = field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    execution_trace: Optional[ExecutionTrace] = None
    is_grounded: bool = True
    is_abstained: bool = False
    latency_sec: float = 0.0
    agent_steps: int = 0
    memory_stats: Dict = field(default_factory=dict)
    audit_trail: Optional[GroundingAudit] = None
    translation_used: bool = False
    original_query: Optional[str] = None
    agent_state: Optional[AgentState] = None

    def to_ragas_contexts(self) -> List[str]:
        """
        Return ALL contexts used across initial retrieval AND agentic tool calls.
        This is the correct 'contexts' field for RAGAS faithfulness / precision.
        """
        seen: set = set()
        all_ctx: List[str] = []

        # Initial retrieval contexts
        if self.retrieval:
            for ctx in self.retrieval.to_ragas_contexts():
                if ctx not in seen:
                    seen.add(ctx)
                    all_ctx.append(ctx)

        # Agentic tool-call contexts (multi-hop loop)
        if self.execution_trace:
            for ctx in self.execution_trace.all_contexts:
                if ctx not in seen:
                    seen.add(ctx)
                    all_ctx.append(ctx)

        return all_ctx
