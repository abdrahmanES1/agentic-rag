# -*- coding: utf-8 -*-
"""
Base class for all RAG baselines.

All baselines share:
  - Same OllamaClient (configured via shared.py constants)
  - Same /api/retrieve endpoint for context retrieval
  - Same gold test set (benchmark_testset_gold.json)
  - Same BaselineResult dataclass so benchmark_runner.py can compare uniformly

Only the generation strategy differs between baselines.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from benchmarking.shared import OllamaClient, api_retrieve, format_context, get_contexts_from_response


@dataclass
class BaselineResult:
    """Uniform output from every baseline — maps directly to RAGAS/ARES input."""

    question: str
    answer: str
    contexts: List[str]               # retrieved chunk texts (List[str] for RAGAS)
    latency_sec: float
    baseline_name: str
    is_outscope: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_ragas_row(self, ground_truth: str = "") -> Dict:
        """RAGAS-compatible dict row."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": ground_truth,
        }

    def to_ares_rows(self, ground_truth: str = "") -> List[Dict]:
        """ARES expects one row per (question, document) pair."""
        return [
            {
                "query": self.question,
                "document": ctx,
                "answer": self.answer,
                "label": ground_truth,
            }
            for ctx in self.contexts
        ]


class BaselineRAG(ABC):
    """
    Abstract base for all RAG baselines.

    Subclasses implement run() and are responsible for:
      1. Retrieving context (via self.retrieve() or custom logic)
      2. Generating an answer (via self.client.generate())
      3. Returning a BaselineResult

    The baseline_name class attribute appears in output tables and RAGAS datasets.
    """

    baseline_name: str = "baseline"

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        retrieve_url: str = None,
        top_k: int = 5,
    ):
        from benchmarking.shared import RETRIEVE_URL
        self.client = client or OllamaClient()
        self.retrieve_url = retrieve_url or RETRIEVE_URL
        self.top_k = top_k

    @abstractmethod
    def run(self, question: str) -> BaselineResult:
        """Run the baseline on a single question and return a BaselineResult."""

    def retrieve(self, question: str) -> Dict:
        """Call /api/retrieve and return the full response dict."""
        return api_retrieve(question, top_k=self.top_k, url=self.retrieve_url)

    def get_contexts(self, retrieve_response: Dict) -> List[str]:
        """Extract List[str] contexts from an /api/retrieve response."""
        return get_contexts_from_response(retrieve_response)

    def build_context_string(self, retrieve_response: Dict) -> str:
        """Format retrieved chunks into a context string."""
        chunks = retrieve_response.get("chunks", [])
        return format_context(chunks)

    def _out_of_scope_result(self, question: str, language: str, latency: float) -> BaselineResult:
        if language == "french":
            msg = "Cette question est hors de mon domaine. Je suis spécialisé dans les services publics marocains."
        elif language == "arabic_msa":
            msg = "عذراً، هذا السؤال خارج نطاق اختصاصي."
        else:
            msg = "هذا السؤال خارج نطاق اختصاصي."
        return BaselineResult(
            question=question,
            answer=msg,
            contexts=[],
            latency_sec=latency,
            baseline_name=self.baseline_name,
            is_outscope=True,
        )
