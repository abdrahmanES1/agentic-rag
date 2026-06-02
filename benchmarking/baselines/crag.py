# -*- coding: utf-8 -*-
"""
CRAG Baseline — Corrective Retrieval Augmented Generation (Shi et al. 2024).

Architecture:
  1. Retrieve initial context from /api/retrieve
  2. Evaluate relevance of retrieved docs (LLM judge):
     CORRECT  → use as-is
     INCORRECT → rephrase query and retry
     AMBIGUOUS → use retrieved + rephrased query results combined
  3. If ambiguous/incorrect: decompose into knowledge strips (key facts)
  4. Generate final answer from the corrected context

Reference: Shi et al., "Corrective Retrieval Augmented Generation", 2024
"""

import re
import time
from typing import List

from benchmarking.baselines.base import BaselineRAG, BaselineResult
from benchmarking.shared import get_contexts_from_response

RELEVANCE_CORRECT = "CORRECT"
RELEVANCE_INCORRECT = "INCORRECT"
RELEVANCE_AMBIGUOUS = "AMBIGUOUS"


class CRAG(BaselineRAG):

    baseline_name = "crag"

    def run(self, question: str) -> BaselineResult:
        t0 = time.time()

        initial_ret = self.retrieve(question)
        if initial_ret.get("is_outscope"):
            return self._out_of_scope_result(question, initial_ret.get("language", "arabic_msa"), time.time() - t0)

        language = initial_ret.get("language", "arabic_msa")
        initial_contexts = get_contexts_from_response(initial_ret)
        all_contexts: List[str] = list(initial_contexts)

        # Step 2: Evaluate relevance
        verdict = self._evaluate_relevance(question, initial_contexts[:3], language)

        if verdict == RELEVANCE_CORRECT:
            final_context = self.build_context_string(initial_ret)

        elif verdict == RELEVANCE_INCORRECT:
            # Rephrase and retry
            rephrased = self._rephrase_query(question, initial_contexts[:2], language)
            retry_ret = self.retrieve(rephrased)
            retry_contexts = get_contexts_from_response(retry_ret)
            for ctx in retry_contexts:
                if ctx not in all_contexts:
                    all_contexts.append(ctx)
            final_context = self.build_context_string(retry_ret) or self.build_context_string(initial_ret)

        else:  # AMBIGUOUS — combine and extract knowledge strips
            rephrased = self._rephrase_query(question, initial_contexts[:2], language)
            retry_ret = self.retrieve(rephrased)
            retry_contexts = get_contexts_from_response(retry_ret)
            for ctx in retry_contexts:
                if ctx not in all_contexts:
                    all_contexts.append(ctx)
            # Extract relevant knowledge strips from combined context
            combined = self.build_context_string(initial_ret) + "\n\n" + self.build_context_string(retry_ret)
            final_context = self._extract_knowledge_strips(question, combined, language)

        answer = self._generate(question, final_context, language)
        return BaselineResult(
            question=question,
            answer=answer,
            contexts=all_contexts,
            latency_sec=time.time() - t0,
            baseline_name=self.baseline_name,
            metadata={"relevance_verdict": verdict},
        )

    def _evaluate_relevance(self, question: str, contexts: List[str], language: str) -> str:
        ctx_preview = "\n\n".join(contexts[:2])[:600]
        if language in ("arabic_msa", "Darija"):
            prompt = (
                "قيّم مدى صلة الوثائق بالسؤال:\n\n"
                f"السؤال: {question}\n\n"
                f"الوثائق:\n{ctx_preview}\n\n"
                "أجب بكلمة واحدة فقط:\n"
                "- CORRECT: الوثائق ذات صلة مباشرة وكاملة\n"
                "- INCORRECT: الوثائق غير ذات صلة\n"
                "- AMBIGUOUS: الوثائق ذات صلة جزئية\n"
                "الحكم:"
            )
        else:
            prompt = (
                "Évaluez la pertinence des documents pour la question:\n\n"
                f"Question: {question}\n\n"
                f"Documents:\n{ctx_preview}\n\n"
                "Répondez avec un seul mot:\n"
                "- CORRECT: les documents sont directement pertinents\n"
                "- INCORRECT: les documents ne sont pas pertinents\n"
                "- AMBIGUOUS: les documents sont partiellement pertinents\n"
                "Verdict:"
            )
        result = (self.client.generate(prompt, temperature=0.0, max_tokens=20) or "").upper().strip()
        if RELEVANCE_CORRECT in result:
            return RELEVANCE_CORRECT
        if RELEVANCE_INCORRECT in result:
            return RELEVANCE_INCORRECT
        return RELEVANCE_AMBIGUOUS

    def _rephrase_query(self, question: str, contexts: List[str], language: str) -> str:
        ctx_hint = "\n".join(contexts[:1])[:300]
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"الوثائق الموجودة تحتوي على:\n{ctx_hint}\n\n"
                f"السؤال الأصلي: {question}\n\n"
                "أعد صياغة السؤال لتحسين البحث (اكتب السؤال المعاد صياغته فقط):"
            )
        else:
            prompt = (
                f"Les documents disponibles contiennent:\n{ctx_hint}\n\n"
                f"Question originale: {question}\n\n"
                "Reformulez la question pour améliorer la recherche (écrivez uniquement la question reformulée):"
            )
        return self.client.generate(prompt, temperature=0.3, max_tokens=150) or question

    def _extract_knowledge_strips(self, question: str, context: str, language: str) -> str:
        """Extract the most relevant knowledge strips from the combined context."""
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"استخرج فقط المعلومات ذات الصلة المباشرة بالسؤال من الوثائق التالية:\n\n"
                f"السؤال: {question}\n\n"
                f"الوثائق:\n{context[:1200]}\n\n"
                "المعلومات ذات الصلة (نقاط مختصرة):"
            )
        else:
            prompt = (
                f"Extrayez uniquement les informations directement liées à la question:\n\n"
                f"Question: {question}\n\n"
                f"Documents:\n{context[:1200]}\n\n"
                "Informations pertinentes (points résumés):"
            )
        strips = self.client.generate(prompt, temperature=0.1, max_tokens=600) or context[:800]
        return strips

    def _generate(self, question: str, context: str, language: str) -> str:
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"أجب على السؤال بناءً على المعلومات المصححة أدناه فقط:\n\n"
                f"{context}\n\n"
                f"السؤال: {question}\n\nالإجابة:"
            )
        else:
            prompt = (
                f"Répondez à la question uniquement à partir des informations corrigées ci-dessous:\n\n"
                f"{context}\n\n"
                f"Question: {question}\n\nRéponse:"
            )
        return self.client.generate(prompt, temperature=0.2, max_tokens=1024) or "[ERROR]"
