# -*- coding: utf-8 -*-
"""
HyDE Baseline — Hypothetical Document Embeddings (Gao et al. 2022).

Architecture:
  1. Generate a hypothetical answer document for the question
  2. Use the hypothetical document as the retrieval query (not the question)
  3. Retrieve using the hypothetical doc → better dense-retrieval alignment
  4. Generate the real answer from retrieved context

Why include: upper bound for dense-only retrieval. If HyDE beats Naive RAG
significantly, it means the question and document embeddings are misaligned
and embedding-side improvements would help.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels", 2022
"""

import time
from typing import List

from benchmarking.baselines.base import BaselineRAG, BaselineResult
from benchmarking.shared import get_contexts_from_response


class HyDE(BaselineRAG):

    baseline_name = "hyde"

    def run(self, question: str) -> BaselineResult:
        t0 = time.time()

        # First retrieve with original question to get language
        initial_ret = self.retrieve(question)
        if initial_ret.get("is_outscope"):
            return self._out_of_scope_result(question, initial_ret.get("language", "arabic_msa"), time.time() - t0)

        language = initial_ret.get("language", "arabic_msa")

        # Step 1: Generate a hypothetical answer document
        hyp_doc = self._generate_hypothetical_doc(question, language)

        # Step 2: Retrieve using the hypothetical document as query
        hyp_ret = self.retrieve(hyp_doc) if hyp_doc else initial_ret

        # Merge initial + hypothetical retrieval contexts
        all_contexts: List[str] = []
        for ctx in get_contexts_from_response(initial_ret):
            if ctx not in all_contexts:
                all_contexts.append(ctx)
        for ctx in get_contexts_from_response(hyp_ret):
            if ctx not in all_contexts:
                all_contexts.append(ctx)

        # Step 3: Generate the real answer from retrieved context
        context = self.build_context_string(hyp_ret) or self.build_context_string(initial_ret)
        answer = self._generate_answer(question, context, language)

        return BaselineResult(
            question=question,
            answer=answer,
            contexts=all_contexts,
            latency_sec=time.time() - t0,
            baseline_name=self.baseline_name,
            metadata={"hyp_doc_preview": hyp_doc[:150] if hyp_doc else ""},
        )

    def _generate_hypothetical_doc(self, question: str, language: str) -> str:
        """Generate a short hypothetical answer that looks like a document excerpt."""
        if language in ("arabic_msa", "Darija"):
            prompt = (
                "اكتب مقتطفاً وثائقياً قصيراً (3-4 جمل) من وثيقة رسمية يجيب على هذا السؤال. "
                "اكتب كأنك تقتبس من وثيقة رسمية مغربية — لا تذكر المصدر.\n\n"
                f"السؤال: {question}\n\nمقتطف من وثيقة رسمية:"
            )
        else:
            prompt = (
                "Écrivez un court extrait documentaire (3-4 phrases) d'un document officiel qui répond à cette question. "
                "Écrivez comme si vous citiez un document officiel marocain — ne mentionnez pas la source.\n\n"
                f"Question: {question}\n\nExtrait d'un document officiel:"
            )
        return self.client.generate(prompt, temperature=0.3, max_tokens=300) or question

    def _generate_answer(self, question: str, context: str, language: str) -> str:
        if language in ("arabic_msa", "Darija"):
            prompt = (
                "أجب على السؤال بناءً على الوثائق الرسمية المقدمة فقط. لا تخترع معلومات.\n\n"
                f"الوثائق:\n{context}\n\n"
                f"السؤال: {question}\n\nالإجابة:"
            )
        else:
            prompt = (
                "Répondez à la question uniquement à partir des documents officiels fournis. N'inventez rien.\n\n"
                f"Documents:\n{context}\n\n"
                f"Question: {question}\n\nRéponse:"
            )
        return self.client.generate(prompt, temperature=0.2, max_tokens=1024) or "[ERROR]"
