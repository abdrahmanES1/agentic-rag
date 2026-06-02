# -*- coding: utf-8 -*-
"""
Naive RAG Baseline — Lewis et al. 2020.

Architecture:
  1. Retrieve: /api/retrieve → top-K chunks
  2. Generate: one Ollama call with [context] + [question]
  No loop, no planning, no tools, no verification.
"""

import time

from benchmarking.baselines.base import BaselineRAG, BaselineResult


class NaiveRAG(BaselineRAG):

    baseline_name = "naive_rag"

    def run(self, question: str) -> BaselineResult:
        t0 = time.time()

        ret = self.retrieve(question)
        if not ret:
            return BaselineResult(
                question=question, answer="[Retrieval failed]",
                contexts=[], latency_sec=time.time() - t0,
                baseline_name=self.baseline_name,
            )
        if ret.get("is_outscope"):
            return self._out_of_scope_result(question, ret.get("language", "arabic_msa"), time.time() - t0)

        context = self.build_context_string(ret)
        contexts = self.get_contexts(ret)
        language = ret.get("language", "arabic_msa")

        if language in ("arabic_msa", "Darija"):
            prompt = (
                "أنت مساعد إداري متخصص في الخدمات العامة المغربية.\n"
                "أجب على السؤال بناءً على الوثائق الرسمية المقدمة فقط. لا تخترع معلومات.\n\n"
                f"الوثائق:\n{context}\n\n"
                f"السؤال: {question}\n\nالإجابة:"
            )
        else:
            prompt = (
                "Vous êtes un assistant administratif spécialisé dans les services publics marocains.\n"
                "Répondez à la question uniquement à partir des documents officiels fournis. N'inventez rien.\n\n"
                f"Documents:\n{context}\n\n"
                f"Question: {question}\n\nRéponse:"
            )

        answer = self.client.generate(prompt, temperature=0.3, max_tokens=1024)
        if not answer:
            answer = "[ERROR: empty response]"

        return BaselineResult(
            question=question,
            answer=answer,
            contexts=contexts,
            latency_sec=time.time() - t0,
            baseline_name=self.baseline_name,
        )
