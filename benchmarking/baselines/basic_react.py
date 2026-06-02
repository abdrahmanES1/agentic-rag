# -*- coding: utf-8 -*-
"""
Basic ReACT Baseline — Yao et al. 2023.

Architecture:
  Thought/Action/Observation loop (max 3 iterations):
  - THOUGHT: analyze what information is needed
  - ACTION: retrieve using generated sub-query
  - OBSERVATION: evaluate retrieved chunks
  Final: synthesize from all observations.
"""

import re
import time
from typing import List

from benchmarking.baselines.base import BaselineRAG, BaselineResult
from benchmarking.shared import OllamaClient, get_contexts_from_response


class BasicReACT(BaselineRAG):

    baseline_name = "basic_react"

    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def run(self, question: str) -> BaselineResult:
        t0 = time.time()

        ret = self.retrieve(question)
        if ret.get("is_outscope"):
            return self._out_of_scope_result(question, ret.get("language", "arabic_msa"), time.time() - t0)

        language = ret.get("language", "arabic_msa")
        all_contexts: List[str] = []
        scratchpad: List[str] = []

        # Seed with initial retrieval
        initial_contexts = get_contexts_from_response(ret)
        all_contexts.extend(initial_contexts)

        for iteration in range(self.max_iterations):
            # THOUGHT: what sub-question do we need to answer next?
            thought_prompt = self._thought_prompt(question, scratchpad, language)
            thought = self.client.generate(thought_prompt, temperature=0.2, max_tokens=200)
            if not thought:
                break
            scratchpad.append(f"THOUGHT [{iteration + 1}]: {thought}")

            # ACTION: extract sub-query from thought and retrieve
            sub_query = self._extract_subquery(thought, question)
            action_ret = self.retrieve(sub_query)
            new_contexts = get_contexts_from_response(action_ret)
            for ctx in new_contexts:
                if ctx not in all_contexts:
                    all_contexts.append(ctx)
            scratchpad.append(f"ACTION [{iteration + 1}]: retrieve('{sub_query[:60]}')")

            # OBSERVATION: summarize retrieved info
            obs_prompt = self._observation_prompt(sub_query, new_contexts[:3], language)
            obs = self.client.generate(obs_prompt, temperature=0.1, max_tokens=300)
            scratchpad.append(f"OBSERVATION [{iteration + 1}]: {obs or '[empty]'}")

            if obs and ("غير موجود" in obs or "non disponible" in obs.lower() or "not found" in obs.lower()):
                break

        # Final synthesis from all observations
        synthesis_prompt = self._synthesis_prompt(question, scratchpad, all_contexts[:5], language)
        answer = self.client.generate(synthesis_prompt, temperature=0.2, max_tokens=1024)
        if not answer:
            answer = "[ERROR: empty response]"

        return BaselineResult(
            question=question,
            answer=answer,
            contexts=all_contexts,
            latency_sec=time.time() - t0,
            baseline_name=self.baseline_name,
            metadata={"scratchpad": scratchpad, "iterations": len(scratchpad) // 3},
        )

    def _thought_prompt(self, question: str, scratchpad: List[str], language: str) -> str:
        prior = "\n".join(scratchpad[-3:]) if scratchpad else "None"
        if language in ("arabic_msa", "Darija"):
            return (
                f"السؤال الأصلي: {question}\n"
                f"الخطوات السابقة:\n{prior}\n\n"
                "فكر: ما هي المعلومة التالية التي تحتاجها للإجابة الكاملة؟ اكتب سؤالاً فرعياً واحداً بإيجاز."
            )
        return (
            f"Question originale: {question}\n"
            f"Étapes précédentes:\n{prior}\n\n"
            "Réfléchis: quelle information supplémentaire te manque-t-il? Écris une sous-question brève."
        )

    def _extract_subquery(self, thought: str, fallback: str) -> str:
        for pattern in [r'"([^"]+)"', r"'([^']+)'", r":\s*(.{10,80})$"]:
            m = re.search(pattern, thought)
            if m:
                return m.group(1).strip()
        first_sentence = re.split(r"[.؟!]", thought)[0].strip()
        return first_sentence if len(first_sentence) > 10 else fallback

    def _observation_prompt(self, sub_query: str, contexts: List[str], language: str) -> str:
        ctx_text = "\n\n".join(contexts[:3]) if contexts else "(no results)"
        if language in ("arabic_msa", "Darija"):
            return f"السؤال: {sub_query}\nالوثائق:\n{ctx_text}\n\nأجب بجملة أو جملتين فقط:"
        return f"Question: {sub_query}\nDocuments:\n{ctx_text}\n\nRépondez en une ou deux phrases seulement:"

    def _synthesis_prompt(self, question: str, scratchpad: List[str], contexts: List[str], language: str) -> str:
        reasoning = "\n".join(scratchpad)
        ctx_text = "\n\n".join(contexts)
        if language in ("arabic_msa", "Darija"):
            return (
                "اجمع المعلومات التالية في إجابة شاملة. لا تخترع معلومات غير موجودة.\n\n"
                f"عملية التفكير:\n{reasoning}\n\n"
                f"الوثائق:\n{ctx_text}\n\n"
                f"السؤال: {question}\n\nالإجابة الشاملة:"
            )
        return (
            "Assemblez les informations suivantes en une réponse complète. N'inventez rien.\n\n"
            f"Raisonnement:\n{reasoning}\n\n"
            f"Documents:\n{ctx_text}\n\n"
            f"Question: {question}\n\nRéponse complète:"
        )
