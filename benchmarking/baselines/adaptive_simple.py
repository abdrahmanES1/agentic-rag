# -*- coding: utf-8 -*-
"""
Adaptive Simple Baseline — Jeong et al. 2024.

Architecture:
  Classify question complexity first, then route:
  - SIMPLE  → single retrieval + one-shot generation (same as Naive RAG)
  - COMPLEX → two-stage: decompose sub-questions → retrieve per sub-Q → synthesize

This is the classification-based routing baseline.
"""

import re
import time
from typing import List, Tuple

from benchmarking.baselines.base import BaselineRAG, BaselineResult
from benchmarking.shared import get_contexts_from_response


class AdaptiveSimple(BaselineRAG):

    baseline_name = "adaptive_simple"

    def run(self, question: str) -> BaselineResult:
        t0 = time.time()

        ret = self.retrieve(question)
        if ret.get("is_outscope"):
            return self._out_of_scope_result(question, ret.get("language", "arabic_msa"), time.time() - t0)

        language = ret.get("language", "arabic_msa")
        is_complex = self._classify_complexity(question, language)

        if not is_complex:
            return self._simple_path(question, ret, language, t0)
        return self._complex_path(question, ret, language, t0)

    def _classify_complexity(self, question: str, language: str) -> bool:
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"هل هذا السؤال يحتاج لإجابات متعددة (مثلاً يسأل عن وثائق وتكلفة ومدة معاً)؟\n"
                f"السؤال: {question}\n\n"
                "أجب فقط: SIMPLE أو COMPLEX"
            )
        else:
            prompt = (
                f"Cette question nécessite-t-elle plusieurs informations différentes (ex: documents ET coût ET délai)?\n"
                f"Question: {question}\n\n"
                "Répondez uniquement: SIMPLE ou COMPLEX"
            )
        result = self.client.generate(prompt, temperature=0.0, max_tokens=10)
        return "COMPLEX" in (result or "").upper()

    def _simple_path(self, question: str, ret: dict, language: str, t0: float) -> BaselineResult:
        context = self.build_context_string(ret)
        contexts = self.get_contexts(ret)
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"أجب على السؤال بناءً على الوثائق فقط:\n\n{context}\n\n"
                f"السؤال: {question}\n\nالإجابة:"
            )
        else:
            prompt = (
                f"Répondez uniquement à partir des documents:\n\n{context}\n\n"
                f"Question: {question}\n\nRéponse:"
            )
        answer = self.client.generate(prompt, temperature=0.2, max_tokens=1024) or "[ERROR]"
        return BaselineResult(
            question=question, answer=answer, contexts=contexts,
            latency_sec=time.time() - t0, baseline_name=self.baseline_name,
            metadata={"path": "simple"},
        )

    def _complex_path(self, question: str, ret: dict, language: str, t0: float) -> BaselineResult:
        sub_questions = self._decompose(question, language)
        all_contexts: List[str] = self.get_contexts(ret)
        sub_answers: List[Tuple[str, str]] = []

        for sub_q in sub_questions:
            sub_ret = self.retrieve(sub_q)
            sub_contexts = get_contexts_from_response(sub_ret)
            for ctx in sub_contexts:
                if ctx not in all_contexts:
                    all_contexts.append(ctx)
            sub_ctx = self.build_context_string(sub_ret)
            if language in ("arabic_msa", "Darija"):
                sub_prompt = f"أجب بإيجاز:\n{sub_ctx}\n\nالسؤال: {sub_q}\n\nالإجابة المختصرة:"
            else:
                sub_prompt = f"Répondez brièvement:\n{sub_ctx}\n\nQuestion: {sub_q}\n\nRéponse courte:"
            sub_ans = self.client.generate(sub_prompt, temperature=0.1, max_tokens=400) or ""
            sub_answers.append((sub_q, sub_ans))

        synthesis = self._synthesize(question, sub_answers, language)
        return BaselineResult(
            question=question, answer=synthesis, contexts=all_contexts,
            latency_sec=time.time() - t0, baseline_name=self.baseline_name,
            metadata={"path": "complex", "sub_questions": [sq for sq, _ in sub_answers]},
        )

    def _decompose(self, question: str, language: str) -> List[str]:
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"اكتب 2-3 أسئلة فرعية للإجابة على:\n{question}\n\n"
                "اكتب فقط الأسئلة، رقم ونقطة:"
            )
        else:
            prompt = (
                f"Écrivez 2-3 sous-questions pour répondre à:\n{question}\n\n"
                "Écrivez uniquement les questions, numérotées:"
            )
        raw = self.client.generate(prompt, temperature=0.2, max_tokens=200) or ""
        sub_qs = []
        for line in raw.split("\n"):
            m = re.match(r"^\d+[\.\)]\s*(.+)$", line.strip())
            if m and len(m.group(1).split()) >= 2:
                sub_qs.append(m.group(1).strip())
        return sub_qs[:3] if sub_qs else [question]

    def _synthesize(self, question: str, sub_answers: List[Tuple[str, str]], language: str) -> str:
        parts = "\n".join(f"  {i + 1}. {sq}: {ans[:200]}" for i, (sq, ans) in enumerate(sub_answers))
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"اجمع الإجابات التالية في جواب شامل:\n{parts}\n\n"
                f"السؤال الأصلي: {question}\n\nالإجابة الشاملة:"
            )
        else:
            prompt = (
                f"Assemblez les réponses suivantes:\n{parts}\n\n"
                f"Question originale: {question}\n\nRéponse complète:"
            )
        return self.client.generate(prompt, temperature=0.2, max_tokens=1024) or "[ERROR]"
