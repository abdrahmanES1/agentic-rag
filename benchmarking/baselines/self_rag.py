# -*- coding: utf-8 -*-
"""
Self-RAG Baseline — Asai et al. 2023.

Architecture (prompt-based, no special fine-tuned model needed):
  1. Reflection token 1: [Retrieve?]  — should we retrieve for this question?
  2. If yes → retrieve context
  3. Reflection token 2: [IsRel?]    — is the retrieved doc relevant?
  4. Reflection token 3: [IsSup?]    — is the generated answer supported?
  5. Reflection token 4: [IsUse?]    — is the answer useful to the user?

We simulate the reflection tokens via explicit LLM calls (not logit-level
token generation). This gives us the Self-RAG behavior without requiring
a fine-tuned model.

Reference: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", 2023
"""

import time
from typing import List, Tuple

from benchmarking.baselines.base import BaselineRAG, BaselineResult
from benchmarking.shared import get_contexts_from_response


class SelfRAG(BaselineRAG):

    baseline_name = "self_rag"

    def run(self, question: str) -> BaselineResult:
        t0 = time.time()

        # First retrieve to get language signal
        initial_ret = self.retrieve(question)
        if initial_ret.get("is_outscope"):
            return self._out_of_scope_result(question, initial_ret.get("language", "arabic_msa"), time.time() - t0)

        language = initial_ret.get("language", "arabic_msa")
        all_contexts: List[str] = []
        reflections: List[str] = []

        # Reflection 1: should we retrieve?
        should_retrieve = self._reflect_retrieve(question, language)
        reflections.append(f"[Retrieve]: {'YES' if should_retrieve else 'NO'}")

        if should_retrieve:
            all_contexts = get_contexts_from_response(initial_ret)
            context = self.build_context_string(initial_ret)

            # Reflection 2: is the retrieved doc relevant?
            is_relevant = self._reflect_relevance(question, all_contexts[:3], language)
            reflections.append(f"[IsRel]: {'YES' if is_relevant else 'NO'}")

            if not is_relevant:
                # Retry with a rephrased query
                rephrased = self._rephrase_query(question, language)
                retry_ret = self.retrieve(rephrased)
                new_contexts = get_contexts_from_response(retry_ret)
                all_contexts = list(dict.fromkeys(all_contexts + new_contexts))
                context = self.build_context_string(retry_ret) or context
        else:
            context = ""

        # Generate answer
        answer = self._generate(question, context, language)

        # Reflection 3: is the answer supported by context?
        is_supported = self._reflect_supported(answer, all_contexts[:3], language) if all_contexts else False
        reflections.append(f"[IsSup]: {'YES' if is_supported else 'NO'}")

        # Reflection 4: is the answer useful?
        is_useful = self._reflect_useful(question, answer, language)
        reflections.append(f"[IsUse]: {'YES' if is_useful else 'NO'}")

        if not is_supported and all_contexts:
            answer = self._regenerate_grounded(question, context, answer, language)

        return BaselineResult(
            question=question,
            answer=answer,
            contexts=all_contexts,
            latency_sec=time.time() - t0,
            baseline_name=self.baseline_name,
            metadata={
                "reflections": reflections,
                "is_supported": is_supported,
                "is_useful": is_useful,
            },
        )

    def _reflect_retrieve(self, question: str, language: str) -> bool:
        if language in ("arabic_msa", "Darija"):
            prompt = f"هل يحتاج هذا السؤال للبحث في وثائق للإجابة؟\n{question}\nأجب: نعم أو لا"
        else:
            prompt = f"Cette question nécessite-t-elle de chercher dans des documents?\n{question}\nRépondez: oui ou non"
        result = self.client.generate(prompt, temperature=0.0, max_tokens=10) or ""
        return "نعم" in result or "oui" in result.lower() or "yes" in result.lower()

    def _reflect_relevance(self, question: str, contexts: List[str], language: str) -> bool:
        ctx = "\n\n".join(contexts[:2])
        if language in ("arabic_msa", "Darija"):
            prompt = f"هل هذه الوثائق ذات صلة بالسؤال؟\nسؤال: {question}\nوثائق: {ctx[:500]}\nأجب: نعم أو لا"
        else:
            prompt = f"Ces documents sont-ils pertinents pour la question?\nQuestion: {question}\nDocs: {ctx[:500]}\nRépondez: oui ou non"
        result = self.client.generate(prompt, temperature=0.0, max_tokens=10) or ""
        return "نعم" in result or "oui" in result.lower() or "yes" in result.lower()

    def _reflect_supported(self, answer: str, contexts: List[str], language: str) -> bool:
        ctx = "\n\n".join(contexts[:2])
        if language in ("arabic_msa", "Darija"):
            prompt = f"هل الجواب مدعوم بالوثائق؟\nجواب: {answer[:300]}\nوثائق: {ctx[:500]}\nأجب: نعم أو لا"
        else:
            prompt = f"La réponse est-elle soutenue par les documents?\nRéponse: {answer[:300]}\nDocs: {ctx[:500]}\nRépondez: oui ou non"
        result = self.client.generate(prompt, temperature=0.0, max_tokens=10) or ""
        return "نعم" in result or "oui" in result.lower() or "yes" in result.lower()

    def _reflect_useful(self, question: str, answer: str, language: str) -> bool:
        if language in ("arabic_msa", "Darija"):
            prompt = f"هل الإجابة مفيدة للسؤال؟\nسؤال: {question}\nإجابة: {answer[:200]}\nأجب: نعم أو لا"
        else:
            prompt = f"La réponse est-elle utile pour la question?\nQuestion: {question}\nRéponse: {answer[:200]}\nRépondez: oui ou non"
        result = self.client.generate(prompt, temperature=0.0, max_tokens=10) or ""
        return "نعم" in result or "oui" in result.lower() or "yes" in result.lower()

    def _rephrase_query(self, question: str, language: str) -> str:
        if language in ("arabic_msa", "Darija"):
            prompt = f"أعد صياغة هذا السؤال بكلمات مختلفة للبحث:\n{question}\nالصياغة الجديدة:"
        else:
            prompt = f"Reformulez cette question avec des mots différents pour la recherche:\n{question}\nNouvelle formulation:"
        return self.client.generate(prompt, temperature=0.3, max_tokens=100) or question

    def _generate(self, question: str, context: str, language: str) -> str:
        if context:
            if language in ("arabic_msa", "Darija"):
                prompt = f"أجب على السؤال من الوثائق فقط:\n{context}\n\nسؤال: {question}\nإجابة:"
            else:
                prompt = f"Répondez à partir des documents uniquement:\n{context}\n\nQuestion: {question}\nRéponse:"
        else:
            if language in ("arabic_msa", "Darija"):
                prompt = f"أجب على هذا السؤال المتعلق بالخدمات المغربية:\n{question}\nإجابة:"
            else:
                prompt = f"Répondez à cette question sur les services marocains:\n{question}\nRéponse:"
        return self.client.generate(prompt, temperature=0.2, max_tokens=1024) or "[ERROR]"

    def _regenerate_grounded(self, question: str, context: str, prev_answer: str, language: str) -> str:
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"الإجابة السابقة لم تكن مدعومة بالوثائق. أعد الإجابة بناءً على الوثائق فقط.\n\n"
                f"الوثائق:\n{context}\n\nالسؤال: {question}\nالإجابة الجديدة المدعومة:"
            )
        else:
            prompt = (
                f"La réponse précédente n'était pas soutenue. Répondez uniquement à partir des documents.\n\n"
                f"Documents:\n{context}\n\nQuestion: {question}\nNouvelle réponse soutenue:"
            )
        return self.client.generate(prompt, temperature=0.1, max_tokens=1024) or prev_answer
