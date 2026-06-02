# -*- coding: utf-8 -*-
"""
FLARE Baseline — Forward-Looking Active REtrieval (Jiang et al. 2023).

Architecture:
  Generate the answer sentence by sentence. When the model's confidence
  for the next sentence is low (simulated via explicit LLM self-assessment),
  retrieve additional context before continuing.

  Since we don't have access to per-token log probabilities through Ollama's
  OpenAI-compatible API, we simulate low confidence by asking the LLM to
  explicitly flag uncertain sentences.

Reference: Jiang et al., "Active Retrieval Augmented Generation", 2023
"""

import re
import time
from typing import List, Tuple

from benchmarking.baselines.base import BaselineRAG, BaselineResult
from benchmarking.shared import get_contexts_from_response

MAX_SENTENCES = 5
UNCERTAINTY_MARKERS_AR = ["غير متأكد", "لا أعلم", "قد يكون", "من المحتمل", "يُحتمل"]
UNCERTAINTY_MARKERS_FR = ["je ne suis pas sûr", "peut-être", "il est possible", "probablement", "incertain"]


class FLARE(BaselineRAG):

    baseline_name = "flare"

    def run(self, question: str) -> BaselineResult:
        t0 = time.time()

        initial_ret = self.retrieve(question)
        if initial_ret.get("is_outscope"):
            return self._out_of_scope_result(question, initial_ret.get("language", "arabic_msa"), time.time() - t0)

        language = initial_ret.get("language", "arabic_msa")
        all_contexts: List[str] = list(get_contexts_from_response(initial_ret))
        current_context = self.build_context_string(initial_ret)

        answer_sentences: List[str] = []
        retrieval_count = 0

        for i in range(MAX_SENTENCES):
            # Ask model to generate the next sentence and flag uncertainty
            next_sent, is_uncertain, uncertain_phrase = self._generate_next_sentence(
                question, " ".join(answer_sentences), current_context, language
            )
            if not next_sent:
                break

            # Check for end signals
            if any(marker in next_sent for marker in ["[END]", "[FIN]", "[نهاية]"]):
                break

            if is_uncertain and retrieval_count < 3:
                retrieval_count += 1
                query = uncertain_phrase or next_sent
                extra_ret = self.retrieve(query)
                new_contexts = get_contexts_from_response(extra_ret)
                for ctx in new_contexts:
                    if ctx not in all_contexts:
                        all_contexts.append(ctx)
                extra_context = self.build_context_string(extra_ret)
                current_context = extra_context or current_context

                # Regenerate the sentence with additional context
                next_sent, _, _ = self._generate_next_sentence(
                    question, " ".join(answer_sentences), current_context, language
                )

            if next_sent:
                answer_sentences.append(next_sent)

            if len(answer_sentences) >= 3 and self._is_complete(answer_sentences, language):
                break

        answer = " ".join(answer_sentences) if answer_sentences else "[ERROR: no answer generated]"

        return BaselineResult(
            question=question,
            answer=answer,
            contexts=all_contexts,
            latency_sec=time.time() - t0,
            baseline_name=self.baseline_name,
            metadata={"sentences": len(answer_sentences), "retrieval_count": retrieval_count},
        )

    def _generate_next_sentence(
        self, question: str, answer_so_far: str, context: str, language: str
    ) -> Tuple[str, bool, str]:
        """
        Generate the next sentence of the answer.
        Returns (sentence, is_uncertain, uncertain_phrase).
        """
        if language in ("arabic_msa", "Darija"):
            prompt = (
                f"أنت تجيب على هذا السؤال جملة جملة.\n"
                f"سؤال: {question}\n"
                f"الجواب حتى الآن: {answer_so_far or '(بداية الجواب)'}\n"
                f"الوثائق: {context[:800]}\n\n"
                "اكتب الجملة التالية فقط. "
                "إذا لم تكن متأكداً، ضع [UNCERTAIN: ...الجزء غير المؤكد...] في الجملة. "
                "إذا انتهى الجواب اكتب [نهاية].\n"
                "الجملة التالية:"
            )
        else:
            prompt = (
                f"Vous répondez à cette question phrase par phrase.\n"
                f"Question: {question}\n"
                f"Réponse jusqu'ici: {answer_so_far or '(début de la réponse)'}\n"
                f"Documents: {context[:800]}\n\n"
                "Écrivez uniquement la phrase suivante. "
                "Si vous n'êtes pas sûr, mettez [UNCERTAIN: ...partie incertaine...]. "
                "Si la réponse est complète, écrivez [FIN].\n"
                "Phrase suivante:"
            )
        raw = self.client.generate(prompt, temperature=0.2, max_tokens=200) or ""

        is_uncertain = False
        uncertain_phrase = ""
        m = re.search(r"\[UNCERTAIN:\s*([^\]]+)\]", raw)
        if m:
            is_uncertain = True
            uncertain_phrase = m.group(1).strip()
            raw = raw.replace(m.group(0), uncertain_phrase)

        # Also check for natural uncertainty markers
        for marker in (UNCERTAINTY_MARKERS_AR if language in ("arabic_msa", "Darija") else UNCERTAINTY_MARKERS_FR):
            if marker in raw.lower():
                is_uncertain = True
                uncertain_phrase = uncertain_phrase or raw.strip()
                break

        return raw.strip(), is_uncertain, uncertain_phrase

    def _is_complete(self, sentences: List[str], language: str) -> bool:
        last = sentences[-1] if sentences else ""
        end_markers = [".", "!", "؟"]
        return any(last.rstrip().endswith(m) for m in end_markers)
