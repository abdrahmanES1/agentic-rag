# -*- coding: utf-8 -*-
"""
QAGenerator — uses the already-running Ollama model to produce Q&A items from
scraped text sections, in the exact schema of benchmark_testset_gold.json.

For each text section, the LLM is asked to generate 2–4 Q&A items covering:
  - SIMPLE factual question (documents needed, deadline, cost, eligibility)
  - LEGAL  if the section cites a decree/article number
  - MULTIHOP if the section links two steps or references another procedure

The output is a JSON array parsed directly — no post-processing required.
"""

import json
import logging
import re
import time
from typing import List, Optional, Optional

log = logging.getLogger("scraper.qa_generator")

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior NLP engineer building a state-of-the-art evaluation benchmark for a \
Moroccan government RAG (Retrieval-Augmented Generation) system.

Your task: given a passage from a Moroccan government procedure page, generate high-quality \
Q&A items that rigorously test the RAG system.

QUALITY RULES — follow all of them:
1. Questions must sound like real citizen questions (not academic).
2. ATOMIC & SELF-CONTAINED: Every question must be fully understandable on its own — \
   no references to "the mentioned channels", "the above", "instead of what was mentioned", \
   "السابق ذكره", "المذكور/المذكورة", "القنوات المذكورة", "ci-dessus", "susmentionné", \
   "بدلاً من القنوات", or ANY implicit context from a previous question. \
   Name everything explicitly in the question itself.
3. gold_answer must include the EXACT names, numbers, costs, deadlines, or document titles \
   from the passage — not generic descriptions.
4. gold_keywords must be 2-5 SPECIFIC terms that will definitely appear in a correct answer \
   (e.g. "صورتان شمسيتان", "30 يوما", "بالمجان") — NOT generic words like "وثائق".
5. For Arabic passages → also generate a French translation of the best question.
6. For each passage → include one OUTSCOPE question about something plausibly related \
   but NOT answered by this passage (should_abstain: true).
7. category values: SIMPLE | LEGAL | MULTIHOP | OUTSCOPE
   - SIMPLE: factual — documents needed, cost, deadline, eligibility
   - LEGAL: question specifically about a cited decree/article/loi number
   - MULTIHOP: requires combining two facts from the passage or a linked procedure
   - OUTSCOPE: question the system should refuse to answer (not in passage)
8. language values: arabic_msa | french | Darija | Arabizi
9. Start your response with the [ character immediately. No preamble, no explanation, no thinking text."""

_USER_PROMPT = """\
Source URL: {url}
Source site: {source}
Section heading: {heading}

Text passage ({lang}):
\"\"\"
{text}
\"\"\"

Generate 5-7 Q&A items. Return a JSON array — EXACTLY this schema, no extra fields:
[
  {{
    "category": "SIMPLE",
    "language": "arabic_msa",
    "question": "ما هي الوثائق المطلوبة للحصول على رخصة البناء؟",
    "gold_answer": "وفقاً للإجراء، الوثائق المطلوبة هي: 1. نسخة من بطاقة التعريف الوطنية 2. شهادة الملكية حديثة 3. التصميم المعماري الموقع من مهندس معماري معتمد. التكلفة: بالمجان. المدة: 30 يوما.",
    "gold_keywords": ["بطاقة التعريف الوطنية", "شهادة الملكية حديثة", "التصميم المعماري", "30 يوما"],
    "source": "{url} — {heading}",
    "should_abstain": false,
    "is_multihop": false
  }},
  {{
    "category": "SIMPLE",
    "language": "french",
    "question": "Quels sont les documents requis pour obtenir un permis de construire?",
    "gold_answer": "Selon la procédure, les documents requis sont: 1. Copie de la carte nationale d'identité 2. Certificat de propriété récent 3. Plan architectural signé par un architecte agréé. Coût: gratuit. Délai: 30 jours.",
    "gold_keywords": ["carte nationale d'identité", "certificat de propriété", "plan architectural", "30 jours"],
    "source": "{url} — {heading}",
    "should_abstain": false,
    "is_multihop": false
  }},
  {{
    "category": "DARIJA",
    "language": "Darija",
    "question": "شنو هي الوراق اللي خاصني باش نجيب رخصة البناء؟",
    "gold_answer": "حسب الإجراء، خاصك تجيب: 1. نسخة من بطاقة التعريف الوطنية 2. شهادة الملكية حديثة 3. التصميم المعماري الموقع من مهندس معماري معتمد. التكلفة: بالمجان. المدة: 30 يوما.",
    "gold_keywords": ["بطاقة التعريف الوطنية", "شهادة الملكية حديثة", "30 يوما"],
    "source": "{url} — {heading}",
    "should_abstain": false,
    "is_multihop": false
  }},
  {{
    "category": "ARABIZI",
    "language": "Arabizi",
    "question": "chno hia lwaraq li khassni bash njib rkhssat lbna?",
    "gold_answer": "حسب الإجراء، خاصك تجيب: 1. نسخة من بطاقة التعريف الوطنية 2. شهادة الملكية حديثة 3. التصميم المعماري الموقع من مهندس معماري معتمد. التكلفة: بالمجان. المدة: 30 يوما.",
    "gold_keywords": ["بطاقة التعريف الوطنية", "شهادة الملكية حديثة", "30 يوما"],
    "source": "{url} — {heading}",
    "should_abstain": false,
    "is_multihop": false
  }}
]

MANDATORY — generate exactly these 4 items, in this exact order:
1. SIMPLE arabic_msa  — about documents/cost/deadline, use EXACT values from the text
2. SIMPLE french      — same question in French, gold_answer in French with exact terms
3. DARIJA Darija      — same question rephrased in Moroccan dialect (واش/خاصني/باش/دابا/كاين/شنو/كيفاش), gold_answer MUST be in arabic_msa (Arabic script)
4. ARABIZI Arabizi    — same question romanized (wach/khassni/bash/daba/kayen/chno/kifach), gold_answer MUST be in arabic_msa (Arabic script ع ا ر ب ي — NEVER Latin/romanized)

OPTIONAL — add AFTER the 4 above if applicable:
5. LEGAL arabic_msa   — IF the passage cites a specific decree/article/loi number
6. MULTIHOP arabic_msa — IF passage has an "إجراءات ذات صلة" section with linked procedures.
   MULTIHOP rules:
   - The question MUST combine facts from TWO DIFFERENT procedures or sources
   - Do NOT reuse the pattern "المغاربة المقيمون في الخارج + بطاقة التعريف" — this is overused
   - Instead use: comparing costs/deadlines of two linked procedures, eligibility across steps,
     or documents required for the linked procedure referenced in "إجراءات ذات صلة"

KEY RULES:
- gold_keywords must be EXACT strings copied from the passage (not paraphrased)
- gold_answer for DARIJA and ARABIZI must be in arabic_msa (Arabic script ع ا ر ب ي — NEVER Latin)
- gold_answer must contain ALL gold_keywords
- Stop after item 4 (plus optional 5-6). Do not add OUTSCOPE here."""


_OUTSCOPE_SYSTEM = """\
You generate exactly ONE out-of-scope question for a Moroccan government RAG benchmark.
The question must be plausible (sounds related) but NOT answerable from the given procedure.
Output ONLY a valid JSON object — no markdown, no explanation, start with {"""

_OUTSCOPE_PROMPT = """\
Procedure title: {heading}
Procedure summary: {summary}

Generate 1 question that sounds related to this procedure but is NOT answerable from it.
Examples of good OUTSCOPE topics: appeals process, online submission availability, \
exceptions for special cases, related procedures not mentioned, legal consequences.

Return ONLY this JSON object:
{{
  "category": "OUTSCOPE",
  "language": "arabic_msa",
  "question": "[plausible question NOT in the procedure]",
  "gold_answer": "هذه المعلومة غير متوفرة في هذا الإجراء.",
  "gold_keywords": ["غير متوفر"],
  "source": "{url} — {heading}",
  "should_abstain": true,
  "is_multihop": false
}}"""


_MULTIHOP_SYSTEM = """\
You generate exactly ONE multi-hop question for a Moroccan government RAG benchmark.
The question MUST require combining facts from TWO different procedures.
Output ONLY a valid JSON object — no markdown, no explanation, start with {"""

_MULTIHOP_PROMPT = """\
Main procedure title: {heading}
Main procedure text:
\"\"\"
{text}
\"\"\"

The text above contains an "إجراءات ذات صلة" (related procedures) section listing linked procedures.

Generate 1 multi-hop question that requires information from BOTH the main procedure AND \
one of the linked procedures. Use one of these patterns:
- Compare the cost or deadline of the main procedure vs. the linked procedure
- Ask what prerequisite documents are needed for the linked procedure before starting the main one
- Ask about the total timeline if both procedures must be completed sequentially
- Ask about eligibility conditions that span both procedures

STRICT RULES:
- The question must NAME both procedures explicitly (do NOT say "هذا الإجراء")
- Do NOT use the overused pattern "المغاربة المقيمون في الخارج + بطاقة التعريف الوطنية"
- gold_answer must contain EXACT values (costs, delays, doc names) from the text
- gold_keywords: 3-5 exact strings from the passage

Return ONLY this JSON object:
{{
  "category": "MULTIHOP",
  "language": "arabic_msa",
  "question": "[self-contained question naming both procedures]",
  "gold_answer": "[detailed answer combining facts from both procedures, with exact values]",
  "gold_keywords": ["[exact term 1]", "[exact term 2]", "[exact term 3]"],
  "source": "{url} — {heading}",
  "should_abstain": false,
  "is_multihop": true
}}"""


_LEGAL_SYSTEM = """\
You generate exactly ONE legal-reference question for a Moroccan government RAG benchmark.
The question must ask specifically about a cited decree, article, or law number in the text.
Output ONLY a valid JSON object — no markdown, no explanation, start with {"""

_LEGAL_PROMPT = """\
Procedure title: {heading}
Procedure text:
\"\"\"
{text}
\"\"\"

The text cites specific legal references (مرسوم/قانون رقم/ظهير/منشور/قرار).

Generate 1 question about one of those specific legal references. Examples:
- "ما هو رقم المرسوم الذي ينظم هذه العملية؟"
- "ما هي الشروط التي يحددها قانون رقم X؟"
- "متى صدر الظهير المنظم لهذا الإجراء؟"

STRICT RULES:
- The question must cite the EXACT law/decree number or name from the text
- gold_answer must quote the EXACT legal reference string from the passage
- gold_keywords: include the exact decree/law number/name

Return ONLY this JSON object:
{{
  "category": "LEGAL",
  "language": "arabic_msa",
  "question": "[question citing specific legal reference from text]",
  "gold_answer": "[answer quoting exact legal reference from the text]",
  "gold_keywords": ["[exact decree/law number]", "[exact article or date if present]"],
  "source": "{url} — {heading}",
  "should_abstain": false,
  "is_multihop": false
}}"""


class QAGenerator:
    """
    Generates Q&A pairs from text sections using Ollama.

    Reuses the OllamaClient from benchmarking/shared.py — no new dependency.
    """

    # Generation hyper-parameters tuned for structured JSON Q&A output.
    # temperature=0.2  : reliable JSON schema adherence + enough variety across 2,533 procedures
    # repetition_penalty=1.0 : NO penalty — gold_answer must repeat exact document names from the
    #                          procedure body; any penalty > 1.05 breaks gold_keywords matching
    _TEMPERATURE        = 0.2
    _REPETITION_PENALTY = 1.0
    _MAX_TOKENS         = 3072   # thinking text can be 800+ tokens; need room for the JSON

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434/v1",
        model: str = "gemma4:e4b",
        max_retries: int = 2,
        delay_between_calls: float = 0.5,
    ) -> None:
        from benchmarking.shared import OllamaClient
        self.client = OllamaClient(base_url=ollama_base_url, model=model)
        self.model = model
        self.max_retries = max_retries
        self.delay = delay_between_calls

    def generate(
        self,
        text: str,
        url: str,
        heading: str = "",
        lang: str = "ar",
        source_site: str = "",
    ) -> List[dict]:
        """
        Generate Q&A items from a single text section.

        Parameters
        ----------
        text        : the passage content
        url         : source URL (used in source field)
        heading     : section heading (used in source field)
        lang        : "ar" or "fr" (hint for the LLM)
        source_site : domain name (e.g. "idarati.ma")

        Returns
        -------
        List of raw Q&A dicts (not yet assigned IDs)
        """
        if len(text.split()) < 30:
            return []   # too short to generate meaningful questions

        lang_label = "Arabic" if lang == "ar" else "French"
        prompt = _USER_PROMPT.format(
            url=url,
            source=source_site or url,
            text=text[:3000],   # cap to avoid context overflow
            lang=lang_label,
            heading=heading or url,
        )

        raw = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.generate(
                    prompt=prompt,
                    system=_SYSTEM_PROMPT,
                    max_tokens=self._MAX_TOKENS,
                    temperature=self._TEMPERATURE,
                    repetition_penalty=self._REPETITION_PENALTY,
                )
                break
            except Exception as exc:
                log.warning("[qa] LLM call failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                time.sleep(1.0)

        if not raw:
            return []

        items = self._parse_json(raw)

        # Second call: OUTSCOPE (always — model skips it when bundled with 4+ items)
        outscope = self._generate_outscope(heading=heading, summary=text[:300], url=url)
        if outscope:
            items.append(outscope)

        # Third call: MULTIHOP — only when passage has linked procedures section
        if "إجراءات ذات صلة" in text:
            multihop = self._generate_multihop(heading=heading, text=text[:3000], url=url)
            if multihop:
                items.append(multihop)

        # Fourth call: LEGAL — only when passage cites a specific decree/law/article
        _legal_markers = ("مرسوم", "قانون رقم", "ظهير", "منشور", "قرار وزير")
        if any(m in text for m in _legal_markers):
            legal = self._generate_legal(heading=heading, text=text[:3000], url=url)
            if legal:
                items.append(legal)

        time.sleep(self.delay)
        return items

    def _generate_outscope(self, heading: str, summary: str, url: str) -> Optional[dict]:
        """
        Generate a single OUTSCOPE item via a short dedicated prompt.
        Returns a valid item dict or None on failure.
        """
        prompt = _OUTSCOPE_PROMPT.format(
            heading=heading[:200],
            summary=summary,
            url=url,
        )
        raw = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.generate(
                    prompt=prompt,
                    system=_OUTSCOPE_SYSTEM,
                    max_tokens=400,
                    temperature=self._TEMPERATURE,
                    repetition_penalty=self._REPETITION_PENALTY,
                )
                break
            except Exception as exc:
                log.warning("[qa] OUTSCOPE call failed (attempt %d/%d): %s",
                            attempt, self.max_retries, exc)
                time.sleep(1.0)

        if not raw:
            return None

        # Parse single JSON object (not array)
        # Strip <think> blocks first — they may contain { chars that break extraction
        raw = self._strip_thinking(raw)
        # Find first { ... } block
        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            return None
        snippet = re.sub(r",\s*([}\]])", r"\1", raw[start: end + 1])
        try:
            item = json.loads(snippet)
            if not isinstance(item, dict):
                return None
            question = item.get("question", "").strip()
            if len(question.split()) < 4:
                return None
            # Force correct values — OUTSCOPE items have fixed answer/keywords
            item["category"]      = "OUTSCOPE"
            item["language"]      = self._normalise_language(item.get("language", "arabic_msa"))
            item["gold_answer"]   = "هذه المعلومة غير متوفرة في هذا الإجراء."
            item["gold_keywords"] = ["غير متوفر", "غير متاح"]
            item["should_abstain"] = True
            item["is_multihop"]   = False
            item.setdefault("source", f"{url} — {heading}")
            return item
        except json.JSONDecodeError:
            pass
        return None

    def _generate_multihop(self, heading: str, text: str, url: str) -> Optional[dict]:
        """
        Generate a single MULTIHOP item for passages with إجراءات ذات صلة.
        Returns a valid item dict or None on failure.
        """
        prompt = _MULTIHOP_PROMPT.format(
            heading=heading[:200],
            text=text,
            url=url,
        )
        raw = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.generate(
                    prompt=prompt,
                    system=_MULTIHOP_SYSTEM,
                    max_tokens=500,
                    temperature=self._TEMPERATURE,
                    repetition_penalty=self._REPETITION_PENALTY,
                )
                break
            except Exception as exc:
                log.warning("[qa] MULTIHOP call failed (attempt %d/%d): %s",
                            attempt, self.max_retries, exc)
                time.sleep(1.0)
        if not raw:
            return None
        raw = self._strip_thinking(raw)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return None
        snippet = re.sub(r",\s*([}\]])", r"\1", raw[start: end + 1])
        try:
            item = json.loads(snippet)
            if not isinstance(item, dict):
                return None
            item["category"]   = "MULTIHOP"
            item["is_multihop"] = True
            item["language"] = self._normalise_language(item.get("language", "arabic_msa"))
            item.setdefault("should_abstain", False)
            item.setdefault("source", f"{url} — {heading}")
            if self._is_valid(item):
                return item
        except json.JSONDecodeError:
            pass
        return None

    def _generate_legal(self, heading: str, text: str, url: str) -> Optional[dict]:
        """
        Generate a single LEGAL item for passages that cite a decree/law/article.
        Returns a valid item dict or None on failure.
        """
        prompt = _LEGAL_PROMPT.format(
            heading=heading[:200],
            text=text,
            url=url,
        )
        raw = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.generate(
                    prompt=prompt,
                    system=_LEGAL_SYSTEM,
                    max_tokens=400,
                    temperature=self._TEMPERATURE,
                    repetition_penalty=self._REPETITION_PENALTY,
                )
                break
            except Exception as exc:
                log.warning("[qa] LEGAL call failed (attempt %d/%d): %s",
                            attempt, self.max_retries, exc)
                time.sleep(1.0)
        if not raw:
            return None
        raw = self._strip_thinking(raw)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return None
        snippet = re.sub(r",\s*([}\]])", r"\1", raw[start: end + 1])
        try:
            item = json.loads(snippet)
            if not isinstance(item, dict):
                return None
            item["category"]    = "LEGAL"
            item["is_multihop"] = False
            item["language"] = self._normalise_language(item.get("language", "arabic_msa"))
            item.setdefault("should_abstain", False)
            item.setdefault("source", f"{url} — {heading}")
            if self._is_valid(item):
                return item
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _strip_thinking(raw: str) -> str:
        """Remove <think>...</think> / <thinking>...</thinking> blocks before parsing."""
        raw = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"```(?:json)?", "", raw)
        return raw.strip()

    @staticmethod
    def _normalise_language(lang: str) -> str:
        """Map model language variants to canonical _VALID_LANGUAGES values."""
        _MAP = {
            "arabic_msa": "arabic_msa", "arabic": "arabic_msa", "ar": "arabic_msa",
            "msa": "arabic_msa", "arabic msa": "arabic_msa",
            "french": "french", "fr": "french", "français": "french", "francais": "french",
            "darija": "Darija", "moroccan arabic": "Darija", "moroccan": "Darija",
            "arabizi": "Arabizi", "arab romanized": "Arabizi",
        }
        return _MAP.get(lang.lower().strip(), lang)

    def _parse_json(self, raw: str) -> List[dict]:
        """
        Extract and parse the JSON array from the LLM response.
        Handles: <think> blocks, markdown fences, leading text, trailing commas.
        """
        if not raw:
            return []

        # Strip <think>...</think> blocks FIRST — they may contain { and [ chars
        # that confuse the bracket scanner below.
        raw = self._strip_thinking(raw)

        # Find the LAST [ ... ] block — model sometimes outputs explanatory text
        # before the actual JSON, so we want the last well-formed array.
        # Try from each '[' occurrence, pick the first that parses cleanly.
        bracket_positions = [i for i, c in enumerate(raw) if c == "["]
        for start in bracket_positions:
            candidate = raw[start:]
            # Find the matching closing bracket
            end = candidate.rfind("]")
            if end == -1:
                continue
            snippet = candidate[: end + 1]
            # Fix common trailing-comma issues before parsing
            snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                items = json.loads(snippet)
                if not isinstance(items, list):
                    continue
                valid = []
                for item in items:
                    if isinstance(item, dict):
                        # Normalise category to uppercase
                        if "category" in item:
                            item["category"] = item["category"].upper()
                        # Normalise language to canonical value
                        if "language" in item:
                            item["language"] = self._normalise_language(item["language"])
                        if self._is_valid(item):
                            valid.append(item)
                        else:
                            log.debug("[qa] Skipping invalid item: cat=%s lang=%s q_words=%d a_words=%d kws=%d",
                                      item.get("category"), item.get("language"),
                                      len((item.get("question") or "").split()),
                                      len((item.get("gold_answer") or "").split()),
                                      len(item.get("gold_keywords") or []))
                if valid:
                    return valid
            except json.JSONDecodeError:
                continue

        log.debug("[qa] No valid JSON array found in response (len=%d)", len(raw))
        return []

    _VALID_CATEGORIES = {"SIMPLE", "MULTIHOP", "LEGAL", "DARIJA", "ARABIZI", "OUTSCOPE", "MULTIHOP_HARD", "EDGE"}
    _VALID_LANGUAGES  = {"arabic_msa", "french", "Darija", "Arabizi"}

    # Phrases that make a question context-dependent (non-atomic).
    # Any question containing one of these is rejected.
    _NON_ATOMIC_PATTERNS = [
        # Arabic — bare demonstratives referencing unnamed prior context
        "هذا الإجراء", "بهذا الإجراء", "لهذا الإجراء", "من هذا الإجراء",
        "المذكورة", "المذكور", "السابق ذكره", "المشار إليه", "المشار إليها",
        "القنوات المذكورة", "الطريقة المذكورة", "الخطوات المذكورة",
        "بدلاً من القنوات", "بدلاً من الطريقة", "بدلاً من ما ذكر",
        "المذكورة أعلاه", "أعلاه", "ما سبق",
        # French
        "susmentionné", "susmentionnée", "susmentionnés", "ci-dessus",
        "mentionné précédemment", "mentionnée précédemment",
        "au lieu des canaux", "au lieu de la méthode",
        # English (model sometimes slips)
        "mentioned above", "aforementioned", "instead of the mentioned",
    ]

    @staticmethod
    def _is_valid(item: dict) -> bool:
        required = {"category", "language", "question", "gold_answer", "gold_keywords", "source"}
        if not required.issubset(item.keys()):
            return False
        keywords    = item.get("gold_keywords") or []
        gold_answer = item.get("gold_answer") or ""
        question    = item.get("question") or ""
        language    = item.get("language", "")

        # Reject placeholder / truncated content
        if "..." in gold_answer or "..." in question:
            return False
        if any("..." in kw for kw in keywords if kw):
            return False
        # Reject bracket placeholders the model forgot to fill in
        if "[exact" in gold_answer or "[exact" in question:
            return False
        # Reject non-atomic questions that reference prior context
        q_lower = question.lower()
        if any(pat.lower() in q_lower for pat in QAGenerator._NON_ATOMIC_PATTERNS):
            log.debug("[qa] Rejected non-atomic question: %s", question[:80])
            return False
        # Reject ARABIZI / DARIJA items whose gold_answer is in Latin script.
        # Count Arabic chars vs ASCII-alpha chars in the first 80 characters.
        if language in ("Arabizi", "Darija"):
            sample = gold_answer[:80]
            arabic = sum(1 for c in sample if "؀" <= c <= "ۿ")
            latin  = sum(1 for c in sample if c.isascii() and c.isalpha())
            # Reject if Latin dominates OR if the answer barely has any Arabic at all
            if latin > arabic or arabic < 5:
                log.debug("[qa] Rejected %s item with Latin/mixed gold_answer: %s",
                          language, gold_answer[:60])
                return False
        # Reject gold_answer that references a list not fully spelled out
        if "باقي الوثائق المذكورة" in gold_answer or "الوثائق المذكورة أعلاه" in gold_answer:
            return False

        category = item.get("category", "").upper()
        # OUTSCOPE has a fixed short gold_answer — relax word count for that category
        min_answer_words = 3 if category == "OUTSCOPE" else 10

        return (
            isinstance(keywords, list)
            and 1 <= len(keywords) <= 8           # must have specific keywords
            and len(question.split()) >= 4        # not a one-liner
            and len(gold_answer.split()) >= min_answer_words
            and category in QAGenerator._VALID_CATEGORIES
            and language in QAGenerator._VALID_LANGUAGES
        )

    # ── Batch generation ──────────────────────────────────────────────────────

    def generate_batch(
        self,
        sections,    # List[Section] from html_scraper OR List[dict] from idarati_api
        already_done_count: int = 0,
        progress_callback=None,
    ) -> List[dict]:
        """
        Generate Q&A from a list of sections with progress logging.

        Parameters
        ----------
        sections          : iterable of Section objects or dicts with body/url/lang_hint
        already_done_count: number of Q&A items already generated (for resume)
        progress_callback : optional callable(qa_so_far: List[dict]) called after each item

        Returns
        -------
        All newly generated Q&A dicts (does NOT include already_done items)
        """
        all_new: List[dict] = []
        total = len(sections)

        for idx, section in enumerate(sections, 1):
            # Normalise input — accept both Section objects and dicts
            if isinstance(section, dict):
                text       = section.get("body", section.get("body_ar", ""))
                url        = section.get("url", section.get("url_ar", ""))
                heading    = section.get("heading", section.get("title_ar", ""))
                lang       = section.get("lang_hint", "ar")
                site       = section.get("source_site", "")
            else:
                text    = section.body
                url     = section.url
                heading = section.heading
                lang    = section.lang_hint or "ar"
                site    = section.source_site

            items = self.generate(text=text, url=url, heading=heading, lang=lang, source_site=site)
            all_new.extend(items)

            if items:
                log.info("[qa] %d/%d sections → +%d items (total new: %d)",
                         idx, total, len(items), len(all_new))

            if progress_callback and items:
                progress_callback(all_new)

        return all_new
