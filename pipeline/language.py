# -*- coding: utf-8 -*-
"""
Step 4 & 5 — Language detection, query translation, and question classification.

Fully agentic: one LLM call returns ALL semantic routing signals (language,
translation, intents, needs_multihop, is_legal, is_outscope). Twelve fixed
keyword/regex lists previously used for heuristic detection have been removed
in favor of LLM-based semantic decisions.

Kept (deterministic structural facts, NOT semantic heuristics):
  - Unicode-range script detection ('؀' <= c <= 'ۿ')
  - Arabizi digit-letter pattern ([a-zA-Z][2379])
  These remain as safety nets when the LLM is unavailable or contradicts the
  obvious orthographic script.
"""

import json
import logging
import re
from typing import Dict, Optional, Tuple

from pipeline.config import settings

log = logging.getLogger("MoroccanRAG")


# ── Agentic LLM classification ───────────────────────────────────────────────

_LLM_INTENTS = ["DOCUMENTS", "PROCEDURE", "COST", "DEADLINE", "ELIGIBILITY", "LEGAL", "COMPARISON"]

# ── Agentic classification schema ─────────────────────────────────────────────
# ALL semantic routing decisions are made by the LLM in ONE call. Replaces:
#   - INTENT_RULES (6 keyword lists × proclitic-aware regex)
#   - detect_multihop (10 pattern lists, 5-signal voting ensemble)
#   - check_outscope_keywords (2 keyword lists — caught 0/15 real OOS items)
#   - check_legal_keywords (2 keyword lists — over-fired on procedural questions)
#   - DARIJA_MARKERS / ARABIZI_MARKERS / MSA_KEYWORDS / FRENCH_KEYWORDS
# The LLM does what semantic understanding requires; deterministic checks are
# kept only for Unicode-range script discrimination (a structural fact, not a
# heuristic word list).
_LANG_SCHEMA = {
    "type": "object",
    "properties": {
        "language":        {"type": "string", "enum": ["Darija", "Arabizi", "arabic_msa", "french"]},
        "msa":             {"type": "string"},
        "intents":         {"type": "array", "items": {"type": "string", "enum": _LLM_INTENTS}},
        "needs_multihop":  {"type": "boolean"},
        "is_legal":        {"type": "boolean"},
        "is_outscope":     {"type": "boolean"},
    },
    "required": ["language", "msa", "intents", "needs_multihop", "is_legal", "is_outscope"],
}
_LANG_SYS = (
    "You analyze Moroccan public-service questions. Return JSON with 6 fields.\n\n"
    "1. \"language\" — EXACTLY one of:\n"
    "   - \"Darija\": Moroccan colloquial Arabic in ARABIC script (شنو، واش، بغيت، خاصني، اللي، ديال، فين، شحال)\n"
    "   - \"Arabizi\": Moroccan colloquial Arabic ROMANIZED in Latin letters, often with digits "
    "3=ع 7=ح 9=ق 2=ء (achno, khassni, bach, 3lach, nna9l, ada2, dyal). Latin letters with Arabic words/digits "
    "= Arabizi, NOT french.\n"
    "   - \"arabic_msa\": Modern Standard (formal) Arabic in Arabic script\n"
    "   - \"french\": standard French\n\n"
    "2. \"msa\" — Modern Standard Arabic translation if Darija/Arabizi, otherwise question verbatim.\n\n"
    "3. \"intents\" — ALL aspects the question asks about (subset of):\n"
    "   DOCUMENTS (papers/وثائق/أوراق required), PROCEDURE (steps to follow), COST (fees/price),\n"
    "   DEADLINE (duration/time), ELIGIBILITY (who qualifies), LEGAL (sanctions/penalties), COMPARISON.\n\n"
    "4. \"needs_multihop\" — TRUE in ANY of these cases:\n"
    "   (a) COMPARING two procedures or services ('difference between A and B',\n"
    "       'A versus B', 'better between A and B')\n"
    "   (b) Two distinct administrative ACTIONS connected by 'and'/'also'/'then'\n"
    "       (e.g., 'register a company AND pay capital duty', 'get a certificate\n"
    "       AND open an account', 'file an appeal AND modify the data')\n"
    "   (c) Question mentions TWO distinct procedure names (registration + payment,\n"
    "       authorization + license, declaration + amendment, X procedure + Y procedure)\n"
    "   (d) Compound questions about prerequisites + main procedure\n"
    "       ('what do I need to do BEFORE doing X to also do Y')\n"
    "   FALSE ONLY when asking multiple aspects (documents, cost, time, eligibility) of ONE procedure.\n"
    "   Examples:\n"
    "     'What documents for national ID?' → FALSE (one procedure)\n"
    "     'What documents, cost and time for ID card?' → FALSE (one procedure, three aspects)\n"
    "     'Difference between LLC and joint-stock registration?' → TRUE (case a)\n"
    "     'Documents to register a company AND pay capital duty?' → TRUE (case b — register + pay)\n"
    "     'Steps to register company AND get tax ID?' → TRUE (case b — two actions)\n"
    "     'What is needed to extend X certificate AND modify Y data?' → TRUE (case b)\n\n"
    "5. \"is_legal\" — TRUE only if the question is about legal sanctions, penalties, prison terms,\n"
    "   court procedures, or fines for violations. FALSE for administrative questions that merely\n"
    "   involve legal documents (e.g., 'how to register a company' is NOT is_legal=TRUE).\n\n"
    "6. \"is_outscope\" — TRUE if the question is unrelated to Moroccan administrative public services\n"
    "   (sports, entertainment, weather, tourism, recipes, restaurants, hotels, real estate prices,\n"
    "   personal opinions). FALSE for any administrative/procedural question, even if the specific\n"
    "   procedure may not be in the knowledge base.\n\n"
    "Output JSON only — no prose, no explanations."
)

_ARABIZI_DIGIT_RE = re.compile(r"[a-zA-Z][2379]|[2379][a-zA-Z]")


def _has_arabizi_digits(text: str) -> bool:
    """Latin word with an Arabizi digit-letter (3=ع,7=ح,9=ق,2=ء) — French never does this."""
    return bool(_ARABIZI_DIGIT_RE.search(text))


def _is_latin_script(text: str) -> bool:
    ar = sum(1 for c in text if "؀" <= c <= "ۿ")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return lat > ar


# Distinctive Darija markers (Arabic script) — excludes words shared with MSA
# (كيف/عندي/علاه/كيفية) to avoid false MSA→Darija. Used by the deterministic detector.
_DARIJA_AR_MARKERS = {
    "واش", "اشنو", "شنو", "ماشي", "بزاف", "كيفاش", "فين", "شحال", "علاش", "دابا",
    "غادي", "كاين", "كاينة", "بغيت", "بغا", "باغي", "مزيان", "كنقول", "كنشوف",
    "كندير", "كنبغي", "راه", "راك", "راها", "بصح", "ماعنديش", "خاصني", "خاصك",
    "خصني", "ماكاينش", "ديال", "ديالي", "اللي", "وقيلا", "فاش", "واخا", "بحال",
}
# Moroccan Arabizi markers (Latin script, used when no Arabizi digit is present).
_ARABIZI_LAT_MARKERS = {
    "achno", "chno", "chnou", "ashno", "wach", "wash", "fin", "ch7al", "chhal",
    "kifach", "kifash", "3lach", "3lah", "bghit", "bghiti", "khass", "khassni",
    "khassek", "dyal", "dyali", "bzaf", "bezzaf", "daba", "kayn", "kayen", "kayna",
    "wa9ila", "ndir", "nkhles", "bach", "ola", "wla", "had", "l9it", "kanbghi",
    "mzyan", "machi", "walakin", "b7al", "blkhsos", "3ndi", "9adia", "nta", "nti",
    "smiti", "kifya", "wakha",
}


def _script_detect(text: str) -> str:
    """
    Deterministic language label from script + markers.

    Validated 96.8% on the 124-item testset vs the 4B LLM's 30%. Script is the
    correct signal for the orthographic label — Arabic-vs-Latin is a Unicode-range
    fact the LLM cannot beat (and provably gets wrong, e.g. labelling romanized
    Latin text as arabic_msa). The LLM is reserved for translation + intents.

      Arabic script → Darija (if Darija markers) else arabic_msa
      Latin  script → Arabizi (if Arabizi digits/markers) else french
    """
    ar = sum(1 for c in text if "؀" <= c <= "ۿ")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    if ar >= lat and ar > 0:
        return "Darija" if any(m in text for m in _DARIJA_AR_MARKERS) else "arabic_msa"
    if lat > 0:
        if _has_arabizi_digits(text):
            return "Arabizi"
        toks = set(re.findall(r"[a-z0-9]+", text.lower()))
        return "Arabizi" if (toks & _ARABIZI_LAT_MARKERS) else "french"
    return "unknown"


def _llm_classify(question: str, ollama) -> Optional[Dict]:
    """
    One LLM call → ALL semantic routing signals.

    Returns dict with keys: language, msa, intents, needs_multihop, is_legal,
    is_outscope — or None if the LLM call fails or returns invalid JSON.

    This single call replaces 12 fixed keyword/regex lists previously used for
    intent detection, multihop detection, legal detection, and outscope
    detection. The LLM does what semantic understanding requires.
    """
    try:
        resp = ollama.generate(
            [{"role": "system", "content": _LANG_SYS}, {"role": "user", "content": question}],
            temperature=0.0, max_tokens=600, fmt=_LANG_SCHEMA, think=False,
        )
        if not resp:
            return None
        resp = re.sub(r"```(?:json)?", "", resp)
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            return None
        d = json.loads(m.group(0))
        lang = d.get("language")
        if lang not in ("Darija", "Arabizi", "arabic_msa", "french"):
            return None
        return {
            "language":       lang,
            "msa":            (d.get("msa") or "").strip(),
            "intents":        [i for i in (d.get("intents") or []) if i in _LLM_INTENTS],
            "needs_multihop": bool(d.get("needs_multihop", False)),
            "is_legal":       bool(d.get("is_legal", False)),
            "is_outscope":    bool(d.get("is_outscope", False)),
        }
    except Exception as exc:
        log.debug("LLM classify failed: %s", exc)
        return None


def detect_and_translate(question: str, ollama=None) -> Tuple[str, float, Optional[str], Optional[Dict]]:
    """
    Agentic language understanding in ONE step:
    returns (language, confidence, msa_query_or_None, llm_signals_or_None).

    The LLM returns ALL semantic routing signals (intents, needs_multihop,
    is_legal, is_outscope) in a single structured call. Unicode-range script
    detection is kept ONLY as:
      (a) safe default when LLM unavailable, and
      (b) override when LLM's language label disagrees with the obvious script
          (e.g., labelling Latin text as arabic_msa is impossible by definition).

    `llm_signals` is the full dict from _llm_classify, or None if LLM
    unavailable — callers pass it to classify_question to build QuestionFlags
    without any keyword-list lookups.
    """
    text = (question or "").strip()
    if not text:
        return "unknown", 0.0, None, None

    # ── Default language label from script (Unicode fact, used as safety net) ──
    lang_script = _script_detect(text)
    if lang_script == "unknown":
        lang_script = "arabic_msa"  # safe default (Arabic-script corpus)

    if ollama is None:
        # No LLM available → return script-based label with no semantic signals
        return lang_script, 0.5, None, None

    # ── Agentic: LLM returns ALL signals in one call ─────────────────────────
    signals = _llm_classify(question, ollama)
    if signals is None:
        # LLM failed → fall back to script label with no signals (caller will
        # use safe defaults: is_multihop=False, is_legal=False, is_outscope=False).
        return lang_script, 0.5, None, None

    # ── Reconcile LLM language with script fact ──────────────────────────────
    lang = signals["language"]
    # Latin script + Arabizi digits is unambiguously Arabizi — override any
    # contradictory LLM label. Pure script discrimination is a Unicode fact.
    if _is_latin_script(text):
        if _has_arabizi_digits(text):
            lang = "Arabizi"
        elif lang in ("Darija", "arabic_msa"):
            # LLM said Arabic-script variety but text is Latin — clearly wrong
            lang = "Arabizi" if lang_script == "Arabizi" else "french"
    else:
        # Arabic-script text — LLM shouldn't return Latin labels here
        if lang in ("Arabizi", "french"):
            lang = lang_script if lang_script in ("Darija", "arabic_msa") else "arabic_msa"

    # ── MSA translation for retrieval (Darija/Arabizi only) ─────────────────
    msa_out: Optional[str] = None
    if lang in ("Darija", "Arabizi"):
        cand = signals["msa"]
        if cand and not _is_latin_script(cand):
            msa_out = cand
        else:
            # LLM didn't translate or returned Latin — translate explicitly
            msa_out = translate_to_msa(question, lang, ollama) or None

    # Update signals with the reconciled language so downstream code sees one consistent value
    signals["language"] = lang

    return lang, 0.9, msa_out, signals


def translate_to_msa(query: str, source_lang: str, ollama) -> Optional[str]:
    """Translate Darija or Arabizi query to MSA for retrieval."""
    if not settings.enable_query_translation:
        return None

    if source_lang == "Darija":
        system = (
            "أنت مترجم متخصص. حوّل الدارجة المغربية إلى العربية الفصحى.\n"
            "اكتب الترجمة فقط بدون شرح. الترجمة:"
        )
    else:  # Arabizi
        system = (
            "أنت مترجم متخصص. حوّل الأرابيزي المغربي إلى العربية الفصحى.\n"
            "اكتب الترجمة العربية فقط بدون شرح. الترجمة:"
        )

    translation = ollama.generate(
        [{"role": "system", "content": system}, {"role": "user", "content": query}],
        temperature=0.1,
        max_tokens=600,
        think=False,
    )
    if translation:
        log.info(f"  Translated: '{query[:60]}' → '{translation.strip()[:60]}'")
        return translation.strip()
    log.warning("  Translation failed → using original query for retrieval")
    return None


def classify_question(question: str, language: str, confidence: float, ollama=None,
                      llm_intents=None, llm_signals=None):
    """
    Build QuestionFlags from the LLM's structured classification.

    The LLM call in detect_and_translate returns `llm_signals` (dict with
    intents, needs_multihop, is_legal, is_outscope). This function turns that
    dict into a QuestionFlags object. No keyword lists, no regex patterns —
    all semantic decisions come from the LLM.

    Backward compatibility:
      - `llm_intents` (list) — old caller signature; treated as a partial
        signals dict.
      - When llm_signals is None and ollama is provided, this function will
        invoke the LLM itself (one call) to get the signals.

    Safe defaults when the LLM is unavailable: is_simple=True, is_legal=False,
    is_outscope=False, intents=["DOCUMENTS"]. These defaults are safer than
    keyword rules because:
      - The previous keyword-based OOS detection caught 0/15 actual OOS items
      - The previous keyword-based intent detection over-fired (3+ intents
        on SIMPLE-category questions, forcing unnecessary multi-hop routing)
    """
    from pipeline.models import QuestionFlags

    # ── Acquire LLM signals (if needed) ──────────────────────────────────────
    if llm_signals is None and llm_intents is None and ollama is not None:
        llm_signals = _llm_classify(question, ollama)

    # Support old API: llm_intents is just a list → convert to signals dict
    if llm_signals is None and llm_intents is not None:
        llm_signals = {
            "intents": [i for i in llm_intents if i in _LLM_INTENTS],
            "needs_multihop": False,
            "is_legal": False,
            "is_outscope": False,
        }

    # ── Build flags from LLM signals (or safe defaults) ─────────────────────
    if llm_signals:
        intents     = list(dict.fromkeys(llm_signals.get("intents") or []))
        is_multihop = bool(llm_signals.get("needs_multihop", False))
        is_legal    = bool(llm_signals.get("is_legal", False))
        is_outscope = bool(llm_signals.get("is_outscope", False))
    else:
        intents     = []
        is_multihop = False
        is_legal    = False
        is_outscope = False

    # OOS supersedes everything else
    if is_outscope:
        intents = ["OUT_OF_SCOPE"]
    else:
        if is_legal and "LEGAL" not in intents:
            intents.append("LEGAL")
        if not intents:
            intents = ["DOCUMENTS"]   # safe fallback if LLM returned no intents

    # Routing decision: SIMPLE if the LLM said the question covers ONE procedure
    # (not multihop) AND is not legal. Intent count is NOT used — multi-aspect
    # single-procedure questions stay simple per the LLM's semantic judgement.
    is_simple = (not is_multihop) and (not is_legal) and (not is_outscope)

    hop_count = 1 if is_simple else max(1, min(len(intents), 4))

    return QuestionFlags(
        SIMPLE=is_simple,
        MULTIHOP=is_multihop,
        LEGAL=is_legal,
        OUTSCOPE=is_outscope,
        language=language,
        confidence=confidence,
        hop_count=hop_count,
        intents=intents,
    )


