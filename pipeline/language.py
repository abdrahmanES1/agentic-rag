# -*- coding: utf-8 -*-
"""
Step 4 & 5 — Language detection and query translation.

5-signal ensemble (Darija/Arabizi first):
  Signal 1 — Darija markers
  Signal 2 — Arabizi markers
  Signal 3 — Script ratio (Arabic vs Latin)
  Signal 4 — Domain keyword lists (MSA / French)
  Signal 5 — ML detectors (Lingua, langdetect, langid)
"""

import json
import logging
import re
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

from pipeline.config import settings

log = logging.getLogger("MoroccanRAG")

# ── Optional ML detectors ─────────────────────────────────────────────────────

try:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import langid
    langid.set_languages(["ar", "fr"])
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False

try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_DETECTOR = (
        LanguageDetectorBuilder.from_languages(Language.ARABIC, Language.FRENCH)
        .with_preloaded_language_models()
        .build()
    )
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False

# ── Marker sets ───────────────────────────────────────────────────────────────

DARIJA_MARKERS: Set[str] = {
    "واش", "ماشي", "بزاف", "كيف", "فين", "شحال", "علاش", "درابا", "دابا",
    "غادي", "كاين", "كاينة", "بغيت", "بغا", "مزيان", "كنت", "كنقول",
    "كنشوف", "كندير", "كنبغي", "راه", "راك", "راها", "ياللاه", "زوين",
    "بصح", "ماعنديش", "خاصني", "خاصك", "ماكاينش",
}

ARABIZI_MARKERS: Set[str] = {
    "chkoun", "kifach", "kifash", "ndir", "khass", "bghit", "bghiti",
    "wach", "3lach", "3lah", "bzaf", "bezzaf", "daba", "draba", "ghadi",
    "kayen", "kayna", "mzyan", "zwine", "machi", "wallah", "yallah",
    "bsa7", "mchi", "sir",
}

# ── Domain keyword lists ───────────────────────────────────────────────────────

MSA_KEYWORDS = [
    "ما هي", "ما هو", "كيف يمكن", "كيف أحصل", "متى يمكن",
    "الوثائق المطلوبة", "الإجراءات", "للحصول على", "مدة الإنجاز", "الرسوم",
]
FRENCH_KEYWORDS = [
    "comment", "quels", "quelle", "quelles", "documents", "procédure",
    "dossier", "formulaire", "carte nationale", "permis", "acte",
    "certificat", "délai", "frais", "pièces justificatives",
]

OUTSCOPE_KEYWORDS_AR = [
    "كرة القدم", "مباراة كرة", "نتيجة مباراة", "ترتيب الفرق",
    "فيلم سينمائي", "مسلسل تلفزيوني", "أغنية", "موسيقى", "وصفة طبخ",
]
OUTSCOPE_KEYWORDS_FR = [
    "match de football", "résultat sportif", "classement équipe",
    "film cinéma", "série télévisée", "recette cuisine", "chanteur",
    "chanson", "restaurant", "meilleur restaurant", "hôtel", "hébergement",
    "tourisme", "voyage", "vacances", "météo", "température",
    "prix immobilier", "loyer",
]
LEGAL_KEYWORDS_AR = [
    "حكم بالسجن", "السجن", "الاعتقال", "المحكمة الجنائية",
    "النيابة العامة", "جريمة جنائية", "جنحة", "إدانة جنائية",
    "عقوبة جنائية", "غرامة جنائية",
]
LEGAL_KEYWORDS_FR = [
    "peine d'emprisonnement", "prison", "tribunal correctionnel",
    "procureur", "infraction pénale", "condamnation pénale",
    "casier judiciaire", "amende pénale",
]

# ── Multi-hop / intent detection patterns ────────────────────────────────────

AR_QUESTION_WORDS = {"ما", "ماذا", "من", "كيف", "متى", "إمتى", "أين", "فين", "لماذا", "كم", "شحال", "هل", "واش", "أي"}
FR_QUESTION_WORDS = {"comment", "quels", "quelle", "quelles", "quel", "quand", "où", "pourquoi", "combien", "qui", "que"}
AR_CONJUNCTIVE_PATTERNS = [r"و\s*(كم|ما|كيف|أين|متى|هل|من|لماذا|أي)", r"وكذلك\s", r"وأيضا\s", r"فضلا\s+عن\s+ذلك", r"ثم\s+(ما|كيف|أين|كم|هل)"]
FR_CONJUNCTIVE_PATTERNS = [r"et\s+(comment|quels?|quelle|quand|où|combien|qui|pourquoi)", r"ainsi\s+que", r"de\s+plus", r"également"]
AR_COMPARISON_PATTERNS = [r"الفرق\s+بين", r"مقارنة\s+بين", r"\bأم\b", r"الأفضل\s+بين"]
FR_COMPARISON_PATTERNS = [r"différence\s+entre", r"comparaison\s+entre", r"ou\s+bien\b", r"plutôt\s+que", r"versus\b"]
AR_ENUMERATION_PATTERNS = [r"(الوثائق|الشروط|الخطوات).{0,30}(الرسوم|المدة|المكان|الإجراءات)"]
FR_ENUMERATION_PATTERNS = [r"(documents|conditions|étapes).{0,40}(frais|délai|lieu|procédure)"]
AR_MULTI_ENTITY_PATTERNS = [r"(البطاقة|الجواز|الرخصة|الشهادة).{2,40}(و|أو).{2,40}(البطاقة|الجواز|الرخصة|الشهادة)"]
FR_MULTI_ENTITY_PATTERNS = [r"(carte|passeport|permis|certificat).{2,50}(et|ou).{2,50}(carte|passeport|permis|certificat)"]


# ── Public API ────────────────────────────────────────────────────────────────


def detect_language(question: str) -> Tuple[str, float]:
    """
    5-signal ensemble language detection.
    Returns (language, confidence).
    language in: "Darija" | "Arabizi" | "arabic_msa" | "french" | "mixed" | "unknown"
    """
    text = question.strip()
    if not text:
        return "unknown", 0.0

    # Signal 1: Darija markers (highest priority)
    if settings.enable_darija:
        darija_count = sum(1 for m in DARIJA_MARKERS if m in text)
        if darija_count >= settings.darija_marker_min:
            conf = min(0.85 + (darija_count - settings.darija_marker_min) * 0.03, 0.95)
            log.info(f"  Language: Darija ({darija_count} markers, conf={conf:.2f})")
            return "Darija", conf

    # Signal 2: Arabizi markers
    if settings.enable_arabizi:
        arabizi_count = sum(1 for m in ARABIZI_MARKERS if m.lower() in text.lower())
        if arabizi_count >= settings.arabizi_marker_min:
            conf = min(0.80 + (arabizi_count - settings.arabizi_marker_min) * 0.03, 0.92)
            log.info(f"  Language: Arabizi ({arabizi_count} markers, conf={conf:.2f})")
            return "Arabizi", conf

    # Signals 3-5: weighted ensemble
    votes: Dict[str, float] = defaultdict(float)

    s1_lang, s1_conf = _detect_by_script(text)
    if s1_lang != "unknown":
        votes[s1_lang] += 1.0 * s1_conf

    s2_lang, s2_conf = _detect_by_keywords(text)
    if s2_lang != "unknown":
        votes[s2_lang] += 1.5 * s2_conf

    if LINGUA_AVAILABLE:
        s3_lang, s3_conf = _detect_by_lingua(text)
        if s3_lang != "unknown":
            votes[s3_lang] += 2.0 * s3_conf

    if LANGDETECT_AVAILABLE:
        s4_lang, s4_conf = _detect_by_langdetect(text)
        if s4_lang != "unknown":
            votes[s4_lang] += 1.0 * s4_conf

    if LANGID_AVAILABLE:
        s5_lang, s5_conf = _detect_by_langid(text)
        if s5_lang != "unknown":
            votes[s5_lang] += 1.0 * s5_conf

    if not votes:
        return "arabic_msa", 0.3

    total_weight = sum(votes.values())
    final_lang = max(votes, key=votes.get)
    confidence = votes[final_lang] / total_weight if total_weight > 0 else 0.5

    if votes.get("arabic_msa", 0) > 0 and votes.get("french", 0) > 0:
        ar_share = votes["arabic_msa"] / total_weight
        fr_share = votes["french"] / total_weight
        if 0.25 < ar_share < 0.75 and 0.25 < fr_share < 0.75:
            final_lang = "mixed"
            confidence = min(ar_share, fr_share) * 2

    log.info(f"  Language: {final_lang} (conf={confidence:.2f})")
    return final_lang, confidence


# ── Agentic LLM detect + translate ────────────────────────────────────────────

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
    "4. \"needs_multihop\" — TRUE ONLY if the question requires combining information from MULTIPLE\n"
    "   distinct administrative procedures, or comparing two different services.\n"
    "   FALSE when asking about multiple aspects (docs+cost+time) of a SINGLE procedure.\n"
    "   Examples:\n"
    "     'What documents are required for the national ID?' → FALSE (one procedure)\n"
    "     'What documents, fees and time for the ID card?' → FALSE (one procedure, three aspects)\n"
    "     'Difference between LLC and joint-stock company registration?' → TRUE (two procedures)\n"
    "     'What documents to register a company AND pay capital duty?' → TRUE (two procedures)\n\n"
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


def check_outscope_keywords(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in question for kw in OUTSCOPE_KEYWORDS_AR) or any(
        kw in q_lower for kw in OUTSCOPE_KEYWORDS_FR
    )


def check_legal_keywords(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in question for kw in LEGAL_KEYWORDS_AR) or any(
        kw in q_lower for kw in LEGAL_KEYWORDS_FR
    )


_AR_LETTER = "ء-ي٠-٩"
# Arabic proclitics that may attach before a stem (article + و/ف/ب/ل/ك …).
_AR_PROCLITIC = r"(?:بال|وال|فال|كال|لل|ال|و|ف|ب|ل|ك)?"


def _intent_kw_hit(keywords, text: str, text_lower: str) -> bool:
    """
    True if any keyword is present. Latin keywords use lowercase substring;
    Arabic keywords use proclitic-aware word boundaries so a fee word (رسوم)
    does NOT match inside a decree word (مرسوم), nor مدة inside عمدة.
    """
    for kw in keywords:
        if kw.isascii():
            if kw in text_lower:
                return True
        else:
            pat = (r"(?<![" + _AR_LETTER + r"])" + _AR_PROCLITIC
                   + re.escape(kw) + r"(?![" + _AR_LETTER + r"])")
            if re.search(pat, text):
                return True
    return False


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


def detect_multihop(question: str) -> Tuple[bool, float, list, int]:
    q_lower = question.lower().strip()
    signals_fired, votes = [], 0
    ar_count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", question)) for w in AR_QUESTION_WORDS)
    fr_count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", q_lower)) for w in FR_QUESTION_WORDS)
    total_qwords = ar_count + fr_count
    if total_qwords >= 2:
        votes += 1; signals_fired.append(f"S1:qwords({total_qwords})")
    if any(re.search(p, question) for p in AR_CONJUNCTIVE_PATTERNS) or any(re.search(p, q_lower) for p in FR_CONJUNCTIVE_PATTERNS):
        votes += 1; signals_fired.append("S2:conjunctive")
    if any(re.search(p, question) for p in AR_COMPARISON_PATTERNS) or any(re.search(p, q_lower) for p in FR_COMPARISON_PATTERNS):
        votes += 1; signals_fired.append("S3:comparison")
    if any(re.search(p, question) for p in AR_ENUMERATION_PATTERNS) or any(re.search(p, q_lower) for p in FR_ENUMERATION_PATTERNS):
        votes += 1; signals_fired.append("S4:enumeration")
    if any(re.search(p, question) for p in AR_MULTI_ENTITY_PATTERNS) or any(re.search(p, q_lower) for p in FR_MULTI_ENTITY_PATTERNS):
        votes += 1; signals_fired.append("S5:multi_entity")
    ar_conj = sum(len(re.findall(p, question)) for p in [r"و\s*(كم|ما|كيف|أين|متى|هل|من|لماذا)", r"وما\s+هي", r"وما\s+هو"])
    fr_conj = sum(len(re.findall(p, q_lower)) for p in [r"et\s+(comment|quels?|quelle|quand|où|combien|qui)", r"ainsi\s+que"])
    hop_count = max(1, min(total_qwords + ar_conj + fr_conj, 4))
    return votes >= 1, votes / 5, signals_fired, hop_count


# ── Private helpers ───────────────────────────────────────────────────────────


def _detect_by_script(text: str) -> Tuple[str, float]:
    ar = sum(1 for c in text if "؀" <= c <= "ۿ")
    la = sum(1 for c in text if "a" <= c.lower() <= "z")
    total = sum(1 for c in text if not c.isspace() and not c.isdigit())
    if total < 3:
        return "unknown", 0.0
    ar_ratio, la_ratio = ar / total, la / total
    if ar_ratio > settings.arabic_script_min:
        return "arabic_msa", min(1.0, (ar_ratio - settings.arabic_script_min) / (1 - settings.arabic_script_min) + 0.5)
    if la_ratio > settings.french_latin_min:
        return "french", min(1.0, (la_ratio - settings.french_latin_min) / (1 - settings.french_latin_min) + 0.5)
    return ("arabic_msa" if ar_ratio > la_ratio else "french"), 0.4


def _detect_by_keywords(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    ar_hits = sum(1 for kw in MSA_KEYWORDS if kw in text)
    fr_hits = sum(1 for kw in FRENCH_KEYWORDS if kw in text_lower)
    if ar_hits == 0 and fr_hits == 0:
        return "unknown", 0.0
    total = ar_hits + fr_hits
    if ar_hits >= fr_hits:
        return "arabic_msa", min(1.0, 0.5 + ar_hits / (total * 2))
    return "french", min(1.0, 0.5 + fr_hits / (total * 2))


def _detect_by_lingua(text: str) -> Tuple[str, float]:
    try:
        from lingua import Language
        result = LINGUA_DETECTOR.detect_language_of(text)
        conf_values = LINGUA_DETECTOR.compute_language_confidence_values(text)
        lang_map = {Language.ARABIC: "arabic_msa", Language.FRENCH: "french"}
        if result is None:
            return "unknown", 0.0
        detected = lang_map.get(result, "unknown")
        conf = 0.5
        for cv in conf_values:
            if cv.language == result:
                conf = cv.value
                break
        return detected, float(conf)
    except Exception as exc:
        log.debug("Lingua detection failed: %s", exc)
        return "unknown", 0.0


def _detect_by_langdetect(text: str) -> Tuple[str, float]:
    try:
        results = detect_langs(text)
        lang_map = {"ar": "arabic_msa", "fr": "french"}
        for r in results:
            mapped = lang_map.get(r.lang)
            if mapped:
                return mapped, float(r.prob)
        return "unknown", 0.0
    except Exception as exc:
        log.debug("langdetect detection failed: %s", exc)
        return "unknown", 0.0


def _detect_by_langid(text: str) -> Tuple[str, float]:
    try:
        lang, conf = langid.classify(text)
        lang_map = {"ar": "arabic_msa", "fr": "french"}
        return lang_map.get(lang, "unknown"), max(0.0, min(1.0, 1.0 + conf / 10.0))
    except Exception as exc:
        log.debug("langid detection failed: %s", exc)
        return "unknown", 0.0
