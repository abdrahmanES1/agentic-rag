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


def classify_question(question: str, language: str, confidence: float, ollama=None):
    """
    Classify question into QuestionFlags (SIMPLE/MULTIHOP/LEGAL/OUTSCOPE + intents).

    Called after language detection. ollama is accepted but not used — reserved for
    future LLM-based intent classification.
    """
    from pipeline.models import QuestionFlags

    q_lower = question.lower()

    is_outscope = check_outscope_keywords(question)
    is_legal = check_legal_keywords(question)
    is_multihop, _conf, _signals, hop_count = detect_multihop(question)

    # Intent detection via keyword matching
    INTENT_RULES = [
        ("PROCEDURE", ["إجراء", "خطوات", "طريقة", "كيفية", "procédure", "étapes", "comment faire", "démarche"]),
        ("COST",      ["رسوم", "تكلفة", "سعر", "ثمن", "درهم", "frais", "coût", "prix", "tarif", "dirhams"]),
        ("DEADLINE",  ["مدة", "أجل", "وقت", "متى", "délai", "durée", "jours", "semaines", "date limite"]),
        ("ELIGIBILITY", ["شروط", "أهلية", "يحق", "من يستفيد", "conditions", "éligibilité", "qui peut", "bénéficier"]),
        ("LEGAL",     LEGAL_KEYWORDS_AR + LEGAL_KEYWORDS_FR),
        ("DOCUMENTS", ["وثيقة", "وثائق", "ملف", "مستند", "document", "pièce", "dossier", "formulaire"]),
    ]

    intents = []
    for intent, keywords in INTENT_RULES:
        if any(kw in question or kw in q_lower for kw in keywords):
            intents.append(intent)

    if is_legal and "LEGAL" not in intents:
        intents.append("LEGAL")
    if not intents:
        intents = ["DOCUMENTS"]
    if is_outscope:
        intents = ["OUT_OF_SCOPE"]

    is_simple = not is_multihop and not is_legal and len(intents) <= 1

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
