# -*- coding: utf-8 -*-
"""
Step 10 — 11-Layer best-in-class grounding verification.

Key changes vs the monolith:
  - NLIVerifier class owns the model lifecycle (no global state, no monkey-patch).
  - The softmax fix (FIX 60) is built-in: mDeBERTa-XNLI returns raw logits,
    not probabilities; we apply softmax before reading the entailment index.
  - verify_output() accepts a PipelineResult so it can use ALL contexts
    (initial retrieval + agentic tool calls) for grounding, not just memory.
"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from pipeline.config import settings
from pipeline.models import (
    Chunk,
    ClaimVerification,
    EntityVerification,
    GroundingAudit,
    QuestionFlags,
    ScoredChunk,
    short_source,
)

log = logging.getLogger("MoroccanRAG")

_MAX_CLAIM_CHARS = 600
_NLI_MAX_CHUNKS = 5
_JUDGE_MAX_CHUNKS = 3


def _resolve_device() -> str:
    """
    Resolve the torch device. 'auto' → cuda if available. Even when 'cuda' is
    requested explicitly, verify CUDA is actually usable — otherwise loading on
    'cuda' raises and the NLI model silently degrades to word-overlap.
    """
    d = settings.device
    if d in ("auto", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if d == "cuda":
                log.warning("DEVICE=cuda but torch reports no CUDA — falling back to CPU. "
                            "Install a CUDA torch build to use the GPU.")
            return "cpu"
        except Exception:
            return "cpu"
    return d

DARIJA_STOPWORDS: Set[str] = {
    "واش", "غادي", "كاين", "كاينة", "بزاف", "شوية", "غير",
    "حتى", "راه", "راك", "راها", "داك", "ديك", "هاد", "ليكان",
    "ماكاينش", "ماشي", "هوما", "دابا",
}

def get_abstain_message(language: str) -> str:
    from pipeline.prompts import ABSTAIN
    return ABSTAIN.get(language, ABSTAIN["french"])


# ── NLI verifier ─────────────────────────────────────────────────────────────


class NLIVerifier:
    """
    Multilingual NLI claim verifier (mDeBERTa-XNLI).

    Owns the model lifecycle — no global state. The softmax fix (FIX 60) is
    built-in so callers never need to monkey-patch anything.
    """

    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(self):
        self._model = None
        self._available = False

    def _load(self, device: str = None) -> bool:
        if self._model is not None:
            return self._available
        if device is None:
            device = _resolve_device()   # was "cpu" — now honors DEVICE=cuda
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.MODEL_NAME, device=device, max_length=512)
            if settings.model_fp16 and device == "cuda":
                self._model.model.half()
            self._available = True
            log.info(f"  NLI loaded (mDeBERTa-XNLI AR+FR) on {device}"
                     f"{', fp16' if settings.model_fp16 and device == 'cuda' else ''}")
        except Exception as exc:
            log.warning(f"  NLI unavailable: {exc} — word-overlap fallback")
            self._available = False
        return self._available

    @staticmethod
    def _softmax_entailment(raw) -> float:
        """Apply softmax to raw logits; return p(entailment). Index 0 = entailment."""
        logits = np.array(raw, dtype=float)
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / exp_l.sum()
        return float(probs[0])

    def verify_claim(self, claim: str, chunks: List[Chunk]) -> ClaimVerification:
        """NLI-based verification with automatic word-overlap fallback."""
        self._load()
        if self._available and self._model is not None:
            try:
                chunk_list = chunks[:_NLI_MAX_CHUNKS]
                context = " ".join(c.text for c in chunk_list)
                context_trunc = " ".join(context.split()[:400])
                claim_trunc = " ".join(claim.split()[:100])

                # Batch ALL pairs (combined context + each chunk) into ONE
                # forward pass instead of 1 + N sequential calls — big speedup,
                # especially on GPU.
                pairs = [(context_trunc, claim_trunc)] + [
                    (c.text[:400], claim_trunc) for c in chunk_list
                ]
                raw_all = self._model.predict(pairs)

                entailment = self._softmax_entailment(raw_all[0])
                grounded = entailment >= settings.nli_grounding_threshold

                best_chunk, best_score = None, 0.0
                for chunk, raw in zip(chunk_list, raw_all[1:]):
                    ce = self._softmax_entailment(raw)
                    if ce > best_score:
                        best_score, best_chunk = ce, chunk

                return ClaimVerification(
                    claim=claim,
                    grounded=grounded,
                    nli_score=round(entailment, 4),
                    supporting_chunk=best_chunk,
                    evidence_text=best_chunk.text[:200] if best_chunk else "",
                    confidence=round(entailment, 4),
                    method="nli",
                )
            except Exception as exc:
                log.warning(f"  NLI predict failed: {exc} — word-overlap")

        return _verify_claim_word_overlap(claim, chunks)


# Module-level singleton — shared by verify_output() callers
_nli_verifier = NLIVerifier()


# ── Entity extraction & verification ─────────────────────────────────────────


def extract_entities(text: str) -> List[EntityVerification]:
    entities: List[EntityVerification] = []

    for pattern in [
        r"\b(20\d{2})\b",
        r"\b(\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b",
        r"[٠-٩]{4}",
        r"[٠-٩]{1,2}/[٠-٩]{1,2}/[٠-٩]{4}",
    ]:
        for m in re.finditer(pattern, text):
            entities.append(
                EntityVerification(entity=m.group(0), entity_type="DATE",
                                   found_in_context=False, exact_match=False)
            )

    for pattern in [
        r"\b(\d+(?:[,\.]\d+)?)\s*(?:MAD|DH|درهم|dirhams?)\b",
        r"\b(?:درهم|dirhams?)\s*(\d+(?:[,\.]\d+)?)\b",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(
                EntityVerification(entity=m.group(0), entity_type="AMOUNT",
                                   found_in_context=False, exact_match=False)
            )

    for city in [
        "Rabat", "Casablanca", "Fès", "Fes", "Marrakech", "Tanger",
        "Agadir", "Meknès", "Oujda", "Kenitra", "Tétouan",
        "الرباط", "الدار البيضاء", "فاس", "مراكش", "طنجة",
        "أكادير", "مكناس", "وجدة", "القنيطرة", "تطوان",
    ]:
        if city in text:
            entities.append(
                EntityVerification(entity=city, entity_type="LOCATION",
                                   found_in_context=False, exact_match=False)
            )

    # ── DURATION (deadlines / délais — critical for administrative procedures) ──
    _dur_ar = (r"(?:يوم(?:ًا|اً|ا)?|أيام|أسبوع(?:ًا|اً)?|أسابيع|"
               r"شهر(?:ًا|اً|ا)?|أشهر|سنة|سنوات|سنوية|عام(?:ًا|اً)?|أعوام|ساعة|ساعات)")
    _dur_lat = r"(?:jours?|semaines?|mois|ans?|années?|heures?|days?|weeks?|months?|years?|hours?)"
    for pattern in [
        rf"\b(\d+)\s*{_dur_lat}\b",
        rf"(\d+)\s*{_dur_ar}",
        rf"([٠-٩]+)\s*{_dur_ar}",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(
                EntityVerification(entity=m.group(0).strip(), entity_type="DURATION",
                                   found_in_context=False, exact_match=False)
            )

    # ── LEGAL_REF (decree / law / dahir / arrêté numbers) ──────────────────────
    for pattern in [
        r"(?:مرسوم|قانون|قرار|ظهير|منشور|دورية)\s*(?:ملكي\s*)?(?:شريف\s*)?(?:رقم\s*)?"
        r"[\d٠-٩][\d٠-٩./\-]*",
        r"(?:décret|loi|arrêté|dahir|circulaire)\s*(?:royal\s*)?n\s*[°ºo]?\s*"
        r"[\d][\d./\-]*",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(
                EntityVerification(entity=m.group(0).strip(), entity_type="LEGAL_REF",
                                   found_in_context=False, exact_match=False)
            )
    return entities


_TASHKEEL_RE = re.compile(r"[ً-ْٰ]")
_A2W_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize_for_match(s: str) -> str:
    """
    Normalize for entity grounding: Arabic-Indic → Western digits, strip Arabic
    diacritics (tashkeel), collapse whitespace, lowercase. This lets multi-word
    entities ('30 يوماً' vs '30 يوما', 'décret n° 2.04.564') match across the
    spacing/diacritic/digit-script variation found between answers and contexts.
    """
    s = s.translate(_A2W_DIGITS)
    s = _TASHKEEL_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_entities_in_chunks(
    entities: List[EntityVerification], chunks: List[Chunk]
) -> Tuple[List[EntityVerification], float]:
    if not entities:
        return [], 1.0

    norm_context = _normalize_for_match(" ".join(c.text for c in chunks))
    norm_chunks = [(c, _normalize_for_match(c.text)) for c in chunks]

    for entity in entities:
        norm_entity = _normalize_for_match(entity.entity)
        if not norm_entity:
            continue
        if norm_entity in norm_context:
            entity.found_in_context = entity.exact_match = True
            for chunk, nch in norm_chunks:
                if norm_entity in nch:
                    entity.source_chunk = chunk
                    break

    matched = sum(1 for e in entities if e.exact_match)
    return entities, matched / len(entities)


# ── Claim decomposition ───────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[Source:[^\]]*\]", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")


def _strip_citations(text: str) -> str:
    """Remove [Source:…] tags and bare URLs so they don't become 'claims'."""
    text = _CITATION_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return text


def _is_noise_claim(claim: str) -> bool:
    """True for non-claims: citations, URLs, JSON artifacts, or 'source is …' lines."""
    c = claim.strip().strip('",[]{}').strip()
    if len(c) <= 10:
        return True
    if _URL_RE.search(c) or "idarati.ma" in c.lower():
        return True
    if c.startswith(("{", "[", '"claims"', '"ادعاءات"', "claims", "ادعاءات", "réponse", "json")):
        return True
    # "the source is …" / "المصدر …" / "الرابط …" style meta-claims
    low = c.lower()
    if any(k in low for k in ("the source", "source for", "reference link", "le source")) \
       or c.startswith(("المصدر", "الرابط", "المرجع")):
        return True
    return False


def decompose_into_claims(answer: str, language: str, ollama) -> List[str]:
    """Break answer into max 5 atomic verifiable claims via LLM (JSON output)."""
    answer_trunc = _strip_citations(answer)[:_MAX_CLAIM_CHARS]

    if language in ("arabic_msa", "Darija"):
        system = (
            "أنت محلل نصوص متخصص في التحقق من الحقائق.\n"
            "مهمتك: فكك الإجابة إلى ادعاءات بسيطة مستقلة قابلة للتحقق.\n\n"
            "قواعد:\n"
            "1. كل ادعاء يحتوي على حقيقة واحدة فقط.\n"
            "2. الحد الأقصى 5 ادعاءات — ركز على الأهم.\n"
            "3. حافظ على الأرقام والتواريخ والمبالغ بدقة.\n"
            "4. أخرج JSON فقط."
        )
        user = f"فكك هذه الإجابة إلى ادعاءات:\n{answer_trunc}"
    else:
        system = (
            "You are a fact-checking text analyzer.\n"
            "Task: decompose the answer into simple independent verifiable claims.\n\n"
            "Rules:\n"
            "1. Each claim contains exactly one verifiable fact.\n"
            "2. Maximum 5 claims — focus on the most important.\n"
            "3. Preserve numbers, dates, and amounts exactly.\n"
            "4. Output JSON only."
        )
        user = f"Decompose into claims:\n{answer_trunc}"

    response = ollama.generate(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=400,
        fmt={
            "type": "json_schema",
            "json_schema": {
                "name": "claims",
                "schema": {
                    "type": "object",
                    "properties": {
                        "claims": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["claims"],
                },
            },
        },
    )
    if not response:
        sep = r"[.؟!]" if language in ("arabic_msa", "Darija") else r"[.?!]"
        cands = [s.strip() for s in re.split(sep, answer_trunc)]
        return [c for c in cands if not _is_noise_claim(c)][:5]

    # Strip any stray thinking / code fences before JSON parsing.
    response = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", response, flags=re.DOTALL)
    response = re.sub(r"```(?:json)?", "", response)

    try:
        parsed = (
            json.loads(response)
            if response.lstrip().startswith("{")
            else json.loads(re.search(r"\{.*\}", response, re.DOTALL).group(0))
        )
        # Model may use a localized key ("ادعاءات") instead of "claims".
        raw_claims = (parsed.get("claims") or parsed.get("ادعاءات")
                      or parsed.get("réponses") or [])
        claims = [c.strip() for c in raw_claims
                  if isinstance(c, str) and not _is_noise_claim(c)]
        return claims[:5]
    except Exception as exc:
        log.debug("Claim decompose JSON parse failed, using line split: %s", exc)
        cands = [
            re.sub(r"^[\d\.\)\-•*]+\s*", "", line.strip())
            for line in response.split("\n")
            if line.strip()
        ]
        return [c for c in cands if not _is_noise_claim(c)][:5]


def verify_claim_llm_judge(
    claim: str, chunks: List[Chunk], ollama, language: str
) -> ClaimVerification:
    """LLM-as-judge for ambiguous cases where NLI confidence is low."""
    context = "\n\n".join(c.text for c in chunks[:_JUDGE_MAX_CHUNKS])

    if language in ("arabic_msa", "Darija"):
        system = (
            "أنت خبير في التحقق من الحقائق للخدمات الإدارية المغربية.\n"
            "حدد ما إذا كان الادعاء مدعوماً بالسياق المقدم.\n\n"
            "تعريف الأحكام:\n"
            "- SUPPORTED: الادعاء موجود بوضوح في السياق\n"
            "- PARTIALLY: الادعاء موجود جزئياً أو بصياغة مختلفة\n"
            "- NOT_SUPPORTED: الادعاء غير موجود في السياق\n"
            "- CONTRADICTED: السياق يناقض الادعاء صراحة\n\n"
            "أخرج JSON فقط."
        )
        user = f"السياق:\n{context}\n\nالادعاء:\n{claim}"
    else:
        system = (
            "You are a fact-checker for Moroccan administrative services.\n"
            "Determine if the claim is supported by the provided context.\n\n"
            "Verdict definitions:\n"
            "- SUPPORTED: claim is clearly present in the context\n"
            "- PARTIALLY: claim is partially present or differently worded\n"
            "- NOT_SUPPORTED: claim is absent from the context\n"
            "- CONTRADICTED: context explicitly contradicts the claim\n\n"
            "Output JSON only."
        )
        user = f"Context:\n{context}\n\nClaim:\n{claim}"

    response = ollama.generate(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,   # deterministic — single verdict enum, no creativity needed
        max_tokens=64,     # {"verdict": "NOT_SUPPORTED"} is ~15 tokens; was 1000 (40x over)
        fmt={
            "type": "json_schema",
            "json_schema": {
                "name": "verdict",
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["SUPPORTED", "PARTIALLY", "NOT_SUPPORTED", "CONTRADICTED"],
                        }
                    },
                    "required": ["verdict"],
                },
            },
        },
    )
    if not response:
        return _nli_verifier.verify_claim(claim, chunks)

    try:
        parsed = (
            json.loads(response)
            if response.startswith("{")
            else json.loads(re.search(r"\{.*?\}", response, re.DOTALL).group(0))
        )
        verdict = parsed.get("verdict", "NOT_SUPPORTED").upper()
    except Exception as exc:
        log.debug("LLM judge JSON parse failed, using raw verdict: %s", exc)
        verdict = response.strip().upper()

    _VERDICT_MAP = {
        "SUPPORTED": (True, 0.90),
        "PARTIALLY": (False, 0.60),
        "CONTRADICTED": (False, 0.10),
    }
    grounded, conf = _VERDICT_MAP.get(verdict, (False, 0.30))
    return ClaimVerification(
        claim=claim,
        grounded=grounded,
        nli_score=conf,
        supporting_chunk=chunks[0] if grounded and chunks else None,
        evidence_text=context[:300] if grounded else "",
        confidence=conf,
        method="llm_judge",
    )


def _verify_claim_word_overlap(claim: str, chunks: List[Chunk]) -> ClaimVerification:
    all_stopwords = {
        "the", "a", "an", "is", "in", "on", "at", "to", "of",
        "و", "في", "من", "على", "أن", "إلى", "هذا", "هذه",
        "le", "la", "les", "de", "du", "des", "en", "un", "une",
    } | DARIJA_STOPWORDS
    claim_words = {
        w for w in re.sub(r"[^\w\s؀-ۿ]", "", claim.lower()).split()
        if len(w) > 2
    } - all_stopwords
    if not claim_words:
        return ClaimVerification(
            claim=claim, grounded=True, nli_score=1.0,
            supporting_chunk=None, evidence_text="", confidence=1.0,
            method="word_overlap",
        )
    best_chunk, best_overlap = None, 0.0
    for chunk in chunks:
        cw = set(re.sub(r"[^\w\s؀-ۿ]", "", chunk.text.lower()).split())
        ov = len(claim_words & cw) / len(claim_words)
        if ov > best_overlap:
            best_overlap, best_chunk = ov, chunk
    grounded = best_overlap >= settings.grounding_threshold
    return ClaimVerification(
        claim=claim,
        grounded=grounded,
        nli_score=best_overlap,
        supporting_chunk=best_chunk,
        evidence_text=best_chunk.text[:200] if best_chunk else "",
        confidence=best_overlap,
        method="word_overlap",
    )


# ── Answer cleaning ───────────────────────────────────────────────────────────


def _clean_answer(answer: str) -> str:
    """Remove Gemma generation artifacts, duplicate lines, and score tags."""
    def _remove_content_brackets(m):
        inner = m.group(1)
        if inner.startswith("Source:") or inner.startswith("UNVERIFIED"):
            return m.group(0)
        return inner

    citation_re = re.compile(r"\[Source:[^\]]+\]")
    lines = answer.split("\n")
    cleaned, prev_stripped, seen_content = [], None, set()
    for line in lines:
        stripped = line.strip()
        line_no_cit = citation_re.sub("", stripped).strip()
        if stripped and not line_no_cit:
            continue
        if stripped and stripped == prev_stripped:
            continue
        prev_stripped = stripped
        if line_no_cit and len(line_no_cit) > 15:
            content_key = re.sub(r"\s+", " ", line_no_cit.lower()).strip()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
        line = re.sub(r"\[([^\]]+)\]", _remove_content_brackets, line)
        cleaned.append(line)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    result = re.sub(r"(\])\s*Score:\s*\d+\.\d+\s*(\[)", r"\1 \2", result)
    result = re.sub(r"(\])\s*Score:\s*\d+\.\d+\s*$", r"\1", result, flags=re.MULTILINE)
    for pattern in [
        r"\n.*?(?:la réponse s\'arrête|cette information n\'est pas fournie).*?$",
        r"\n.*?(?:لا تتوفر هذه المعلومات|الإجابة النهائية تتوقف).*?$",
        r"\n.*?\[Source:\.\.\.\].*?(?:pas fournie|not provided).*?$",
    ]:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
    result = result.strip()
    if result and not result.endswith((".", "!", "؟", ":", "]")):
        last_punct = max(result.rfind("."), result.rfind("؟"), result.rfind("!"))
        if last_punct > len(result) // 2:
            result = result[: last_punct + 1]
    return result


# ── Citation injection & validation ──────────────────────────────────────────


def _inject_citations(answer: str, chunks: List[Chunk], language: str) -> Tuple[str, int]:
    if not chunks:
        return answer, 0
    source_map: Dict[str, List[Chunk]] = defaultdict(list)
    for chunk in chunks:
        source_map[short_source(chunk.source)].append(chunk)
    has_citation = re.compile(r"\[Source:\s*[^\]]+\]")
    is_content_line = re.compile(r"^\s*([-•*]\s+|\d+\.\s+)")
    lines, new_lines, injected = answer.split("\n"), [], 0
    for line in lines:
        stripped = line.strip()
        if (
            not stripped
            or stripped.endswith(":")
            or len(stripped) < 15
            or not has_citation.sub("", stripped).strip()
        ):
            new_lines.append(line)
            continue
        if has_citation.search(line):
            m = re.search(r"\[Source:\s*([^\]|]+?)(?:\s*\|[^\]]*)?\]", line)
            if m and m.group(1).strip() not in source_map and m.group(1).strip() != "UNVERIFIED":
                line = re.sub(r"\[Source:[^\]]+\]", "[Source: UNVERIFIED]", line)
            new_lines.append(line)
            continue
        if (is_content_line.match(line) or len(stripped) > 40) and injected < 8:
            line_words = {
                w for w in set(
                    re.sub(r"[^\w؀-ۿ]", " ", stripped.lower()).split()
                ) if len(w) > 2
            }
            best_src, best_overlap = None, 0
            for source, src_chunks in source_map.items():
                combined = " ".join(c.text for c in src_chunks)
                doc_words = set(re.sub(r"[^\w؀-ۿ]", " ", combined.lower()).split())
                if len(line_words & doc_words) > best_overlap:
                    best_overlap, best_src = len(line_words & doc_words), source
            if best_overlap > 0:
                line = line.rstrip() + f" [Source: {best_src}]"
                injected += 1
        new_lines.append(line)
    return "\n".join(new_lines), injected


def _normalize_src(name: str) -> str:
    return unicodedata.normalize("NFC", name).strip().lower()


def _validate_citations(answer: str, valid_sources: set) -> Tuple[str, int, int]:
    valid_normalized = {_normalize_src(s): s for s in valid_sources}
    valid_stems = {
        _normalize_src(s).replace(".pdf", "").replace("_", " "): s
        for s in valid_sources
    }
    matches = list(set(re.findall(r"\[Source:\s*([^\]|]+?)(?:\s*\|[^\]]*)?\]", answer)))
    valid = invalid = 0
    for src in matches:
        src_clean = src.strip()
        if src_clean in ("UNVERIFIED", ""):
            continue
        src_norm = _normalize_src(src_clean)
        if src_norm in valid_normalized:
            valid += 1
            continue
        src_stem = src_norm.replace(".pdf", "").replace("_", " ")
        if any(
            (src_stem in v or v in src_stem)
            for v in valid_stems
            if len(src_stem) >= 6 and len(v) >= 6
        ):
            valid += 1
            continue
        answer = re.sub(
            re.escape("[Source: ") + re.escape(src) + r"(?:\s*\|[^\]]*)?\]",
            "[Source: UNVERIFIED]",
            answer,
        )
        invalid += 1
    return answer, valid, invalid


# ── Audit trail ───────────────────────────────────────────────────────────────


def _save_audit(audit: GroundingAudit, agent_state=None) -> None:
    ts = audit.timestamp.replace(":", "-").replace(".", "-")
    qhash = abs(hash(audit.question)) % 100000
    filepath = Path(settings.audit_log_dir) / f"audit_{ts}_{qhash}.json"
    try:
        data = {
            "timestamp": audit.timestamp,
            "question": audit.question,
            "answer": audit.answer[:500],
            "claims": [
                {
                    "claim": c.claim,
                    "grounded": c.grounded,
                    "score": round(c.nli_score, 3),
                    "method": c.method,
                    "evidence": c.evidence_text[:150],
                }
                for c in audit.claims
            ],
            "entities": [
                {
                    "entity": e.entity,
                    "type": e.entity_type,
                    "found": e.found_in_context,
                    "exact": e.exact_match,
                }
                for e in audit.entities
            ],
            "summary": {
                "overall_grounded": audit.overall_grounded,
                "claim_ratio": round(audit.claim_grounded_ratio, 3),
                "entity_ratio": round(audit.entity_match_ratio, 3),
                "cfi": round(audit.composite_fidelity_index, 3),
            },
            "warnings": audit.warnings,
            "metadata": audit.metadata,
            "planning": agent_state.to_audit_dict() if agent_state else None,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"  Audit saved: {filepath.name}")
    except Exception as exc:
        log.warning(f"  Audit save failed: {exc}")


# ── Main verification orchestrator ────────────────────────────────────────────


def _run_claim_verification(
    answer: str,
    all_chunks: List[Chunk],
    flags: QuestionFlags,
    ollama,
) -> Tuple[List[ClaimVerification], float]:
    """Layers 1+2+7: decompose claims, NLI verify each, LLM-judge ambiguous ones."""
    log.info("  Layer 1: Decomposing into atomic claims...")
    claims_text = decompose_into_claims(answer, flags.language, ollama)
    log.info(f"  → {len(claims_text)} claims")

    log.info("  Layers 2+7: Verifying claims (NLI + LLM-judge)...")
    claim_verifications: List[ClaimVerification] = []
    for claim in claims_text:
        cv = _nli_verifier.verify_claim(claim, all_chunks)
        if settings.enable_llm_judge and cv.confidence < settings.ambiguous_nli_threshold:
            log.info(f"  → LLM-judge (NLI={cv.confidence:.2f}<{settings.ambiguous_nli_threshold})")
            cv = verify_claim_llm_judge(claim, all_chunks, ollama, flags.language)
        claim_verifications.append(cv)

    avg_conf = sum(cv.confidence for cv in claim_verifications) / max(len(claim_verifications), 1)
    return claim_verifications, avg_conf


def _run_entity_verification(
    answer: str,
    all_chunks: List[Chunk],
) -> Tuple[List[EntityVerification], float]:
    """Layers 3+4: entity extraction + grounding check."""
    if not settings.enable_entity_verification:
        return [], 1.0

    log.info("  Layers 3+4: Entity verification...")
    entities = extract_entities(answer)
    entity_match_ratio = 1.0
    if entities:
        entities, entity_match_ratio = verify_entities_in_chunks(entities, all_chunks)
        unmatched = [e.entity for e in entities if not e.exact_match]
        if unmatched:
            log.warning(f"  Unmatched entities: {unmatched}")
    log.info(f"  → entities={len(entities)}, match={entity_match_ratio:.1%}")
    return entities, entity_match_ratio


def _run_chain_verification(
    intermediate_answers: List[str],
    all_chunks: List[Chunk],
    flags: QuestionFlags,
    ollama,
) -> bool:
    """Layer 6: consistency check across multi-hop intermediate answers."""
    if not (settings.enable_chain_verification and intermediate_answers and flags.MULTIHOP):
        return True

    log.info("  Layer 6: Chain verification...")
    chain_verified = True
    for i, inter in enumerate(intermediate_answers):
        inter_claims = decompose_into_claims(inter, flags.language, ollama)
        for ic in inter_claims:
            icv = _nli_verifier.verify_claim(ic, all_chunks)
            if not icv.grounded:
                log.warning(f"  Intermediate {i + 1} ungrounded: {ic[:50]}")
                chain_verified = False
    return chain_verified


def _compute_cfi(
    claim_grounded_ratio: float,
    entity_match_ratio: float,
    num_entities: int,
    chain_verified: bool,
) -> float:
    """
    Layers 8+10: Composite Fidelity Index — a single faithfulness scalar in [0,1].

    Combines three fidelity signals:
      • claim grounding  (entailment-based faithfulness — the PRIMARY signal)
      • entity fidelity  (numbers/dates/amounts found verbatim in context)
      • relational consistency (multi-hop chain agreement)

    The earlier version derived CFI purely from ``entity_match_ratio``, which
    DEFAULTS to 1.0 when an answer has no extractable entities — making CFI
    vacuously high even for completely ungrounded answers (the CFI=0.89 vs
    claim_ratio=0.15 contradiction). CFI is now anchored on claim grounding,
    and the entity term is renormalized away when no entities exist instead of
    contributing a free 1.0.
    """
    relation_score = 1.0 if chain_verified else 0.5
    w_claim = settings.cfi_weight_claim
    w_entity = settings.cfi_weight_entity
    w_relation = settings.cfi_weight_relation

    if num_entities > 0:
        total = w_claim + w_entity + w_relation
        cfi = (
            w_claim * claim_grounded_ratio
            + w_entity * entity_match_ratio
            + w_relation * relation_score
        ) / total
    else:
        # No extractable entities → the entity term is undefined (not a free
        # 1.0). Renormalize over the claim + relation components only.
        total = w_claim + w_relation
        cfi = (w_claim * claim_grounded_ratio + w_relation * relation_score) / total

    return round(cfi, 4)


def _inject_and_validate_citations(
    answer: str,
    all_chunks: List[Chunk],
    generation_chunks: Optional[List[Chunk]],
    flags: QuestionFlags,
) -> str:
    """Layer 9: inject [Source:] tags and strip invalid ones."""
    valid_sources = {c.source for c in all_chunks}
    cite_chunks = generation_chunks if generation_chunks else all_chunks
    if not flags.MULTIHOP:
        answer, _ = _inject_citations(answer, cite_chunks, flags.language)
    answer, _, _ = _validate_citations(answer, valid_sources)
    return answer


def verify_output(
    answer: str,
    all_chunks: List[Chunk],
    flags: QuestionFlags,
    ollama,
    question: str = "",
    generation_chunks: Optional[List[Chunk]] = None,
    intermediate_answers: Optional[List[str]] = None,
    agent_state=None,
) -> Tuple[str, bool, bool, Optional[GroundingAudit]]:
    """
    Orchestrator: calls the layer helpers in sequence.

    all_chunks — ALL chunks across the full pipeline (initial + agentic tool calls).
    generation_chunks — chunks used for the final synthesis call (citation injection).
    Returns (verified_answer, is_grounded, is_abstained, audit_trail).
    """
    answer_clean = _clean_answer(answer)

    if not all_chunks:
        return get_abstain_message(flags.language), False, True, None

    # Layers 1+2+7: claim decomposition + NLI + LLM-judge
    claim_verifications, avg_conf = _run_claim_verification(answer_clean, all_chunks, flags, ollama)

    # Layers 3+4: entity extraction + verification
    entities, entity_match_ratio = _run_entity_verification(answer_clean, all_chunks)

    # Layer 6: multi-hop chain verification
    chain_verified = _run_chain_verification(intermediate_answers or [], all_chunks, flags, ollama)

    # Claim grounding ratio (primary faithfulness signal) — needed for CFI.
    grounded_claims_n = sum(1 for cv in claim_verifications if cv.grounded)
    claim_grounded_ratio = grounded_claims_n / max(len(claim_verifications), 1)

    # Layers 8+10: Composite Fidelity Index (anchored on claim grounding).
    cfi = _compute_cfi(claim_grounded_ratio, entity_match_ratio, len(entities), chain_verified)

    # Overall grounding decision
    entity_ok = (
        entity_match_ratio >= 1.0 if settings.entity_exact_match
        else entity_match_ratio >= 0.80
    )
    is_grounded = (
        claim_grounded_ratio >= settings.claim_grounded_ratio
        and entity_ok
        and cfi >= 0.70
    )
    log.info(
        f"  → claim_ratio={claim_grounded_ratio:.1%} | "
        f"entity={entity_match_ratio:.1%} | CFI={cfi:.2f} | grounded={is_grounded}"
    )

    # Layer 9: citation injection + validation
    answer_clean = _inject_and_validate_citations(answer_clean, all_chunks, generation_chunks, flags)

    # Layer 11: audit trail
    warnings: List[str] = []
    if not is_grounded:
        if claim_grounded_ratio < settings.claim_grounded_ratio:
            warnings.append(
                f"Low claim ratio: {claim_grounded_ratio:.1%} < {settings.claim_grounded_ratio:.1%}"
            )
        if not entity_ok:
            warnings.append(f"Entity mismatch: {entity_match_ratio:.1%}")
        if not chain_verified:
            warnings.append("Multi-hop chain verification failed")
        if cfi < 0.70:
            warnings.append(f"CFI too low: {cfi:.2f}")

    audit: Optional[GroundingAudit] = None
    if settings.enable_audit_trail:
        audit = GroundingAudit(
            timestamp=datetime.now().isoformat(),
            question=question,
            answer=answer_clean,
            claims=claim_verifications,
            entities=entities,
            overall_grounded=is_grounded,
            claim_grounded_ratio=claim_grounded_ratio,
            entity_match_ratio=entity_match_ratio,
            composite_fidelity_index=cfi,
            warnings=warnings,
            metadata={
                "avg_confidence": avg_conf,
                "chain_verified": chain_verified,
                "num_claims": len(claim_verifications),
                "num_entities": len(entities),
            },
        )
        _save_audit(audit, agent_state)

    return answer_clean, is_grounded, False, audit
