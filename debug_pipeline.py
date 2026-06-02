# -*- coding: utf-8 -*-
"""
Moroccan RAG V12 — Pipeline Debugger + SOTA OCR Loader

TWO THINGS IN ONE FILE:
 1. PipelineDebugger  — full execution trace via pipeline.ask(), returns
                        FullDebugReport with per-tool latency, intermediate
                        answers, reflection verdicts, and per-layer verification.
 2. SOTA OCR loader   — three-tier strategy for PDF text extraction:
       Tier 1: pymupdf4llm   (markdown-aware, very fast)
       Tier 2: marker-pdf    (layout-preserving ML OCR)
       Tier 3: fitz + Tesseract (always-available fallback)

USAGE:
  from debug_pipeline import PipelineDebugger
  debugger = PipelineDebugger(pipeline)
  report = debugger.debug("ما هي الوثائق المطلوبة؟")
"""

import json
import logging
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("MoroccanRAG")

# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OcrPageResult:
    source: str
    page: int
    text: str
    tier_used: str
    char_count: int
    word_count: int
    is_ocr: bool
    language_guess: str


@dataclass
class ToolCallDebug:
    step_index: int
    tool_name: str
    intent: str
    query: str
    latency_ms: float
    chunks_returned: int
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_answer: str = ""
    reflection: str = ""


@dataclass
class VerificationLayerDebug:
    layer: str
    passed: bool
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FullDebugReport:
    question: str
    language: str
    flags: Dict[str, Any]
    total_latency_ms: float
    total_duration_ms: float          # alias — debug.html reads this name

    # Planning
    plan: List[Dict[str, Any]]
    plan_source: str
    plan_latency_ms: float

    # Tool execution
    tool_calls: List[ToolCallDebug]

    # Retrieval — flat lists for debug.html renderRetCard()
    retrieved: List[Dict[str, Any]]   # all chunks after initial retrieval
    reranked: List[Dict[str, Any]]    # top-N after reranking (subset of retrieved)
    context_sent: str                 # exact context string fed to LLM
    initial_retrieval: Dict[str, Any]
    all_contexts: List[str]
    synthesis_context: str

    # Answer
    raw_answer: str
    final_answer: str

    # Verification
    verification_layers: List[VerificationLayerDebug]
    claim_results: List[Dict[str, Any]]
    entity_results: List[Dict[str, Any]]
    cfi: float
    claim_grounded_ratio: float

    # Diagnostics
    root_causes: List[str]
    recommendations: List[str]

    # KB overview (arabic_chunks, french_chunks counts + samples for debug.html)
    kb_summary: Dict[str, Any] = field(default_factory=dict)

    # Backward-compat: flat steps list for debug.html
    steps: List[Dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# SOTA OCR — THREE-TIER LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def _guess_language(text: str) -> str:
    ar = sum(1 for c in text if '؀' <= c <= 'ۿ')
    la = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    total = max(ar + la, 1)
    if ar / total > 0.55:
        return "arabic"
    if la / total > 0.65:
        return "french"
    return "mixed"


def _clean_ocr_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s؀-ۿÀ-ɏ.,،؟?!:;«»()\-/\n]', '', text)
    return text.strip()


def _load_with_pymupdf4llm(pdf_path: Path) -> List[OcrPageResult]:
    try:
        import pymupdf
        import pymupdf4llm

        doc = pymupdf.open(str(pdf_path))
        page_count = doc.page_count
        doc.close()

        results = []
        for page_num in range(page_count):
            md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
            text = _clean_ocr_text(md)
            if len(text.split()) >= 15:
                results.append(OcrPageResult(
                    source=pdf_path.name, page=page_num + 1,
                    text=text, tier_used="pymupdf4llm",
                    char_count=len(text), word_count=len(text.split()),
                    is_ocr=False, language_guess=_guess_language(text)
                ))
        return results
    except Exception as exc:
        log.debug("pymupdf4llm failed for %s: %s", pdf_path.name, exc)
        return []


def _load_with_marker(pdf_path: Path) -> List[OcrPageResult]:
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=None, renderer=None
        )
        rendered = converter(str(pdf_path))
        full_text, _, _ = text_from_rendered(rendered)

        pages_raw = re.split(r'---\s*Page\s+\d+\s*---', full_text)
        if len(pages_raw) <= 1:
            pages_raw = [full_text]

        results = []
        for page_num, page_text in enumerate(pages_raw, 1):
            text = _clean_ocr_text(page_text)
            if len(text.split()) >= 15:
                results.append(OcrPageResult(
                    source=pdf_path.name, page=page_num,
                    text=text, tier_used="marker",
                    char_count=len(text), word_count=len(text.split()),
                    is_ocr=True, language_guess=_guess_language(text)
                ))
        return results
    except Exception as exc:
        log.debug("marker failed for %s: %s", pdf_path.name, exc)
        return []


def _load_with_fitz_fallback(pdf_path: Path) -> List[OcrPageResult]:
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        results = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = _clean_ocr_text(page.get_text("text").strip())
            if len(text.split()) >= 15:
                results.append(OcrPageResult(
                    source=pdf_path.name, page=page_num + 1,
                    text=text, tier_used="fitz_text",
                    char_count=len(text), word_count=len(text.split()),
                    is_ocr=False, language_guess=_guess_language(text)
                ))
            else:
                try:
                    import pytesseract
                    from io import BytesIO
                    from PIL import Image
                    mat = pymupdf.Matrix(2.0, 2.0)
                    img_bytes = page.get_pixmap(matrix=mat).tobytes("png")
                    img = Image.open(BytesIO(img_bytes))
                    ocr_text = _clean_ocr_text(pytesseract.image_to_string(img, lang="ara+fra"))
                    if len(ocr_text.split()) >= 15:
                        results.append(OcrPageResult(
                            source=pdf_path.name, page=page_num + 1,
                            text=ocr_text, tier_used="tesseract",
                            char_count=len(ocr_text), word_count=len(ocr_text.split()),
                            is_ocr=True, language_guess=_guess_language(ocr_text)
                        ))
                except Exception as exc:
                    log.debug("Tesseract page OCR failed: %s", exc)
        doc.close()
        return results
    except Exception as exc:
        log.debug("fitz fallback failed for %s: %s", pdf_path.name, exc)
        return []


def _page_has_text(pdf_path: Path, page_num: int, min_chars: int = 80) -> bool:
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        text = doc[page_num].get_text("text").strip()
        doc.close()
        return len(text) >= min_chars
    except Exception:
        return False


_MARKER_MODELS = None
_MARKER_AVAILABLE = None


def _check_marker_available() -> bool:
    global _MARKER_AVAILABLE
    if _MARKER_AVAILABLE is not None:
        return _MARKER_AVAILABLE
    try:
        import os
        cache_dirs = [
            Path.home() / ".cache" / "surya",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path(os.environ.get("HF_HOME", "")) / "hub",
        ]
        has_cache = any(d.exists() and any(d.iterdir()) for d in cache_dirs if d.exists())
        if not has_cache:
            _MARKER_AVAILABLE = False
            return False
        from marker.converters.pdf import PdfConverter  # noqa
        from marker.models import create_model_dict      # noqa
        _MARKER_AVAILABLE = True
        return True
    except Exception:
        _MARKER_AVAILABLE = False
        return False


def _ocr_single_page_marker(pdf_path: Path, page_num: int) -> str:
    global _MARKER_MODELS
    try:
        import os
        import tempfile
        import pymupdf
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        if _MARKER_MODELS is None:
            _MARKER_MODELS = create_model_dict()

        src = pymupdf.open(str(pdf_path))
        tmp = pymupdf.open()
        tmp.insert_pdf(src, from_page=page_num, to_page=page_num)
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        tmp.save(tmp_path)
        tmp.close()
        src.close()

        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=_MARKER_MODELS,
            processor_list=None, renderer=None
        )
        rendered = converter(tmp_path)
        full_text, _, _ = text_from_rendered(rendered)
        os.unlink(tmp_path)
        return full_text.strip()
    except Exception as exc:
        log.debug("Marker single-page OCR failed: %s", exc)
        return ""


def _ocr_single_page_tesseract(pdf_path: Path, page_num: int) -> str:
    try:
        import pymupdf
        import pytesseract
        from io import BytesIO
        from PIL import Image
        doc = pymupdf.open(str(pdf_path))
        mat = pymupdf.Matrix(2.5, 2.5)
        img_bytes = doc[page_num].get_pixmap(matrix=mat).tobytes("png")
        doc.close()
        img = Image.open(BytesIO(img_bytes))
        return pytesseract.image_to_string(img, lang="ara+fra", config="--psm 3 --oem 1")
    except Exception as exc:
        log.debug("Tesseract single-page OCR failed: %s", exc)
        return ""


def load_pdf_sota(pdf_path: Path, preferred_tier: str = "auto") -> List[OcrPageResult]:
    if preferred_tier == "pymupdf4llm":
        r = _load_with_pymupdf4llm(pdf_path)
        return r if r else _load_with_fitz_fallback(pdf_path)
    if preferred_tier == "marker":
        r = _load_with_marker(pdf_path)
        return r if r else _load_with_fitz_fallback(pdf_path)
    if preferred_tier == "fitz":
        return _load_with_fitz_fallback(pdf_path)

    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        total_pages = doc.page_count
        doc.close()
    except Exception as exc:
        log.warning("Cannot open PDF %s: %s", pdf_path.name, exc)
        return []

    digital_pages = [i for i in range(total_pages) if _page_has_text(pdf_path, i)]
    scanned_pages = [i for i in range(total_pages) if i not in digital_pages]

    results: List[OcrPageResult] = []

    if digital_pages:
        try:
            import pymupdf
            import pymupdf4llm
            for page_num in digital_pages:
                md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
                text = _clean_ocr_text(md)
                garbage_ratio = sum(1 for c in text if ord(c) > 0x10000 or c == '\x00') / max(len(text), 1)
                if len(text.split()) < 15 or garbage_ratio > 0.05:
                    doc = pymupdf.open(str(pdf_path))
                    text = _clean_ocr_text(doc[page_num].get_text("text").strip())
                    doc.close()
                    tier = "fitz_text"
                else:
                    tier = "pymupdf4llm"
                if len(text.split()) >= 15:
                    results.append(OcrPageResult(
                        source=pdf_path.name, page=page_num + 1, text=text,
                        tier_used=tier, char_count=len(text), word_count=len(text.split()),
                        is_ocr=False, language_guess=_guess_language(text)
                    ))
        except Exception as exc:
            log.warning("pymupdf4llm error for %s: %s — using fitz fallback", pdf_path.name, exc)
            import pymupdf
            doc = pymupdf.open(str(pdf_path))
            for page_num in digital_pages:
                text = _clean_ocr_text(doc[page_num].get_text("text").strip())
                if len(text.split()) >= 15:
                    results.append(OcrPageResult(
                        source=pdf_path.name, page=page_num + 1, text=text,
                        tier_used="fitz_text", char_count=len(text), word_count=len(text.split()),
                        is_ocr=False, language_guess=_guess_language(text)
                    ))
            doc.close()

    if scanned_pages:
        use_marker = _check_marker_available()
        for page_num in scanned_pages:
            text = ""
            if use_marker:
                text = _clean_ocr_text(_ocr_single_page_marker(pdf_path, page_num))
            if len(text.split()) < 15:
                text = _clean_ocr_text(_ocr_single_page_tesseract(pdf_path, page_num))
                tier = "tesseract"
            else:
                tier = "marker"
            if len(text.split()) >= 15:
                results.append(OcrPageResult(
                    source=pdf_path.name, page=page_num + 1, text=text,
                    tier_used=tier, char_count=len(text), word_count=len(text.split()),
                    is_ocr=True, language_guess=_guess_language(text)
                ))

    return results


def load_documents_sota(
    pdf_dir: str,
    preferred_tier: str = "auto",
    min_words_per_page: int = 15,
) -> List[Dict]:
    """Drop-in replacement for the old load_documents() — returns same List[Dict] format."""
    pdf_dir_path = Path(pdf_dir)
    pdf_files = list(pdf_dir_path.glob("*.pdf"))

    if not pdf_files:
        return [
            {"text": "الحصول على البطاقة الوطنية للتعريف الإلكترونية CNIE. الوثائق المطلوبة: شهادة الميلاد الكاملة، صورتان فوتوغرافيتان، إثبات الإقامة. مدة الإنجاز: 30 يوم عمل. الرسوم: 75 درهم.",
             "source": "sample_cnie_ar.pdf", "page": 1, "is_ocr": False},
            {"text": "Obtenir la Carte Nationale d'Identité Électronique CNIE. Documents requis: acte de naissance complet, deux photos d'identité récentes, justificatif de domicile. Délai: 30 jours ouvrables. Frais: 75 dirhams.",
             "source": "sample_cnie_fr.pdf", "page": 1, "is_ocr": False},
        ]

    all_pages: List[Dict] = []
    for pdf_path in pdf_files:
        for r in load_pdf_sota(pdf_path, preferred_tier):
            if r.word_count >= min_words_per_page:
                all_pages.append({
                    "text": r.text, "source": r.source, "page": r.page,
                    "is_ocr": r.is_ocr, "_tier": r.tier_used, "_lang_guess": r.language_guess,
                })
    return all_pages


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE DEBUGGER
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineDebugger:
    """
    Wraps a MoroccanRAGPipeline and produces a FullDebugReport from a single
    pipeline.ask() call. Includes per-tool latency, intermediate answers,
    reflection verdicts, and per-layer verification breakdown.

    Usage:
        debugger = PipelineDebugger(pipeline)
        report = debugger.debug("ما هي الوثائق المطلوبة؟")
        debugger.save_report(report)
    """

    def __init__(self, pipeline):
        self.pipe = pipeline

    def debug(self, question: str) -> FullDebugReport:
        t0 = time.time()
        result = self.pipe.ask(question)
        total_ms = round((time.time() - t0) * 1000)

        flags_dict = result.flags.to_dict() if hasattr(result.flags, "to_dict") else {}
        language = result.language or flags_dict.get("language", "unknown")

        # ── Planning ──────────────────────────────────────────────────────────
        trace = result.execution_trace
        plan = trace.plan if trace else []
        agent_plan = result.agent_state.plan if result.agent_state else None
        plan_source = agent_plan.plan_source if agent_plan else (
            "simple_path" if (trace and trace.agent_path == "simple") else "unknown"
        )
        plan_latency_ms = trace.plan_latency_ms if trace else 0.0

        # ── Tool calls ────────────────────────────────────────────────────────
        tool_calls: List[ToolCallDebug] = []
        for tc in (trace.tool_calls if trace else []):
            chunks_debug = [
                {
                    "text_preview": ctx[:200],
                    "score": score,
                }
                for ctx, score in zip(tc.contexts, tc.scores)
            ]
            tool_calls.append(ToolCallDebug(
                step_index=tc.step_index,
                tool_name=tc.tool_name,
                intent=tc.intent,
                query=tc.query,
                latency_ms=tc.latency_ms,
                chunks_returned=tc.chunks_returned,
                chunks=chunks_debug,
                intermediate_answer=tc.intermediate_answer,
                reflection=tc.reflection,
            ))

        # ── Retrieval ─────────────────────────────────────────────────────────
        initial_retrieval = result.retrieval.to_dict() if result.retrieval else {}
        all_contexts = trace.all_contexts if trace else []
        synthesis_context = trace.synthesis_context if trace else ""

        # Flat chunk list for debug.html renderRetCard() — reads d.retrieved / d.reranked
        def _sc_to_dict(sc) -> Dict[str, Any]:
            c = sc.chunk
            return {
                "chunk_id":       c.chunk_id,
                "source":         c.source,
                "page":           getattr(c, "page", 0),
                "language":       c.language,
                "text_preview":   c.text[:400],
                "bm25_score":     round(sc.bm25_score, 4) if sc.bm25_score else None,
                "dense_score":    round(sc.dense_score, 4) if sc.dense_score else None,
                "rrf_score":      round(sc.rrf_score, 4) if sc.rrf_score else None,
                "reranker_score": round(sc.reranker_score, 4) if sc.reranker_score else None,
            }

        retrieval_chunks = result.retrieval.chunks if result.retrieval else []
        retrieved = [_sc_to_dict(sc) for sc in retrieval_chunks]
        # Reranked = top-N (compress_top_n) subset — after reranking these are the first N
        from pipeline.config import settings as _cfg
        reranked = retrieved[:_cfg.compress_top_n] if retrieved else []

        # context_sent: prefer the traced synthesis context, fall back to formatted chunks
        context_sent = synthesis_context
        if not context_sent and retrieval_chunks:
            context_sent = result.retrieval.context if result.retrieval else ""

        # ── Verification breakdown ────────────────────────────────────────────
        audit = result.audit_trail
        verification_layers: List[VerificationLayerDebug] = []
        claim_results: List[Dict] = []
        entity_results: List[Dict] = []
        cfi = 0.0
        claim_grounded_ratio = 0.0

        if audit:
            cfi = round(audit.composite_fidelity_index, 4)
            claim_grounded_ratio = round(audit.claim_grounded_ratio, 4)

            claim_layer_details = []
            for cv in (audit.claims or []):
                row = {
                    "claim": cv.claim,
                    "grounded": cv.grounded,
                    "nli_score": round(cv.nli_score, 4),
                    "method": cv.method,
                    "evidence_preview": cv.evidence_text[:200] if cv.evidence_text else "",
                }
                claim_layer_details.append(row)
                claim_results.append(row)

            verification_layers.append(VerificationLayerDebug(
                layer="claim_nli",
                passed=claim_grounded_ratio >= 0.5,
                details=claim_layer_details,
            ))

            entity_layer_details = []
            for ev in (audit.entities or []):
                row = {
                    "entity": ev.entity,
                    "type": ev.entity_type,
                    "found": ev.found_in_context,
                    "exact_match": ev.exact_match,
                }
                entity_layer_details.append(row)
                entity_results.append(row)

            verification_layers.append(VerificationLayerDebug(
                layer="entity_grounding",
                passed=audit.entity_match_ratio >= 0.5,
                details=entity_layer_details,
            ))

            # citation_coverage is derived from metadata if present
            cit_cov = audit.metadata.get("citation_coverage", 0.0) if audit.metadata else 0.0
            verification_layers.append(VerificationLayerDebug(
                layer="citation_coverage",
                passed=cit_cov >= 0.5,
                details=[{"citation_coverage": round(cit_cov, 4)}],
            ))

        # ── Diagnostics ───────────────────────────────────────────────────────
        root_causes = self._analyze_root_causes(result, tool_calls, audit)
        recommendations = self._build_recommendations(result, audit)

        # ── KB summary for debug.html overview panel ──────────────────────────
        kb_summary: Dict[str, Any] = {}
        kb = self.pipe.kb
        if kb is not None:
            def _chunk_sample(chunks, n=5):
                return [
                    {
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "page": getattr(c, "page", 0),
                        "language": c.language,
                        "word_count": len(c.text.split()),
                        "text_preview": c.text[:300],
                    }
                    for c in (chunks or [])[:n]
                ]

            kb_summary = {
                "arabic_chunks": len(kb.arabic_chunks or []),
                "french_chunks": len(kb.french_chunks or []),
                "arabic_sample": _chunk_sample(kb.arabic_chunks),
                "french_sample": _chunk_sample(kb.french_chunks),
            }

        # ── Backward-compat steps list for debug.html ─────────────────────────
        steps = self._build_steps(result, tool_calls, audit, total_ms)

        return FullDebugReport(
            question=question,
            language=language,
            flags=flags_dict,
            total_latency_ms=total_ms,
            total_duration_ms=total_ms,       # alias — debug.html reads this name
            plan=plan,
            plan_source=plan_source,
            plan_latency_ms=plan_latency_ms,
            tool_calls=tool_calls,
            retrieved=retrieved,              # flat chunk list for renderRetCard()
            reranked=reranked,               # top-N subset after reranking
            context_sent=context_sent,       # exact string fed to LLM
            initial_retrieval=initial_retrieval,
            all_contexts=all_contexts,
            synthesis_context=synthesis_context,
            raw_answer=result.answer,
            final_answer=result.answer,
            verification_layers=verification_layers,
            claim_results=claim_results,
            entity_results=entity_results,
            cfi=cfi,
            claim_grounded_ratio=claim_grounded_ratio,
            root_causes=root_causes,
            recommendations=recommendations,
            kb_summary=kb_summary,
            steps=steps,
        )

    def save_report(self, report: FullDebugReport, out_dir: Optional[str] = None) -> str:
        from pipeline.config import settings
        out_path = Path(out_dir or settings.audit_log_dir) / "debug"
        out_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        qhash = abs(hash(report.question)) % 10000
        filepath = out_path / f"debug_{ts}_{qhash}.json"

        def _serial(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=_serial)
        log.info("Debug report saved: %s", filepath.name)
        return str(filepath)

    def snapshot_kb(self) -> Dict:
        if not self.pipe.kb:
            return {"error": "KB not initialised"}

        def chunk_info(c):
            return {
                "chunk_id": c.chunk_id, "source": c.source,
                "page": c.page, "language": c.language,
                "word_count": len(c.text.split()),
                "text_preview": c.text[:300],
            }

        return {
            "arabic_chunks": [chunk_info(c) for c in self.pipe.kb.arabic_chunks],
            "french_chunks": [chunk_info(c) for c in self.pipe.kb.french_chunks],
            "arabic_count": len(self.pipe.kb.arabic_chunks),
            "french_count": len(self.pipe.kb.french_chunks),
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_steps(self, result, tool_calls, audit, total_ms) -> List[Dict]:
        """Build a flat steps list for debug.html backward compatibility."""
        steps = []
        trace = result.execution_trace
        agent_path = trace.agent_path if trace else "unknown"
        plan_ms = trace.plan_latency_ms if trace else 0.0
        gen_ms = trace.generation_latency_ms if trace else 0.0
        ver_ms = trace.verification_latency_ms if trace else 0.0

        # ── Step 1: Language detection & retrieval ────────────────────────────
        retrieval = result.retrieval
        steps.append({
            "step": 1, "name": "Language Detection + Retrieval",
            "status": "ok" if retrieval and retrieval.chunks else "warn",
            "duration_ms": 0,
            "inputs": {
                "question": result.question,
                "language": result.language,
                "translation_used": result.translation_used,
                "retrieval_query": retrieval.retrieval_query if retrieval else "",
            },
            "outputs": {
                "chunks_retrieved": len(retrieval.chunks) if retrieval else 0,
                "reranker_applied": retrieval.reranker_applied if retrieval else False,
                "strategy": retrieval.retriever_strategy if retrieval else "—",
                "flags": result.flags.summary() if hasattr(result.flags, "summary") else str(result.flags),
            },
            "warnings": [] if (retrieval and retrieval.chunks) else ["No chunks retrieved — OOS threshold too high or KB empty"],
        })

        # ── Step 2: Planning (multihop) or Direct (simple) ───────────────────
        plan = result.agent_state.plan if result.agent_state else None
        if agent_path == "simple":
            steps.append({
                "step": 2, "name": "Generation (Simple path — no planning)",
                "status": "ok",
                "duration_ms": gen_ms,
                "inputs": {
                    "path": "simple",
                    "reason": "SIMPLE flag set — single intent, not legal, not multihop",
                    "context_words": len((trace.synthesis_context if trace else "").split()),
                },
                "outputs": {
                    "answer_length": len(result.answer),
                    "is_abstained": result.is_abstained,
                },
                "warnings": [],
            })
        else:
            # Planning step
            steps.append({
                "step": 2, "name": "Planning",
                "status": "ok" if (plan and plan.steps) else "warn",
                "duration_ms": plan_ms,
                "inputs": {
                    "path": agent_path,
                    "flags": result.flags.summary() if hasattr(result.flags, "summary") else "",
                },
                "outputs": {
                    "plan_steps": len(plan.steps) if plan else 0,
                    "plan_source": plan.plan_source if plan else "unknown",
                    "intents": [s.intent for s in plan.steps] if plan else [],
                },
                "warnings": [] if (plan and plan.steps) else ["Planning returned 0 steps — fallback plan used"],
            })

            # Per-tool call sub-steps
            for tc in tool_calls:
                steps.append({
                    "step": 3,
                    "name": f"Tool · {tc.tool_name} [{tc.intent}]",
                    "status": "ok" if tc.chunks_returned > 0 else "warn",
                    "duration_ms": tc.latency_ms,
                    "inputs": {
                        "query": tc.query,
                        "intent": tc.intent,
                    },
                    "outputs": {
                        "chunks_returned": tc.chunks_returned,
                        "intermediate_answer": tc.intermediate_answer[:300] if tc.intermediate_answer else "—",
                        "reflection": tc.reflection or "—",
                    },
                    "warnings": [] if tc.chunks_returned > 0 else [f"Tool returned 0 chunks for intent {tc.intent}"],
                })

            # Synthesis step (after tools)
            synth_step = max(3, 3 + len(tool_calls))
            steps.append({
                "step": synth_step, "name": "Synthesis",
                "status": "ok" if result.answer and len(result.answer) > 20 else "warn",
                "duration_ms": gen_ms,
                "inputs": {
                    "facts_collected": len(result.agent_state.facts) if result.agent_state else 0,
                    "context_words": len((trace.synthesis_context if trace else "").split()),
                },
                "outputs": {
                    "answer_length": len(result.answer),
                    "is_abstained": result.is_abstained,
                },
                "warnings": [],
            })

        # ── Final step: Verification ──────────────────────────────────────────
        ver_step = len(steps) + 1
        ver_status = "ok" if result.is_grounded else (
            "warn" if (audit and audit.entity_match_ratio >= 0.5) else "error"
        )
        steps.append({
            "step": ver_step, "name": "Verification (11 layers)",
            "status": ver_status,
            "duration_ms": ver_ms,
            "inputs": {
                "claims_checked": len(audit.claims) if audit else 0,
                "entities_checked": len(audit.entities) if audit else 0,
            },
            "outputs": {
                "cfi": round(audit.composite_fidelity_index, 3) if audit else 0,
                "claim_grounded_ratio": f"{audit.claim_grounded_ratio:.0%}" if audit else "—",
                "entity_match_ratio": f"{audit.entity_match_ratio:.0%}" if audit else "—",
                "is_grounded": result.is_grounded,
                "warnings": audit.warnings if audit else [],
            },
            "warnings": audit.warnings if audit else [],
        })

        return steps

    def _analyze_root_causes(self, result, tool_calls, audit) -> List[str]:
        causes = []
        if result.is_abstained:
            causes.append("Pipeline abstained — no relevant information found in knowledge base.")
        if not result.is_grounded and audit:
            n_claims = len(audit.claims or [])
            if n_claims == 0:
                causes.append(
                    f"0 claims extracted from answer — NLI grounding check was skipped. "
                    f"This can happen with Darija/dialect text where claim decomposition returns empty. "
                    f"CFI={audit.composite_fidelity_index:.2f}"
                )
            else:
                causes.append(
                    f"Answer not grounded: {n_claims} claims checked, "
                    f"claim_grounded_ratio={audit.claim_grounded_ratio:.0%}, "
                    f"CFI={audit.composite_fidelity_index:.2f}"
                )
        no_chunk_tools = [tc for tc in tool_calls if tc.chunks_returned == 0]
        if no_chunk_tools:
            causes.append(
                f"{len(no_chunk_tools)} tool call(s) returned 0 chunks: "
                + ", ".join(f"{tc.tool_name}[{tc.intent}]" for tc in no_chunk_tools)
            )
        insufficient = [tc for tc in tool_calls if "partial" in tc.reflection.lower() or "insufficient" in tc.reflection.lower()]
        if insufficient:
            causes.append(
                f"{len(insufficient)} step(s) had insufficient reflection: "
                + ", ".join(f"{tc.intent}" for tc in insufficient)
            )
        if not causes:
            causes.append("Pipeline completed normally — answer retrieved and grounded.")
        return causes

    def _build_recommendations(self, result, audit) -> List[str]:
        recs = []
        if result.is_abstained:
            recs.append("Add more government PDFs to the knowledge base and rebuild the index.")
        if audit and audit.claim_grounded_ratio < 0.5:
            recs.append(
                "Low claim grounding ratio — verify NLI model is loaded (check pipeline logs for 'NLI loaded')."
            )
        if audit and audit.composite_fidelity_index < 0.3:
            recs.append(
                "Very low CFI — the answer may not be supported by retrieved documents. "
                "Try lowering retrieval thresholds or adding more relevant PDFs."
            )
        if not recs:
            recs.append("Pipeline completed normally — review tool_calls and verification_layers for details.")
        return recs


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK STANDALONE RUN — python debug_pipeline.py "your question"
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "ما هي الوثائق المطلوبة للحصول على البطاقة الوطنية للتعريف؟"

    logging.basicConfig(level=logging.INFO)
    print(f"\nDebugging pipeline for: {question}\n")

    try:
        from pipeline.pipeline import MoroccanRAGPipeline
        pipe = MoroccanRAGPipeline()
        pipe.setup()
        pipe.build_knowledge_base(force_rebuild=False)

        debugger = PipelineDebugger(pipe)
        report = debugger.debug(question)
        print(f"Total latency: {report.total_latency_ms:.0f} ms")
        print(f"Plan steps: {len(report.plan)}")
        print(f"Tool calls: {len(report.tool_calls)}")
        print(f"CFI: {report.cfi:.3f}")
        print(f"Final answer: {report.final_answer[:300]}")
        print(f"Root causes: {report.root_causes}")
        debugger.save_report(report)

    except ImportError as exc:
        print(f"Could not import pipeline: {exc}")
        print("Run from the v12/ directory: python debug_pipeline.py")
