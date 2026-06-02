# -*- coding: utf-8 -*-
"""
Moroccan RAG V11 — Flask API v2
================================
Adds:
  - /api/debug          — full step-by-step pipeline trace
  - /api/debug/kb       — knowledge base snapshot (all chunks + text)
  - /api/debug/ollama   — direct Ollama connectivity test
  - /api/ocr-tiers      — test all OCR tiers on a single PDF
  Uses SOTA OCR (pymupdf4llm → marker → surya → fitz) for document loading.
  Includes the NLI softmax fix in-process.

Run:
  python api_v2.py
"""

import json
import os
import queue
import threading
import time
import traceback
from dataclasses import asdict, fields
from pathlib import Path

# ── APPLY NLI SOFTMAX FIX before anything runs ───────────────────────────────
# This patches verify_claim_nli in the already-imported module so that
# mDeBERTa-XNLI logits are properly converted to probabilities.
import moroccan_rag_v12 as _rag_module
import numpy as _np
from debug_pipeline import (
    PipelineDebugger,
    load_documents_sota,
    load_pdf_sota,
)
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

# ── pipeline & debug imports ──────────────────────────────────────────────────
from moroccan_rag_v12 import CONFIG, Config, MoroccanRAGPipeline


def _verify_claim_nli_fixed(claim, chunks):
    import logging

    from moroccan_rag_v12 import (
        CONFIG,
        NLI_AVAILABLE,
        NLI_MODEL,
        ClaimVerification,
        _load_nli_model,
        _verify_claim_word_overlap,
    )

    log = logging.getLogger("MoroccanRAG_V11")
    _load_nli_model()
    if NLI_AVAILABLE and NLI_MODEL is not None:
        try:
            context = " ".join(c.text for c in chunks[:5])
            context_trunc = " ".join(context.split()[:400])
            claim_trunc = " ".join(claim.split()[:100])
            raw = NLI_MODEL.predict([(context_trunc, claim_trunc)])[0]
            # ── softmax to get probabilities ──────────────────────────────
            logits = _np.array(raw, dtype=float)
            exp_l = _np.exp(logits - _np.max(logits))
            probs = exp_l / exp_l.sum()
            # mDeBERTa-XNLI id2label: {0: "entailment", 1: "neutral", 2: "contradiction"}
            entailment = float(probs[0])
            grounded = entailment >= CONFIG.NLI_GROUNDING_THRESHOLD
            # best supporting chunk
            best_chunk, best_score = None, 0.0
            for chunk in chunks[:5]:
                cs = NLI_MODEL.predict([(chunk.text[:400], claim_trunc)])[0]
                cs_l = _np.array(cs, dtype=float)
                cs_e = _np.exp(cs_l - _np.max(cs_l))
                cs_p = cs_e / cs_e.sum()
                ce = float(cs_p[0])
                if ce > best_score:
                    best_score, best_chunk = ce, chunk
            return ClaimVerification(
                claim=claim,
                grounded=grounded,
                nli_score=round(entailment, 4),
                supporting_chunk=best_chunk,
                evidence_text=best_chunk.text[:200] if best_chunk else "",
                confidence=round(entailment, 4),
                method="nli_softmax",
            )
        except Exception as e:
            log.warning(f"  NLI (fixed) failed: {e} → word_overlap")
    return _verify_claim_word_overlap(claim, chunks)


# Monkey-patch: replace the raw-logit version with the softmax-corrected one
_rag_module.verify_claim_nli = _verify_claim_nli_fixed
print("[api_v2] NLI softmax fix applied ✓")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

_pipeline = MoroccanRAGPipeline()
_debugger = None
_pipeline_ready = False
_pipeline_error = ""

# ── helpers ───────────────────────────────────────────────────────────────────


def _result_to_dict(r) -> dict:
    audit = None
    if r.audit_trail:
        a = r.audit_trail
        audit = {
            "timestamp": a.timestamp,
            "overall_grounded": a.overall_grounded,
            "claim_grounded_ratio": round(a.claim_grounded_ratio, 3),
            "entity_match_ratio": round(a.entity_match_ratio, 3),
            "composite_fidelity_index": round(a.composite_fidelity_index, 3),
            "warnings": a.warnings,
            "metadata": a.metadata,
            "claims": [
                {
                    "claim": c.claim,
                    "grounded": c.grounded,
                    "nli_score": round(c.nli_score, 3),
                    "confidence": round(c.confidence, 3),
                    "method": c.method,
                    "evidence": c.evidence_text[:200],
                }
                for c in a.claims
            ],
            "entities": [
                {
                    "entity": e.entity,
                    "entity_type": e.entity_type,
                    "found": e.found_in_context,
                    "exact_match": e.exact_match,
                }
                for e in a.entities
            ],
        }
    flags = None
    if r.flags:
        flags = {
            "SIMPLE": r.flags.SIMPLE,
            "MULTIHOP": r.flags.MULTIHOP,
            "LEGAL": r.flags.LEGAL,
            "OUTSCOPE": r.flags.OUTSCOPE,
            "language": r.flags.language,
            "confidence": round(r.flags.confidence, 3),
            "hop_count": r.flags.hop_count,
            "intents": r.flags.intents,
        }

    # FIX 71-75: Planning trace from AgentState
    planning = None
    if r.agent_state:
        s = r.agent_state
        planning = s.to_audit_dict()  # plan, facts, reflections, scratchpad

    return {
        "question": r.question,
        "answer": r.answer,
        "language": r.language,
        "lang_confidence": round(r.lang_confidence, 3),
        "flags": flags,
        "sources": r.sources,
        "is_grounded": r.is_grounded,
        "is_abstained": r.is_abstained,
        "latency_sec": round(r.latency_sec, 2),
        "agent_steps": r.agent_steps,
        "memory_stats": r.memory_stats,
        "audit_trail": audit,
        "planning": planning,  # NEW — planning trace
        "translation_used": r.translation_used,
        "original_query": r.original_query,
    }


def _config_to_dict() -> dict:
    return {f.name: getattr(CONFIG, f.name) for f in fields(CONFIG)}


def _kb_status() -> dict:
    if _pipeline.kb is None:
        return {
            "arabic": False,
            "french": False,
            "unified": False,
            "arabic_chunks": 0,
            "french_chunks": 0,
            "total_chunks": 0,
        }
    unified_ok = (
        getattr(_pipeline.kb, "unified_faiss", None) is not None
        and len(getattr(_pipeline.kb, "all_chunks", [])) > 0
    )
    return {
        "arabic": _pipeline.kb.arabic_faiss is not None,
        "french": _pipeline.kb.french_faiss is not None,
        "unified": unified_ok,  # FIX 70
        "arabic_chunks": len(_pipeline.kb.arabic_chunks),
        "french_chunks": len(_pipeline.kb.french_chunks),
        "total_chunks": len(getattr(_pipeline.kb, "all_chunks", [])),
    }


def _startup():
    global _pipeline_ready, _pipeline_error, _debugger
    try:
        print("[startup] Loading models…")
        _pipeline.setup()
        print("[startup] Building KB with SOTA OCR…")
        # Use SOTA OCR loader for document loading
        _build_kb_sota(force_rebuild=False)
        _debugger = PipelineDebugger(_pipeline)
        _pipeline_ready = True
        print("[startup] Pipeline ready ✓")
    except Exception as e:
        _pipeline_error = str(e)
        print(f"[startup] ERROR: {e}")
        traceback.print_exc()


def _build_kb_sota(force_rebuild: bool = False, contextual_retrieval: bool = False):
    """
    Build KB using SOTA OCR + optional Contextual Retrieval (FIX 69).
    Also builds the unified FAISS index (FIX 70).
    """
    from moroccan_rag_v12 import KnowledgeBase, chunk_documents

    _pipeline.kb = KnowledgeBase(_pipeline.embedding_model)
    if not force_rebuild and _pipeline.kb.load(CONFIG.INDEX_DIR):
        print("[SOTA KB] Loaded from disk ✓")
        return
    pages = load_documents_sota(CONFIG.PDF_DIR)
    ar_chunks, fr_chunks = chunk_documents(pages)
    # FIX 69: pass ollama for contextual enrichment if requested
    ollama_for_enrichment = _pipeline.ollama if contextual_retrieval else None
    if contextual_retrieval:
        print("[SOTA KB] Contextual Retrieval enabled — enriching chunks…")
    _pipeline.kb.build(ar_chunks, fr_chunks, ollama=ollama_for_enrichment)
    _pipeline.kb.save(CONFIG.INDEX_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# ORIGINAL ENDPOINTS (unchanged from api.py)
# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/api/status", methods=["GET"])
def status():
    ollama_ok = False
    if _pipeline.ollama:
        try:
            test = _pipeline.ollama.generate(
                [{"role": "user", "content": "ping return just one word"}],
                max_tokens=400,
                temperature=0.1,
                fmt={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "heart_beats",
                        "schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                            "type": "string"
                            }
                        },
                        "required": ["status"]
                        }
                    }
                    }
            )
            ollama_ok = test is not None
        except Exception:
            pass
    return jsonify(
        {
            "pipeline_ready": _pipeline_ready,
            "pipeline_error": _pipeline_error,
            "ollama_connected": ollama_ok,
            "ollama_model": CONFIG.GENERATOR_MODEL,
            "ollama_url": CONFIG.OLLAMA_BASE_URL,
            "kb": _kb_status(),
            "nli_fix_applied": True,
            "api_stats": _pipeline.ollama.stats() if _pipeline.ollama else {},
        }
    )


@app.route("/api/ask", methods=["POST"])
def ask():
    """
    POST /api/ask
    Body: { "question": "...", "mode": "fast" | "research" }

    mode=fast     (default, chatbot mode):
        Phase 1 — Generate answer immediately, stream SSE event "answer"
        Phase 2 — Run verification in background thread, stream SSE event "verify"
        Phase 3 — Stream SSE event "done" with full result

    mode=research (thesis/demo mode):
        All 11 layers run sequentially, single JSON response (no streaming).
        Identical to old behaviour.
    """
    if not _pipeline_ready:
        return jsonify({"error": "Pipeline not ready. " + _pipeline_error}), 503
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("question") or "").strip()
    mode = (data.get("mode") or "fast").lower()
    if not q:
        return jsonify({"error": "Missing 'question'"}), 400

    # ── RESEARCH MODE — sequential, single JSON (original behaviour) ──────────
    if mode == "research":
        try:
            result = _pipeline.answer(q)
            return jsonify({**_result_to_dict(result), "mode": "research"})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # ── FAST MODE — two-phase streaming via SSE ───────────────────────────────
    # Phase 1: run pipeline WITHOUT verification (skip layers 1-11)
    # Phase 2: run verification in background, push result when done
    def _generate_stream():
        t0 = time.time()

        # ── PHASE 1: answer (no verify) ──────────────────────────────────────
        try:
            result_phase1 = _pipeline.answer(q, skip_verify=True)
            d1 = _result_to_dict(result_phase1)
            d1["mode"] = "fast"
            d1["verify_status"] = "pending"
            # SSE: send answer immediately so the GUI can render it
            yield f"data: {json.dumps({'event': 'answer', 'payload': d1})}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'event': 'error', 'payload': str(e)})}\n\n"
            return

        # ── PHASE 2: verify in background thread ─────────────────────────────
        verify_q = queue.Queue()

        def _do_verify():
            try:
                result_full = _pipeline.answer(
                    q, skip_verify=False, _cached_result=result_phase1
                )
                verify_q.put(("ok", _result_to_dict(result_full)))
            except Exception as ex:
                verify_q.put(("err", str(ex)))

        t = threading.Thread(target=_do_verify, daemon=True)
        t.start()

        # Poll with timeout — push keepalive comments so connection stays open
        timeout = 120  # seconds
        elapsed = 0
        while elapsed < timeout:
            try:
                status, payload = verify_q.get(timeout=1.0)
                if status == "ok":
                    payload["mode"] = "fast"
                    payload["verify_status"] = "done"
                    payload["total_latency"] = round(time.time() - t0, 2)
                    yield f"data: {json.dumps({'event': 'verify', 'payload': payload})}\n\n"
                else:
                    yield f"data: {json.dumps({'event': 'verify_error', 'payload': payload})}\n\n"
                break
            except queue.Empty:
                elapsed += 1
                # SSE keepalive comment
                yield ": keepalive\n\n"
        else:
            yield f"data: {json.dumps({'event': 'verify_timeout'})}\n\n"

        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return Response(
        stream_with_context(_generate_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    """Steps 4-8 only. Returns chunks, no generation."""
    if not _pipeline_ready:
        return jsonify({"error": "Pipeline not ready. " + _pipeline_error}), 503
    data  = request.get_json(force=True, silent=True) or {}
    q     = (data.get("question") or "").strip()
    top_k = min(int(data.get("top_k", 5)), 10)
    if not q:
        return jsonify({"error": "Missing question"}), 400
    try:
        from moroccan_rag_v12 import (
            detect_language, translate_to_msa, classify_question,
            hybrid_retrieve, compress_context,
        )
        language, confidence = detect_language(q)
        rq = q
        if language in ("Darija", "Arabizi"):
            msa = translate_to_msa(q, language, _pipeline.ollama)
            if msa: rq = msa
        flags = classify_question(rq, language, confidence, _pipeline.ollama)
        fd = {"SIMPLE": flags.SIMPLE, "MULTIHOP": flags.MULTIHOP,
              "OUTSCOPE": flags.OUTSCOPE, "LEGAL": flags.LEGAL,
              "language": flags.language, "intents": flags.intents,
              "hop_count": flags.hop_count}
        if flags.OUTSCOPE:
            return jsonify({"question": q, "language": language, "flags": fd,
                            "is_outscope": True, "chunks": [], "context": ""})
        scored, is_out = hybrid_retrieve(rq, flags, _pipeline.kb, _pipeline.embedding_model)
        if is_out or not scored:
            return jsonify({"question": q, "language": language, "flags": fd,
                            "is_outscope": bool(is_out), "chunks": [], "context": ""})
        selected, _ = compress_context(scored, rq, _pipeline.reranker, flags)
        sc_map = {sc.chunk.chunk_id: sc.rrf_score for sc in scored}
        chunks_out = [{"text": c.text, "source": c.source, "page": c.page,
                        "language": c.language,
                        "score": round(sc_map.get(c.chunk_id, 0.0), 4)}
                       for c in selected[:top_k]]
        sep = "\n\n"
        ctx = sep.join(
            "[Source: " + c["source"] + " | Page: " + str(c["page"]) + "]\n" + c["text"]
            for c in chunks_out)
        return jsonify({"question": q, "language": language, "flags": fd,
                        "is_outscope": False, "chunks": chunks_out, "context": ctx})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




@app.route("/api/build-kb", methods=["POST"])
def build_kb():
    if not _pipeline_ready:
        return jsonify({"error": "Pipeline not ready"}), 503
    data = request.get_json(force=True, silent=True) or {}
    force = bool(data.get("force_rebuild", False))
    tier = data.get("ocr_tier", "auto")
    contextual_retrieval = bool(data.get("contextual_retrieval", False))
    try:
        t0 = time.time()
        if tier == "original":
            _pipeline.build_knowledge_base(
                force_rebuild=force, contextual_retrieval=contextual_retrieval
            )
        else:
            _build_kb_sota(
                force_rebuild=force, contextual_retrieval=contextual_retrieval
            )
        return jsonify(
            {
                "ok": True,
                "elapsed_sec": round(time.time() - t0, 1),
                "kb": _kb_status(),
                "ocr_tier": tier,
                "contextual_retrieval": contextual_retrieval,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


_SAFE_RUNTIME_FIELDS = {
    "TEMPERATURE",
    "MAX_NEW_TOKENS",
    "API_TIMEOUT",
    "API_MAX_RETRIES",
    "API_RETRY_DELAY",
    "RETRIEVE_TOP_K",
    "RERANKER_TOP_N",
    "COMPRESS_TOP_N",
    "BM25_WEIGHT",
    "DENSE_WEIGHT",
    "NLI_GROUNDING_THRESHOLD",
    "GROUNDING_THRESHOLD",
    "CLAIM_GROUNDED_RATIO",
    "AMBIGUOUS_NLI_THRESHOLD",
    "ENTITY_EXACT_MATCH",
    "CFI_WEIGHT_ENTITY",
    "CFI_WEIGHT_RELATION",
    "MEMORY_MAX_CHUNKS",
    "MEMORY_MIN_SCORE",
    "MEMORY_MAX_PER_SRC",
    "OUTSCOPE_SCORE_THRESHOLD",
    "DARIJA_MARKER_MIN",
    "ARABIZI_MARKER_MIN",
    "ENABLE_DARIJA",
    "ENABLE_ARABIZI",
    "ENABLE_QUERY_TRANSLATION",
    "ENABLE_LLM_JUDGE",
    "ENABLE_AUDIT_TRAIL",
    "ENABLE_ENTITY_VERIFICATION",
    "ENABLE_CHAIN_VERIFICATION",
}
_FIELD_TYPES = {f.name: f.type for f in fields(Config)}


def _coerce(name, value):
    typ = _FIELD_TYPES.get(name, "str")
    if typ in ("bool", bool):
        return bool(value)
    if typ in ("int", int):
        return int(value)
    if typ in ("float", float):
        return float(value)
    return value


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(_config_to_dict())


@app.route("/api/config", methods=["POST"])
def set_config():
    data = request.get_json(force=True, silent=True) or {}
    updated, rejected = {}, {}
    for key, value in data.items():
        if key not in _SAFE_RUNTIME_FIELDS:
            rejected[key] = "not a safe runtime field"
            continue
        if not hasattr(CONFIG, key):
            rejected[key] = "unknown field"
            continue
        try:
            coerced = _coerce(key, value)
            setattr(CONFIG, key, coerced)
            updated[key] = coerced
        except Exception as e:
            rejected[key] = str(e)
    return jsonify(
        {"updated": updated, "rejected": rejected, "config": _config_to_dict()}
    )


@app.route("/api/audit", methods=["GET"])
def list_audits():
    files = sorted(Path(CONFIG.AUDIT_LOG_DIR).glob("audit_*.json"), reverse=True)[:50]
    return jsonify([f.name for f in files])


@app.route("/api/audit/<path:filename>", methods=["GET"])
def get_audit(filename):
    p = Path(CONFIG.AUDIT_LOG_DIR) / filename
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    with open(p, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════════
# NEW DEBUG ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/api/debug", methods=["POST"])
def debug_ask():
    """
    Full step-by-step pipeline trace.
    Same as /api/ask but returns every internal step, retrieved chunks,
    context sent to LLM, and root cause analysis.

    Body: { "question": "..." }
    """
    if not _pipeline_ready:
        return jsonify({"error": "Pipeline not ready. " + _pipeline_error}), 503
    if not _debugger:
        return jsonify({"error": "Debugger not initialised"}), 503

    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "Missing 'question'"}), 400

    try:
        report = _debugger.debug(q)
        # Save debug report to disk
        debug_dir = Path(CONFIG.AUDIT_LOG_DIR) / "debug"
        _debugger.save_report(report, str(debug_dir))

        return jsonify(
            {
                "question": report.question,
                "total_duration_ms": report.total_duration_ms,
                "steps": [asdict(s) for s in report.steps],
                "kb_summary": {
                    "arabic_chunks": len(report.chunks_arabic),
                    "french_chunks": len(report.chunks_french),
                    "arabic_sample": [asdict(c) for c in report.chunks_arabic[:5]],
                    "french_sample": [asdict(c) for c in report.chunks_french[:5]],
                },
                "retrieved": [asdict(r) for r in report.retrieved],
                "reranked": [asdict(r) for r in report.reranked],
                "context_sent": report.context_sent,
                "raw_answer": report.raw_answer,
                "final_answer": report.final_answer,
                "root_causes": report.root_causes,
                "recommendations": report.recommendations,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/kb", methods=["GET"])
def debug_kb():
    """
    Full KB snapshot: every chunk with its full text.
    Useful for verifying OCR extraction quality.
    """
    if not _debugger:
        return jsonify({"error": "Debugger not initialised"}), 503
    return jsonify(_debugger.snapshot_kb())


@app.route("/api/debug/ollama", methods=["GET"])
def debug_ollama():
    """
    Direct Ollama connectivity and generation test.
    Returns the model's response to a simple prompt.
    """
    if not _pipeline.ollama:
        return jsonify({"ok": False, "error": "No Ollama client"})
    try:
        t0 = time.time()
        resp = _pipeline.ollama.generate(
            [{"role": "user", "content": "Reply with exactly: OLLAMA_OK"}],
            temperature=0.0,
            max_tokens=20,
        )
        latency = round((time.time() - t0) * 1000)
        return jsonify(
            {
                "ok": resp is not None,
                "response": resp,
                "latency_ms": latency,
                "model": CONFIG.GENERATOR_MODEL,
                "url": CONFIG.OLLAMA_BASE_URL,
                "api_stats": _pipeline.ollama.stats(),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/debug/ocr", methods=["POST"])
def debug_ocr():
    """
    Test all OCR tiers on a single PDF from PDF_DIR.
    Body: { "filename": "myfile.pdf", "tier": "auto" }
    Returns per-tier extraction results for comparison.
    """
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename", "")
    tier = data.get("tier", "auto")

    pdf_path = Path(CONFIG.PDF_DIR) / filename
    if not filename or not pdf_path.exists():
        # List available PDFs
        pdfs = [f.name for f in Path(CONFIG.PDF_DIR).glob("*.pdf")]
        return jsonify({"error": f"File not found: {filename}", "available_pdfs": pdfs})

    results = {}
    from debug_pipeline import (
        _load_with_fitz_fallback,
        _load_with_marker,
        _load_with_pymupdf4llm,
        _load_with_surya,
    )

    if tier in ("auto", "pymupdf4llm"):
        t0 = time.time()
        r = _load_with_pymupdf4llm(pdf_path)
        results["pymupdf4llm"] = {
            "pages": len(r),
            "ok": len(r) > 0,
            "duration_ms": round((time.time() - t0) * 1000),
            "sample": [
                {
                    "page": p.page,
                    "words": p.word_count,
                    "lang": p.language_guess,
                    "preview": p.text[:300],
                }
                for p in r[:3]
            ],
        }
    if tier in ("auto", "marker"):
        t0 = time.time()
        r = _load_with_marker(pdf_path)
        results["marker"] = {
            "pages": len(r),
            "ok": len(r) > 0,
            "duration_ms": round((time.time() - t0) * 1000),
            "sample": [
                {
                    "page": p.page,
                    "words": p.word_count,
                    "lang": p.language_guess,
                    "preview": p.text[:300],
                }
                for p in r[:3]
            ],
        }
    if tier in ("auto", "fitz"):
        t0 = time.time()
        r = _load_with_fitz_fallback(pdf_path)
        results["fitz_fallback"] = {
            "pages": len(r),
            "ok": len(r) > 0,
            "duration_ms": round((time.time() - t0) * 1000),
            "sample": [
                {
                    "page": p.page,
                    "words": p.word_count,
                    "lang": p.language_guess,
                    "preview": p.text[:300],
                }
                for p in r[:3]
            ],
        }

    return jsonify({"filename": filename, "tiers": results})


@app.route("/api/debug/list-pdfs", methods=["GET"])
def list_pdfs():
    """List all PDFs in PDF_DIR with basic stats."""
    pdf_dir = Path(CONFIG.PDF_DIR)
    if not pdf_dir.exists():
        return jsonify({"error": f"PDF_DIR does not exist: {CONFIG.PDF_DIR}"})
    pdfs = []
    for f in sorted(pdf_dir.glob("*.pdf")):
        pdfs.append(
            {
                "filename": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "path": str(f),
            }
        )
    return jsonify({"pdf_dir": str(pdf_dir), "count": len(pdfs), "files": pdfs})


@app.route("/api/debug/reports", methods=["GET"])
def list_debug_reports():
    """List all saved debug report JSON files."""
    debug_dir = Path(CONFIG.AUDIT_LOG_DIR) / "debug"
    if not debug_dir.exists():
        return jsonify([])
    files = sorted(debug_dir.glob("debug_*.json"), reverse=True)[:30]
    return jsonify([f.name for f in files])


@app.route("/api/debug/reports/<path:filename>", methods=["GET"])
def get_debug_report(filename):
    """Fetch a saved debug report."""
    p = Path(CONFIG.AUDIT_LOG_DIR) / "debug" / filename
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    with open(p, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/api/plan", methods=["POST"])
def preview_plan():
    """
    FIX 74: Generate the execution plan for a question WITHOUT running it.
    Useful for the debug console to show what the planner would do.

    Body: { "question": "..." }
    Returns: { "plan": [{step_id, intent, tool, sub_question, rationale}...],
               "plan_source": "llm"|"fallback", "flags": {...} }
    """
    if not _pipeline_ready:
        return jsonify({"error": "Pipeline not ready"}), 503
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "Missing 'question'"}), 400
    try:
        from moroccan_rag_v12 import (
            AgentState,
            PlannerAgent,
            classify_question,
            detect_language,
        )

        language, confidence = detect_language(q)
        flags = classify_question(q, language, confidence, _pipeline.ollama)
        agent = PlannerAgent(_pipeline.ollama, _pipeline.kb, _pipeline.embedding_model)
        state = AgentState(question=q, language=language, flags=flags)
        plan = agent._plan(q, flags, state)
        return jsonify(
            {
                "plan": [
                    {
                        "step_id": s.step_id,
                        "intent": s.intent,
                        "tool": s.tool,
                        "tool_args": s.tool_args,
                        "sub_question": s.sub_question,
                        "rationale": s.rationale,
                        "depends_on": s.depends_on,
                    }
                    for s in plan.steps
                ],
                "plan_source": plan.plan_source,
                "flags": {
                    "SIMPLE": flags.SIMPLE,
                    "MULTIHOP": flags.MULTIHOP,
                    "LEGAL": flags.LEGAL,
                    "intents": flags.intents,
                    "hop_count": flags.hop_count,
                },
                "language": language,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _startup()
    app.run(host="0.0.0.0", port=5000, debug=False)
