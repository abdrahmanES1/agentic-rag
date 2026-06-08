# -*- coding: utf-8 -*-
"""
API routes for the Moroccan RAG pipeline.

Endpoints:
  POST /api/ask                    — full pipeline (answer + retrieval + verification)
  POST /api/retrieve               — retrieval only (for benchmarking baselines)
  POST /api/build-kb               — trigger KB rebuild
  GET  /api/status                 — health check
  GET  /api/config                 — current settings
  POST /api/config                 — update runtime settings
  GET  /api/audit                  — list audit log files
  GET  /api/audit/<filename>       — fetch a specific audit log
  POST /api/debug                  — full pipeline trace
  GET  /api/debug/ollama           — Ollama connectivity test
  GET  /api/debug/kb               — KB snapshot
  GET  /api/debug/list-pdfs        — list source PDFs
  POST /api/debug/ocr              — test OCR tiers on a PDF
  GET  /api/debug/reports          — list saved debug reports
  GET  /api/debug/reports/<name>   — fetch a debug report
  POST /api/plan                   — preview execution plan (no generation)

Key difference from api_v2.py:
  /api/ask response now includes retrieval.contexts (List[str]) and
  execution_trace.all_contexts â€" everything RAGAS/ARES needs.
  No monkey-patching: NLI softmax is built into NLIVerifier.
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context
from pipeline.config import settings
from pipeline.language import classify_question, detect_and_translate, detect_language, translate_to_msa
from pipeline.models import QuestionFlags
from pipeline.pipeline import MoroccanRAGPipeline

from api.schemas import pipeline_result_to_dict, retrieval_result_to_dict

log = logging.getLogger("MoroccanRAG")

_SSE_TIMEOUT_SEC = 120
_SSE_POLL_SEC = 1.0


class _State:
    ready: bool = False
    error: str = ""


_state = _State()


def set_ready(ready: bool, error: str = "") -> None:
    _state.ready = ready
    _state.error = error


def register_routes(app: Flask, pipeline: MoroccanRAGPipeline) -> None:

    # â"€â"€ /api/status â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @app.route("/api/status", methods=["GET"])
    def status():
        ollama_ok = False
        if pipeline.ollama:
            try:
                test = pipeline.ollama.generate(
                    [{"role": "user", "content": "ping return pong"}],
                    max_tokens=50,
                    temperature=0.0,
                )
                ollama_ok = test is not None
            except Exception as exc:
                log.debug("Ollama ping failed: %s", exc)
        return jsonify(
            {
                "pipeline_ready": _state.ready,
                "pipeline_error": _state.error,
                "ollama_connected": ollama_ok,
                "ollama_model": settings.generator_model,
                "ollama_url": settings.ollama_base_url,
                "kb": pipeline.kb_status(),
                "api_stats": pipeline.ollama.stats() if pipeline.ollama else {},
            }
        )

    # â"€â"€ /api/ask â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @app.route("/api/ask", methods=["POST"])
    def ask():
        """
        POST /api/ask
        Body: {"question": "...", "mode": "fast" | "research"}

        mode=research (default):
            Sequential pipeline â€" returns single JSON with full retrieval context.
            Use this for benchmarking (RAGAS/ARES need the full response).

        mode=fast:
            Phase 1: answer immediately (skip_verify=True), stream SSE "answer"
            Phase 2: verify in background, stream SSE "verify"

        Response includes:
          retrieval.contexts      â€" List[str] for RAGAS
          execution_trace.all_contexts â€" multi-hop contexts for RAGAS
          ragas_contexts          â€" merged deduplicated contexts (initial + agentic)
          audit_trail             â€" CFI, claim_grounded_ratio, entity_match_ratio
        """
        if not _state.ready:
            return jsonify({"error": "Pipeline not ready. " + _state.error}), 503
        data = request.get_json(force=True, silent=True) or {}
        q = (data.get("question") or "").strip()
        mode = (data.get("mode") or "research").lower()
        if not q:
            return jsonify({"error": "Missing 'question'"}), 400

        if mode == "research" or mode != "fast":
            try:
                result = pipeline.ask(q)
                return jsonify({**pipeline_result_to_dict(result), "mode": "research"})
            except Exception as exc:
                log.error("Request failed", exc_info=True)
                return jsonify({"error": str(exc)}), 500

        # â"€â"€ Fast mode: SSE streaming â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
        def _generate_stream():
            t0 = time.time()
            try:
                result_phase1 = pipeline.ask(q, skip_verify=True)
                d1 = pipeline_result_to_dict(result_phase1)
                d1["mode"] = "fast"
                d1["verify_status"] = "pending"
                yield f"data: {json.dumps({'event': 'answer', 'payload': d1})}\n\n"
            except Exception as exc:
                log.error("Request failed", exc_info=True)
                yield f"data: {json.dumps({'event': 'error', 'payload': str(exc)})}\n\n"
                return

            verify_q: queue.Queue = queue.Queue()

            def _do_verify():
                try:
                    result_full = pipeline.ask(q, skip_verify=False)
                    verify_q.put(("ok", pipeline_result_to_dict(result_full)))
                except Exception as ex:
                    verify_q.put(("err", str(ex)))

            threading.Thread(target=_do_verify, daemon=True).start()

            timeout, elapsed = _SSE_TIMEOUT_SEC, 0
            while elapsed < timeout:
                try:
                    status_flag, payload = verify_q.get(timeout=_SSE_POLL_SEC)
                    if status_flag == "ok":
                        payload["mode"] = "fast"
                        payload["verify_status"] = "done"
                        payload["total_latency"] = round(time.time() - t0, 2)
                        yield f"data: {json.dumps({'event': 'verify', 'payload': payload})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'verify_error', 'payload': payload})}\n\n"
                    break
                except queue.Empty:
                    elapsed += 1
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

    # â"€â"€ /api/retrieve â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @app.route("/api/retrieve", methods=["POST"])
    def retrieve():
        """
        POST /api/retrieve
        Body: {"question": "...", "top_k": 5}

        Returns retrieval result with full chunk scores for benchmarking.
        Used by baseline implementations to retrieve context.
        """
        if not _state.ready:
            return jsonify({"error": "Pipeline not ready"}), 503
        data = request.get_json(force=True, silent=True) or {}
        q = (data.get("question") or "").strip()
        top_k = min(int(data.get("top_k", 5)), 20)
        if not q:
            return jsonify({"error": "Missing question"}), 400

        try:
            language, confidence, msa, llm_signals = detect_and_translate(q, pipeline.ollama)
            rq = q
            translated = False
            if language in ("Darija", "Arabizi") and msa and settings.enable_query_translation:
                rq, translated = msa, True

            flags = classify_question(rq, language, confidence, pipeline.ollama, llm_signals=llm_signals)

            if flags.OUTSCOPE:
                return jsonify(
                    {
                        "question": q,
                        "language": language,
                        "flags": flags.to_dict(),
                        "is_outscope": True,
                        "chunks": [],
                        "contexts": [],
                        "context": "",
                        "metadata": {},
                    }
                )

            retrieval = pipeline.retrieve(
                query=q,
                flags=flags,
                query_translated=translated,
                retrieval_query=rq,
            )
            is_oos = pipeline.is_out_of_scope(retrieval)

            resp = retrieval_result_to_dict(retrieval, language, flags, is_oos, top_k)
            resp["question"] = q
            return jsonify(resp)

        except Exception as exc:
            log.error("Request failed", exc_info=True)
            return jsonify({"error": str(exc)}), 500

    # â"€â"€ /api/build-kb â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @app.route("/api/build-kb", methods=["POST"])
    def build_kb():
        if not _state.ready:
            return jsonify({"error": "Pipeline not ready"}), 503
        data = request.get_json(force=True, silent=True) or {}
        force = bool(data.get("force_rebuild", False))
        contextual = bool(data.get("contextual_retrieval", False))
        try:
            t0 = time.time()
            pipeline.build_knowledge_base(
                force_rebuild=force, contextual_retrieval=contextual
            )
            return jsonify(
                {
                    "status": "ok",
                    "kb": pipeline.kb_status(),
                    "duration_sec": round(time.time() - t0, 1),
                }
            )
        except Exception as exc:
            log.error("Request failed", exc_info=True)
            return jsonify({"error": str(exc)}), 500

    # â"€â"€ /api/config â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @app.route("/api/config", methods=["GET"])
    def config():
        return jsonify(
            {
                "generator_model": settings.generator_model,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "retrieve_top_k": settings.retrieve_top_k,
                "reranker_top_n": settings.reranker_top_n,
                "compress_top_n": settings.compress_top_n,
                "bm25_weight": settings.bm25_weight,
                "dense_weight": settings.dense_weight,
                "nli_grounding_threshold": settings.nli_grounding_threshold,
                "claim_grounded_ratio": settings.claim_grounded_ratio,
                "enable_llm_judge": settings.enable_llm_judge,
                "enable_darija": settings.enable_darija,
                "enable_query_translation": settings.enable_query_translation,
            }
        )

    # /api/debug€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @app.route("/api/debug", methods=["POST"])
    def debug():
        """Full pipeline trace — returns FullDebugReport with per-tool latency,
        intermediate answers, reflection verdicts, and per-layer verification."""
        if not _state.ready:
            return jsonify({"error": "Pipeline not ready"}), 503
        data = request.get_json(force=True, silent=True) or {}
        q = (data.get("question") or "").strip()
        if not q:
            return jsonify({"error": "Missing question"}), 400
        try:
            from dataclasses import asdict

            from debug_pipeline import PipelineDebugger

            debugger = PipelineDebugger(pipeline)
            report = debugger.debug(q)
            return jsonify(asdict(report))
        except Exception as exc:
            log.error("Request failed", exc_info=True)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/debug/ollama", methods=["GET"])
    def debug_ollama():
        if pipeline.ollama is None:
            return jsonify({"error": "Ollama not initialized"}), 503
        return jsonify(pipeline.ollama.diagnose())

    @app.route("/api/debug/kb", methods=["GET"])
    def debug_kb():
        if pipeline.kb is None:
            return jsonify({"error": "KB not built"}), 503

        def chunk_dict(c):
            return {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "page": getattr(c, "page", 0),
                "language": c.language,
                "word_count": len(c.text.split()),
                "text_preview": c.text[:300],
            }

        arabic_chunks = [chunk_dict(c) for c in (pipeline.kb.arabic_chunks or [])]
        french_chunks = [chunk_dict(c) for c in (pipeline.kb.french_chunks or [])]

        return jsonify(
            {
                "arabic_chunks": arabic_chunks,
                "french_chunks": french_chunks,
                "kb": pipeline.kb_status(),
            }
        )

    # ── /api/config POST ──────────────────────────────────────────────────────

    _SAFE_RUNTIME_FIELDS = {
        "temperature",
        "max_new_tokens",
        "api_timeout",
        "retrieve_top_k",
        "reranker_top_n",
        "compress_top_n",
        "bm25_weight",
        "dense_weight",
        "outscope_score_threshold",
        "nli_grounding_threshold",
        "grounding_threshold",
        "claim_grounded_ratio",
        "enable_darija",
        "enable_arabizi",
        "enable_query_translation",
        "enable_llm_judge",
        "enable_audit_trail",
    }

    @app.route("/api/config", methods=["POST"])
    def set_config():
        data = request.get_json(force=True, silent=True) or {}
        updated, rejected = {}, {}
        for key, value in data.items():
            key_lower = key.lower()
            if key_lower not in _SAFE_RUNTIME_FIELDS:
                rejected[key] = "not a safe runtime field"
                continue
            if not hasattr(settings, key_lower):
                rejected[key] = "unknown field"
                continue
            try:
                current = getattr(settings, key_lower)
                if isinstance(current, bool):
                    coerced = bool(value)
                elif isinstance(current, int):
                    coerced = int(value)
                elif isinstance(current, float):
                    coerced = float(value)
                else:
                    coerced = value
                setattr(settings, key_lower, coerced)
                updated[key_lower] = coerced
            except Exception as exc:
                rejected[key] = str(exc)
        return jsonify(
            {
                "updated": updated,
                "rejected": rejected,
                "config": {
                    "generator_model": settings.generator_model,
                    "embedding_model": settings.embedding_model,
                    "temperature": settings.temperature,
                    "max_new_tokens": settings.max_new_tokens,
                    "retrieve_top_k": settings.retrieve_top_k,
                    "reranker_top_n": settings.reranker_top_n,
                    "compress_top_n": settings.compress_top_n,
                    "bm25_weight": settings.bm25_weight,
                    "dense_weight": settings.dense_weight,
                    "nli_grounding_threshold": settings.nli_grounding_threshold,
                    "claim_grounded_ratio": settings.claim_grounded_ratio,
                    "enable_llm_judge": settings.enable_llm_judge,
                    "enable_darija": settings.enable_darija,
                    "enable_query_translation": settings.enable_query_translation,
                },
            }
        )

    # ── /api/audit ────────────────────────────────────────────────────────────

    @app.route("/api/audit", methods=["GET"])
    def list_audits():
        audit_dir = Path(settings.audit_log_dir)
        if not audit_dir.exists():
            return jsonify([])
        files = sorted(audit_dir.glob("audit_*.json"), reverse=True)[:50]
        return jsonify([f.name for f in files])

    @app.route("/api/audit/<path:filename>", methods=["GET"])
    def get_audit(filename):
        p = Path(settings.audit_log_dir) / filename
        if not p.exists():
            return jsonify({"error": "Not found"}), 404
        with open(p, encoding="utf-8") as f:
            return jsonify(json.load(f))

    # ── /api/debug/list-pdfs ─────────────────────────────────────────────────

    @app.route("/api/debug/list-pdfs", methods=["GET"])
    def list_pdfs():
        pdf_dir = Path(settings.pdf_dir)
        if not pdf_dir.exists():
            return jsonify(
                {"error": f"pdf_dir does not exist: {settings.pdf_dir}"}
            ), 404
        files = [
            {"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in sorted(pdf_dir.glob("*.pdf"))
        ]
        return jsonify({"pdf_dir": str(pdf_dir), "count": len(files), "files": files})

    # ── /api/debug/ocr ───────────────────────────────────────────────────────

    @app.route("/api/debug/ocr", methods=["POST"])
    def debug_ocr():
        data = request.get_json(force=True, silent=True) or {}
        filename = data.get("filename", "")
        tier = data.get("tier", "auto")
        pdf_path = Path(settings.pdf_dir) / filename
        if not filename or not pdf_path.exists():
            available = [f.name for f in Path(settings.pdf_dir).glob("*.pdf")]
            return jsonify(
                {"error": f"File not found: {filename}", "available_pdfs": available}
            ), 404
        try:
            from debug_pipeline import (
                _load_with_fitz_fallback,
                _load_with_marker,
                _load_with_pymupdf4llm,
            )

            results = {}

            def _tier_result(loader, pdf):
                t0 = time.time()
                pages = loader(pdf)
                return {
                    "pages": len(pages),
                    "ok": len(pages) > 0,
                    "duration_ms": round((time.time() - t0) * 1000),
                    "sample": [
                        {
                            "page": p.page,
                            "words": p.word_count,
                            "lang": p.language_guess,
                            "preview": p.text[:300],
                        }
                        for p in pages[:3]
                    ],
                }

            if tier in ("auto", "pymupdf4llm"):
                results["pymupdf4llm"] = _tier_result(_load_with_pymupdf4llm, pdf_path)
            if tier in ("auto", "marker"):
                results["marker"] = _tier_result(_load_with_marker, pdf_path)
            if tier in ("auto", "fitz"):
                results["fitz_fallback"] = _tier_result(
                    _load_with_fitz_fallback, pdf_path
                )

            return jsonify({"filename": filename, "tiers": results})
        except Exception as exc:
            log.error("OCR test failed", exc_info=True)
            return jsonify({"error": str(exc)}), 500

    # ── /api/debug/reports ───────────────────────────────────────────────────

    @app.route("/api/debug/reports", methods=["GET"])
    def list_debug_reports():
        debug_dir = Path(settings.audit_log_dir) / "debug"
        if not debug_dir.exists():
            return jsonify([])
        files = sorted(debug_dir.glob("debug_*.json"), reverse=True)[:30]
        return jsonify([f.name for f in files])

    @app.route("/api/debug/reports/<path:filename>", methods=["GET"])
    def get_debug_report(filename):
        p = Path(settings.audit_log_dir) / "debug" / filename
        if not p.exists():
            return jsonify({"error": "Not found"}), 404
        with open(p, encoding="utf-8") as f:
            return jsonify(json.load(f))

    # ── /api/plan ─────────────────────────────────────────────────────────────

    @app.route("/api/plan", methods=["POST"])
    def preview_plan():
        """Generate the execution plan for a question without running it."""
        if not _state.ready:
            return jsonify({"error": "Pipeline not ready"}), 503
        data = request.get_json(force=True, silent=True) or {}
        q = (data.get("question") or "").strip()
        if not q:
            return jsonify({"error": "Missing 'question'"}), 400
        try:
            flags, language, plan = pipeline.plan_preview(q)
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
                    "flags": flags.to_dict(),
                    "language": language,
                }
            )
        except Exception as exc:
            log.error("Request failed", exc_info=True)
            return jsonify({"error": str(exc)}), 500
