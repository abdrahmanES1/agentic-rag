# -*- coding: utf-8 -*-
"""
Flask application factory.

Usage:
    flask --app api.app run
    python -m api.app
"""

# Must be set before any torch/faiss import — prevents OpenMP duplicate-lib
# segfault on Windows when CrossEncoder (torch) and faiss both load libiomp5md.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import threading
import traceback
from pathlib import Path

from flask import Flask, send_from_directory
from flask_cors import CORS

from pipeline.config import settings
from pipeline.pipeline import MoroccanRAGPipeline

log = logging.getLogger("MoroccanRAG")

_STATIC_DIR = Path(__file__).parent.parent / "static"

# Module-level pipeline singleton — shared across all requests.
# _startup_lock ensures setup() + build_knowledge_base() complete atomically
# before set_ready(True) is called. Per-request asks are safe because
# pipeline.ask() creates a fresh AgentState and never mutates kb/ollama/_retriever.
_pipeline = MoroccanRAGPipeline()
_startup_lock = threading.Lock()


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(_STATIC_DIR), static_url_path="/static")
    CORS(app)

    # ── Serve UI pages ────────────────────────────────────────────────────────
    @app.route("/")
    def ui_index():
        return send_from_directory(_STATIC_DIR, "index.html")

    @app.route("/debug")
    def ui_debug():
        return send_from_directory(_STATIC_DIR, "debug.html")

    from api.routes import register_routes, set_ready

    register_routes(app, _pipeline)

    return app


def _startup(pipeline: MoroccanRAGPipeline) -> None:
    from api.routes import set_ready

    with _startup_lock:
        try:
            log.info("[startup] Loading models...")
            pipeline.setup()
            log.info("[startup] Building knowledge base...")
            pipeline.build_knowledge_base(force_rebuild=False)
            set_ready(True)
            log.info("[startup] Pipeline ready.")
        except Exception as exc:
            set_ready(False, str(exc))
            log.error("[startup] ERROR: %s", exc, exc_info=True)


def get_pipeline() -> MoroccanRAGPipeline:
    return _pipeline


if __name__ == "__main__":
    # Start pipeline initialization in background thread so Flask starts immediately
    t = threading.Thread(target=_startup, args=(_pipeline,), daemon=True)
    t.start()

    app = create_app()
    app.run(host="0.0.0.0", port=settings.flask_port, debug=False, threaded=True)
