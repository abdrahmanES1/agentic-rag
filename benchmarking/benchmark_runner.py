#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_runner.py — Moroccan RAG Benchmark (v12 refactored)

Compares 8 pipelines on the gold test set:
  1. naive_rag       — retrieve → 1-shot generate            (Lewis 2020)
  2. basic_react     — Thought/Action/Observation loop        (Yao 2023)
  3. adaptive_simple — LLM-based simple/multihop routing      (Jeong 2024)
  4. hyde            — hypothetical document embeddings        (Gao 2022)
  5. self_rag        — reflection token grounding              (Asai 2023)
  6. flare           — forward-looking active retrieval        (Jiang 2023)
  7. crag            — corrective retrieval augmented gen      (Shi 2024)
  8. v12_pipeline    — full agentic pipeline (ours)

REQUIRES: `python -m api.app` running on port 5000 (for /api/ask and /api/retrieve)

USAGE:
  python benchmark_runner.py
  python benchmark_runner.py --quick          # first 10 questions
  python benchmark_runner.py --no-v12         # baselines only
  python benchmark_runner.py --baselines naive_rag,hyde
  python benchmark_runner.py --no-ragas       # skip RAGAS (no LLM eval)
  python benchmark_runner.py --no-ares        # skip ARES LLM judge (slow)
  python benchmark_runner.py --resume         # continue latest interrupted run
  python benchmark_runner.py --resume results/run_20260511_012448  # specific run

  # Ground-truth metrics (disabled by default — enable once you have gold_answer/gold_keywords):
  python benchmark_runner.py --with-gt

Metric tiers
────────────
  ALWAYS ON (reference-free):
    • RAGAS core:        faithfulness, answer_relevancy, context_precision
    • ARES core:         ares_context_relevance, ares_answer_faithfulness, ares_answer_relevance
    • ARES extended:     ares_completeness, ares_dialect_coherence, ares_legal_accuracy,
                         ares_multihop_coverage, ares_abstain_quality
    • G-Eval:            geval_coherence, geval_consistency, geval_fluency, geval_relevance
    • FActScore:         factscore, unsupported_claim_rate, avg_claims_per_answer
    • Retrieval quality: retrieval_context_coverage, retrieval_mrr, retrieval_context_utilization
    • Domain precision:  domain_legal_citation_hit, domain_cost_deadline_hit,
                         domain_dialect_response_match, domain_hallucination_number_rate
    • Efficiency:        avg/p50/p95/p99 latency, abstain_rate, avg_contexts_retrieved
    • V12-only:          v12_cfi, v12_claim_grounded_ratio, v12_is_grounded_rate

  NEEDS GROUND TRUTH (--with-gt):
    • Lexical:           exact_match, token_f1, rouge_l            (needs gold_answer)
    • Semantic:          bertscore_precision/recall/f1             (needs gold_answer)
    • RAGAS ext v2:      context_recall, answer_correctness,
                         context_entity_recall, factual_correctness (needs gold_answer)
    • RGB robustness:    noise_robustness, negative_rejection,
                         information_integration, counterfactual    (needs gold_answer)
    • Cross-lingual:     cross_lingual_consistency                  (needs gold_answer)
    • Domain:            keyword_hit_rate                           (needs gold_keywords)
    • Abstain:           abstain_accuracy/precision/recall/f1       (needs should_abstain)
    • Breakdowns:        per-category, per-language, win-rate       (needs gold_answer)

Checkpoint / resume
───────────────────
  Every baseline's raw answers are saved to results/run_<ts>/ as soon as that
  baseline finishes. V12 answers are saved after EACH question. Scores are saved
  immediately after each baseline is evaluated. A manifest.json tracks what is done.

  On crash: re-run with --resume to skip completed baselines/scores and pick up
  from where the run left off.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path fix ──────────────────────────────────────────────────────────────────
# Allow running as `python benchmark_runner.py` from inside benchmarking/ AND
# as `python benchmarking/benchmark_runner.py` from the v12 root.
_ROOT = Path(__file__).resolve().parent.parent   # v12/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_runner")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
TESTSET_FILE = _HERE / "benchmark_testset_gold.json"
RESULTS_DIR = _HERE / "results"
V12_API = "http://localhost:5000"

ALL_BASELINES = [
    "naive_rag",
    "basic_react",
    "adaptive_simple",
    "hyde",
    "self_rag",
    "flare",
    "crag",
]


# ---------------------------------------------------------------------------
# CheckpointManager — save on every step, resume after a crash
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Saves data incrementally so a crash never loses completed work.

    Directory layout (under run_dir/):
      manifest.json               — tracks which baselines are results/scores done
      raw_<baseline>.json         — answers for a completed baseline (all questions)
      raw_v12_pipeline.json       — v12 answers, updated after EACH question
      scores_<baseline>.json      — metric scores for a completed baseline
      breakdown.json              — per-category/language/win-rate (final step)
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = run_dir / "manifest.json"
        self.manifest: Dict = self._load_manifest()
        log.info("[checkpoint] Run directory: %s", self.run_dir)

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _load_manifest(self) -> Dict:
        if self._manifest_path.exists():
            with open(self._manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {"created_at": datetime.now().isoformat(), "systems": {}}

    def _flush_manifest(self) -> None:
        self.manifest["updated_at"] = datetime.now().isoformat()
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def _sys(self, name: str) -> Dict:
        return self.manifest["systems"].setdefault(name, {})

    # ── Results (raw answers) ─────────────────────────────────────────────────

    def has_results(self, name: str) -> bool:
        return self._sys(name).get("results_done", False)

    def load_results(self, name: str) -> Optional[List[dict]]:
        path = self.run_dir / f"raw_{name}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            log.info("[checkpoint] Loaded %d results for %s from checkpoint", len(data), name)
            return data
        return None

    def save_results(self, name: str, results: List[dict]) -> None:
        """Save all results for a baseline at once (called when baseline finishes)."""
        path = self.run_dir / f"raw_{name}.json"
        _atomic_write(path, results)
        self._sys(name).update({
            "results_done": True,
            "results_count": len(results),
            "results_saved_at": datetime.now().isoformat(),
        })
        self._flush_manifest()
        log.info("[checkpoint] ✓ Results saved: %s (%d items) → %s", name, len(results), path.name)

    # ── V12 partial save (per question) ──────────────────────────────────────

    def load_partial_v12(self) -> List[dict]:
        """Load partially completed v12 results so we can resume mid-run."""
        path = self.run_dir / "raw_v12_pipeline.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            log.info("[checkpoint] Resuming v12_pipeline from question %d", len(data) + 1)
            return data
        return []

    def save_partial_v12(self, results: List[dict]) -> None:
        """Overwrite v12 results after each question (cheap, ~1 ms)."""
        path = self.run_dir / "raw_v12_pipeline.json"
        _atomic_write(path, results)
        # Update manifest count but don't mark done yet
        self._sys("v12_pipeline")["results_count"] = len(results)
        self._sys("v12_pipeline")["last_saved_at"] = datetime.now().isoformat()
        self._flush_manifest()

    # ── Scores ────────────────────────────────────────────────────────────────

    def has_scores(self, name: str) -> bool:
        return self._sys(name).get("scores_done", False)

    def load_scores(self, name: str) -> Optional[Dict[str, float]]:
        path = self.run_dir / f"scores_{name}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            log.info("[checkpoint] Loaded scores for %s from checkpoint", name)
            return data
        return None

    def save_scores(self, name: str, scores: Dict[str, float]) -> None:
        path = self.run_dir / f"scores_{name}.json"
        _atomic_write(path, scores)
        self._sys(name).update({
            "scores_done": True,
            "scores_saved_at": datetime.now().isoformat(),
        })
        self._flush_manifest()
        log.info("[checkpoint] ✓ Scores saved: %s → %s", name, path.name)

    # ── Breakdown (final) ─────────────────────────────────────────────────────

    def save_breakdown(self, cat_scores: Dict, lang_scores: Dict, win_matrix: Dict) -> None:
        path = self.run_dir / "breakdown.json"
        _atomic_write(path, {
            "category_scores": cat_scores,
            "language_scores": lang_scores,
            "win_rate_matrix": win_matrix,
            "saved_at": datetime.now().isoformat(),
        })
        self.manifest["breakdown_done"] = True
        self._flush_manifest()
        log.info("[checkpoint] ✓ Breakdown saved → %s", path.name)

    # ── Final summary ─────────────────────────────────────────────────────────

    def save_final_summary(
        self,
        all_results: Dict[str, List[dict]],
        scores_by_baseline: Dict[str, Dict],
        testset: List[dict],
    ) -> None:
        """Write the consolidated raw_results + scores files (end of run)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _atomic_write(
            self.run_dir / f"all_raw_results_{ts}.json",
            {"timestamp": datetime.now().isoformat(), "testset_n": len(testset), "results": all_results},
        )
        _atomic_write(self.run_dir / f"all_scores_{ts}.json", scores_by_baseline)
        self.manifest["completed_at"] = datetime.now().isoformat()
        self._flush_manifest()
        log.info("[checkpoint] ✓ Final summary saved in %s", self.run_dir)

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def new_run(cls, output_dir: Path) -> "CheckpointManager":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(output_dir / f"run_{ts}")

    @classmethod
    def find_latest(cls, output_dir: Path) -> Optional["CheckpointManager"]:
        """Return the most recent run_* directory that is not fully completed."""
        if not output_dir.exists():
            return None
        candidates = sorted(output_dir.glob("run_*"), reverse=True)
        for d in candidates:
            manifest_path = d / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            if "completed_at" not in m:
                log.info("[checkpoint] Found incomplete run: %s", d)
                return cls(d)
        return None

    @classmethod
    def from_path(cls, path: str, output_dir: Path) -> "CheckpointManager":
        p = Path(path)
        if not p.is_absolute():
            p = output_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {p}")
        return cls(p)


def _atomic_write(path: Path, data) -> None:
    """Write JSON to a temp file then rename — avoids corrupt files on crash."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Lazy baseline loader
# ---------------------------------------------------------------------------

def _load_baseline(name: str, ollama_base_url: str, generator_model: str):
    """Import and instantiate a baseline class by name."""
    from benchmarking.shared import OllamaClient

    client = OllamaClient(base_url=ollama_base_url, model=generator_model)
    retrieve_url = f"{V12_API}/api/retrieve"

    module_map = {
        "naive_rag":       ("benchmarking.baselines.naive_rag",       "NaiveRAG"),
        "basic_react":     ("benchmarking.baselines.basic_react",      "BasicReACT"),
        "adaptive_simple": ("benchmarking.baselines.adaptive_simple",  "AdaptiveSimple"),
        "hyde":            ("benchmarking.baselines.hyde",             "HyDE"),
        "self_rag":        ("benchmarking.baselines.self_rag",         "SelfRAG"),
        "flare":           ("benchmarking.baselines.flare",            "FLARE"),
        "crag":            ("benchmarking.baselines.crag",             "CRAG"),
    }
    if name not in module_map:
        raise ValueError(f"Unknown baseline: {name!r}")

    import importlib
    mod_path, cls_name = module_map[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(client=client, retrieve_url=retrieve_url)


# ---------------------------------------------------------------------------
# V12 pipeline runner (via /api/ask)
# ---------------------------------------------------------------------------

def _run_v12(question: str, timeout: int = 900) -> dict:
    t0 = time.time()
    try:
        resp = requests.post(
            f"{V12_API}/api/ask",
            json={"question": question, "mode": "research", "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        data.setdefault("latency_sec", round(time.time() - t0, 2))
        return data
    except requests.exceptions.ConnectionError:
        return {
            "answer": "[ERROR: API not running — start `python -m api.app`]",
            "latency_sec": round(time.time() - t0, 2),
            "sources": [], "contexts": [], "ragas_contexts": [],
        }
    except Exception as exc:
        return {
            "answer": f"[ERROR: {exc}]",
            "latency_sec": round(time.time() - t0, 2),
            "sources": [], "contexts": [], "ragas_contexts": [],
        }


# ---------------------------------------------------------------------------
# Check API availability
# ---------------------------------------------------------------------------

def _check_api() -> bool:
    try:
        r = requests.get(f"{V12_API}/api/status", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Run all baselines  — saves each baseline's results immediately on completion
# ---------------------------------------------------------------------------

def run_baselines(
    baselines: List[str],
    testset: List[dict],
    ollama_base_url: str,
    generator_model: str,
    ckpt: CheckpointManager,
) -> Dict[str, List[dict]]:
    """
    Returns {baseline_name: [result_dict, ...]} aligned with testset order.

    Checkpoint behaviour:
      - If a baseline's results are already in the checkpoint, load and skip it.
      - After every baseline finishes all questions, save to checkpoint immediately.
    """
    results: Dict[str, List[dict]] = {}

    for b_name in baselines:
        # ── Resume: skip if already done ─────────────────────────────────────
        if ckpt.has_results(b_name):
            cached = ckpt.load_results(b_name)
            if cached is not None:
                log.info("[checkpoint] Skipping %s — already completed (%d results)", b_name, len(cached))
                results[b_name] = cached
                continue

        # ── Load baseline ─────────────────────────────────────────────────────
        log.info("Loading baseline: %s", b_name)
        try:
            baseline = _load_baseline(b_name, ollama_base_url, generator_model)
        except Exception as exc:
            log.error("Failed to load baseline %s: %s", b_name, exc)
            error_results = [
                {"answer": f"[LOAD ERROR: {exc}]", "question": item["question"],
                 "contexts": [], "latency_sec": 0.0, "is_outscope": False,
                 "baseline_name": b_name, "metadata": {"error": str(exc)}}
                for item in testset
            ]
            ckpt.save_results(b_name, error_results)
            results[b_name] = error_results
            continue

        # ── Run all questions ─────────────────────────────────────────────────
        log.info("Running %s on %d questions...", b_name, len(testset))
        b_results: List[dict] = []
        for idx, item in enumerate(testset, 1):
            q = item["question"]
            log.info("[%d/%d] %s | %s...", idx, len(testset), b_name, q[:60])
            try:
                br = baseline.run(q)
                result = {
                    "question": q,
                    "answer": br.answer,
                    "contexts": br.contexts,
                    "latency_sec": br.latency_sec,
                    "is_outscope": br.is_outscope,
                    "baseline_name": br.baseline_name,
                    "metadata": br.metadata,
                }
            except Exception as exc:
                log.warning("Error in %s on q%d: %s", b_name, idx, exc)
                result = {
                    "question": q,
                    "answer": f"[ERROR: {exc}]",
                    "contexts": [],
                    "latency_sec": 0.0,
                    "is_outscope": False,
                    "baseline_name": b_name,
                    "metadata": {"error": str(exc)},
                }
            b_results.append(result)

        # ── Save immediately after this baseline finishes ─────────────────────
        ckpt.save_results(b_name, b_results)
        results[b_name] = b_results

    return results


# ---------------------------------------------------------------------------
# V12 pipeline — saves after EACH question (most granular checkpoint)
# ---------------------------------------------------------------------------

def run_v12_pipeline(
    testset: List[dict],
    ckpt: CheckpointManager,
) -> List[dict]:
    """
    Call /api/ask for every test item.

    Checkpoint behaviour:
      - Loads any partially completed v12 results from checkpoint.
      - Saves to checkpoint after EACH question so a crash loses at most 1 answer.
      - If all questions are already done (has_results), loads and returns immediately.
    """
    # Full resume: all questions already done
    if ckpt.has_results("v12_pipeline"):
        cached = ckpt.load_results("v12_pipeline")
        if cached is not None and len(cached) == len(testset):
            log.info("[checkpoint] Skipping v12_pipeline — already completed (%d results)", len(cached))
            return cached

    # Partial resume: some questions already answered
    v12_results = ckpt.load_partial_v12()
    already_done = len(v12_results)
    remaining = testset[already_done:]

    if already_done:
        log.info("[checkpoint] Resuming v12_pipeline from question %d / %d", already_done + 1, len(testset))

    n = len(testset)
    for idx, item in enumerate(remaining, start=already_done + 1):
        q = item["question"]
        log.info("[%d/%d] v12_pipeline | %s...", idx, n, q[:60])
        result = _run_v12(q)
        result.setdefault("question", q)
        v12_results.append(result)
        # Save after every single question
        ckpt.save_partial_v12(v12_results)

    # Mark as fully done
    ckpt.save_results("v12_pipeline", v12_results)
    return v12_results


# ---------------------------------------------------------------------------
# Compute all scores — saves each baseline's scores immediately
# ---------------------------------------------------------------------------

def compute_all_scores(
    all_results: Dict[str, List[dict]],
    testset: List[dict],
    run_ragas: bool,
    run_ares: bool,
    with_gt: bool,
    ollama_url: str,
    model: str,
    ckpt: CheckpointManager,
    judge_url: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-baseline scores dict.
    Keys: baseline name → {metric: score}.

    Checkpoint behaviour:
      - If a baseline's scores are already in the checkpoint, load and skip scoring.
      - After each baseline is scored, save scores to checkpoint immediately.

    Metric tiers:
      ALWAYS ON  — efficiency stats, RAGAS core (faithfulness/answer_relevancy/
                   context_precision), ARES judge + extended, G-Eval, FActScore,
                   retrieval quality, domain precision, V12-specific
      --with-gt  — lexical, BERTScore, RAGAS extended (v2), keyword_hit_rate,
                   abstain_accuracy, RGB robustness, cross-lingual consistency
    """
    # Judge endpoint — falls back to generation endpoint if not specified separately
    _judge_url   = judge_url   or ollama_url
    _judge_model = judge_model or model
    if _judge_url != ollama_url:
        log.info("[judge] Using separate judge endpoint: %s  model=%s", _judge_url, _judge_model)

    from benchmarking.adapters.ragas_adapter import build_ragas_dataset
    from benchmarking.metrics import (
        aggregate_baseline_results,
        compute_ares_scores,
        compute_ares_extended,
        compute_ragas_scores,
        compute_ragas_extended,
        compute_geval_scores,
        compute_factscore,
        compute_rgb_robustness,
        compute_retrieval_quality,
        compute_domain_precision,
        compute_cross_lingual_consistency,
        compute_v12_specific,
    )

    scores_by_baseline: Dict[str, Dict[str, float]] = {}

    for b_name, results in all_results.items():
        # ── Resume: skip if scores already computed ───────────────────────────
        if ckpt.has_scores(b_name):
            cached = ckpt.load_scores(b_name)
            if cached is not None:
                log.info("[checkpoint] Skipping scoring for %s — already done", b_name)
                scores_by_baseline[b_name] = cached
                continue

        log.info("Computing scores for: %s", b_name)
        b_scores: Dict[str, float] = {}

        # ── Efficiency / timing stats (always, no ground truth) ───────────────
        try:
            class _R:
                def __init__(self, d):
                    self.latency_sec = d.get("latency_sec", 0.0)
                    self.is_outscope = d.get("is_outscope", False)
                    self.answer = d.get("answer", "")
                    self.contexts = d.get("contexts", []) or []
            agg = aggregate_baseline_results([_R(r) for r in results])
            b_scores.update(agg)
        except Exception as exc:
            log.warning("Aggregate stats failed for %s: %s", b_name, exc)

        # ── RAGAS reference-free core ─────────────────────────────────────────
        if run_ragas:
            try:
                from ragas.metrics import answer_relevancy, context_precision, faithfulness
                ds = build_ragas_dataset(results, testset)
                ragas = compute_ragas_scores(
                    ds,
                    metrics=[faithfulness, answer_relevancy, context_precision],
                    extended=False,
                )
                b_scores.update(ragas)
            except Exception as exc:
                log.warning("RAGAS eval failed for %s: %s", b_name, exc)

        # ── ARES-style LLM judge (reference-free) ─────────────────────────────
        if run_ares:
            try:
                log.info("[ARES] Running LLM judge for %s…", b_name)
                ares = compute_ares_scores(
                    results, testset,
                    ollama_base_url=_judge_url,
                    model=_judge_model,
                    max_contexts=3,
                )
                b_scores.update(ares)
            except Exception as exc:
                log.warning("ARES eval failed for %s: %s", b_name, exc)

            # ARES-extended: 5 domain-specific dimensions (Moroccan RAG)
            try:
                log.info("[ARES-ext] Running extended ARES for %s…", b_name)
                b_scores.update(compute_ares_extended(
                    results, testset,
                    ollama_base_url=_judge_url, model=_judge_model,
                ))
            except Exception as exc:
                log.warning("ARES-extended failed for %s: %s", b_name, exc)

            # G-Eval: CoT LLM judge — coherence, consistency, fluency, relevance
            try:
                log.info("[G-Eval] Running G-Eval (Wang 2023) for %s…", b_name)
                b_scores.update(compute_geval_scores(
                    results, testset,
                    ollama_base_url=_judge_url, model=_judge_model,
                ))
            except Exception as exc:
                log.warning("G-Eval failed for %s: %s", b_name, exc)

            # FActScore: atomic claim decomposition + context verification
            try:
                log.info("[FActScore] Running FActScore (Min 2023) for %s…", b_name)
                b_scores.update(compute_factscore(
                    results, testset,
                    ollama_base_url=_judge_url, model=_judge_model,
                ))
            except Exception as exc:
                log.warning("FActScore failed for %s: %s", b_name, exc)

        # ── V12-specific (from audit_trail — reference-free) ──────────────────
        if b_name == "v12_pipeline":
            try:
                v12 = compute_v12_specific(results)
                b_scores.update(v12)
            except Exception as exc:
                log.warning("V12-specific metrics failed: %s", exc)

        # ── Reference-free SOTA metrics (no LLM, no ground truth needed) ──────
        # Retrieval quality: coverage, redundancy, MRR, utilization
        try:
            b_scores.update(compute_retrieval_quality(results, testset))
        except Exception as exc:
            log.warning("Retrieval quality metrics failed for %s: %s", b_name, exc)

        # Domain precision: legal citations, costs, dialect match, hallucinated numbers
        try:
            b_scores.update(compute_domain_precision(results, testset))
        except Exception as exc:
            log.warning("Domain precision metrics failed for %s: %s", b_name, exc)

        # ══════════════════════════════════════════════════════════════════════
        # GROUND-TRUTH TIER — disabled until solid dataset is ready.
        # Enable with --with-gt once gold_answer / gold_keywords are in place.
        # ══════════════════════════════════════════════════════════════════════
        if with_gt:
            from benchmarking.metrics import (
                compute_abstain_accuracy,
                compute_arabizi_normalized,
                compute_bertscore,
                compute_keyword_hit_rate,
                compute_lexical_scores,
            )

            try:
                b_scores.update(compute_lexical_scores(results, testset))
            except Exception as exc:
                log.warning("[GT] Lexical scores failed for %s: %s", b_name, exc)

            try:
                b_scores.update(compute_bertscore(results, testset))
            except Exception as exc:
                log.warning("[GT] BERTScore failed for %s: %s", b_name, exc)

            try:
                b_scores.update(compute_arabizi_normalized(results, testset))
            except Exception as exc:
                log.warning("[GT] Arabizi-normalized F1 failed for %s: %s", b_name, exc)

            if run_ragas:
                try:
                    from ragas.metrics import (
                        answer_relevancy, context_precision,
                        context_recall, faithfulness,
                    )
                    ds = build_ragas_dataset(results, testset)
                    b_scores.update(compute_ragas_scores(
                        ds,
                        metrics=[faithfulness, answer_relevancy,
                                 context_precision, context_recall],
                        extended=True,
                    ))
                except Exception as exc:
                    log.warning("[GT] RAGAS extended failed for %s: %s", b_name, exc)

            try:
                b_scores.update(compute_keyword_hit_rate(results, testset))
            except Exception as exc:
                log.warning("[GT] Keyword hit rate failed for %s: %s", b_name, exc)

            try:
                b_scores.update(compute_abstain_accuracy(results, testset))
            except Exception as exc:
                log.warning("[GT] Abstain accuracy failed for %s: %s", b_name, exc)

            # RGB robustness: noise robustness, negative rejection, integration, counterfactual
            try:
                b_scores.update(compute_rgb_robustness(results, testset))
            except Exception as exc:
                log.warning("[GT] RGB robustness (Chen 2023) failed for %s: %s", b_name, exc)

            # Cross-lingual consistency: Arabic vs French answer quality on same URL
            try:
                b_scores.update(compute_cross_lingual_consistency(results, testset))
            except Exception as exc:
                log.warning("[GT] Cross-lingual consistency failed for %s: %s", b_name, exc)

            # RAGAS-extended v2: context_entity_recall, noise_sensitivity, factual_correctness
            if run_ragas:
                try:
                    from benchmarking.adapters.ragas_adapter import build_ragas_dataset as _build_ragas
                    b_scores.update(compute_ragas_extended(_build_ragas(results, testset)))
                except Exception as exc:
                    log.warning("[GT] RAGAS-extended (v2) failed for %s: %s", b_name, exc)

        # ── Save scores immediately after this baseline is scored ─────────────
        ckpt.save_scores(b_name, b_scores)
        scores_by_baseline[b_name] = b_scores

    return scores_by_baseline


# ---------------------------------------------------------------------------
# Per-category / language breakdowns + win-rate matrix
# ---------------------------------------------------------------------------

def compute_all_breakdowns(
    all_results: Dict[str, List[dict]],
    testset: List[dict],
) -> Tuple[Dict, Dict, Dict]:
    """
    NOTE: Uses token_f1 and keyword_hit_rate — requires gold_answer / gold_keywords.
    Only call when --with-gt is active.
    """
    from benchmarking.metrics import (
        per_category_scores,
        per_language_scores,
        win_rate_matrix,
    )

    cat_scores: Dict[str, Dict] = {}
    lang_scores: Dict[str, Dict] = {}

    for b_name, results in all_results.items():
        try:
            cat_scores[b_name] = per_category_scores(results, testset)
        except Exception as exc:
            log.warning("Category breakdown failed for %s: %s", b_name, exc)
            cat_scores[b_name] = {}
        try:
            lang_scores[b_name] = per_language_scores(results, testset)
        except Exception as exc:
            log.warning("Language breakdown failed for %s: %s", b_name, exc)
            lang_scores[b_name] = {}

    try:
        matrix = win_rate_matrix(all_results, testset, metric="token_f1")
    except Exception as exc:
        log.warning("Win-rate matrix failed: %s", exc)
        matrix = {}

    return cat_scores, lang_scores, matrix


# ---------------------------------------------------------------------------
# Save RAGAS rows + ARES input files (adapter outputs)
# ---------------------------------------------------------------------------

def _save_adapter_outputs(
    all_results: Dict[str, List[dict]],
    testset: List[dict],
    run_dir: Path,
) -> None:
    from benchmarking.adapters.ares_adapter import build_ares_input, save_ares_input
    from benchmarking.adapters.ragas_adapter import build_ragas_rows, save_ragas_rows

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for b_name, results in all_results.items():
        safe = b_name.replace(" ", "_")
        try:
            ragas_rows = build_ragas_rows(results, testset)
            save_ragas_rows(ragas_rows, str(run_dir / f"ragas_{safe}_{ts}.json"))
        except Exception as exc:
            log.warning("Could not save RAGAS rows for %s: %s", b_name, exc)
        try:
            ares_rows = build_ares_input(results, testset)
            save_ares_input(ares_rows, str(run_dir / f"ares_{safe}_{ts}.json"))
        except Exception as exc:
            log.warning("Could not save ARES input for %s: %s", b_name, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Moroccan RAG Benchmark Runner")
    parser.add_argument("--quick", action="store_true", help="Run only first 10 questions")
    parser.add_argument("--no-v12", action="store_true", help="Skip V12 pipeline (no API needed)")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS evaluation (much faster)")
    parser.add_argument("--no-ares", action="store_true", help="Skip all LLM-judge metrics: ARES core+extended, G-Eval, FActScore (saves many Ollama calls)")
    parser.add_argument(
        "--with-gt",
        action="store_true",
        help=(
            "Enable ground-truth metrics (requires gold_answer + gold_keywords): "
            "exact_match, token_f1, rouge_l, BERTScore, context_recall, "
            "answer_correctness, keyword_hit_rate, abstain_accuracy, breakdowns"
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="RUN_DIR",
        help=(
            "Resume an interrupted run. "
            "With no argument: auto-detects the latest incomplete run. "
            "With a path: resumes that specific run directory."
        ),
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help=f"Comma-separated list of baselines to run. Available: {','.join(ALL_BASELINES)}",
    )
    parser.add_argument("--testset", default=None, help="Path to gold testset JSON")
    parser.add_argument("--output", default=str(RESULTS_DIR), help="Output directory for run folders")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1", help="Ollama base URL for answer generation")
    parser.add_argument("--model", default="gemma4:e4b", help="Generator model name for answer generation")
    parser.add_argument(
        "--judge-url",
        default=None,
        help=(
            "LLM judge endpoint for ARES/G-Eval/FActScore. "
            "Defaults to --ollama-url (local). "
            "Use https://api.openai.com/v1 + OPENAI_API_KEY env var for GPT-4o mini."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "LLM judge model. Defaults to --model. "
            "Examples:  gpt-4o-mini | gemini-2.0-flash | gemini-1.5-flash | llama-3.3-70b-versatile"
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        metavar="KEY",
        help=(
            "API key for the judge endpoint. "
            "If omitted, auto-detected from env vars: "
            "JUDGE_API_KEY > GEMINI_API_KEY > OPENAI_API_KEY > GROQ_API_KEY. "
            "Not needed for local Ollama."
        ),
    )
    args = parser.parse_args()

    # Judge URL/model fall back to generation URL/model if not specified
    args.judge_url   = args.judge_url   or args.ollama_url
    args.judge_model = args.judge_model or args.model

    # Inject explicit api key into environment so _openai_client picks it up
    if args.judge_api_key:
        os.environ["JUDGE_API_KEY"] = args.judge_api_key

    output_dir = Path(args.output)

    # ── Load testset ──────────────────────────────────────────────────────────
    ts_path = Path(args.testset) if args.testset else TESTSET_FILE
    if not ts_path.exists():
        log.error("Testset not found: %s", ts_path)
        sys.exit(1)
    with open(ts_path, encoding="utf-8") as f:
        testset: List[dict] = json.load(f)
    if args.quick:
        testset = testset[:10]

    # ── Checkpoint: new run or resume ─────────────────────────────────────────
    if args.resume:
        if args.resume == "auto":
            ckpt = CheckpointManager.find_latest(output_dir)
            if ckpt is None:
                log.warning("No incomplete run found — starting a new run.")
                ckpt = CheckpointManager.new_run(output_dir)
        else:
            ckpt = CheckpointManager.from_path(args.resume, output_dir)
        log.info("[checkpoint] Resuming run: %s", ckpt.run_dir)
    else:
        ckpt = CheckpointManager.new_run(output_dir)

    # ── Baselines ─────────────────────────────────────────────────────────────
    if args.baselines:
        baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
        unknown = [b for b in baselines if b not in ALL_BASELINES]
        if unknown:
            log.error("Unknown baseline(s): %s", unknown)
            sys.exit(1)
    else:
        baselines = list(ALL_BASELINES)

    # ── V12 API check ─────────────────────────────────────────────────────────
    include_v12 = not args.no_v12
    if include_v12:
        if ckpt.has_results("v12_pipeline"):
            log.info("V12 already completed in checkpoint — will load from disk.")
        else:
            log.info("Checking V12 API at %s...", V12_API)
            if _check_api():
                log.info("API is up")
            else:
                log.warning("API not reachable — V12 pipeline will be skipped.")
                include_v12 = False

    run_ragas = not args.no_ragas
    run_ares  = not args.no_ares
    with_gt   = args.with_gt

    log.info("=" * 70)
    log.info("MOROCCAN RAG BENCHMARK  (v12 — SOTA metrics enabled)")
    log.info("Run dir      : %s", ckpt.run_dir)
    log.info("Questions    : %d", len(testset))
    log.info("Baselines    : %s", baselines)
    log.info("V12          : %s", "yes" if include_v12 else "no")
    log.info("RAGAS        : %s", "yes (faithfulness/relevancy/precision)" if run_ragas else "no (--no-ragas)")
    log.info("LLM judges   : %s", "ARES+ext, G-Eval, FActScore" if run_ares else "disabled (--no-ares, faster)")
    log.info("Ref-free SOTA: retrieval_quality + domain_precision (always on)")
    log.info("Ground-truth : %s", "ENABLED — RGB, cross-lingual, RAGAS-v2, lexical" if with_gt else "disabled (add --with-gt)")
    log.info("=" * 70)

    t_start = time.time()
    all_results: Dict[str, List[dict]] = {}

    # ── Run baselines (checkpoint-aware) ──────────────────────────────────────
    if baselines:
        all_results.update(
            run_baselines(baselines, testset, args.ollama_url, args.model, ckpt)
        )

    # ── Run V12 (checkpoint-aware, per-question save) ─────────────────────────
    if include_v12:
        all_results["v12_pipeline"] = run_v12_pipeline(testset, ckpt)

    if not all_results:
        log.error("No pipelines ran — nothing to report.")
        sys.exit(1)

    log.info("All pipelines done in %.0fs", time.time() - t_start)

    # ── Compute scores (checkpoint-aware) ─────────────────────────────────────
    scores_by_baseline = compute_all_scores(
        all_results,
        testset,
        run_ragas=run_ragas,
        run_ares=run_ares,
        with_gt=with_gt,
        ollama_url=args.ollama_url,
        model=args.model,
        ckpt=ckpt,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
    )

    # ── Print comparison table ─────────────────────────────────────────────────
    from benchmarking.metrics import (
        print_category_table,
        print_comparison_table,
        print_language_table,
        print_win_rate_matrix,
    )
    print_comparison_table(scores_by_baseline)

    # ── Per-category / language breakdowns + win-rate matrix ──────────────────
    if with_gt:
        cat_scores, lang_scores, win_matrix = compute_all_breakdowns(all_results, testset)
        print_category_table(cat_scores)
        print_language_table(lang_scores)
        if win_matrix:
            print_win_rate_matrix(win_matrix)
        ckpt.save_breakdown(cat_scores, lang_scores, win_matrix)
    else:
        log.info("Skipping breakdowns / win-rate matrix (add --with-gt to enable)")

    # ── Final consolidated save + adapter outputs ──────────────────────────────
    ckpt.save_final_summary(all_results, scores_by_baseline, testset)
    _save_adapter_outputs(all_results, testset, ckpt.run_dir)

    log.info("Done. Results in: %s", ckpt.run_dir)


if __name__ == "__main__":
    main()
