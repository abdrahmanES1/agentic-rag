# -*- coding: utf-8 -*-
"""
RAGAS adapter — builds ragas.Dataset from pipeline or baseline results.

RAGAS requires four fields per row:
  question     : str
  answer       : str
  contexts     : List[str]    ← retrieved chunk texts
  ground_truth : str          ← gold answer from test set

Usage:
    from benchmarking.adapters.ragas_adapter import build_ragas_dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    dataset = build_ragas_dataset(pipeline_responses, gold_items)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

Pipeline response format (from /api/ask?mode=research):
    {
        "answer": str,
        "ragas_contexts": List[str],   ← merged initial + agentic contexts
        "retrieval": {"contexts": List[str]},
        "execution_trace": {"all_contexts": List[str]},
        ...
    }

Baseline result format (BaselineResult.to_ragas_row()):
    {
        "question": str,
        "answer": str,
        "contexts": List[str],
        "ground_truth": str,
    }

RAGAS faithfulness root-cause fixes (applied here):
  1. Citation stripping: v12 answers contain [Source: ...] tags. RAGAS's LLM
     claim extractor sees these as part of the claim text → a claim like
     "نسخة من البطاقة [Source: CIN.pdf]" can never be found in any context →
     NLI marks it unsupported → faithfulness is artificially lowered. Citations
     are 30% of v12's answer tokens. Strip them before claim extraction.
  2. Context deduplication + reranker sorting: ragas_contexts = initial 5
     (ranked by reranker) + execution_trace N (execution order, not ranked).
     Execution_trace may add duplicate chunks and unranked low-quality contexts.
     Deduplicate and sort all contexts by reranker score so RAGAS evaluates
     the most relevant evidence first (and identical chunks are not double-counted).
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("benchmarking")

_CIT_RE = re.compile(r"\[Source:[^\]]*\]", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")


def _strip_answer(text: str) -> str:
    """
    Strip [Source:...] citation tags and bare URLs from an answer before RAGAS
    claim extraction. These tags are 30% of v12's answer tokens and contaminate
    every extracted claim — the claim 'X [Source: file.pdf]' can never entail
    from any context chunk → artificial faithfulness undercount.
    """
    text = _CIT_RE.sub("", text or "")
    text = _URL_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _rank_contexts(result: Dict, contexts: List[str]) -> List[str]:
    """
    Sort contexts by reranker score descending (most relevant first) and
    deduplicate. For v12 multi-hop results, ragas_contexts = initial 5
    (reranker-sorted) + execution_trace N (execution order). After merging,
    re-ranking ensures RAGAS evaluates the highest-quality evidence first and
    identical chunks retrieved in multiple hops are not double-counted.
    Falls back to original order if no scores are available.
    """
    scores_dict = (result.get("retrieval") or {}).get("scores") or {}
    reranker_scores = scores_dict.get("reranker") or []

    # Build a score lookup: first 5 contexts (initial retrieval) have known scores
    score_map: Dict[str, float] = {}
    initial_ctxs = (
        (result.get("retrieval") or {}).get("contexts")
        or (result.get("retrieval") or {}).get("contexts")
        or []
    )
    for i, (ctx, sc) in enumerate(zip(initial_ctxs, reranker_scores)):
        if ctx:
            score_map[ctx.strip()] = float(sc)

    # Deduplicate (preserve first occurrence order for unseen contexts)
    seen: set = set()
    unique: List[str] = []
    for ctx in contexts:
        key = ctx.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(ctx)

    # Sort by reranker score; contexts not in score_map get 0.0 (end of list)
    unique.sort(key=lambda c: score_map.get(c.strip(), 0.0), reverse=True)
    return unique


def build_ragas_dataset(
    results: List[Dict],
    gold_items: List[Dict],
    gold_answer_key: str = "gold_answer",
    question_key: str = "question",
):
    """
    Build a ragas.Dataset from pipeline responses and gold test items.

    Parameters
    ----------
    results : List[Dict]
        API responses from /api/ask (or baseline result dicts).
        Must align positionally with gold_items.
    gold_items : List[Dict]
        Items from benchmark_testset_gold.json.
    gold_answer_key : str
        Key in gold_items for the reference answer.
    question_key : str
        Key in gold_items for the question text.

    Returns
    -------
    datasets.Dataset (RAGAS-compatible)
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("pip install datasets ragas")

    rows = []
    for result, gold in zip(results, gold_items):
        question    = gold.get(question_key, result.get("question", ""))
        answer_raw  = result.get("answer", "")
        ground_truth = gold.get(gold_answer_key, "")

        # Fix 1: strip citation tags from answer before RAGAS claim extraction
        answer = _strip_answer(answer_raw)

        # Fix 2: extract, deduplicate, and reranker-sort contexts
        contexts_raw = _extract_contexts(result)
        contexts     = _rank_contexts(result, contexts_raw)

        rows.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })

    log.info(f"[RAGAS adapter] Built dataset: {len(rows)} rows "
             f"(citations stripped, contexts deduped+ranked)")
    return Dataset.from_list(rows)


def _extract_contexts(result: Dict) -> List[str]:
    """Extract the best available List[str] contexts from a result dict."""
    # Best: ragas_contexts = merged initial + all agentic tool-call contexts
    if result.get("ragas_contexts"):
        return [c for c in result["ragas_contexts"] if c]

    # Next: execution_trace.all_contexts (multi-hop)
    if result.get("execution_trace", {}).get("all_contexts"):
        return [c for c in result["execution_trace"]["all_contexts"] if c]

    # Fallback: retrieval.contexts (initial only)
    if result.get("retrieval", {}).get("contexts"):
        return [c for c in result["retrieval"]["contexts"] if c]

    # Last resort: baseline "contexts" key
    return [c for c in result.get("contexts", []) if c]


def save_ragas_rows(rows: List[Dict], output_path: str) -> None:
    """Save RAGAS-format rows to a JSON file for inspection or offline eval."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log.info(f"[RAGAS adapter] Saved {len(rows)} rows to {output_path}")


def build_ragas_rows(
    results: List[Dict],
    gold_items: List[Dict],
    gold_answer_key: str = "gold_answer",
) -> List[Dict]:
    """Return raw dicts (no datasets dependency) for inspection."""
    return [
        {
            "question":     gold.get("question", result.get("question", "")),
            "answer":       _strip_answer(result.get("answer", "")),
            "contexts":     _rank_contexts(result, _extract_contexts(result)),
            "ground_truth": gold.get(gold_answer_key, ""),
        }
        for result, gold in zip(results, gold_items)
    ]
