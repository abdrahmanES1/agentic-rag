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
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("benchmarking")


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
        question = gold.get(question_key, result.get("question", ""))
        answer = result.get("answer", "")
        ground_truth = gold.get(gold_answer_key, "")

        # Priority: ragas_contexts (merged initial + agentic) > retrieval.contexts > execution_trace
        contexts = _extract_contexts(result)

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    log.info(f"[RAGAS adapter] Built dataset: {len(rows)} rows")
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


def build_ragas_rows(results: List[Dict], gold_items: List[Dict], gold_answer_key: str = "gold_answer") -> List[Dict]:
    """Return raw dicts (no datasets dependency) for inspection."""
    return [
        {
            "question": gold.get("question", result.get("question", "")),
            "answer": result.get("answer", ""),
            "contexts": _extract_contexts(result),
            "ground_truth": gold.get(gold_answer_key, ""),
        }
        for result, gold in zip(results, gold_items)
    ]
