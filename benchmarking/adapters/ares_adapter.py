# -*- coding: utf-8 -*-
"""
ARES adapter — builds ARES evaluation input from pipeline or baseline results.

ARES requires one row per (question, document) pair:
  query    : str   ← the question
  document : str   ← one retrieved chunk text
  answer   : str   ← generated answer
  label    : str   ← gold answer (used by ARES UES/IDP scoring)

ARES scores three dimensions:
  - Context Relevance:   Is the document relevant to the query?
  - Answer Faithfulness: Is the answer grounded in the document?
  - Answer Relevance:    Does the answer address the query?

Usage:
    from benchmarking.adapters.ares_adapter import build_ares_input, save_ares_input
    rows = build_ares_input(pipeline_responses, gold_items)
    save_ares_input(rows, "benchmarking/ares_eval_input.json")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("benchmarking")


def build_ares_input(
    results: List[Dict],
    gold_items: List[Dict],
    gold_answer_key: str = "gold_answer",
    question_key: str = "question",
    max_contexts_per_question: int = 5,
) -> List[Dict]:
    """
    Build ARES evaluation input rows from pipeline responses and gold items.

    Parameters
    ----------
    results : List[Dict]
        API responses from /api/ask or baseline result dicts.
    gold_items : List[Dict]
        Items from benchmark_testset_gold.json.
    max_contexts_per_question : int
        Cap how many contexts are included per question (ARES can be slow).

    Returns
    -------
    List of dicts with keys: query, document, answer, label
    """
    rows = []
    for result, gold in zip(results, gold_items):
        question = gold.get(question_key, result.get("question", ""))
        answer = result.get("answer", "")
        label = gold.get(gold_answer_key, "")
        contexts = _extract_contexts(result)[:max_contexts_per_question]

        if not contexts:
            rows.append({
                "query": question,
                "document": "",
                "answer": answer,
                "label": label,
            })
        else:
            for ctx in contexts:
                rows.append({
                    "query": question,
                    "document": ctx,
                    "answer": answer,
                    "label": label,
                })

    log.info(f"[ARES adapter] Built {len(rows)} rows from {len(results)} results")
    return rows


def _extract_contexts(result: Dict) -> List[str]:
    if result.get("ragas_contexts"):
        return [c for c in result["ragas_contexts"] if c]
    if result.get("execution_trace", {}).get("all_contexts"):
        return [c for c in result["execution_trace"]["all_contexts"] if c]
    if result.get("retrieval", {}).get("contexts"):
        return [c for c in result["retrieval"]["contexts"] if c]
    return [c for c in result.get("contexts", []) if c]


def save_ares_input(rows: List[Dict], output_path: str) -> None:
    """Save ARES input rows to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log.info(f"[ARES adapter] Saved {len(rows)} rows to {output_path}")


def build_ares_tsv(rows: List[Dict], output_path: str) -> None:
    """Save ARES input as TSV (alternative format some ARES versions expect)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query\tdocument\tanswer\tlabel\n")
        for row in rows:
            query = row["query"].replace("\t", " ").replace("\n", " ")
            doc = row["document"].replace("\t", " ").replace("\n", " ")
            answer = row["answer"].replace("\t", " ").replace("\n", " ")
            label = str(row["label"]).replace("\t", " ").replace("\n", " ")
            f.write(f"{query}\t{doc}\t{answer}\t{label}\n")
    log.info(f"[ARES adapter] Saved TSV {len(rows)} rows to {output_path}")
