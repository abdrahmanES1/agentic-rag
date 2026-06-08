# -*- coding: utf-8 -*-
"""
Validate the fully-agentic classification (fix/fully-agentic-classification
branch) BEFORE committing to a 4-hour benchmark rerun.

Runs the new _llm_classify on all 124 testset items, compares the LLM's
signals against the expected category labels, and reports accuracy.

Decision rule (printed at the end):
  - All categories ≥ 80% → GO (run the full benchmark)
  - Any category < 70%   → ABORT (LLM not accurate enough; keep main branch)

Cost: ~3-5 minutes, zero $ (local LLM only).

Prerequisites:
  - Ollama running on localhost:11434 with gemma4:e4b
  - Be on branch: fix/fully-agentic-classification

Usage:  python validate_agentic_classification.py
"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.CRITICAL)
import json, time
from collections import Counter

from pipeline.language import _llm_classify
from pipeline.generation import OllamaClient

ts = json.load(open("benchmarking/benchmark_testset_v1.0.json", encoding="utf-8"))
items = ts["items"] if isinstance(ts, dict) and "items" in ts else ts
ollama = OllamaClient()

print(f"Validating agentic classification on {len(items)} items...\n")
t0 = time.time()

# Tally: for each (expected_category, signal) → counter
results = {
    "SIMPLE":   Counter(),
    "DARIJA":   Counter(),
    "ARABIZI":  Counter(),
    "MULTIHOP": Counter(),
    "LEGAL":    Counter(),
    "OUTSCOPE": Counter(),
}
failed_classifications = []

for i, it in enumerate(items):
    cat = it.get("category", "?")
    q = it.get("question", "")
    if (i + 1) % 20 == 0:
        print(f"  ...{i+1}/{len(items)} done")

    signals = _llm_classify(q, ollama)
    if signals is None:
        results[cat]["LLM_FAILED"] += 1
        failed_classifications.append((i, cat, q[:50]))
        continue

    needs_mh = signals["needs_multihop"]
    is_legal = signals["is_legal"]
    is_outscope = signals["is_outscope"]

    # Per category, what's correct?
    if cat == "SIMPLE":
        results[cat]["simple_correct"] += int(not needs_mh and not is_outscope)
    elif cat in ("DARIJA", "ARABIZI"):
        # These should NOT be multihop (single procedure, multiple aspects)
        results[cat]["simple_correct"] += int(not needs_mh)
    elif cat == "MULTIHOP":
        results[cat]["multihop_correct"] += int(needs_mh)
    elif cat == "LEGAL":
        results[cat]["legal_correct"] += int(is_legal)
    elif cat == "OUTSCOPE":
        # OUTSCOPE items should have is_outscope=True
        # The testset's "should_abstain" matches OUTSCOPE category
        results[cat]["outscope_correct"] += int(is_outscope)

    results[cat]["total"] += 1

# Report
print(f"\nClassification complete in {time.time()-t0:.0f}s")
print()
print("=" * 70)
print(" PER-CATEGORY ACCURACY")
print("=" * 70)

accuracies = {}
for cat in ["SIMPLE", "DARIJA", "ARABIZI", "MULTIHOP", "LEGAL", "OUTSCOPE"]:
    r = results[cat]
    total = r.get("total", 0)
    if total == 0: continue

    if cat == "SIMPLE":
        correct = r.get("simple_correct", 0)
        label = "routed simple"
    elif cat in ("DARIJA", "ARABIZI"):
        correct = r.get("simple_correct", 0)
        label = "NOT routed multihop (one procedure, multi-aspect)"
    elif cat == "MULTIHOP":
        correct = r.get("multihop_correct", 0)
        label = "routed multihop"
    elif cat == "LEGAL":
        correct = r.get("legal_correct", 0)
        label = "flagged is_legal"
    elif cat == "OUTSCOPE":
        correct = r.get("outscope_correct", 0)
        label = "flagged is_outscope"

    pct = correct / total * 100
    accuracies[cat] = pct
    mark = "✓ OK" if pct >= 80 else ("⚠ MARGINAL" if pct >= 70 else "✗ FAIL")
    failed = r.get("LLM_FAILED", 0)
    print(f"  {cat:10s} {correct:>2d}/{total:<3d} {pct:>3.0f}%  {label}   [{mark}]" +
          (f"  ({failed} LLM_failed)" if failed else ""))

# Failed classifications
total_failed = sum(r.get("LLM_FAILED", 0) for r in results.values())
if total_failed:
    print(f"\n  ⚠ {total_failed} items where the LLM call failed entirely")
    for i, cat, q in failed_classifications[:5]:
        print(f"     [{i}] cat={cat} Q: {q}")

# Decision
print()
print("=" * 70)
print(" RECOMMENDATION")
print("=" * 70)
min_acc = min(accuracies.values()) if accuracies else 0
critical_cats = ["MULTIHOP", "OUTSCOPE"]   # the cats where wrong = catastrophic
critical_min = min(accuracies.get(c, 0) for c in critical_cats)

if min_acc >= 80 and critical_min >= 80:
    print(f"  ✅ GO — all categories ≥80% accurate (min={min_acc:.0f}%)")
    print(f"  Run the full benchmark:")
    print(f"    git checkout fix/fully-agentic-classification")
    print(f"    export OPENROUTER_API_KEY=...")
    print(f"    python benchmarking/benchmark_runner.py --baselines none --with-gt \\")
    print(f"      --judge-url https://openrouter.ai/api/v1 --judge-model openai/gpt-4o-mini \\")
    print(f"      --testset benchmarking/benchmark_testset_v1.0.json")
elif min_acc >= 70:
    print(f"  ⚠ MARGINAL — min accuracy {min_acc:.0f}%, critical cats {critical_min:.0f}%")
    print(f"  Likely improvement on SIMPLE but real risk on MULTIHOP. Consider:")
    print(f"    - Trying a stronger LLM (gpt-4o-mini via OpenRouter for classification)")
    print(f"    - Reverting to main branch and using the current FINAL results")
elif critical_min < 70:
    print(f"  ✗ ABORT — critical categories ({critical_cats}) below 70%")
    print(f"  The 4B LLM is not accurate enough at this classification task.")
    print(f"  Recommendation: stay on main branch with current FINAL results.")
else:
    print(f"  ✗ ABORT — accuracy {min_acc:.0f}% is too low to risk a rerun")
