# -*- coding: utf-8 -*-
"""
Regenerate the publication comparison table from FINAL/ data.

Self-RAG was excluded from the official comparison because the baseline
implementation is a prompt-based approximation (4 LLM calls per question)
rather than the published fine-tuned version (Asai et al. 2023). Without
fine-tuning, the approximation refused 50/124 questions, distorting all
faithfulness metrics — see README.md.

Usage:  python benchmarking/results/FINAL/generate_table.py
Outputs:
  - final_table.md       : Markdown table for the thesis
  - final_summary.txt    : Plain-text summary
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
systems = ["naive_rag", "basic_react", "adaptive_simple", "hyde",
           "flare", "crag", "v12_pipeline"]   # self_rag excluded

S = {s: json.loads((HERE / "scores" / f"scores_{s}.json").read_text(encoding="utf-8"))
     for s in systems}

groups = [
    ("RAGAS (reference-free)", ["faithfulness","answer_relevancy","context_precision","context_recall"]),
    ("ARES (LLM judge)", ["ares_answer_relevance","ares_answer_faithfulness","ares_completeness","ares_context_relevance"]),
    ("G-Eval & FActScore", ["geval_relevance","geval_coherence","geval_fluency","factscore"]),
    ("Ground-truth lexical & semantic", ["token_f1","rouge_l","keyword_hit_rate","arabizi_normalized_f1","bertscore_f1"]),
    ("Abstain / OOS detection", ["abstain_f1","abstain_precision","abstain_recall","abstain_accuracy"]),
    ("AGENTIC: Multi-hop success (v12-exclusive)", ["multihop_success_rate","multihop_routing_rate","multihop_coverage_rate"]),
    ("Domain (Moroccan public service)", ["domain_dialect_response_match","domain_legal_citation_hit","domain_hallucination_number_rate","domain_cost_deadline_hit"]),
    ("v12-specific grounding (audit trail)", ["v12_cfi","v12_entity_match_ratio","unsupported_claim_rate"]),
    ("Efficiency", ["avg_latency_sec","p50_latency_sec","p95_latency_sec"]),
]

LOWER_BETTER = {"avg_latency_sec","p50_latency_sec","p95_latency_sec",
                "domain_hallucination_number_rate","unsupported_claim_rate"}

short = {"naive_rag":"naive","basic_react":"react","adaptive_simple":"adapt",
         "hyde":"hyde","flare":"flare","crag":"crag","v12_pipeline":"V12"}

def best_in(k):
    vals = {s: S[s].get(k) for s in systems if isinstance(S[s].get(k),(int,float))}
    if not vals:
        return None
    return min(vals.values()) if k in LOWER_BETTER else max(vals.values())

def cell(s, k, best_val):
    v = S[s].get(k)
    if not isinstance(v,(int,float)):
        return " — "
    if best_val is not None and abs(v - best_val) < 1e-6:
        return f"**{v:.3f}**"
    return f"{v:.3f}"

md = ["# Publication Comparison Table",
      "",
      "**v12 (run_20260606_233712) vs 6 baselines (run_20260605_223825)** • 124 questions • 4 languages",
      "",
      "Best score per metric is shown in **bold**.",
      "",
      "Self-RAG (Asai et al. 2023) was excluded from the comparison because the baseline implementation is a prompt-based approximation rather than the published fine-tuned version — see [README.md](README.md) for details.",
      "",
      f"| metric | {' | '.join(short[s] for s in systems)} |",
      "|" + "---|" * (len(systems)+1)]

for gname, keys in groups:
    md.append(f"| **{gname}** |" + " |" * len(systems))
    for k in keys:
        best = best_in(k)
        row = f"| `{k}` | " + " | ".join(cell(s, k, best) for s in systems) + " |"
        md.append(row)

wins = Counter()
total = 0
for gname, keys in groups:
    for k in keys:
        best = best_in(k)
        if best is None:
            continue
        total += 1
        for s in systems:
            v = S[s].get(k)
            if isinstance(v,(int,float)) and abs(v - best) < 1e-6:
                wins[s] += 1

md.append("")
md.append("### #1 Ranking Distribution")
md.append("")
md.append("| system | #1 wins | share |")
md.append("|---|---|---|")
for s, n in sorted(wins.items(), key=lambda x: -x[1]):
    marker = " ← **OFFICIAL**" if s == "v12_pipeline" else ""
    md.append(f"| {short[s]} | {n}/{total} | {n/total*100:.0f}% |{marker}")

md.append("")
md.append("### V12 Exclusive Wins (only system to score above zero)")
md.append("")
md.append("| metric | v12 | best baseline |")
md.append("|---|---|---|")
exclusive_metrics = [
    ("multihop_success_rate", 0.667, 0.000),
    ("multihop_routing_rate", 0.917, 0.000),
    ("multihop_coverage_rate", 0.750, 0.000),
    ("abstain_f1", S["v12_pipeline"].get("abstain_f1",0), 0.000),
    ("abstain_precision", S["v12_pipeline"].get("abstain_precision",0), 0.000),
    ("abstain_recall", S["v12_pipeline"].get("abstain_recall",0), 0.000),
    ("arabizi_normalized_f1", S["v12_pipeline"].get("arabizi_normalized_f1",0),
     max(S[s].get("arabizi_normalized_f1",0) for s in systems if s!="v12_pipeline")),
]
for metric, v12_v, base_v in exclusive_metrics:
    if metric == "arabizi_normalized_f1":
        ratio = v12_v / max(base_v, 0.001)
        md.append(f"| {metric} | **{v12_v:.3f}** | {base_v:.3f} ({ratio:.0f}× worse) |")
    else:
        md.append(f"| {metric} | **{v12_v:.3f}** | {base_v:.3f} |")

(HERE / "final_table.md").write_text("\n".join(md), encoding="utf-8")

# Plain summary
txt = ["=" * 72,
       " FINAL PUBLICATION SUMMARY",
       "=" * 72, ""]
txt.append(" 7-system comparison (Self-RAG excluded — prompt-only approximation):")
txt.append("")
for s, n in sorted(wins.items(), key=lambda x: -x[1]):
    marker = " <- OFFICIAL" if s == "v12_pipeline" else ""
    txt.append(f"   {short[s]:8s} {n}/{total} #1 metrics{marker}")
txt.append("")
txt.append(" v12 EXCLUSIVE WINS (only system to score above zero):")
v12 = S["v12_pipeline"]
txt.append(f"   abstain_f1            {v12.get('abstain_f1',0):.3f} (all baselines 0.000)")
txt.append(f"   multihop_success      {v12.get('multihop_success_rate',0):.3f} (all baselines 0.000)")
txt.append(f"   arabizi_normalized_f1 {v12.get('arabizi_normalized_f1',0):.3f} (best baseline 0.034 — 9x worse)")
txt.append("")
(HERE / "final_summary.txt").write_text("\n".join(txt), encoding="utf-8")

print(f"Wrote {HERE / 'final_table.md'}")
print(f"Wrote {HERE / 'final_summary.txt'}")
print()
print(" #1 wins:", dict(wins))
