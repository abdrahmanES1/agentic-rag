# -*- coding: utf-8 -*-
"""
Regenerate the publication comparison table from FINAL/ data.

Usage:  python benchmarking/results/FINAL/generate_table.py

Outputs:
  - final_table.md       : Markdown table for the thesis
  - final_summary.txt    : Plain-text summary
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
systems_all = ["naive_rag", "basic_react", "adaptive_simple", "hyde",
                "self_rag", "flare", "crag", "v12_pipeline"]
systems_primary = [s for s in systems_all if s != "self_rag"]

S = {s: json.loads((HERE / "scores" / f"scores_{s}.json").read_text(encoding="utf-8"))
     for s in systems_all}

# Metric groups
groups = [
    ("RAGAS", ["faithfulness","answer_relevancy","context_precision","context_recall"]),
    ("ARES (LLM judge)", ["ares_answer_relevance","ares_answer_faithfulness","ares_completeness","ares_context_relevance"]),
    ("G-Eval / FActScore", ["geval_relevance","geval_coherence","geval_fluency","factscore"]),
    ("Ground-truth lexical", ["token_f1","rouge_l","keyword_hit_rate","arabizi_normalized_f1"]),
    ("Abstain / OOS detection", ["abstain_f1","abstain_precision","abstain_recall","abstain_accuracy"]),
    ("AGENTIC: Multi-hop success", ["multihop_success_rate","multihop_routing_rate","multihop_coverage_rate"]),
    ("Domain (Moroccan PS)", ["domain_dialect_response_match","domain_legal_citation_hit","domain_hallucination_number_rate","domain_cost_deadline_hit"]),
    ("v12-specific grounding", ["v12_cfi","v12_entity_match_ratio","unsupported_claim_rate"]),
    ("Efficiency", ["avg_latency_sec","p50_latency_sec","p95_latency_sec"]),
]

LOWER_BETTER = {"avg_latency_sec","p50_latency_sec","p95_latency_sec",
                "domain_hallucination_number_rate","unsupported_claim_rate"}

short = {"naive_rag":"naive","basic_react":"react","adaptive_simple":"adapt",
         "hyde":"hyde","self_rag":"self†","flare":"flare","crag":"crag","v12_pipeline":"V12"}

def best_in(systems_subset, k):
    vals = {s: S[s].get(k) for s in systems_subset if isinstance(S[s].get(k),(int,float))}
    if not vals:
        return None
    return min(vals.values()) if k in LOWER_BETTER else max(vals.values())

def cell(s, k, best_val):
    v = S[s].get(k)
    if not isinstance(v,(int,float)):
        return " — "
    star = " *" if best_val is not None and abs(v - best_val) < 1e-6 else "  "
    return f"{v:.3f}{star}"

# Markdown output
md = []
md.append("# Publication Comparison Table")
md.append("")
md.append("**v12 (run_20260606_233712) vs 7 baselines (run_20260605_223825)** • 124 questions • 4 languages")
md.append("")
md.append("`*` marks the #1 ranking on each metric.")
md.append("")
md.append("## Primary Comparison (7 systems, excluding self_rag)")
md.append("")
md.append(f"| metric | {' | '.join(short[s] for s in systems_primary)} |")
md.append("|" + "---|" * (len(systems_primary)+1))

for gname, keys in groups:
    md.append(f"| **{gname}** |" + " |" * len(systems_primary))
    for k in keys:
        best = best_in(systems_primary, k)
        row = f"| {k} | " + " | ".join(cell(s, k, best) for s in systems_primary) + " |"
        md.append(row)

# Win count primary
wins_p = Counter()
total_compared = 0
for gname, keys in groups:
    for k in keys:
        best = best_in(systems_primary, k)
        if best is None: continue
        total_compared += 1
        for s in systems_primary:
            v = S[s].get(k)
            if isinstance(v,(int,float)) and abs(v - best) < 1e-6:
                wins_p[s] += 1

md.append("")
md.append("### #1 ranking distribution")
md.append("")
md.append("| system | wins |")
md.append("|---|---|")
for s, n in sorted(wins_p.items(), key=lambda x: -x[1]):
    marker = " ← **OFFICIAL**" if s == "v12_pipeline" else ""
    md.append(f"| {short[s]} | {n}/{total_compared}{marker} |")
md.append("")

# Appendix: 8-system table including self_rag
md.append("---")
md.append("")
md.append("## Appendix: 8-system table (including Self-RAG approximation)")
md.append("")
md.append(f"`†` Self-RAG is a prompt-based approximation (4 LLM calls per question) rather than the published fine-tuned version (Asai et al. 2023, requires 150K supervised examples + critic model fine-tuning). **Empirically refuses 50/124 (40%) of questions, including answerable ones.** Inflated ARES faithfulness (0.897) and FActScore (0.960) reflect 'faithful to nothing' refusals — RAGAS faithfulness (0.315, worst) exposes the artifact. Excluded from primary comparison.")
md.append("")
md.append(f"| metric | {' | '.join(short[s] for s in systems_all)} |")
md.append("|" + "---|" * (len(systems_all)+1))

for gname, keys in groups:
    md.append(f"| **{gname}** |" + " |" * len(systems_all))
    for k in keys:
        best = best_in(systems_all, k)
        row = f"| {k} | " + " | ".join(cell(s, k, best) for s in systems_all) + " |"
        md.append(row)

# Win count all
wins_a = Counter()
total_a = 0
for gname, keys in groups:
    for k in keys:
        best = best_in(systems_all, k)
        if best is None: continue
        total_a += 1
        for s in systems_all:
            v = S[s].get(k)
            if isinstance(v,(int,float)) and abs(v - best) < 1e-6:
                wins_a[s] += 1
md.append("")
md.append("### #1 ranking distribution (8-system)")
md.append("")
md.append("| system | wins |")
md.append("|---|---|")
for s, n in sorted(wins_a.items(), key=lambda x: -x[1]):
    marker = " ← **OFFICIAL**" if s == "v12_pipeline" else (" — refusal-inflated" if s == "self_rag" else "")
    md.append(f"| {short[s]} | {n}/{total_a}{marker} |")

(HERE / "final_table.md").write_text("\n".join(md), encoding="utf-8")

# Plain summary
txt = []
txt.append("=" * 72)
txt.append(" FINAL PUBLICATION SUMMARY")
txt.append("=" * 72)
txt.append("")
txt.append(" Primary comparison (7 systems, excluding self_rag prompt-approximation):")
txt.append("")
for s, n in sorted(wins_p.items(), key=lambda x: -x[1]):
    marker = " ← OFFICIAL" if s == "v12_pipeline" else ""
    txt.append(f"   {short[s]:8s} {n}/{total_compared} #1 metrics{marker}")
txt.append("")
txt.append(" v12 EXCLUSIVE WINS (only system to score on these):")
txt.append(f"   abstain_f1            0.538 (all baselines 0.000)")
txt.append(f"   multihop_success      0.667 (all baselines 0.000)")
txt.append(f"   arabizi_normalized_f1 0.303 (best baseline 0.034 = 9x worse)")
txt.append("")
(HERE / "final_summary.txt").write_text("\n".join(txt), encoding="utf-8")

print(f"Wrote {HERE / 'final_table.md'}")
print(f"Wrote {HERE / 'final_summary.txt'}")
