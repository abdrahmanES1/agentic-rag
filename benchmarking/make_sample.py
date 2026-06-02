# -*- coding: utf-8 -*-
"""
make_sample.py — Build the language-balanced 124-item benchmark from the gold pool.

Design (decided for v1.0 of the *final* benchmark):
  • French / Darija / Arabizi: 28 each, drawn from the SAME 28 source procedures
    (parallel) → enables cross-lingual consistency across the three.
  • arabic_msa: 40, used for the Arabic-only capability tracks that don't exist in
    the other languages: all 13 LEGAL + 12 MULTIHOP + 15 OUTSCOPE (abstention).
  → 124 items. Language mix ~32% MSA / 22.6% each of fr/Darija/Arabizi
    (vs 58% MSA in a naive category-stratified draw).

Reproducible: fixed seed → identical sample every run.

Usage:
    python benchmarking/make_sample.py            # writes benchmark_testset_sample.json
    python benchmarking/make_sample.py --seed 7
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
GOLD = _HERE / "benchmark_testset_gold.json"
OUT  = _HERE / "benchmark_testset_sample.json"

PARALLEL_PER_LANG = 28          # french / Darija / Arabizi each
MSA_TRACKS = {"LEGAL": 13, "MULTIHOP": 12, "OUTSCOPE": 15}   # = 40 arabic_msa


def _proc(item) -> str:
    return item.get("source", "").split(" —")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(args.seed)

    gold = json.load(open(GOLD, encoding="utf-8"))

    # index: procedure -> {language -> item}
    proc_lang = defaultdict(dict)
    for it in gold:
        proc_lang[_proc(it)].setdefault(it["language"], it)

    # ── 1. Parallel core: procedures present in french + Darija + Arabizi ──────
    parallel = [p for p, lm in proc_lang.items()
                if {"french", "Darija", "Arabizi"} <= set(lm)]
    chosen_procs = rng.sample(parallel, PARALLEL_PER_LANG)
    sample = []
    for p in chosen_procs:
        for lang in ("french", "Darija", "Arabizi"):
            sample.append(proc_lang[p][lang])

    # ── 2. arabic_msa capability tracks ───────────────────────────────────────
    by_cat_msa = defaultdict(list)
    for it in gold:
        if it["language"] == "arabic_msa":
            by_cat_msa[it["category"]].append(it)
    for cat, k in MSA_TRACKS.items():
        pool = by_cat_msa.get(cat, [])
        sample += rng.sample(pool, min(k, len(pool)))

    rng.shuffle(sample)

    # ── report ────────────────────────────────────────────────────────────────
    from collections import Counter
    langs = Counter(g["language"] for g in sample)
    cats  = Counter(g["category"] for g in sample)
    print(f"Total: {len(sample)} items")
    print("By language:", dict(langs))
    print("By category:", dict(cats))
    print(f"Parallel procedures (fr/Darija/Arabizi share these): {len(chosen_procs)}")

    json.dump(sample, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nWrote → {OUT}")


if __name__ == "__main__":
    main()
