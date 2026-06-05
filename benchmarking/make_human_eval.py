# -*- coding: utf-8 -*-
"""
Generate a BLIND human-evaluation sheet for the PFE.

Samples a stratified set of questions, pulls each system's answer, anonymizes
the system identity, shuffles the rows, and writes:
  - human_eval_sheet.csv  — the rater fills correctness / fluency / dialect_quality (1-5)
  - human_eval_key.csv    — row_id -> system (de-anonymize AFTER rating)

Blind + shuffled so the rater can't tell which system produced an answer.

Usage:  python benchmarking/make_human_eval.py
        python benchmarking/make_human_eval.py --per-lang 3 --systems v12_pipeline,naive_rag,crag
"""
import argparse, csv, glob, json, os, random, re
from pathlib import Path

_HERE = Path(__file__).parent
CIT = re.compile(r"\[Source:[^\]]*\]", re.IGNORECASE)
URL = re.compile(r"https?://\S+")
strip = lambda t: URL.sub("", CIT.sub("", t or "")).strip()


def latest_run():
    dirs = sorted(glob.glob(str(_HERE / "results" / "run_*")), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: latest)")
    ap.add_argument("--testset", default=str(_HERE / "benchmark_testset_v1.0.json"))
    ap.add_argument("--per-lang", type=int, default=3, help="questions sampled per language")
    ap.add_argument("--systems", default="v12_pipeline,naive_rag,crag")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = args.run or latest_run()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    ts = json.load(open(args.testset, encoding="utf-8"))
    items = ts["items"] if isinstance(ts, dict) and "items" in ts else ts
    raws = {s: json.load(open(os.path.join(run, f"raw_{s}.json"), encoding="utf-8")) for s in systems}

    random.seed(args.seed)
    by_lang = {}
    for i, it in enumerate(items):
        by_lang.setdefault(it.get("language") or it.get("expected_language") or "?", []).append(i)
    picked = []
    for lang, idxs in by_lang.items():
        picked += random.sample(idxs, min(args.per_lang, len(idxs)))

    rows, key = [], []
    rid = 1000
    for i in picked:
        it = items[i]
        for s in systems:
            if i >= len(raws[s]):
                continue
            rid += 1
            rows.append({
                "row_id": rid,
                "language": it.get("language") or it.get("expected_language"),
                "category": it.get("category", ""),
                "question": it.get("question", ""),
                "gold_answer": it.get("gold_answer", ""),
                "answer": strip(raws[s][i].get("answer", "")),
                "correctness_1to5": "", "fluency_1to5": "",
                "dialect_quality_1to5": "", "notes": "",
            })
            key.append({"row_id": rid, "system": s, "item_id": it.get("id", i)})
    random.shuffle(rows)  # blind: rater can't infer system from order

    sheet = _HERE / "human_eval_sheet.csv"
    keyf = _HERE / "human_eval_key.csv"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(keyf, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "system", "item_id"])
        w.writeheader(); w.writerows(key)

    print(f"Wrote {len(rows)} blind rows ({len(picked)} questions x {len(systems)} systems)")
    print(f"  rater sheet : {sheet}")
    print(f"  de-anon key : {keyf}")
    print("Rate correctness/fluency 1-5 for all; dialect_quality 1-5 only for Darija/Arabizi.")


if __name__ == "__main__":
    main()
