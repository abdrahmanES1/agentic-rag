# -*- coding: utf-8 -*-
"""
audit_gold.py — Language-consistency & quality audit for benchmark_testset_gold.json

The benchmark requires every item to be language-consistent:

    question_language == answer_language == keywords_language == expected_language

Per declared language, the EXPECTED script/language is:
    arabic_msa  -> Arabic script   (MSA)
    Darija      -> Arabic script   (Moroccan Darija, NOT MSA)
    Arabizi     -> Latin script    (romanized Darija: 3=ع 7=ح 9=ق, no Arabic chars)
    french      -> Latin script    (French)

This script does NOT modify the dataset. It only reports problems so you can
review them and decide a fix strategy.

Outputs:
    - console summary table
    - benchmarking/gold_audit_report.md    (human-readable, grouped by issue)
    - benchmarking/gold_audit_report.json  (machine-readable: per-item flags)

Usage:
    python benchmarking/audit_gold.py
    python benchmarking/audit_gold.py --gold path/to/gold.json
    python benchmarking/audit_gold.py --show ARABIZI   # print all flagged items of one language
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
GOLD_DEFAULT = _HERE / "benchmark_testset_gold.json"

# Expected script per declared language
_EXPECTED_SCRIPT = {
    "arabic_msa": "arabic",
    "Darija":     "arabic",
    "Arabizi":    "latin",
    "french":     "latin",
}

# Abstention markers (an OUTSCOPE answer SHOULD contain one; others should NOT)
_ABSTAIN_MARKERS = [
    # explicit "not available"
    "غير متوفر", "غير متاح", "لا تتوفر", "لا يتوفر", "ماكاينش", "ماجدتش", "خارج نطاق",
    # "this text does not address / contain / mention / provide …"
    "لا يتناول", "لا يتضمن", "لا يقدم", "لا يشير", "لا يحتوي", "لا يوجد",
    "لا يذكر", "لا تذكر", "لم يرد", "ليس في", "لا علاقة",
    # French
    "non disponible", "informations insuffisantes", "pas disponible",
    "ne traite pas", "n'aborde pas", "ne contient pas", "ne mentionne pas",
    # Darija / Arabizi
    "ma kaynach", "machi", "n/a", "ma3andi", "makay/n",
]

# Dangling demonstratives that often signal a NON-self-contained question
# (they point at "this licence / this procedure" without naming which one).
_DANGLING = {
    "arabic": [r"هذه الرخصة", r"هذا الإجراء", r"هذه الخدمة", r"هذا القرار",
               r"هذه الوثيقة", r"هذا الطلب", r"هذه العملية", r"المذكورة?",
               r"المذكور", r"هاد الإجراء", r"هاد الرخصة", r"هاد الخدمة"],
    "latin":  [r"\bcette licence\b", r"\bcette démarche\b", r"\bcette procédure\b",
               r"\bce service\b", r"\bcette décision\b", r"\bce document\b",
               r"\bsusmentionnée?\b", r"\bhad l['’]?ijra", r"\bhad rokhsa\b",
               r"\bhad l['’]?khedma\b"],
}


# ── Script / language helpers ─────────────────────────────────────────────────

def script_of(text: str) -> str:
    """Return 'arabic' | 'latin' | 'mixed' | 'none' based on alphabetic chars."""
    text = text or ""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "none"
    ar = sum(1 for c in alpha if "؀" <= c <= "ۿ")
    ratio = ar / len(alpha)
    if ratio > 0.6:
        return "arabic"
    if ratio < 0.2:
        return "latin"
    return "mixed"


def has_arabic(text: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", text or ""))


def has_latin(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text or ""))


def has_ipa_diacritics(text: str) -> bool:
    """Academic/IPA transliteration marks that should NOT appear in Arabizi."""
    return any(c in set("āēīōūḥḍṭẓṣġʿʾ") for c in (text or "").lower())


# ── Per-item audit ────────────────────────────────────────────────────────────

def audit_item(item: dict) -> list:
    """Return a list of issue-code strings for one gold item."""
    issues = []

    lang     = item.get("language", "")
    expected = item.get("expected_language", "")
    category = item.get("category", "")
    q        = item.get("question", "") or ""
    a        = str(item.get("gold_answer", "") or "")
    kws      = item.get("gold_keywords", []) or []
    kw_text  = " ".join(str(k) for k in kws)
    should_abstain = item.get("should_abstain", False)

    exp_script = _EXPECTED_SCRIPT.get(lang)

    # 1) declared language must equal expected_language
    if expected and expected != lang:
        issues.append("LANG_FIELD_MISMATCH")

    # 2) QUESTION script must match expected
    if exp_script:
        qs = script_of(q)
        if exp_script == "latin" and has_arabic(q):
            issues.append("Q_HAS_ARABIC")          # Arabizi/French Q with Arabic chars
        if exp_script == "arabic" and qs == "latin":
            issues.append("Q_WRONG_SCRIPT")        # Arabic/Darija Q in Latin

    # 3) ANSWER script must match expected (skip pure OUTSCOPE abstentions)
    if exp_script and not should_abstain:
        as_ = script_of(a)
        if exp_script == "latin" and as_ == "arabic":
            issues.append("A_IS_ARABIC")           # Arabizi/French answer is Arabic
        elif exp_script == "latin" and as_ == "mixed":
            issues.append("A_MIXED_SCRIPT")
        elif exp_script == "arabic" and as_ == "latin":
            issues.append("A_IS_LATIN")

    # 4) KEYWORDS script must match expected
    if kws:
        if exp_script == "latin" and has_arabic(kw_text):
            issues.append("KW_HAS_ARABIC")         # French/Arabizi kw in Arabic
        if exp_script == "arabic" and has_latin(kw_text) and not has_arabic(kw_text):
            issues.append("KW_IS_LATIN")
    else:
        if not should_abstain:
            issues.append("KW_EMPTY")

    # 5) Arabizi-specific quality: no IPA diacritics
    if lang == "Arabizi":
        if has_ipa_diacritics(q):
            issues.append("ARABIZI_IPA_IN_Q")
        if has_ipa_diacritics(a):
            issues.append("ARABIZI_IPA_IN_A")

    # 6) Answerability
    if not should_abstain:
        if not a.strip():
            issues.append("ANSWER_EMPTY")
        elif any(m in a.lower() for m in _ABSTAIN_MARKERS):
            issues.append("ANSWER_LOOKS_ABSTAIN")   # non-OUTSCOPE but answer abstains
    else:
        if a.strip() and not any(m in a.lower() for m in _ABSTAIN_MARKERS):
            issues.append("OUTSCOPE_NOT_ABSTAINING")  # should abstain but gives content

    # 7) Self-contained heuristic — dangling demonstrative
    dialect = "latin" if exp_script == "latin" else "arabic"
    for pat in _DANGLING.get(dialect, []):
        if re.search(pat, q, re.IGNORECASE):
            issues.append("MAYBE_NOT_SELF_CONTAINED")
            break

    return issues


# ── Report generation ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Audit gold dataset language consistency.")
    ap.add_argument("--gold", default=str(GOLD_DEFAULT))
    ap.add_argument("--show", default=None,
                    help="Print every flagged item of one language (e.g. ARABIZI, french)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    gold_path = Path(args.gold)
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    flagged = []
    issue_counts = Counter()
    by_lang_issues = defaultdict(Counter)

    for idx, item in enumerate(gold):
        issues = audit_item(item)
        if issues:
            flagged.append({"index": idx, "id": item.get("id"),
                            "language": item.get("language"),
                            "category": item.get("category"),
                            "issues": issues,
                            "question": item.get("question", ""),
                            "gold_answer": str(item.get("gold_answer", "")),
                            "gold_keywords": item.get("gold_keywords", [])})
            for code in issues:
                issue_counts[code] += 1
                by_lang_issues[item.get("language")][code] += 1

    # ── Console summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  GOLD DATASET LANGUAGE-CONSISTENCY AUDIT")
    print("=" * 70)
    print(f"  Total items        : {len(gold)}")
    print(f"  Items with issues  : {len(flagged)}")
    print(f"  Clean items        : {len(gold) - len(flagged)}")
    print()
    print("  ── Issue counts (most common first) ──")
    for code, n in issue_counts.most_common():
        print(f"    {code:28s} {n:4d}")
    print()
    print("  ── Issues by language ──")
    for lang in ["arabic_msa", "french", "Darija", "Arabizi"]:
        c = by_lang_issues.get(lang, {})
        total = sum(c.values())
        n_items = sum(1 for fl in flagged if fl["language"] == lang)
        print(f"    {lang:12s}: {n_items:3d} flagged items, {total} total issues")
        for code, n in sorted(c.items(), key=lambda x: -x[1]):
            print(f"        {code:28s} {n:4d}")
    print("=" * 70)

    # ── Markdown report ──────────────────────────────────────────────────────
    md_path = _HERE / "gold_audit_report.md"
    by_issue = defaultdict(list)
    for fl in flagged:
        for code in fl["issues"]:
            by_issue[code].append(fl)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gold Dataset Audit Report\n\n")
        f.write(f"- Total items: **{len(gold)}**\n")
        f.write(f"- Flagged items: **{len(flagged)}**\n\n")
        f.write("## Issue legend\n\n")
        f.write("| Code | Meaning |\n|---|---|\n")
        f.write("| LANG_FIELD_MISMATCH | language != expected_language |\n")
        f.write("| Q_HAS_ARABIC | Arabizi/French question contains Arabic script |\n")
        f.write("| Q_WRONG_SCRIPT | Arabic/Darija question is in Latin |\n")
        f.write("| A_IS_ARABIC | answer is Arabic but should be Latin (Arabizi/French) |\n")
        f.write("| A_MIXED_SCRIPT | answer mixes scripts |\n")
        f.write("| A_IS_LATIN | answer is Latin but should be Arabic |\n")
        f.write("| KW_HAS_ARABIC | keywords contain Arabic but should be Latin |\n")
        f.write("| KW_IS_LATIN | keywords are Latin but should be Arabic |\n")
        f.write("| KW_EMPTY | no keywords |\n")
        f.write("| ARABIZI_IPA_IN_Q/A | IPA diacritics (ā/ḥ) instead of standard Arabizi |\n")
        f.write("| ANSWER_EMPTY | non-OUTSCOPE with empty answer |\n")
        f.write("| ANSWER_LOOKS_ABSTAIN | non-OUTSCOPE answer looks like an abstention |\n")
        f.write("| OUTSCOPE_NOT_ABSTAINING | should_abstain but answer gives content |\n")
        f.write("| MAYBE_NOT_SELF_CONTAINED | dangling 'this licence/procedure' reference |\n\n")

        for code, items in sorted(by_issue.items(), key=lambda x: -len(x[1])):
            f.write(f"\n## {code}  ({len(items)} items)\n\n")
            for fl in items:
                f.write(f"### #{fl['index']} · {fl['language']} · {fl['category']}\n")
                f.write(f"- **Q:** {fl['question']}\n")
                f.write(f"- **A:** {fl['gold_answer'][:300]}\n")
                f.write(f"- **KW:** {fl['gold_keywords']}\n")
                f.write(f"- **all issues:** {fl['issues']}\n\n")

    print(f"\n  Markdown report → {md_path}")

    # ── JSON report ──────────────────────────────────────────────────────────
    json_path = _HERE / "gold_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(gold), "flagged": len(flagged),
                   "issue_counts": dict(issue_counts), "items": flagged},
                  f, ensure_ascii=False, indent=2)
    print(f"  JSON report     → {json_path}")

    # ── Optional: print all flagged items of one language ────────────────────
    if args.show:
        target = args.show
        print(f"\n{'='*70}\n  FLAGGED ITEMS — {target}\n{'='*70}")
        for fl in flagged:
            if fl["language"].lower() == target.lower():
                print(f"\n#{fl['index']} [{fl['category']}] issues={fl['issues']}")
                print(f"  Q : {fl['question']}")
                print(f"  A : {fl['gold_answer'][:200]}")
                print(f"  KW: {fl['gold_keywords']}")


if __name__ == "__main__":
    main()
