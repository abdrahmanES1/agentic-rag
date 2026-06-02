# -*- coding: utf-8 -*-
"""
DatasetBuilder — deduplicates, assigns IDs, and merges scraped Q&A items
into benchmark_testset_gold.json.

ID scheme (matches existing testset):
  S{n}  — SIMPLE
  M{n}  — MULTIHOP
  L{n}  — LEGAL
  D{n}  — DARIJA
  A{n}  — ARABIZI
  O{n}  — OUTSCOPE
  E{n}  — EDGE

Deduplication: normalized token overlap > 85% between any new question and
any existing question → drop the new item (no LLM needed).
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("scraper.dataset_builder")

_CATEGORY_PREFIX = {
    "SIMPLE":        "S",
    "MULTIHOP":      "M",
    "MULTIHOP_HARD": "M",   # same prefix as MULTIHOP, harder variant
    "LEGAL":         "L",
    "DARIJA":        "D",
    "ARABIZI":       "A",
    "OUTSCOPE":      "O",
    "EDGE":          "E",
}

_DEDUP_THRESHOLD = 0.85   # token overlap above this → duplicate


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_overlap(a: str, b: str) -> float:
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _atomic_write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


class DatasetBuilder:
    """
    Loads the existing gold testset, deduplicates new items,
    assigns IDs, and writes back.
    """

    def __init__(self, gold_path: Path, fresh: bool = False) -> None:
        self.gold_path = gold_path
        self.fresh     = fresh
        self.existing: List[dict] = [] if fresh else self._load()
        if fresh:
            log.info("[builder] --fresh-gold: starting with empty gold testset (IDs from S01)")
        self._existing_questions = [
            _normalize(item.get("question", "")) for item in self.existing
        ]

    def _load(self) -> List[dict]:
        if not self.gold_path.exists():
            log.info("[builder] No existing gold testset — starting empty")
            return []
        with open(self.gold_path, encoding="utf-8") as f:
            data = json.load(f)
        log.info("[builder] Loaded %d existing items from %s", len(data), self.gold_path.name)
        return data

    # ── ID management ─────────────────────────────────────────────────────────

    def _next_id(self, category: str) -> str:
        prefix = _CATEGORY_PREFIX.get(category.upper(), "X")
        # Find max existing number for this prefix
        max_n = 0
        for item in self.existing:
            item_id = item.get("id", "")
            if item_id.startswith(prefix) and item_id[1:].isdigit():
                max_n = max(max_n, int(item_id[1:]))
        return f"{prefix}{max_n + 1:02d}"

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _is_duplicate(self, question: str) -> bool:
        q_norm = _normalize(question)
        return any(
            _token_overlap(q_norm, eq) >= _DEDUP_THRESHOLD
            for eq in self._existing_questions
        )

    # ── Build final items ─────────────────────────────────────────────────────

    def build_items(self, raw_qa: List[dict]) -> Tuple[List[dict], int]:
        """
        Assign IDs, fill default fields, deduplicate.

        Returns
        -------
        (new_items, n_duplicates_dropped)
        """
        new_items: List[dict] = []
        n_dupes = 0

        for raw in raw_qa:
            question = (raw.get("question") or "").strip()
            if not question:
                continue

            if self._is_duplicate(question):
                n_dupes += 1
                continue

            category = raw.get("category", "SIMPLE").upper()
            language = raw.get("language", "arabic_msa")

            # Auto-correct category from language when the model sets SIMPLE
            # for Darija/Arabizi questions (common model mistake).
            if language == "Darija" and category not in {"DARIJA", "OUTSCOPE"}:
                category = "DARIJA"
            elif language == "Arabizi" and category not in {"ARABIZI", "OUTSCOPE"}:
                category = "ARABIZI"

            # Arabizi gold_answer must be in arabic_msa — the RAG system answers
            # in MSA regardless of question language. If model responded in Arabizi,
            # flag it but keep the item (keywords are still valid for evaluation).
            gold_answer = raw.get("gold_answer", "")
            if language == "Arabizi" and gold_answer:
                # Detect if answer is in Latin script (Arabizi) instead of Arabic
                arabic_chars = sum(1 for c in gold_answer if "؀" <= c <= "ۿ")
                latin_chars  = sum(1 for c in gold_answer if c.isascii() and c.isalpha())
                if latin_chars > arabic_chars:
                    # Answer is in Arabizi — mark it for review but don't drop it
                    raw = dict(raw)
                    raw["gold_answer"] = gold_answer  # keep as-is, reviewer can fix
                    raw["review_needed"] = "arabizi_answer_not_msa"

            is_multihop = raw.get("is_multihop", False) or category == "MULTIHOP"

            item = {
                "id": self._next_id(category),
                "category": category,
                "language": language,
                "question": question,
                "gold_answer": raw.get("gold_answer", ""),
                "gold_keywords": raw.get("gold_keywords", []),
                "source": raw.get("source", ""),
                "should_abstain": raw.get("should_abstain", False) or category == "OUTSCOPE",
                "is_multihop": is_multihop,
                "expected_flags": {
                    "SIMPLE":        category == "SIMPLE",
                    "MULTIHOP":      is_multihop,
                    "MULTIHOP_HARD": category == "MULTIHOP_HARD",
                    "OUTSCOPE":      category == "OUTSCOPE",
                    "LEGAL":         category == "LEGAL",
                    "DARIJA":        category == "DARIJA",
                    "ARABIZI":       category == "ARABIZI",
                    "EDGE":          category == "EDGE",
                    "language":      language,
                },
                "expected_language": language,
                "expected_detected_language": language,
                "scraped": True,
                "scraped_at": datetime.now().isoformat(),
            }

            new_items.append(item)
            # Track for intra-batch deduplication
            self._existing_questions.append(_normalize(question))
            self.existing.append(item)   # needed for _next_id counter

        return new_items, n_dupes

    # ── Persist ───────────────────────────────────────────────────────────────

    def save_separate(self, new_items: List[dict], output_dir: Path) -> Path:
        """Write new items to a standalone file (for review)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"scraped_dataset_{ts}.json"
        _atomic_write(path, new_items)
        log.info("[builder] ✓ Separate file: %s (%d items)", path, len(new_items))
        return path

    def merge_into_gold(self, new_items: List[dict]) -> None:
        """
        Write new_items into benchmark_testset_gold.json (atomic write).
        With fresh=True: replaces the file entirely with the new items.
        With fresh=False: appends to existing items (dedup already done in build_items).
        Items are sorted by category prefix then numeric ID.
        """
        merged = self.existing   # already contains new_items from build_items()
        merged_sorted = sorted(
            merged,
            key=lambda x: (
                _CATEGORY_PREFIX.get(x.get("category", "X"), "Z"),
                int(re.sub(r"\D", "", x.get("id", "0")) or 0),
            ),
        )
        _atomic_write(self.gold_path, merged_sorted)
        action = "created" if self.fresh else "updated"
        log.info("[builder] ✓ Gold testset %s: %d items → %s",
                 action, len(merged_sorted), self.gold_path)

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self, new_items: List[dict], n_dupes: int) -> None:
        from collections import Counter
        cats = Counter(i["category"] for i in new_items)
        langs = Counter(i["language"] for i in new_items)
        print(f"\n{'='*60}")
        print(f"  DATASET BUILD SUMMARY")
        print(f"{'='*60}")
        print(f"  New items added : {len(new_items)}")
        print(f"  Duplicates dropped: {n_dupes}")
        print(f"  Total in gold   : {len(self.existing)}")
        print(f"\n  By category:")
        for cat, n in sorted(cats.items()):
            print(f"    {cat:<12} {n}")
        print(f"\n  By language:")
        for lang, n in sorted(langs.items()):
            print(f"    {lang:<16} {n}")
        print(f"{'='*60}\n")
