# -*- coding: utf-8 -*-
"""
Scrape-progress checkpoint.

Saves state after every page / procedure so a crash can be resumed without
restarting from zero.

Layout under progress_dir/:
  manifest.json          — what has been scraped so far
  raw_idarati.json       — accumulated idarati procedure texts
  raw_html.json          — accumulated HTML-scraped sections
  generated_qa.json      — Q&A items produced by the LLM so far
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("scraper.checkpoint")


def _atomic_write(path: Path, data: Any) -> None:
    """Write JSON to .tmp then rename — never leaves a corrupt file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


class ScrapeCheckpoint:
    """
    Persists scraping progress to disk so interrupted runs resume cleanly.

    Usage
    -----
    ckpt = ScrapeCheckpoint(Path("benchmarking/scraper_runs/run_20260513"))
    ckpt.save_idarati(procedure_list)   # after every page of API results
    ckpt.save_html(section_list)        # after every crawled page
    ckpt.save_qa(qa_items)              # after every LLM generation call
    ckpt.mark_done("idarati")           # when a source is fully scraped
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = run_dir / "manifest.json"
        self.manifest: Dict = self._load_manifest()
        log.info("[checkpoint] Scrape run dir: %s", self.run_dir)

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _load_manifest(self) -> Dict:
        if self._manifest_path.exists():
            with open(self._manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "sources": {},
        }

    def _flush(self) -> None:
        self.manifest["updated_at"] = datetime.now().isoformat()
        _atomic_write(self._manifest_path, self.manifest)

    def is_done(self, source: str) -> bool:
        return self.manifest["sources"].get(source, {}).get("done", False)

    def mark_done(self, source: str) -> None:
        self.manifest["sources"].setdefault(source, {})["done"] = True
        self.manifest["sources"][source]["done_at"] = datetime.now().isoformat()
        self._flush()
        log.info("[checkpoint] ✓ Source done: %s", source)

    # ── idarati API procedures ─────────────────────────────────────────────────

    def load_idarati(self) -> List[dict]:
        p = self.run_dir / "raw_idarati.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            log.info("[checkpoint] Loaded %d idarati procedures from checkpoint", len(data))
            return data
        return []

    def save_idarati(self, procedures: List[dict]) -> None:
        _atomic_write(self.run_dir / "raw_idarati.json", procedures)
        self.manifest["sources"].setdefault("idarati", {})["count"] = len(procedures)
        self.manifest["sources"]["idarati"]["last_saved"] = datetime.now().isoformat()
        self._flush()

    def idarati_count(self) -> int:
        return self.manifest["sources"].get("idarati", {}).get("count", 0)

    # ── idarati administrations ───────────────────────────────────────────────

    def load_administrations(self) -> List[dict]:
        p = self.run_dir / "raw_administrations.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            log.info("[checkpoint] Loaded %d administrations from checkpoint", len(data))
            return data
        return []

    def save_administrations(self, admins: List[dict]) -> None:
        _atomic_write(self.run_dir / "raw_administrations.json", admins)
        self.manifest["sources"].setdefault("administrations", {})["count"] = len(admins)
        self.manifest["sources"]["administrations"]["last_saved"] = datetime.now().isoformat()
        self._flush()

    def load_reference_data(self) -> dict:
        """Load admin_types + beneficiary_categories reference data."""
        p = self.run_dir / "raw_reference.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_reference_data(self, data: dict) -> None:
        """Save admin_types + beneficiary_categories (small, fetched once)."""
        _atomic_write(self.run_dir / "raw_reference.json", data)
        self._flush()

    # ── HTML-scraped sections ─────────────────────────────────────────────────

    def load_html(self) -> List[dict]:
        p = self.run_dir / "raw_html.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            log.info("[checkpoint] Loaded %d HTML sections from checkpoint", len(data))
            return data
        return []

    def save_html(self, sections: List[dict]) -> None:
        _atomic_write(self.run_dir / "raw_html.json", sections)
        self.manifest["sources"].setdefault("html", {})["count"] = len(sections)
        self.manifest["sources"]["html"]["last_saved"] = datetime.now().isoformat()
        self._flush()

    # ── Generated Q&A items ───────────────────────────────────────────────────

    def load_qa(self) -> List[dict]:
        p = self.run_dir / "generated_qa.json"
        if p.exists() and p.stat().st_size > 0:
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                log.info("[checkpoint] Loaded %d Q&A items from checkpoint", len(data))
                return data
            except json.JSONDecodeError:
                log.warning("[checkpoint] generated_qa.json is corrupt — starting QA from scratch")
        return []

    def save_qa(self, qa_items: List[dict]) -> None:
        _atomic_write(self.run_dir / "generated_qa.json", qa_items)
        self.manifest.setdefault("qa", {})["count"] = len(qa_items)
        self.manifest["qa"]["last_saved"] = datetime.now().isoformat()
        self._flush()

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def new_run(cls, base_dir: Path) -> "ScrapeCheckpoint":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(base_dir / f"scrape_{ts}")

    @classmethod
    def find_latest(cls, base_dir: Path) -> Optional["ScrapeCheckpoint"]:
        """
        Return the best scrape checkpoint for QA generation.

        Selection order (highest priority first):
        1. Incomplete run (no completed_at) with the MOST idarati procedures
        2. If all runs are finalized, pick the most-data finalized run
           (so QA can still resume against a complete dataset)

        This prevents silently falling back to an old 100-procedure test run
        when the main 2,500-procedure run has been finalized.
        """
        if not base_dir.exists():
            return None

        best_incomplete: Optional[Path] = None
        best_incomplete_procs: int = -1
        best_any: Optional[Path] = None
        best_any_procs: int = -1

        for d in sorted(base_dir.glob("scrape_*"), reverse=True):
            if not d.is_dir():
                continue
            # Count procedures (fast — just check file size as proxy if JSON parse is slow)
            idarati_file = d / "raw_idarati.json"
            proc_count = 0
            if idarati_file.exists() and idarati_file.stat().st_size > 0:
                try:
                    with open(idarati_file, encoding="utf-8") as f:
                        proc_count = len(json.load(f))
                except Exception:
                    pass

            manifest_path = d / "manifest.json"
            is_complete = False
            if manifest_path.exists():
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        m = json.load(f)
                    is_complete = "completed_at" in m
                except Exception:
                    pass
            else:
                # No manifest → treat as incomplete (enrichment runs don't always create one)
                pass

            if proc_count > 0 and proc_count > best_any_procs:
                best_any_procs = proc_count
                best_any = d

            if proc_count > 0 and not is_complete and proc_count > best_incomplete_procs:
                best_incomplete_procs = proc_count
                best_incomplete = d

        if best_incomplete is not None:
            log.info("[checkpoint] Found incomplete scrape: %s (%d procedures)",
                     best_incomplete, best_incomplete_procs)
            return cls(best_incomplete)

        if best_any is not None:
            log.info("[checkpoint] All runs finalized — using largest: %s (%d procedures)",
                     best_any, best_any_procs)
            return cls(best_any)

        return None

    def finalize(self) -> None:
        self.manifest["completed_at"] = datetime.now().isoformat()
        self._flush()
        log.info("[checkpoint] Scrape run complete: %s", self.run_dir)
