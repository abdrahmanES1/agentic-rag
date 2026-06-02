#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_documents.py -- Add required-documents data to already-scraped procedures.

The main scrape already collected 2,533 procedures via the detail API
(/api/informational/procedures/{id}) but missed the documents endpoint
(/api/informational/procedures/{id}/documents) which contains the
"الوثائق المطلوبة" (Required Documents) section.

This script:
  1. Loads existing procedures from the latest checkpoint
  2. Fetches /documents for each procedure (with resume support)
  3. Rebuilds body_ar to include the required documents list
  4. Saves enriched data back to the checkpoint

Run from the repo root:
    python benchmarking/scraper/enrich_documents.py

Or to limit how many are processed (for testing):
    python benchmarking/scraper/enrich_documents.py --limit 50

After this completes, run QA generation:
    python benchmarking/scraper/run_scraper.py --qa-only --resume
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent   # v12/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("enrich")

_HERE     = Path(__file__).parent
_RUNS_DIR = _HERE / "runs"

DELAY = 0.4   # seconds between document API calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N procedures (default: all)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N procedures (default: 50)")
    args = parser.parse_args()

    from benchmarking.scraper.checkpoint import ScrapeCheckpoint
    from benchmarking.scraper.idarati_api import IdaratiAPIScraper

    # ── Find the latest checkpoint ────────────────────────────────────────────
    ckpt = ScrapeCheckpoint.find_latest(_RUNS_DIR)
    if ckpt is None:
        log.error("No scrape checkpoint found. Run the scraper first:")
        log.error("  python benchmarking/scraper/run_scraper.py --scrape-only --idarati-only")
        sys.exit(1)

    log.info("Using checkpoint: %s", ckpt.run_dir)

    # ── Load existing procedures ──────────────────────────────────────────────
    procedures = ckpt.load_idarati()
    if not procedures:
        log.error("No procedures found in checkpoint.")
        sys.exit(1)

    total = len(procedures)
    log.info("Loaded %d procedures", total)

    if args.limit:
        procedures = procedures[: args.limit]
        log.info("Processing first %d procedures (--limit)", len(procedures))

    # ── Load admin lookup for contact info enrichment ─────────────────────────
    raw_admins = ckpt.load_administrations()
    admin_lookup = {a["admin_id"]: {
        "emails":   a.get("emails",   "").split(", ") if a.get("emails")   else [],
        "phones":   a.get("phones",   "").split(", ") if a.get("phones")   else [],
        "websites": a.get("websites", "").split(", ") if a.get("websites") else [],
    } for a in raw_admins if a.get("admin_id")}
    log.info("Admin lookup: %d entries", len(admin_lookup))

    # ── Count how many already have documents (body contains "الوثائق المطلوبة") ─
    already_enriched = sum(
        1 for p in procedures if "الوثائق المطلوبة" in p.get("body_ar", "")
    )
    log.info("Already enriched with documents: %d / %d", already_enriched, len(procedures))

    # ── Enrich ────────────────────────────────────────────────────────────────
    scraper  = IdaratiAPIScraper()
    n_done   = 0
    n_had_docs = 0
    n_no_docs  = 0
    n_skipped  = 0

    for idx, proc in enumerate(procedures):
        body = proc.get("body_ar", "")

        # Skip if already enriched (resume support)
        if "الوثائق المطلوبة" in body:
            n_skipped += 1
            continue

        pid = proc.get("procedure_id", "")
        if not pid:
            continue

        # Fetch documents
        documents = scraper.fetch_documents(pid)

        if documents:
            n_had_docs += 1
        else:
            n_no_docs += 1

        # Re-fetch detail to rebuild body properly (we have it cached in body_ar already,
        # but we need the dict form for _build_procedure_body. Re-use existing text +
        # append documents section to avoid extra API calls.)
        if documents:
            # Build just the documents section and append it
            docs_section = _build_documents_section(documents)
            if docs_section:
                proc["body_ar"] = body.rstrip() + "\n" + docs_section

        # Fix Arabic URL while we're at it
        thematic_id = proc.get("thematic_id", "")
        if thematic_id and f"/ar/thematique/" not in proc.get("url_ar", ""):
            proc["url_ar"] = f"https://idarati.ma/informationnel/ar/thematique/{thematic_id}/{pid}"

        n_done += 1
        time.sleep(DELAY)

        if n_done % args.save_every == 0:
            _save_all(ckpt, procedures, total)
            log.info("[%d/%d] saved — had docs: %d, no docs: %d, skipped: %d",
                     idx + 1, len(procedures), n_had_docs, n_no_docs, n_skipped)

    # Final save
    _save_all(ckpt, procedures, total)

    log.info("")
    log.info("=" * 60)
    log.info("  ENRICHMENT COMPLETE")
    log.info("=" * 60)
    log.info("  Processed  : %d", n_done)
    log.info("  Had docs   : %d", n_had_docs)
    log.info("  No docs    : %d", n_no_docs)
    log.info("  Skipped    : %d  (already had documents)", n_skipped)
    log.info("  Run dir    : %s", ckpt.run_dir)
    log.info("")
    log.info("  Next step:")
    log.info("    python benchmarking/scraper/run_scraper.py --qa-only --resume")
    log.info("=" * 60)


def _build_documents_section(documents: list) -> str:
    """Build just the الوثائق المطلوبة text block from a documents list."""
    from benchmarking.scraper.idarati_api import IdaratiAPIScraper
    # Re-use the static method to build a consistent format
    # We pass empty detail dict so only the documents section is built
    full = IdaratiAPIScraper._build_procedure_body(
        detail={},
        documents=documents,
        admin_lookup=None,
    )
    return full.strip()


def _save_all(ckpt, procedures: list, original_total: int) -> None:
    """
    Save procedures back to checkpoint.
    If we only processed a subset (--limit), merge back with the full list.
    """
    if len(procedures) < original_total:
        # We processed a subset — load full list and merge
        all_procs = ckpt.load_idarati()
        proc_by_id = {p["procedure_id"]: p for p in procedures}
        for i, p in enumerate(all_procs):
            pid = p.get("procedure_id", "")
            if pid in proc_by_id:
                all_procs[i] = proc_by_id[pid]
        ckpt.save_idarati(all_procs)
    else:
        ckpt.save_idarati(procedures)


if __name__ == "__main__":
    main()
