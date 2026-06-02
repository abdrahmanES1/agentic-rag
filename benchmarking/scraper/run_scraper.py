#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_scraper.py -- Moroccan government website scraper + Q&A benchmark builder

Two-step workflow (recommended)
--------------------------------
  Step 1 -- scrape only (no Ollama needed):
    python benchmarking/scraper/run_scraper.py --scrape-only --idarati-only

  Step 2 -- build Q&A from saved data (Ollama must be running):
    python benchmarking/scraper/run_scraper.py --qa-only

  Resume either step after a crash:
    python benchmarking/scraper/run_scraper.py --scrape-only --resume
    python benchmarking/scraper/run_scraper.py --qa-only --resume

One-shot (scrape + QA in a single run):
    python benchmarking/scraper/run_scraper.py --idarati-only

idarati.ma data collected
--------------------------
  2,533 procedures   (description, price, delay, administrations, forms)
  2,446 administrations (emails, phones, websites)
    208 admin organisations (procedure counts)
      5 administration type categories
     10 beneficiary sub-categories

Additional HTML sources
-----------------------
  justice.gov.ma  BFS HTML crawl
  cnss.ma         BFS HTML crawl
  --urls          User-supplied URLs

Output
------
  benchmarking/scraper/runs/scrape_<ts>/
    raw_idarati.json         -- procedures (saved every 10)
    raw_administrations.json -- all administrations
    raw_reference.json       -- admin types + beneficiary categories
    raw_html.json            -- HTML sections
    generated_qa.json        -- Q&A items (saved after every LLM call)
    manifest.json            -- progress tracker

  benchmarking/scraped_dataset_<ts>.json   -- new items (always written)
  benchmarking/benchmark_testset_gold.json -- merged in-place (unless --no-merge)
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List


def _atomic_write(path: Path, data) -> None:
    """Write JSON to .tmp then rename — never leaves a corrupt file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

# ── Path fix ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent   # v12/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")

_HERE      = Path(__file__).parent           # benchmarking/scraper/
_BENCH_DIR = _HERE.parent                    # benchmarking/
_RUNS_DIR  = _HERE / "runs"                  # benchmarking/scraper/runs/
_GOLD_FILE = _BENCH_DIR / "benchmark_testset_gold.json"

DEFAULT_HTML_SITES = [
    "https://www.justice.gov.ma",
    "https://www.cnss.ma",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: scrape only  (no Ollama required)
# ─────────────────────────────────────────────────────────────────────────────

def run_scrape(args: argparse.Namespace) -> None:
    """
    Fetch all data from idarati.ma and/or HTML sites.
    Saves everything to a checkpoint directory.
    Does NOT call Ollama — safe to run without LM Studio.
    """
    from benchmarking.scraper.checkpoint import ScrapeCheckpoint
    from benchmarking.scraper.html_scraper import HTMLScraper
    from benchmarking.scraper.idarati_api import IdaratiAPIScraper

    ckpt = _get_checkpoint(args)

    # ══ idarati.ma ════════════════════════════════════════════════════════════
    if not args.html_only:
        scraper = IdaratiAPIScraper()

        # -- Reference data --
        if ckpt.is_done("reference"):
            log.info("[idarati] Reference data already done.")
            ref = ckpt.load_reference_data()
        else:
            log.info("[idarati] Fetching reference data (admin types, orgs, beneficiaries)...")
            ref = {
                "admin_types":            scraper.fetch_admin_types(),
                "admin_organisations":    scraper.fetch_admin_organisations(),
                "beneficiary_categories": scraper.fetch_beneficiary_categories(),
            }
            ckpt.save_reference_data(ref)
            ckpt.mark_done("reference")

        log.info("[idarati] Reference: %d admin types, %d admin orgs, %d beneficiary cats",
                 len(ref.get("admin_types", [])),
                 len(ref.get("admin_organisations", [])),
                 len(ref.get("beneficiary_categories", [])))

        # -- Administrations --
        if ckpt.is_done("administrations"):
            log.info("[idarati] Administrations already done.")
        else:
            log.info("[idarati] Fetching all administrations (2,446 expected)...")
            raw_admins = [asdict(a) for a in
                          _build_admin_objects(scraper, ref.get("admin_organisations", []))]
            ckpt.save_administrations(raw_admins)
            ckpt.mark_done("administrations")
            log.info("[idarati] Saved %d administrations.", len(raw_admins))

        # -- Procedures --
        if ckpt.is_done("idarati"):
            log.info("[idarati] Procedures already done.")
        else:
            log.info("[idarati] Fetching procedures (up to %s)...",
                     args.idarati_limit or "all 2,533")
            already = ckpt.load_idarati()
            result  = scraper.scrape(
                limit           = args.idarati_limit,
                fetch_details   = True,
                already_scraped = already or None,
            )
            idarati_procs = [asdict(p) for p in result.procedures]
            ckpt.save_idarati(idarati_procs)
            ckpt.mark_done("idarati")
            log.info("[idarati] Saved %d procedures.", len(idarati_procs))

    # ══ HTML sites ════════════════════════════════════════════════════════════
    if not args.idarati_only:
        html_scraper = HTMLScraper(use_playwright=args.playwright)

        if ckpt.is_done("html"):
            log.info("[html] HTML scraping already done.")
        else:
            existing_html        = ckpt.load_html()
            already_visited_urls = {s.get("url", "") for s in existing_html}
            html_sections        = list(existing_html)

            sites_to_crawl = list(DEFAULT_HTML_SITES)
            if args.urls:
                sites_to_crawl.extend(args.urls)

            for site_url in sites_to_crawl:
                log.info("[html] Crawling: %s (max %d pages)", site_url, args.max_pages)
                pages = html_scraper.scrape_site(
                    base_url        = site_url,
                    max_pages       = args.max_pages,
                    already_visited = already_visited_urls,
                )
                for page in pages:
                    for section in page.sections:
                        html_sections.append({
                            "body":        section.body,
                            "url":         section.url,
                            "heading":     section.heading,
                            "lang_hint":   section.lang_hint,
                            "source_site": section.source_site,
                        })
                    already_visited_urls.add(page.url)
                ckpt.save_html(html_sections)

            ckpt.mark_done("html")
            log.info("[html] Saved %d HTML sections.", len(html_sections))

    # ── Summary ───────────────────────────────────────────────────────────────
    procs  = ckpt.load_idarati()
    admins = ckpt.load_administrations()
    html   = ckpt.load_html()
    log.info("")
    log.info("=" * 60)
    log.info("  SCRAPE COMPLETE")
    log.info("=" * 60)
    log.info("  Procedures       : %d", len(procs))
    log.info("  Administrations  : %d", len(admins))
    log.info("  HTML sections    : %d", len(html))
    log.info("  Run dir          : %s", ckpt.run_dir)
    log.info("")
    log.info("  Next step -- start Ollama/LM Studio, then run:")
    log.info("    python benchmarking/scraper/run_scraper.py --qa-only --resume")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: QA only  (Ollama must be running)
# ─────────────────────────────────────────────────────────────────────────────

def run_qa(args: argparse.Namespace) -> None:
    """
    Load scraped data from the latest checkpoint and generate Q&A pairs.
    Requires Ollama / LM Studio to be running.
    """
    from benchmarking.scraper.checkpoint import ScrapeCheckpoint
    from benchmarking.scraper.dataset_builder import DatasetBuilder
    from benchmarking.scraper.qa_generator import QAGenerator

    # Always resume -- QA step is meaningless without existing scraped data
    ckpt = ScrapeCheckpoint.find_latest(_RUNS_DIR)
    if ckpt is None:
        log.error("No scrape checkpoint found. Run step 1 first:")
        log.error("  python benchmarking/scraper/run_scraper.py --scrape-only --idarati-only")
        sys.exit(1)

    log.info("Loading scraped data from: %s", ckpt.run_dir)

    # ── Reconstruct sections from checkpoint ──────────────────────────────────
    # NOTE: Procedures come FIRST so that --limit N picks the richest content.
    # Administrations and HTML sections are appended after.
    all_sections: List[dict] = []

    # Procedures (richest content — must be first so --limit hits them)
    procs = ckpt.load_idarati()
    for proc in procs:
        body = proc.get("body_ar", "")
        if body and len(body.split()) >= 30:
            all_sections.append({
                "body":        body,
                "url":         proc.get("url_ar", ""),
                "heading":     proc.get("title_ar", ""),
                "lang_hint":   "ar",
                "source_site": "idarati.ma",
            })
    log.info("[qa] %d procedure sections loaded", len(all_sections))

    # HTML sections
    html_sections = ckpt.load_html()
    html_start = len(all_sections)
    all_sections.extend(html_sections)
    log.info("[qa] %d HTML sections loaded", len(all_sections) - html_start)

    # Administrations (shorter bodies — appended last)
    raw_admins = ckpt.load_administrations()
    admin_start = len(all_sections)
    for adm in raw_admins:
        body = adm.get("body", "")
        if len(body.split()) >= 30:
            all_sections.append({
                "body":        body,
                "url":         "https://idarati.ma/",
                "heading":     adm.get("title", ""),
                "lang_hint":   "ar",
                "source_site": "idarati.ma",
            })
    log.info("[qa] %d administration sections loaded", len(all_sections) - admin_start)

    log.info("[qa] Total sections available: %d", len(all_sections))

    # Apply --limit cap (for test runs)
    if getattr(args, "limit", None):
        all_sections = all_sections[: args.limit]
        log.info("[qa] --limit %d: capped to %d sections", args.limit, len(all_sections))

    log.info("[qa] Sections to process: %d", len(all_sections))

    # ── Q&A generation ────────────────────────────────────────────────────────
    qa_gen  = QAGenerator(ollama_base_url=args.ollama_url, model=args.model)
    builder = DatasetBuilder(gold_path=_GOLD_FILE,
                             fresh=getattr(args, "fresh_gold", False))

    existing_qa       = ckpt.load_qa()
    already_generated = len(existing_qa)

    if already_generated:
        log.info("[qa] Resuming from section %d / %d", already_generated + 1, len(all_sections))

    sections_to_process = all_sections[already_generated:]

    # Live output file — written after every LLM response so no data is lost on crash.
    # Uses a fixed name so resume appends to the same file, not a new timestamped one.
    live_path = _BENCH_DIR / f"scraped_dataset_live_{ckpt.run_dir.name}.json"

    _fresh = getattr(args, "fresh_gold", False)

    # Seed the live file with any items already generated in a previous run
    if existing_qa and not live_path.exists():
        _existing_items, _ = DatasetBuilder(gold_path=_GOLD_FILE, fresh=_fresh).build_items(existing_qa)
        _atomic_write(live_path, _existing_items)
        log.info("[qa] Live output seeded with %d existing items: %s",
                 len(_existing_items), live_path.name)

    def _save_progress(new_qa: List[dict]) -> None:
        """Called after every successful LLM response. Saves to both checkpoint and live file."""
        all_so_far = existing_qa + new_qa
        # 1. Checkpoint save (raw, for resume)
        ckpt.save_qa(all_so_far)
        # 2. Live output save (processed, with IDs — for visibility and crash safety)
        live_builder = DatasetBuilder(gold_path=_GOLD_FILE, fresh=_fresh)
        live_items, n_dropped = live_builder.build_items(all_so_far)
        _atomic_write(live_path, live_items)
        log.info("[qa] Live file: %d items saved to %s (%d dropped as dupes)",
                 len(live_items), live_path.name, n_dropped)

    new_qa_raw = qa_gen.generate_batch(
        sections           = sections_to_process,
        already_done_count = already_generated,
        progress_callback  = _save_progress,
    )

    all_qa_raw = existing_qa + new_qa_raw
    ckpt.save_qa(all_qa_raw)
    log.info("[qa] Generation complete: %d raw items", len(all_qa_raw))

    # ── Build + deduplicate + assign IDs + save ───────────────────────────────
    new_items, n_dupes = builder.build_items(all_qa_raw)
    log.info("[qa] After deduplication: %d new items (%d dropped)", len(new_items), n_dupes)

    sep_path = builder.save_separate(new_items, output_dir=_BENCH_DIR)
    log.info("[qa] Saved: %s", sep_path)

    # Remove live file now that the proper timestamped file exists
    if live_path.exists():
        live_path.unlink()
        log.info("[qa] Live file removed (superseded by %s)", sep_path.name)

    if not args.no_merge:
        builder.merge_into_gold(new_items)
        log.info("[qa] Gold testset updated.")
    else:
        log.info("[qa] --no-merge: gold testset NOT updated.")

    builder.print_summary(new_items, n_dupes)
    ckpt.finalize()
    log.info("Done. Run dir: %s", ckpt.run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Combined (original one-shot mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_all(args: argparse.Namespace) -> None:
    """Scrape + QA in a single run (original behaviour)."""
    from benchmarking.scraper.checkpoint import ScrapeCheckpoint
    from benchmarking.scraper.dataset_builder import DatasetBuilder
    from benchmarking.scraper.html_scraper import HTMLScraper
    from benchmarking.scraper.idarati_api import IdaratiAPIScraper
    from benchmarking.scraper.qa_generator import QAGenerator

    ckpt    = _get_checkpoint(args)
    qa_gen  = QAGenerator(ollama_base_url=args.ollama_url, model=args.model)
    builder = DatasetBuilder(gold_path=_GOLD_FILE)

    all_sections: List[dict] = []

    # ══ idarati.ma ════════════════════════════════════════════════════════════
    if not args.html_only:
        scraper = IdaratiAPIScraper()

        if ckpt.is_done("reference"):
            ref = ckpt.load_reference_data()
        else:
            ref = {
                "admin_types":            scraper.fetch_admin_types(),
                "admin_organisations":    scraper.fetch_admin_organisations(),
                "beneficiary_categories": scraper.fetch_beneficiary_categories(),
            }
            ckpt.save_reference_data(ref)
            ckpt.mark_done("reference")

        if ckpt.is_done("administrations"):
            raw_admins = ckpt.load_administrations()
        else:
            raw_admins = [asdict(a) for a in
                          _build_admin_objects(scraper, ref.get("admin_organisations", []))]
            ckpt.save_administrations(raw_admins)
            ckpt.mark_done("administrations")

        for adm in raw_admins:
            body = adm.get("body", "")
            if len(body.split()) >= 10:
                all_sections.append({
                    "body":        body,
                    "url":         "https://idarati.ma/",
                    "heading":     adm.get("title", ""),
                    "lang_hint":   "ar",
                    "source_site": "idarati.ma",
                })

        if ckpt.is_done("idarati"):
            idarati_procs = ckpt.load_idarati()
        else:
            already = ckpt.load_idarati()
            result  = scraper.scrape(
                limit           = args.idarati_limit,
                fetch_details   = True,
                already_scraped = already or None,
            )
            idarati_procs = [asdict(p) for p in result.procedures]
            ckpt.save_idarati(idarati_procs)
            ckpt.mark_done("idarati")

        for proc in idarati_procs:
            body = proc.get("body_ar", "")
            if body:
                all_sections.append({
                    "body":        body,
                    "url":         proc.get("url_ar", ""),
                    "heading":     proc.get("title_ar", ""),
                    "lang_hint":   "ar",
                    "source_site": "idarati.ma",
                })

        log.info("[idarati] %d total sections", len(all_sections))

    # ══ HTML sites ════════════════════════════════════════════════════════════
    if not args.idarati_only:
        html_scraper  = HTMLScraper(use_playwright=args.playwright)
        html_sections: List[dict] = []

        if ckpt.is_done("html"):
            html_sections = ckpt.load_html()
        else:
            existing_html        = ckpt.load_html()
            already_visited_urls = {s.get("url", "") for s in existing_html}
            html_sections        = list(existing_html)

            for site_url in (list(DEFAULT_HTML_SITES) + (args.urls or [])):
                pages = html_scraper.scrape_site(
                    base_url        = site_url,
                    max_pages       = args.max_pages,
                    already_visited = already_visited_urls,
                )
                for page in pages:
                    for section in page.sections:
                        html_sections.append({
                            "body":        section.body,
                            "url":         section.url,
                            "heading":     section.heading,
                            "lang_hint":   section.lang_hint,
                            "source_site": section.source_site,
                        })
                    already_visited_urls.add(page.url)
                ckpt.save_html(html_sections)
            ckpt.mark_done("html")

        all_sections.extend(html_sections)

    # ══ Q&A generation ════════════════════════════════════════════════════════
    log.info("Starting Q&A generation for %d sections...", len(all_sections))
    existing_qa       = ckpt.load_qa()
    already_generated = len(existing_qa)
    sections_to_process = all_sections[already_generated:]

    if already_generated:
        log.info("Resuming Q&A from section %d / %d", already_generated + 1, len(all_sections))

    def _save_progress(new_qa: List[dict]) -> None:
        ckpt.save_qa(existing_qa + new_qa)

    new_qa_raw = qa_gen.generate_batch(
        sections           = sections_to_process,
        already_done_count = already_generated,
        progress_callback  = _save_progress,
    )
    all_qa_raw = existing_qa + new_qa_raw
    ckpt.save_qa(all_qa_raw)
    log.info("Q&A generation complete: %d raw items", len(all_qa_raw))

    # ══ Build + save ══════════════════════════════════════════════════════════
    new_items, n_dupes = builder.build_items(all_qa_raw)
    log.info("After dedup: %d new items (%d dropped)", len(new_items), n_dupes)

    sep_path = builder.save_separate(new_items, output_dir=_BENCH_DIR)
    log.info("Saved: %s", sep_path)

    if not args.no_merge:
        builder.merge_into_gold(new_items)
    else:
        log.info("--no-merge: gold testset NOT updated.")

    builder.print_summary(new_items, n_dupes)
    ckpt.finalize()
    log.info("Done. Run dir: %s", ckpt.run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_checkpoint(args):
    from benchmarking.scraper.checkpoint import ScrapeCheckpoint
    if args.resume:
        ckpt = ScrapeCheckpoint.find_latest(_RUNS_DIR)
        if ckpt is None:
            log.warning("No incomplete run found -- starting fresh.")
            return ScrapeCheckpoint.new_run(_RUNS_DIR)
        log.info("Resuming: %s", ckpt.run_dir)
        return ckpt
    return ScrapeCheckpoint.new_run(_RUNS_DIR)


def _build_admin_objects(scraper, admin_orgs: list) -> list:
    from benchmarking.scraper.idarati_api import IdaratiAdministration
    proc_count_by_id = {a["id"]: a.get("proceduresCount", 0) for a in admin_orgs}
    raw_admins       = scraper.fetch_all_administrations()
    results = []
    for raw in raw_admins:
        aid   = raw.get("id", "")
        title = (raw.get("title") or "").strip()
        count = proc_count_by_id.get(aid, 0)
        body  = scraper._build_admin_body(raw, count)
        results.append(IdaratiAdministration(
            admin_id         = aid,
            title            = title,
            emails           = ", ".join(raw.get("emails", [])),
            phones           = ", ".join(raw.get("phones", [])),
            websites         = ", ".join(raw.get("websites", [])),
            is_active        = raw.get("isActive", True),
            procedures_count = count,
            body             = body,
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape idarati.ma + HTML sites and generate Q&A benchmark items.\n"
            "Two-step usage (recommended):\n"
            "  Step 1 (no Ollama needed): --scrape-only\n"
            "  Step 2 (Ollama required):  --qa-only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scrape-only", action="store_true",
                      help="Step 1: fetch all data, save to checkpoint. No Ollama needed.")
    mode.add_argument("--qa-only", action="store_true",
                      help="Step 2: load checkpoint, generate Q&A. Ollama must be running.")

    # ── Source control (ignored by --qa-only) ─────────────────────────────────
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--idarati-only", action="store_true",
                     help="Only idarati.ma API (procedures + administrations)")
    src.add_argument("--html-only", action="store_true",
                     help="Only crawl HTML sites (justice, cnss, --urls)")

    parser.add_argument("--idarati-limit", type=int, default=None, metavar="N",
                        help="Cap idarati.ma procedures at N (default: all 2,533)")
    parser.add_argument("--urls", nargs="+", metavar="URL",
                        help="Extra URLs to scrape via HTML crawl")
    parser.add_argument("--max-pages", type=int, default=30, metavar="N",
                        help="Max pages per HTML site (default: 30)")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--no-merge", action="store_true",
                        help="Write scraped_dataset_<ts>.json only; skip gold update")
    parser.add_argument("--fresh-gold", action="store_true",
                        help="Ignore existing benchmark_testset_gold.json — IDs start from S01. "
                             "Use when rebuilding the gold testset from scratch.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the latest interrupted run (auto-set for --qa-only)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N sections for QA (default: all)")

    # ── Browser ───────────────────────────────────────────────────────────────
    parser.add_argument("--playwright", action="store_true",
                        help="Use headless Chromium for --urls (requires playwright)")

    # ── Ollama ────────────────────────────────────────────────────────────────
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1",
                        help="Ollama/LM Studio base URL (default: http://localhost:11434/v1)")
    parser.add_argument("--model", default="gemma4:e4b",
                        help="Generator model (default: gemma4:e4b)")

    args = parser.parse_args()

    if args.scrape_only:
        run_scrape(args)
    elif args.qa_only:
        run_qa(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
