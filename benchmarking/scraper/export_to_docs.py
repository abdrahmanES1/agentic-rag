# -*- coding: utf-8 -*-
"""
export_to_docs.py — Export scraped idarati.ma procedures to ./docs as .txt files.

Creates two files in ./docs:
  idarati_procedures_ar.txt  — Arabic procedure texts  (~2,500 sections)
  idarati_administrations.txt — Administration descriptions (~2,400 sections)

Each section is separated by a 60-dash line so _load_txt() can split them.

Usage:
    python benchmarking/scraper/export_to_docs.py
    python benchmarking/scraper/export_to_docs.py --runs-dir benchmarking/scraper/runs
    python benchmarking/scraper/export_to_docs.py --docs-dir ./docs
"""

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent  # v12/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("export_to_docs")

_SEPARATOR = "-" * 60


def _section_block(title: str, url: str, body: str) -> str:
    return f"Title: {title}\nURL: {url}\n---\n{body}\n{_SEPARATOR}\n"


def export_procedures(checkpoint, docs_dir: Path) -> int:
    procs = checkpoint.load_idarati()
    if not procs:
        log.warning("No procedures found in checkpoint")
        return 0

    out_path = docs_dir / "idarati_procedures_ar.txt"
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for proc in procs:
            body = (proc.get("body_ar") or "").strip()
            if len(body.split()) < 10:
                continue
            title = (proc.get("title_ar") or proc.get("title") or "").strip()
            url   = (proc.get("url_ar") or proc.get("url") or "").strip()
            f.write(_section_block(title, url, body))
            count += 1

    log.info("Wrote %d procedures → %s", count, out_path)
    return count


def export_administrations(checkpoint, docs_dir: Path) -> int:
    admins = checkpoint.load_administrations()
    if not admins:
        log.warning("No administrations found in checkpoint")
        return 0

    out_path = docs_dir / "idarati_administrations.txt"
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for adm in admins:
            body = (adm.get("body") or "").strip()
            if len(body.split()) < 10:
                continue
            title = (adm.get("title") or "").strip()
            url   = "https://idarati.ma/"
            f.write(_section_block(title, url, body))
            count += 1

    log.info("Wrote %d administrations → %s", count, out_path)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Export scraped idarati.ma data to ./docs as .txt files for KB indexing."
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a specific scrape checkpoint directory (bypasses auto-detect)")
    parser.add_argument("--runs-dir", default=None,
                        help="Path to scraper/runs/ directory (default: auto-detect)")
    parser.add_argument("--docs-dir", default=None,
                        help="Path to ./docs directory (default: v12/docs)")
    parser.add_argument("--no-admins", action="store_true",
                        help="Skip administration export")
    args = parser.parse_args()

    from benchmarking.scraper.checkpoint import ScrapeCheckpoint

    docs_dir = Path(args.docs_dir) if args.docs_dir else (_ROOT / "docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
        if not ckpt_dir.exists():
            log.error("Checkpoint directory not found: %s", ckpt_dir)
            sys.exit(1)
        ckpt = ScrapeCheckpoint(ckpt_dir)
        log.info("Using checkpoint (explicit): %s", ckpt_dir.name)
    else:
        runs_dir = Path(args.runs_dir) if args.runs_dir else (_ROOT / "benchmarking" / "scraper" / "runs")
        ckpt = ScrapeCheckpoint.find_latest(runs_dir)
        if ckpt is None:
            log.error("No scrape checkpoint found in %s", runs_dir)
            log.error("Run scraping first: python benchmarking/scraper/run_scraper.py --scrape-only")
            sys.exit(1)

    log.info("Using checkpoint: %s", ckpt.run_dir.name)

    n_procs = export_procedures(ckpt, docs_dir)
    n_admins = 0 if args.no_admins else export_administrations(ckpt, docs_dir)

    log.info("")
    log.info("=" * 60)
    log.info("  EXPORT COMPLETE")
    log.info("=" * 60)
    log.info("  Procedures    : %d sections → idarati_procedures_ar.txt", n_procs)
    log.info("  Administrations: %d sections → idarati_administrations.txt", n_admins)
    log.info("  Output dir    : %s", docs_dir)
    log.info("")
    log.info("  Next step — rebuild the knowledge base:")
    log.info("    python -m api.app   # then POST /api/build-kb")
    log.info("    OR: pipe.build_knowledge_base(force_rebuild=True)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
