# Moroccan Government Scraper — Q&A Benchmark Dataset Builder

Crawls Moroccan government portals and uses the already-running Ollama model to
auto-generate Q&A pairs in the exact schema of `benchmark_testset_gold.json`.

Goal: grow the benchmark from ~60 hand-curated items to **100+ items** with balanced
coverage across categories (SIMPLE, MULTIHOP, LEGAL, DARIJA, ARABIZI, OUTSCOPE) and
languages (Arabic MSA, French, Darija, Arabizi).

---

## Target Sites

| Site | Tech | Strategy | Scale | Status |
|------|------|----------|-------|--------|
| `idarati.ma` | REST API (JSON) | API-based pagination | **2,533 procedures** | ✅ Works |
| `justice.gov.ma` | WordPress (static HTML) | BFS HTML crawl | ~hundreds of pages | ✅ Works |
| `cnss.ma` | Drupal (static HTML) | BFS HTML crawl + PDF links | ~hundreds of pages | ✅ Works |
| `service-public.ma` | Liferay + Cloudflare | — | — | ❌ DNS Error 1000 (blocked) |
| User URLs | varies | HTML crawl (`--playwright` for JS) | user-defined | ✅ Works |

### idarati.ma — REST API Discovery

The site exposes a public JSON API with **no authentication** and a fully open `robots.txt`:

```
GET https://idarati.ma/api/informational/procedures/search
    ?title=          ← MUST be empty string — title="*" returns only 12 (literal match)
    &pageSize=100
    &pageNumber={n}  ← 0-based

Response: { content: [{id, title, thematicId, administrationId, administrationTitle}],
            totalElements: 2533, totalPages: 26 }

GET https://idarati.ma/api/informational/categories-menu
    → full hierarchical category tree used to build category_path

Procedure detail pages (React SPA — content comes from JSON API, not HTML):
    Arabic: https://idarati.ma/informationnel/ar/{thematicId}/{procedureId}
    French: https://idarati.ma/informationnel/fr/thematique/{thematicId}/{procedureId}
```

26 API calls (pageSize=100) retrieve all 2,533 procedures in ~30 seconds.

> **Windows SSL note:** Moroccan government TLS certificates sometimes fail
> verification on Python 3.12 Windows. All scrapers use `session.verify=False`
> with `urllib3.disable_warnings()` to suppress the noise.

---

## Installation

Dependencies are already in `requirements.txt`:

```bash
pip install beautifulsoup4 lxml requests
```

**Optional — Playwright headless browser** (only needed for JS-heavy `--urls`):

```bash
pip install playwright
playwright install chromium
```

---

## Quick Start

```bash
# From the repo root (v12/)

# Fastest: idarati.ma API only — all 2,533 procedures
python benchmarking/scraper/run_scraper.py --idarati-only

# Quick test: cap at 50 procedures
python benchmarking/scraper/run_scraper.py --idarati-only --idarati-limit 50

# All sources (idarati API + justice.gov.ma + cnss.ma HTML crawl)
python benchmarking/scraper/run_scraper.py

# Add extra URLs
python benchmarking/scraper/run_scraper.py --urls "https://example.gov.ma/procedure/xyz"

# Resume after a crash
python benchmarking/scraper/run_scraper.py --resume

# Save standalone file only — do NOT update the gold testset
python benchmarking/scraper/run_scraper.py --no-merge
```

---

## All CLI Flags

```
python benchmarking/scraper/run_scraper.py [OPTIONS]

Source control (mutually exclusive):
  --idarati-only        Only scrape idarati.ma via REST API (fastest)
  --html-only           Only crawl HTML sites (justice, cnss, --urls)

Scope:
  --idarati-limit N     Cap idarati.ma at N procedures (default: all 2,533)
  --urls URL [URL ...]  Additional URLs to scrape via HTML crawl
  --max-pages N         Max pages crawled per HTML site (default: 30)

Output:
  --no-merge            Write scraped_dataset_<ts>.json only; skip gold testset update

Run control:
  --resume              Continue the latest interrupted scrape run
  --playwright          Use headless Chromium for --urls (requires playwright install)

Ollama / LLM:
  --ollama-url URL      Ollama base URL (default: http://localhost:11434/v1)
  --model NAME          Generator model (default: gemma4:e4b)
```

---

## Output Layout

```
benchmarking/
├── scraper/
│   └── runs/
│       └── scrape_<timestamp>/          ← checkpoint directory (one per run)
│           ├── raw_idarati.json         saved after every API page (100 procs)
│           ├── raw_html.json            saved after every crawled site
│           ├── generated_qa.json        saved after every LLM generation call
│           └── manifest.json            tracks completed phases + timestamps
│
├── scraped_dataset_<timestamp>.json     standalone new items (always written)
└── benchmark_testset_gold.json          updated in-place (unless --no-merge)
```

### scraped_dataset_\<ts\>.json schema (matches gold testset exactly)

```json
[
  {
    "id": "S042",
    "category": "SIMPLE",
    "language": "arabic_msa",
    "question": "ما هي الوثائق المطلوبة للحصول على...؟",
    "gold_answer": "وفقاً للنص، يجب تقديم...",
    "gold_keywords": ["وثيقة الهوية", "شهادة الميلاد"],
    "source": "https://idarati.ma/informationnel/ar/42/11434 — عنوان الإجراء",
    "should_abstain": false,
    "is_multihop": false,
    "expected_flags": {"SIMPLE": true, "MULTIHOP": false, "OUTSCOPE": false, "LEGAL": false, "language": "arabic_msa"},
    "expected_language": "arabic_msa",
    "expected_detected_language": "arabic_msa",
    "scraped": true,
    "scraped_at": "2026-05-13T14:30:00"
  }
]
```

---

## Checkpoint / Resume

Every write is **atomic** (write to `.tmp` → `os.replace()`). A crash at any point
loses at most one LLM call. To resume:

```bash
python benchmarking/scraper/run_scraper.py --resume
```

`--resume` finds the latest `runs/scrape_*/` directory where `manifest.json` does
**not** contain `"finalized": true`, reloads all saved data, and continues from where
it stopped — skipping procedures already fetched and Q&A already generated.

---

## Module Reference

| File | Description |
|------|-------------|
| `run_scraper.py` | CLI entry point — orchestrates all steps end-to-end |
| `idarati_api.py` | `IdaratiAPIScraper` — REST API pagination + bilingual detail page fetch |
| `html_scraper.py` | `HTMLScraper` — BFS crawler for static government sites (BS4 + lxml) |
| `qa_generator.py` | `QAGenerator` — calls Ollama to generate Q&A pairs from text sections |
| `dataset_builder.py` | `DatasetBuilder` — deduplication (85% token overlap), ID assignment, gold merge |
| `checkpoint.py` | `ScrapeCheckpoint` — atomic saves, resume logic, manifest tracking |
| `__init__.py` | Package marker |

---

## How Q&A Generation Works

For each text section (min 30 words) the LLM is asked to produce **2–4 items**:

| Type | Trigger |
|------|---------|
| `SIMPLE` | Always — factual: documents needed, deadlines, costs, eligibility |
| `LEGAL` | Section mentions a decree / article number (e.g. "المرسوم رقم 2.17.552") |
| `MULTIHOP` | Section describes a multi-step process or references another procedure |

- Arabic-language sections → `language: arabic_msa`
- French-language sections → `language: french`
- Each item is validated before acceptance (required fields, min word counts, allowed categories)
- Duplicates are dropped at build time using normalized token overlap (threshold: 85%)

---

## Running the Full Pipeline (Example)

```bash
# 1. Start Ollama / LM Studio (model: gemma4:e4b)

# 2. Quick smoke-test — 10 procedures, no gold update
python benchmarking/scraper/run_scraper.py \
    --idarati-only --idarati-limit 10 --no-merge

# 3. Check output
cat benchmarking/scraped_dataset_*.json | python -m json.tool | head -60

# 4. Full idarati run (all 2,533 procedures — takes several hours)
python benchmarking/scraper/run_scraper.py --idarati-only

# 5. Verify gold testset updated
python -c "import json; d=json.load(open('benchmarking/benchmark_testset_gold.json')); print(len(d), 'items')"

# 6. Run benchmarks with new dataset
python benchmarking/benchmark_runner.py --quick --no-ragas
```
