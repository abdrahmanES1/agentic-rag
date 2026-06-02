# Moroccan Government-Services Multilingual RAG Benchmark — Dataset Card

**Version:** 1.0  **Benchmark items:** 124  **Candidate pool:** 624  **Released:** 2026-05
**Benchmark file:** `benchmark_testset_v1.0.json` — **SHA256** `2b22e5510b36a3c0…`
**Candidate pool:** `benchmark_testset_gold.json` — **SHA256** `5b35bbfbe1f32478…`

> A multilingual (Modern Standard Arabic, French, Moroccan Darija, and Arabizi)
> question-answering benchmark over Moroccan government administrative procedures,
> for evaluating retrieval-augmented generation (RAG) systems. Each item is
> language-consistent: the question, gold answer, and gold keywords are all in
> the same language.
>
> The **evaluation benchmark is 124 items** (`v1.0`), curated for language balance
> from a **624-item annotated candidate pool** (`gold`). Both are released: cite/run
> the 124-item benchmark; the 624 pool documents construction rigor and supports
> extended evaluation.

This card follows the *Datasheets for Datasets* structure (Gebru et al., 2021).

---

## 1. Motivation

- **Purpose.** Evaluate RAG systems on real Moroccan government-service procedures
  in the languages citizens actually use — including **Moroccan Darija** (colloquial
  Arabic) and **Arabizi** (romanized Darija), which are absent from existing Arabic
  QA benchmarks that cover only Modern Standard Arabic (MSA).
- **Gap addressed.** Public Arabic RAG benchmarks are MSA-only and rarely include a
  French track or an explicit *out-of-scope / abstention* track. This benchmark adds
  Darija, Arabizi, a French track, legal-citation questions, multi-hop questions, and
  abstention questions over the same procedure corpus.

## 2. Composition

### 2a. Evaluation benchmark — `benchmark_testset_v1.0.json` (124 items)

Language-balanced for a multilingual benchmark. French / Darija / Arabizi are drawn
from **28 shared (parallel) source procedures** — the same procedure asked in three
languages — which enables cross-lingual consistency analysis. `arabic_msa` carries the
capability tracks that exist only in Arabic (abstention / multi-hop / legal-citation).

| Language | Count | | Category | Count |
|---|---:|---|---|---:|
| arabic_msa | 40 | | SIMPLE (french)   | 28 |
| french     | 28 | | DARIJA            | 28 |
| Darija     | 28 | | ARABIZI           | 28 |
| Arabizi    | 28 | | OUTSCOPE (abstain)| 15 |
|            |    | | LEGAL             | 13 |
|            |    | | MULTIHOP          | 12 |

- Language mix ≈ 32% MSA / 22.6% each fr·Darija·Arabizi.
- Reproducible: `make_sample.py` (seed = 42) regenerates this exact 124-item set.

### 2b. Candidate pool — `benchmark_testset_gold.json` (624 items)

The full annotated set the benchmark was curated from, over **137 source procedures**.

| Category | Count | | Language | Count |
|---|---:|---|---|---:|
| SIMPLE   | 201 | | arabic_msa | 309 |
| OUTSCOPE | 156 | | Darija     | 107 |
| DARIJA   | 107 | | french     | 106 |
| ARABIZI  | 102 | | Arabizi    | 102 |
| MULTIHOP |  45 | | | |
| LEGAL    |  13 | | | |

- Pool `should_abstain = true`: 156 (all OUTSCOPE); `is_multihop = true`: 45.
- **Note:** the capability categories (OUTSCOPE, MULTIHOP, LEGAL) exist only in
  `arabic_msa`; French/Darija/Arabizi are SIMPLE "documents/cost/deadline" questions.
  This is why the benchmark uses MSA for those tracks and the other three for the
  parallel multilingual track.

**Per-item fields**

| Field | Type | Meaning |
|---|---|---|
| `id` | str | Stable identifier (e.g. `S01`, `D34`, `A89`, `O30`, `L05`, `M10`) |
| `category` | str | SIMPLE / OUTSCOPE / DARIJA / ARABIZI / MULTIHOP / LEGAL |
| `language` | str | arabic_msa / french / Darija / Arabizi |
| `question` | str | The user question (in `language`) |
| `gold_answer` | str | Reference answer (in `language`; an abstention for OUTSCOPE) |
| `gold_keywords` | list[str] | Key facts a correct answer must contain (in `language`) |
| `source` | str | Source URL / PDF + procedure title |
| `should_abstain` | bool | True iff the answer is not derivable from the corpus |
| `is_multihop` | bool | True iff combining ≥2 procedures is required |
| `expected_flags` | obj | Expected classifier flags (SIMPLE/MULTIHOP/LEGAL/OUTSCOPE/language) |
| `expected_language` | str | Expected detected language (== `language`) |
| `expected_detected_language` | str | Same; for the language-detection sub-evaluation |
| `scraped`, `scraped_at` | bool/str | Provenance flags |

**Language-consistency guarantee (v1.0).** For every item,
`language(question) == language(gold_answer) == language(gold_keywords) == expected_language`.
Arabizi/French contain no Arabic script and no IPA diacritics; Darija answers use
Moroccan Darija phrasing (not MSA).

## 3. Source corpus

- **idarati.ma** — the official Moroccan public-service portal: ~2,500 administrative
  procedures (Arabic), scraped and exported to text.
- **Official PDFs** — decrees and codes, e.g. CNIE decree, biometric-passport decree,
  the Traffic Code (مدونة السير), the Civil Status law and its implementing decree.

## 4. Collection & annotation process

1. **Scrape** idarati.ma procedures + administration descriptions; load official PDFs
   (three-tier OCR cascade: digital text → VLM OCR → Tesseract).
2. **Generate** candidate QA per procedure with an LLM (`gemma`), targeting the six
   categories and four languages.
3. **Manual correction (this release).** Every Arabizi and French item was corrected
   by hand for language consistency:
   - Arabizi answers + keywords rewritten into natural Moroccan Darija Arabizi
     (numbers-for-Arabic-sounds convention; no Arabic script; no IPA).
   - French keywords translated from Arabic to French; Arabic removed from French answers.
   - Darija answers converted from MSA to Darija phrasing.
   - 15 questions referencing "this licence/procedure" without an antecedent were made
     self-contained by naming the procedure from its `source`.
   - Scraping artifacts removed (e.g. an answer that listed the same document 4×).
   - Per-language spell-check of all Darija and Arabizi questions.
4. **Audit.** A read-only auditor (`audit_gold.py`) checks language/script consistency,
   self-containment, and abstention consistency; remaining flags are documented false
   positives (see repo).

## 5. Recommended use

- **Primary metric tracks:** faithfulness (FActScore / ARES / RAGAS), answer quality
  (token-F1, ROUGE-L, BERTScore, G-Eval), retrieval quality, domain precision
  (legal-citation hit, cost/deadline hit), abstention accuracy (OUTSCOPE), and
  cross-lingual consistency (arabic_msa ↔ french on shared procedures).
- **Report per-category and per-language breakdowns.** Headline numbers use the
  **124-item `v1.0` benchmark**; the 624-item pool is available for extended/robustness
  runs. The benchmark is regenerated deterministically by `make_sample.py` (seed = 42).

## 6. Limitations & ethical notes

- Procedures reflect Moroccan administrative practice at scrape time and may change.
- Darija/Arabizi have no standardized orthography; spellings follow common usage and a
  documented convention, not a single official standard.
- LEGAL (13) is small by design (few procedures cite an exact decree); treat its metric
  as indicative.
- No personal data: questions are about generic procedures, not individuals.

## 7. Files & reproducibility

| File | Role |
|---|---|
| `benchmark_testset_v1.0.json` | **The benchmark — 124 items** (cite & run this; verify SHA256) |
| `benchmark_testset_gold.json` | Candidate pool — 624 annotated items |
| `benchmark_dataset_manifest.json` | Machine-readable stats + checksums + provenance (both files) |
| `CHECKSUMS.txt` | SHA256 of benchmark + pool |
| `make_sample.py` | Regenerates the 124 benchmark from the pool (seed=42) |
| `audit_gold.py` | Read-only language/consistency auditor |
| `DATASET_CARD.md` | This document |

**Run the benchmark:**
```
python benchmarking/benchmark_runner.py \
    --testset benchmarking/benchmark_testset_v1.0.json --with-gt --no-ragas
```

## 8. Citation

```bibtex
@misc{moroccan_rag_benchmark_2026,
  title  = {A Multilingual (MSA/French/Darija/Arabizi) RAG Benchmark over
            Moroccan Government Administrative Procedures},
  year   = {2026},
  note   = {Version 1.0 -- 124-item benchmark (SHA256 2b22e551...),
            curated from a 624-item annotated pool (SHA256 5b35bbfb...)}
}
```
