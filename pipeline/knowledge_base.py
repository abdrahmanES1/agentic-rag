# -*- coding: utf-8 -*-
"""
Step 1-3 — Document loading, chunking, embedding, and indexing.
"""

import logging
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from pipeline.config import settings
from pipeline.models import Chunk

log = logging.getLogger("MoroccanRAG")

# ── BM25 tokenization ─────────────────────────────────────────────────────────

ARABIC_DIACRITICS = re.compile(
    r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]"
)
ARABIC_NORMALIZE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
FRENCH_NORMALIZE = str.maketrans({"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a", "ä": "a", "î": "i", "ï": "i", "ô": "o", "ö": "o", "ù": "u", "û": "u", "ü": "u", "ç": "c"})
FRENCH_STOPWORDS = {"le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "est", "au", "aux", "ce", "qui", "que", "pour", "par", "sur", "dans", "avec", "il", "elle", "ils", "elles", "je", "nous", "vous", "se", "son", "sa", "ses", "mon", "ma", "mes"}
ARABIC_STOPWORDS = {"في", "من", "إلى", "على", "أن", "أو", "لا", "هذا", "هذه", "التي", "الذي", "الذين", "كان", "كانت", "وقد", "قد", "لقد"}

AR_SENTENCE_SEP = re.compile(r"[،؟\n]+")
FR_SENTENCE_SEP = re.compile(r"[.!?\n]+")
_AR_ARTICLE_RE = re.compile(r"(?:^|\n)\s*(?:المادة|الفصل|الباب|القسم)\s+\d+", re.MULTILINE)
_FR_ARTICLE_RE = re.compile(r"(?:^|\n)\s*(?:Article|ART\.?|Chapitre|Titre|Section)\s+\d+", re.MULTILINE | re.IGNORECASE)


def arabic_tokenize(text: str) -> List[str]:
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.translate(ARABIC_NORMALIZE)
    text = re.sub(r"\bال", "", text)
    tokens = re.split(r"[^؀-ۿ]+", text)
    return [t for t in tokens if len(t) >= 2 and t not in ARABIC_STOPWORDS]


def french_tokenize(text: str) -> List[str]:
    text = text.lower().translate(FRENCH_NORMALIZE)
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2 and t not in FRENCH_STOPWORDS]


def tokenize_for_bm25(text: str, language: str) -> List[str]:
    if language in ("arabic_msa", "mixed", "Darija"):
        return arabic_tokenize(text)
    return french_tokenize(text)


# ── Document loading ──────────────────────────────────────────────────────────


def load_documents(pdf_dir: str) -> List[Dict]:
    """Load PDFs (three-tier OCR) + plain-text .txt files (no OCR needed)."""
    pdf_dir_path = Path(pdf_dir)
    pdf_files  = list(pdf_dir_path.glob("*.pdf"))
    txt_files  = list(pdf_dir_path.glob("*.txt"))

    if not pdf_files and not txt_files:
        log.warning(f"No PDFs or .txt files in {pdf_dir} — using built-in sample documents")
        return _create_sample_documents()

    pages = []

    # ── PDFs ──────────────────────────────────────────────────────────────────
    log.info(f"Found {len(pdf_files)} PDF files")
    for pdf_path in pdf_files:
        log.info(f"Loading PDF: {pdf_path.name}")
        file_pages = _load_pdf(pdf_path)
        pages.extend(file_pages)
        log.info(f"  Pages extracted: {len(file_pages)}")

    # ── Plain-text files (pre-extracted, no OCR) ──────────────────────────────
    if txt_files:
        log.info(f"Found {len(txt_files)} .txt files — loading directly (no OCR)")
        for txt_path in txt_files:
            txt_pages = _load_txt(txt_path)
            pages.extend(txt_pages)
        log.info(f"  .txt pages loaded: {sum(1 for _ in txt_files)}")

    log.info(f"Total pages: {len(pages)}")
    return pages


def _load_txt(txt_path: Path) -> List[Dict]:
    """
    Load a plain-text document produced by the idarati.ma exporter.

    Expected format (each section separated by a line of 60 dashes):
        Title: <procedure title>
        URL: <source URL>
        ---
        <body text>
        ------------------------------------------------------------
        Title: ...
    Falls back to treating the whole file as one document if format is absent.
    """
    try:
        raw = txt_path.read_text(encoding="utf-8")
    except Exception as e:
        log.warning(f"Could not read {txt_path.name}: {e}")
        return []

    separator = "-" * 60
    sections  = raw.split(separator)
    pages     = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines  = section.splitlines()
        title  = ""
        url    = ""
        body_lines = []
        in_body = False
        for line in lines:
            if line.startswith("Title: ") and not in_body:
                title = line[7:].strip()
            elif line.startswith("URL: ") and not in_body:
                url = line[5:].strip()
            elif line.strip() == "---":
                in_body = True
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if len(body.split()) < 10:
            continue
        source = url or txt_path.stem
        pages.append({
            "text":   _clean_text(body),
            "source": source,
            "page":   1,
            "is_ocr": False,
            "title":  title,
        })

    if not pages:
        # Fallback: whole file as one chunk
        body = raw.strip()
        if len(body.split()) >= 10:
            pages.append({
                "text":   _clean_text(body),
                "source": txt_path.stem,
                "page":   1,
                "is_ocr": False,
            })

    return pages


def _load_pdf(pdf_path: Path) -> List[Dict]:
    pages = []
    qwen_used = tesseract_used = pymupdf_used = 0
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text").strip()
            confidence = _ocr_confidence(text)

            if confidence > 0.85:
                pages.append({"text": _clean_text(text), "source": pdf_path.name, "page": page_num + 1, "is_ocr": False})
                pymupdf_used += 1
                continue

            mat = pymupdf.Matrix(settings.ocr_scale_factor, settings.ocr_scale_factor)
            img_bytes = page.get_pixmap(matrix=mat).tobytes("png")
            qwen_text = _ocr_with_qwen(img_bytes, page_num + 1)
            if qwen_text and len(qwen_text) >= settings.min_digital_chars:
                pages.append({"text": _clean_text(qwen_text), "source": pdf_path.name, "page": page_num + 1, "is_ocr": True})
                qwen_used += 1
                continue

            tesseract_text = _ocr_tesseract(img_bytes, pdf_path.name)
            if tesseract_text:
                pages.append({"text": _clean_text(tesseract_text), "source": pdf_path.name, "page": page_num + 1, "is_ocr": True})
                tesseract_used += 1
            else:
                log.warning(f"  Page {page_num + 1}: all OCR methods failed")
        doc.close()
        log.info(f"  {pdf_path.name}: pymupdf={pymupdf_used} qwen={qwen_used} tesseract={tesseract_used} failed={doc.page_count - len(pages)}")
    except Exception as e:
        log.error(f"PDF loading failed for {pdf_path.name}: {e}")
    return pages


def _ocr_confidence(text: str) -> float:
    if not text or len(text.strip()) < 50:
        return 0.0
    arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
    latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 0x0600)
    total_chars = arabic_chars + latin_chars
    if total_chars == 0:
        return 0.0
    ar_ratio = arabic_chars / total_chars
    lang_balance = 1.0 - abs(0.5 - ar_ratio)
    words = text.split()
    if len(words) < 10:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_quality = min(avg_word_len / 8.0, 1.0)
    char_diversity = min(len(set(text)) / 100.0, 1.0)
    return max(0.0, min(1.0, lang_balance * 0.4 + word_quality * 0.3 + char_diversity * 0.3))


def _ocr_with_qwen(page_image_bytes: bytes, page_num: int) -> str:
    try:
        import base64
        from openai import OpenAI
        client = OpenAI(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key)
        img_b64 = base64.b64encode(page_image_bytes).decode("utf-8")
        system_prompt = (
            "Extract ALL text from this document image.\n"
            "Requirements:\n"
            "1. Preserve both Arabic and French text exactly as written\n"
            "2. Keep article/section markers: المادة، الفصل، Article, Chapitre, etc.\n"
            "3. Extract numbers and dates exactly\n"
            "4. Output ONLY the extracted text."
        )
        response = client.chat.completions.create(
            model="qwen/qwen3-vl-8b",
            messages=[{"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}],
            max_tokens=4096,
            temperature=0.0,
            # Ollama passes generation options via extra_body.
            # repeat_penalty > 1.0 breaks repetition loops where the model
            # gets stuck on a single token (e.g. "الجمهورية الجمهورية الجمهورية...")
            extra_body={"options": {"repeat_penalty": 1.15, "repeat_last_n": 64}},
        )
        raw = (response.choices[0].message.content or "").strip()
        return _deloop_ocr(raw)
    except Exception as e:
        log.debug(f"Qwen OCR failed on page {page_num}: {e}")
        return ""


def _deloop_ocr(text: str) -> str:
    """
    Post-process OCR output to remove token-repetition loops.

    Detects runs of the same word/token repeated consecutively more than
    3 times and collapses them to a single occurrence.  Also removes lines
    that are pure repetitions of a single short token (common Qwen loop artifact).

    Examples caught:
      "الجمهورية الجمهورية الجمهورية الجمهورية..."  →  "الجمهورية"
      "و و و و و و و و و و..."                     →  removed
    """
    import re

    # 1. Collapse consecutive duplicate tokens (word-level)
    #    Replace 3+ repetitions of the same word with a single copy.
    text = re.sub(r'\b(\S+)(?:\s+\1){3,}\b', r'\1', text)

    # 2. Remove lines that are nothing but one short token repeated
    clean_lines = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 6 and len(set(tokens)) == 1:
            # Entire line is the same token repeated — skip it
            log.debug("[ocr] Removed repetition-loop line: %r", line[:60])
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines)


def _ocr_tesseract(img_bytes: bytes, filename: str) -> str:
    try:
        import io
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        lang = settings.tesseract_lang
        if "_ar" in filename.lower():
            lang = "ara"
        elif "_fr" in filename.lower():
            lang = "fra"
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as exc:
        log.debug("Tesseract OCR failed: %s", exc)
        return ""


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s؀-ۿÀ-ɏ.,،؟?!:;«»()\-/]", "", text)
    return text.strip()


def _create_sample_documents() -> List[Dict]:
    return [
        {"text": "الحصول على البطاقة الوطنية للتعريف الإلكترونية CNIE. الوثائق المطلوبة: شهادة الميلاد الكاملة، صورتان فوتوغرافيتان، إثبات الإقامة. مدة الإنجاز: 30 يوم عمل. الرسوم: 75 درهم.", "source": "sample_cnie_ar.pdf", "page": 1, "is_ocr": False},
        {"text": "Obtenir la Carte Nationale d'Identité Électronique CNIE. Documents requis: acte de naissance complet, deux photos d'identité récentes, justificatif de domicile. Délai: 30 jours ouvrables. Frais: 75 dirhams.", "source": "sample_cnie_fr.pdf", "page": 1, "is_ocr": False},
    ]


# ── Chunking ──────────────────────────────────────────────────────────────────


def chunk_documents(pages: List[Dict]) -> Tuple[List[Chunk], List[Chunk]]:
    """Article-boundary aware chunking (FIX 68)."""
    arabic_chunks: List[Chunk] = []
    french_chunks: List[Chunk] = []
    chunk_counter = 0

    for page in pages:
        text = page["text"]
        if len(text.split()) < settings.min_page_words:
            continue
        page_lang = _detect_chunk_language(text)
        segments = _split_by_articles(text, page_lang)

        for seg in segments:
            ar, fr, chunk_counter = _make_chunks_from_segment(
                seg, page["source"], page["page"], page["is_ocr"], chunk_counter
            )
            arabic_chunks.extend(ar)
            french_chunks.extend(fr)

    log.info(f"Chunking: {chunk_counter} total ({len(arabic_chunks)} ar, {len(french_chunks)} fr)")
    return arabic_chunks, french_chunks


def _detect_chunk_language(text: str) -> str:
    ar = sum(1 for c in text if "؀" <= c <= "ۿ")
    la = sum(1 for c in text if "a" <= c.lower() <= "z")
    total = sum(1 for c in text if not c.isspace())
    if total == 0:
        return "arabic_msa"
    ar_ratio, la_ratio = ar / total, la / total
    if ar_ratio > 0.55:
        return "arabic_msa"
    if la_ratio > 0.65:
        return "french"
    if ar_ratio > 0.2 and la_ratio > 0.2:
        return "mixed"
    return "arabic_msa"


def _split_by_articles(text: str, language: str) -> List[str]:
    pattern = _AR_ARTICLE_RE if language in ("arabic_msa", "mixed") else _FR_ARTICLE_RE
    boundaries = [m.start() for m in pattern.finditer(text)]
    if not boundaries:
        return [text]
    segments = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)
    if boundaries[0] > 0:
        preamble = text[: boundaries[0]].strip()
        if len(preamble.split()) >= 10:
            segments.insert(0, preamble)
    return segments


def _extract_article_metadata(segment: str, language: str) -> Tuple[str, str]:
    article_num = ""
    law_name = ""
    ar_art = re.search(r"(?:المادة|الفصل|الباب|القسم)\s+(\d+(?:\s*مكرر)?)", segment)
    fr_art = re.search(r"(?:Article|ART)\.?\s*(\d+(?:\s*bis)?)", segment, re.IGNORECASE)
    if ar_art:
        article_num = ar_art.group(1).strip()
    elif fr_art:
        article_num = fr_art.group(1).strip()
    ar_law = re.search(r"(?:قانون|ظهير|مرسوم)\s+(?:رقم|شريف)?\s*[\d\.\-]+", segment)
    fr_law = re.search(r"(?:loi|décret|arrêté)\s+n[°\.]?\s*[\d\-]+", segment, re.IGNORECASE)
    if ar_law:
        law_name = ar_law.group(0).strip()
    elif fr_law:
        law_name = fr_law.group(0).strip()
    return article_num, law_name


def _split_into_sentences(text: str, language: str) -> List[str]:
    sep = AR_SENTENCE_SEP if language in ("arabic_msa", "mixed") else FR_SENTENCE_SEP
    return [s.strip() for s in sep.split(text) if s.strip()]


def _make_chunks_from_segment(
    segment: str, source: str, page: int, is_ocr: bool, chunk_counter: int
) -> Tuple[List[Chunk], List[Chunk], int]:
    arabic_out: List[Chunk] = []
    french_out: List[Chunk] = []
    lang = _detect_chunk_language(segment)
    art_num, law_nm = _extract_article_metadata(segment, lang)

    def _save(text: str) -> None:
        nonlocal chunk_counter
        chunk_lang = _detect_chunk_language(text)
        chunk = Chunk(
            text=text, source=source, page=page, language=chunk_lang,
            chunk_id=f"chunk_{chunk_counter:04d}", is_ocr=is_ocr,
            article_number=art_num, law_name=law_nm,
        )
        if chunk_lang in ("arabic_msa", "mixed"):
            arabic_out.append(chunk)
        else:
            french_out.append(chunk)
        chunk_counter += 1

    words = segment.split()
    if len(words) <= settings.chunk_size:
        if len(words) >= settings.min_chunk_words:
            _save(segment)
    else:
        sentences = _split_into_sentences(segment, lang)
        current_words: List[str] = []
        for sentence in sentences:
            s_words = sentence.split()
            if len(current_words) + len(s_words) > settings.chunk_size and current_words:
                _save(" ".join(current_words))
                current_words = current_words[-settings.chunk_overlap:] + s_words
            else:
                current_words.extend(s_words)
        if len(current_words) >= settings.min_chunk_words:
            _save(" ".join(current_words))

    return arabic_out, french_out, chunk_counter


# ── Knowledge Base ────────────────────────────────────────────────────────────


class KnowledgeBase:
    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.arabic_chunks: List[Chunk] = []
        self.arabic_faiss: Optional[faiss.Index] = None
        self.arabic_bm25: Optional[BM25Okapi] = None
        self.arabic_embeddings: Optional[np.ndarray] = None
        self.french_chunks: List[Chunk] = []
        self.french_faiss: Optional[faiss.Index] = None
        self.french_bm25: Optional[BM25Okapi] = None
        self.french_embeddings: Optional[np.ndarray] = None
        # Unified cross-lingual index (FIX 70)
        self.all_chunks: List[Chunk] = []
        self.unified_faiss: Optional[faiss.Index] = None
        self.unified_embeddings: Optional[np.ndarray] = None

    def build(
        self,
        arabic_chunks: List[Chunk],
        french_chunks: List[Chunk],
        ollama=None,
    ) -> None:
        self.arabic_chunks = arabic_chunks
        self.french_chunks = french_chunks
        self.all_chunks = arabic_chunks + french_chunks

        enriched_ar: Optional[List[str]] = None
        enriched_fr: Optional[List[str]] = None

        if arabic_chunks:
            if ollama and settings.enable_contextual_retrieval:
                enriched_ar = self._enrich_chunks(arabic_chunks, ollama)
            self.arabic_embeddings, self.arabic_faiss = self._build_faiss(arabic_chunks, "Arabic", enriched_ar)
            self.arabic_bm25 = self._build_bm25(arabic_chunks, "Arabic", enriched_ar)

        if french_chunks:
            if ollama and settings.enable_contextual_retrieval:
                enriched_fr = self._enrich_chunks(french_chunks, ollama)
            self.french_embeddings, self.french_faiss = self._build_faiss(french_chunks, "French", enriched_fr)
            self.french_bm25 = self._build_bm25(french_chunks, "French", enriched_fr)

        if self.all_chunks:
            all_enriched: Optional[List[str]] = None
            if enriched_ar is not None or enriched_fr is not None:
                all_enriched = (enriched_ar or [c.text for c in arabic_chunks]) + (enriched_fr or [c.text for c in french_chunks])
            self.unified_embeddings, self.unified_faiss = self._build_faiss(self.all_chunks, "Unified AR+FR", all_enriched)

        log.info(f"KB built: {len(self.all_chunks)} total chunks")

    def _enrich_chunks(self, chunks: List[Chunk], ollama) -> List[str]:
        """Contextual Retrieval (Anthropic 2024) — FIX 69."""
        log.info(f"  Contextual enrichment: {len(chunks)} chunks")
        enriched: List[str] = []
        for chunk in chunks:
            meta_parts = []
            if chunk.law_name:
                meta_parts.append(chunk.law_name)
            if chunk.article_number:
                label = f"المادة {chunk.article_number}" if chunk.language in ("arabic_msa", "mixed") else f"Article {chunk.article_number}"
                meta_parts.append(label)
            meta_parts.append(f"[{chunk.source}]")
            meta = " — ".join(meta_parts)

            if chunk.language in ("arabic_msa", "mixed", "Darija"):
                prompt = f"أنت مساعد لبناء قاعدة بيانات. اكتب جملة أو جملتين تصفان هذا المقطع وتضعانه في سياق وثيقته.\n\nالسياق: {meta}\nالمقطع:\n{chunk.text[:600]}\n\nاكتب السياق فقط بالعربية (جملة أو جملتان):"
            else:
                prompt = f"Vous êtes un assistant pour construire une base de données. Rédigez une ou deux phrases situant ce passage dans son document.\n\nContexte: {meta}\nPassage:\n{chunk.text[:600]}\n\nContexte uniquement (une ou deux phrases):"

            try:
                context = ollama.generate([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=600, think=False, num_ctx=2048)
                if context and len(context.strip()) > 10:
                    enriched.append(f"{context.strip()} {chunk.text}")
                else:
                    enriched.append(chunk.text)
            except Exception as e:
                log.warning(f"  Enrichment failed for {chunk.chunk_id}: {e}")
                enriched.append(chunk.text)
        return enriched

    def _build_faiss(self, chunks: List[Chunk], label: str, enriched_texts: Optional[List[str]] = None):
        texts = enriched_texts if enriched_texts else [c.text for c in chunks]
        all_embs = []
        for i in range(0, len(texts), 32):
            embs = self.model.encode(texts[i:i + 32], normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(embs)
        embeddings = np.vstack(all_embs).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        src = "enriched" if enriched_texts else "original"
        log.info(f"  {label} FAISS: {index.ntotal} vectors ({src})")
        return embeddings, index

    def _build_bm25(self, chunks: List[Chunk], label: str, enriched_texts: Optional[List[str]] = None) -> BM25Okapi:
        if enriched_texts:
            tokenized = [tokenize_for_bm25(e, c.language) for e, c in zip(enriched_texts, chunks)]
        else:
            tokenized = [tokenize_for_bm25(c.text, c.language) for c in chunks]
        log.info(f"  {label} BM25: {len(tokenized)} documents")
        return BM25Okapi(tokenized)

    def save(self, index_dir: str) -> None:
        idx = Path(index_dir)
        idx.mkdir(exist_ok=True)
        for lang in ("arabic", "french"):
            fi = getattr(self, f"{lang}_faiss")
            if fi:
                faiss.write_index(fi, str(idx / f"{lang}.faiss"))
                np.save(str(idx / f"{lang}_embs.npy"), getattr(self, f"{lang}_embeddings"))
                with open(idx / f"{lang}_chunks.pkl", "wb") as f:
                    pickle.dump(getattr(self, f"{lang}_chunks"), f)
                with open(idx / f"{lang}_bm25.pkl", "wb") as f:
                    pickle.dump(getattr(self, f"{lang}_bm25"), f)
        if self.unified_faiss is not None:
            faiss.write_index(self.unified_faiss, str(idx / "unified.faiss"))
            np.save(str(idx / "unified_embs.npy"), self.unified_embeddings)
            with open(idx / "all_chunks.pkl", "wb") as f:
                pickle.dump(self.all_chunks, f)
        log.info(f"KB saved to {index_dir}")

    def load(self, index_dir: str) -> bool:
        idx = Path(index_dir)
        if not (idx / "arabic.faiss").exists() and not (idx / "french.faiss").exists():
            return False
        for lang in ("arabic", "french"):
            fp = idx / f"{lang}.faiss"
            if fp.exists():
                setattr(self, f"{lang}_faiss", faiss.read_index(str(fp)))
                setattr(self, f"{lang}_embeddings", np.load(str(idx / f"{lang}_embs.npy")))
                with open(idx / f"{lang}_chunks.pkl", "rb") as f:
                    setattr(self, f"{lang}_chunks", pickle.load(f))
                with open(idx / f"{lang}_bm25.pkl", "rb") as f:
                    setattr(self, f"{lang}_bm25", pickle.load(f))
        unified_fp = idx / "unified.faiss"
        if unified_fp.exists():
            self.unified_faiss = faiss.read_index(str(unified_fp))
            self.unified_embeddings = np.load(str(idx / "unified_embs.npy"))
            with open(idx / "all_chunks.pkl", "rb") as f:
                self.all_chunks = pickle.load(f)
            log.info(f"  Unified FAISS: {self.unified_faiss.ntotal} vectors")
        else:
            self.all_chunks = self.arabic_chunks + self.french_chunks
        log.info(f"KB loaded from {index_dir}: {len(self.all_chunks)} total chunks")
        return True
