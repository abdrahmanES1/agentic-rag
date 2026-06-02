# -*- coding: utf-8 -*-
"""
HTMLScraper — crawls justice.gov.ma, cnss.ma, and arbitrary user-supplied URLs.

Strategy:
  - requests + BeautifulSoup4 (lxml) for static HTML (all target sites are
    server-side rendered — WordPress, Drupal — no JS rendering needed).
  - Optional Playwright fallback for JS-heavy URLs passed via --playwright.
  - BFS crawl within the same domain, following procedure/service/law links.
  - PDF links collected separately (useful for cnss.ma).
  - Sections split into h2/h3-delimited blocks (min 80 words) for LLM input.
"""

import logging
import time
import urllib.robotparser
from dataclasses import dataclass, field
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import urllib3
import requests
from bs4 import BeautifulSoup

# Government sites sometimes have certificate chain issues on Windows —
# suppress the InsecureRequestWarning that would otherwise flood the log.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger("scraper.html")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,fr;q=0.9,en;q=0.8",
}

# Keywords that suggest a URL is a procedure / service / legal-text page
_FOLLOW_KEYWORDS = [
    "procedure", "service", "prestation", "demarche",
    "loi", "decret", "arrete", "circulaire", "texte",
    "قانون", "مرسوم", "قرار", "خدمة", "إجراء",
]

# URL suffixes / segments to skip
_SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".mp4", ".mp3", ".zip", ".rar",
}

# CSS selectors tried in order to find main content (covers WordPress, Drupal, Liferay)
_CONTENT_SELECTORS = [
    "article",
    "main",
    ".entry-content",
    ".field--type-text-with-summary",
    ".journal-content-article",
    ".node__content",
    ".content-area",
    "section.content",
    "#content",
    ".post-content",
]

DELAY_SEC  = 1.5   # seconds between page requests
MIN_WORDS  = 80    # minimum words in a section for it to be useful


@dataclass
class Section:
    """A focused block of text from a web page — the unit given to the LLM."""
    heading: str
    body: str
    url: str
    lang_hint: str = ""     # "ar", "fr", or "" (auto-detected later)
    source_site: str = ""


@dataclass
class PageData:
    url: str
    title: str
    sections: List[Section] = field(default_factory=list)
    pdf_links: List[str] = field(default_factory=list)


class HTMLScraper:
    """
    BFS crawler for static government websites.
    Respects robots.txt; never follows off-domain links.
    """

    def __init__(
        self,
        delay: float = DELAY_SEC,
        use_playwright: bool = False,
    ) -> None:
        self.delay = delay
        self.use_playwright = use_playwright
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._session.verify = False   # Moroccan gov certs often fail on Windows Python 3.12
        self._robots: dict = {}     # base_url → RobotFileParser

    # ── robots.txt ────────────────────────────────────────────────────────────

    def _can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass
            self._robots[base] = rp
        return self._robots[base].can_fetch(_HEADERS["User-Agent"], url)

    # ── Fetch a single page ───────────────────────────────────────────────────

    def fetch_html(self, url: str) -> Optional[str]:
        """Return raw HTML string or None on failure."""
        if not self._can_fetch(url):
            log.debug("[html] robots.txt disallows: %s", url)
            return None
        try:
            if self.use_playwright:
                return self._fetch_playwright(url)
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            log.debug("[html] Fetch failed (%s): %s", url, exc)
            return None

    def _fetch_playwright(self, url: str) -> Optional[str]:
        """Playwright headless-browser fallback for JS-rendered pages."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright not installed. Run:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            html = page.content()
            browser.close()
        return html

    # ── Extract content from HTML ─────────────────────────────────────────────

    def parse_page(self, html: str, url: str) -> PageData:
        """Extract title, sections, and PDF links from raw HTML."""
        soup = BeautifulSoup(html, "lxml")

        # ── Title ──────────────────────────────────────────────────────────────
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # ── Detect language from html[lang] ────────────────────────────────────
        html_lang = soup.find("html", attrs={"lang": True})
        raw_lang = html_lang["lang"].lower() if html_lang else ""
        lang_hint = "ar" if raw_lang.startswith("ar") else ("fr" if raw_lang.startswith("fr") else "")

        # ── Remove noise: nav, header, footer, aside, scripts ─────────────────
        for tag in soup(["nav", "header", "footer", "aside", "script",
                         "style", "noscript", "form", "iframe"]):
            tag.decompose()

        # ── Find main content ──────────────────────────────────────────────────
        content_el = None
        for sel in _CONTENT_SELECTORS:
            content_el = soup.select_one(sel)
            if content_el:
                break
        if not content_el:
            content_el = soup.find("body") or soup

        # ── Split into sections ────────────────────────────────────────────────
        sections = self.extract_sections(content_el, url, lang_hint)

        # ── Collect PDF links ──────────────────────────────────────────────────
        pdf_links = self.collect_pdf_links(soup, url)

        parsed = urlparse(url)
        site = parsed.netloc

        for s in sections:
            s.source_site = site

        return PageData(url=url, title=title, sections=sections, pdf_links=pdf_links)

    def extract_sections(
        self, content_el, url: str, lang_hint: str = ""
    ) -> List[Section]:
        """
        Split content into heading → paragraph blocks (min MIN_WORDS words each).
        Falls back to one big section if no headings exist.
        """
        sections: List[Section] = []
        current_heading = ""
        current_body_parts: List[str] = []

        def _flush():
            body = "\n".join(current_body_parts).strip()
            if len(body.split()) >= MIN_WORDS:
                sections.append(Section(
                    heading=current_heading,
                    body=body,
                    url=url,
                    lang_hint=lang_hint,
                ))

        for el in content_el.descendants:
            if not hasattr(el, "name"):
                continue
            if el.name in ("h2", "h3", "h4"):
                _flush()
                current_heading = el.get_text(strip=True)
                current_body_parts = []
            elif el.name in ("p", "li", "td", "dd"):
                text = el.get_text(strip=True)
                if text:
                    current_body_parts.append(text)

        _flush()  # flush last section

        # If no heading-based sections were found, treat whole content as one section
        if not sections:
            body = content_el.get_text(separator="\n", strip=True)
            if len(body.split()) >= MIN_WORDS:
                sections.append(Section(heading=url, body=body, url=url, lang_hint=lang_hint))

        return sections

    def collect_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Return list of absolute PDF URLs found on the page."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                links.append(urljoin(base_url, href))
        return links

    # ── BFS site crawler ──────────────────────────────────────────────────────

    def scrape_site(
        self,
        base_url: str,
        max_pages: int = 30,
        already_visited: Optional[Set[str]] = None,
    ) -> List[PageData]:
        """
        BFS crawl from base_url, same domain only, following
        procedure/service/law links up to max_pages pages.
        """
        parsed = urlparse(base_url)
        domain = parsed.netloc

        visited: Set[str] = already_visited or set()
        queue = [base_url]
        pages: List[PageData] = []

        while queue and len(pages) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            # Skip non-HTML resources
            path = urlparse(url).path.lower()
            if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
                continue

            log.info("[html] Crawling (%d/%d): %s", len(pages) + 1, max_pages, url)
            html = self.fetch_html(url)
            if not html:
                time.sleep(self.delay)
                continue

            page_data = self.parse_page(html, url)
            if page_data.sections:
                pages.append(page_data)

            # Discover new links
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                link_parsed = urlparse(href)
                if link_parsed.netloc != domain:
                    continue
                if href in visited:
                    continue
                link_path = link_parsed.path.lower()
                if any(link_path.endswith(ext) for ext in _SKIP_EXTENSIONS):
                    continue
                # Prefer pages that look like content (optional filter)
                if _is_content_url(href):
                    queue.insert(0, href)   # content links go to front
                else:
                    queue.append(href)

            time.sleep(self.delay)

        log.info("[html] %s — crawled %d pages, %d total sections",
                 domain, len(pages), sum(len(p.sections) for p in pages))
        return pages

    def scrape_url(self, url: str) -> Optional[PageData]:
        """Scrape a single user-supplied URL."""
        html = self.fetch_html(url)
        if not html:
            return None
        return self.parse_page(html, url)


def _is_content_url(url: str) -> bool:
    """Return True if the URL path suggests it contains procedure/law content."""
    path = urlparse(url).path.lower()
    return any(kw in path for kw in _FOLLOW_KEYWORDS)
