# -*- coding: utf-8 -*-
"""
IdaratiAPIScraper — fetches ALL data from idarati.ma's public REST API.

Discovered endpoints (via JS bundle inspection):

  ── Procedures ──────────────────────────────────────────────────────────────
  GET /api/informational/procedures/search?title=&pageSize=100&pageNumber={n}
      → { content: [{id, title, thematicId, administrationId, administrationTitle}],
          totalElements: 2533, totalPages: 26 }
      IMPORTANT: title="" (empty) = all 2,533; title="*" = only 12 (literal match).

  GET /api/informational/procedures/{id}
      → { id, locale, title, description, price, delay, averageDelay,
          receivingAdministrations, deliveringAdministrations,
          administrationInCharge, downloadableForms, downloadableCosts,
          webSiteUrl, hasComplementStep, validityDurationInDays, ... }

  ── Administrations ─────────────────────────────────────────────────────────
  GET /api/informational/administration/search?title=&pageSize=100&pageNumber={n}
      → { content: [{id, locale, title, hash, emails, phones, websites, isActive}],
          totalElements: 2446 }

  GET /api/informational/administration-type/search
      → [{id, locale, title, proceduresCount}]  — 208 named organisations

  GET /api/informational/administration-type
      → [{id, locale, title, description, administrationTypes}]  — 5 type categories

  ── Categories & beneficiaries ──────────────────────────────────────────────
  GET /api/informational/categories-menu
      → hierarchical tree: category → subCategory → thematic

  GET /api/informational/beneficiary-sub-category
      → [{id, title}]  — 10 beneficiary groups (individuals, companies, …)

  GET /api/informational/procedures/{id}/documents
      → [{id, title, section:{title}, sort, linkedProcedures:[...]}]
        section.title is e.g. "الوثائق المطلوبة" (Required Documents)
        Each document may link to other procedures (useful for MULTIHOP Q&A)

  ── Reference URLs (React SPA — content from API only, not HTML) ────────────
      Arabic: https://idarati.ma/informationnel/ar/thematique/{thematicId}/{procedureId}
      French: https://idarati.ma/informationnel/fr/thematique/{thematicId}/{procedureId}
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger("scraper.idarati")

BASE_URL            = "https://idarati.ma"
API_BASE            = f"{BASE_URL}/api/informational"
SEARCH_API          = f"{API_BASE}/procedures/search"
DETAIL_API          = f"{API_BASE}/procedures"
DOCUMENTS_API       = f"{API_BASE}/procedures"   # + /{id}/documents
CATEGORIES_API      = f"{API_BASE}/categories-menu"
ADMIN_SEARCH_API    = f"{API_BASE}/administration/search"
ADMIN_TYPE_API      = f"{API_BASE}/administration-type"
ADMIN_TYPE_SRCH_API = f"{API_BASE}/administration-type/search"
BENEFICIARY_API     = f"{API_BASE}/beneficiary-sub-category"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,fr;q=0.9,en;q=0.8",
}

DELAY_API  = 0.5   # seconds between paginated API calls
DELAY_PAGE = 1.0   # seconds between procedure detail API calls


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class IdaratiProcedure:
    """One idarati.ma procedure with its full Arabic content."""
    procedure_id:   str
    title_ar:       str
    thematic_id:    str
    administration: str
    body_ar:        str = ""   # assembled from all API fields
    url_ar:         str = ""
    url_fr:         str = ""
    category_path:  str = ""   # "Category > SubCategory > Thematic"


@dataclass
class IdaratiAdministration:
    """One government administration listed on idarati.ma."""
    admin_id:         str
    title:            str
    emails:           str = ""   # comma-separated
    phones:           str = ""   # comma-separated
    websites:         str = ""   # comma-separated
    is_active:        bool = True
    procedures_count: int = 0
    body:             str = ""   # assembled text body for Q&A


# ── Main scraper class ────────────────────────────────────────────────────────

class IdaratiAPIScraper:
    """
    Fetches everything from idarati.ma via its public JSON API.

    Data collected
    --------------
    • 2,533 procedures (description, price, delay, administrations, forms, …)
    • 2,446 administrations (contact info: emails, phones, websites)
    •   208 named admin organisations with procedure counts
    •     5 administration type categories
    •    10 beneficiary sub-categories
    • Full hierarchical category tree
    """

    def __init__(self, delay_api: float = DELAY_API, delay_page: float = DELAY_PAGE) -> None:
        self.delay_api  = delay_api
        self.delay_page = delay_page
        self._session   = requests.Session()
        self._session.headers.update(_HEADERS)
        self._session.verify = False   # Moroccan gov certs fail on Windows Python 3.12

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_json(self, url: str, params: Optional[dict] = None,
                  max_retries: int = 3) -> Optional[object]:
        """GET → parse JSON, with exponential back-off retry."""
        for attempt in range(1, max_retries + 1):
            try:
                r = self._session.get(url, params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                log.debug("[idarati] %s attempt %d/%d failed: %s",
                          url.split("idarati.ma")[-1], attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(1.0 * attempt)
        return None

    def _paginate(self, url: str, extra_params: Optional[dict] = None,
                  page_size: int = 100, limit: Optional[int] = None) -> List[dict]:
        """
        Paginate a search endpoint that returns {content, totalPages, totalElements}.
        Returns a flat list of all content items, deduplicated by 'id'.

        Uses 'size' and 'page' — the param names the idarati.ma JS app uses.
        ('pageSize'/'pageNumber' are silently ignored by the server and cause
        it to return only 5 items/page on every call, giving 507 duplicate pages.)
        """
        all_items: List[dict] = []
        seen_ids: set = set()
        page = 0
        while True:
            params = {"title": "", "size": page_size, "page": page}
            if extra_params:
                params.update(extra_params)
            data = self._get_json(url, params=params)
            if not data:
                break
            items = data.get("content", [])
            if not items:
                break

            # Deduplicate by 'id' — guard against server returning the same
            # page twice if pagination params are mishandled.
            new_items = []
            for item in items:
                item_id = item.get("id")
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                new_items.append(item)
            all_items.extend(new_items)

            total_pages    = data.get("totalPages", 1)
            total_elements = data.get("totalElements", len(all_items))
            if page == 0:
                log.info("[idarati] %s → %d total items across %d pages",
                         url.split("idarati.ma")[-1], total_elements, total_pages)
            log.info("[idarati] Page %d/%d — %d items so far",
                     page + 1, total_pages, len(all_items))
            if limit and len(all_items) >= limit:
                all_items = all_items[:limit]
                break
            page += 1
            if page >= total_pages:
                break
            time.sleep(self.delay_api)
        return all_items

    # ── Step 1: procedures metadata ───────────────────────────────────────────

    def fetch_all_metadata(self, page_size: int = 100,
                           limit: Optional[int] = None) -> List[dict]:
        """Paginate the search API and return all raw procedure metadata dicts."""
        return self._paginate(SEARCH_API, page_size=page_size, limit=limit)

    # ── Step 2: categories ────────────────────────────────────────────────────

    def fetch_categories(self) -> Dict[str, str]:
        """
        Returns {thematicId: "Category > SubCategory > Thematic"}.
        Used to populate category_path on each IdaratiProcedure.
        """
        tree = self._get_json(CATEGORIES_API)
        if not tree:
            return {}
        mapping: Dict[str, str] = {}
        for cat in (tree if isinstance(tree, list) else tree.get("categories", [])):
            cat_name = cat.get("title", "")
            for sub in cat.get("subCategories", []):
                sub_name = sub.get("title", "")
                for thematic in sub.get("thematics", []):
                    tid  = str(thematic.get("id", ""))
                    name = thematic.get("title", "")
                    mapping[tid] = " > ".join(filter(None, [cat_name, sub_name, name]))
        log.info("[idarati] Loaded %d thematic → category mappings", len(mapping))
        return mapping

    # ── Step 3: procedure detail + documents ─────────────────────────────────

    def fetch_detail_api(self, procedure_id: str) -> dict:
        """
        Call /api/informational/procedures/{id} and return the parsed JSON dict.
        Returns {} on failure.
        """
        data = self._get_json(f"{DETAIL_API}/{procedure_id}")
        return data if isinstance(data, dict) else {}

    def fetch_documents(self, procedure_id: str) -> List[dict]:
        """
        Call /api/informational/procedures/{id}/documents.

        Returns a list of required-document dicts, each with:
          - title      : document name (Arabic)
          - section    : {"title": "الوثائق المطلوبة", "sort": 0}
          - sort       : display order
          - linkedProcedures : list of related procedure summaries (for MULTIHOP Q&A)

        Returns [] on failure or when the procedure has no documents.
        """
        data = self._get_json(f"{DETAIL_API}/{procedure_id}/documents")
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _build_procedure_body(detail: dict,
                              documents: Optional[List[dict]] = None,
                              admin_lookup: Optional[Dict[str, dict]] = None) -> str:
        """
        Build a human-readable Arabic text body from the procedure API response.

        Combines:
          - title, description, price, delay
          - administrations (receiving, delivering, in-charge) + contact info
          - required documents list (الوثائق المطلوبة)
          - downloadable forms / costs
          - links to related procedures (from documents[].linkedProcedures)

        Typical output: 80–300 Arabic words.
        """
        if detail is None:
            return ""

        parts: List[str] = []

        def _add(label: str, value) -> None:
            v = (str(value) if value is not None else "").strip()
            if v:
                parts.append(f"{label}: {v}")

        _add("الإجراء",           detail.get("title"))
        _add("الوصف",             detail.get("description"))
        _add("التكلفة",           detail.get("price"))
        _add("المدة",             detail.get("delay"))

        if detail.get("validityDurationInDays"):
            parts.append(f"مدة الصلاحية: {detail['validityDurationInDays']} يوم")

        _add("الموقع الإلكتروني", detail.get("webSiteUrl"))

        recv = detail.get("receivingAdministrations") or []
        recv_names = "، ".join(a.get("title", "") for a in recv if a.get("title"))
        _add("الجهة المستقبلة", recv_names or None)

        deliv = detail.get("deliveringAdministrations") or []
        deliv_names = "، ".join(a.get("title", "") for a in deliv if a.get("title"))
        _add("الجهة المسلِّمة", deliv_names or None)

        admin_charge = detail.get("administrationInCharge") or {}
        _add("الجهة المسؤولة", admin_charge.get("title"))

        # Enrich with contact info from administration lookup
        if admin_lookup and admin_charge.get("id"):
            contact = admin_lookup.get(admin_charge["id"], {})
            emails   = ", ".join(contact.get("emails", []))
            phones   = ", ".join(contact.get("phones", []))
            websites = ", ".join(contact.get("websites", []))
            _add("البريد الإلكتروني", emails or None)
            _add("الهاتف",           phones or None)
            if websites:
                _add("المواقع الإلكترونية للجهة", websites)

        # ── Required documents (الوثائق المطلوبة) ─────────────────────────
        if documents:
            # Group by section title (some procedures have multiple sections)
            sections: Dict[str, List[str]] = {}
            linked_procedures: List[str] = []
            for doc in sorted(documents, key=lambda d: (d.get("section", {}).get("sort", 0),
                                                         d.get("sort", 0))):
                section_title = (doc.get("section") or {}).get("title") or "الوثائق المطلوبة"
                doc_title     = (doc.get("title") or "").strip()
                if doc_title:
                    sections.setdefault(section_title, []).append(doc_title)
                # Collect linked procedure titles for cross-reference context
                for lp in (doc.get("linkedProcedures") or []):
                    lp_title = (lp.get("title") or "").strip()
                    if lp_title and lp_title not in linked_procedures:
                        linked_procedures.append(lp_title)

            for sec_title, doc_list in sections.items():
                parts.append(f"\n{sec_title}:")
                for i, d in enumerate(doc_list, 1):
                    parts.append(f"  {i}. {d}")

            if linked_procedures:
                parts.append("\nإجراءات ذات صلة: " + "، ".join(linked_procedures))

        forms = detail.get("downloadableForms") or []
        form_names = "، ".join(f.get("title", f.get("url", "")) for f in forms if f)
        _add("النماذج القابلة للتحميل", form_names or None)

        costs = detail.get("downloadableCosts") or []
        cost_names = "، ".join(c.get("title", c.get("url", "")) for c in costs if c)
        _add("جداول التكاليف", cost_names or None)

        return "\n".join(parts)

    # ── Step 4: administrations ───────────────────────────────────────────────

    def fetch_all_administrations(self, page_size: int = 100) -> List[dict]:
        """
        Fetch all 2,446 administrations with contact info.
        Returns raw API dicts: {id, locale, title, emails, phones, websites, isActive}.
        """
        return self._paginate(ADMIN_SEARCH_API, page_size=page_size)

    def fetch_admin_organisations(self) -> List[dict]:
        """
        Fetch 208 named admin organisations with their procedure counts.
        Returns raw API dicts: {id, locale, title, proceduresCount}.
        """
        data = self._get_json(ADMIN_TYPE_SRCH_API)
        if isinstance(data, list):
            log.info("[idarati] Loaded %d admin organisations", len(data))
            return data
        return []

    def fetch_admin_types(self) -> List[dict]:
        """
        Fetch 5 top-level administration type categories.
        Returns raw API dicts: {id, locale, title, description, administrationTypes}.
        """
        data = self._get_json(ADMIN_TYPE_API)
        if isinstance(data, list):
            log.info("[idarati] Loaded %d administration types", len(data))
            return data
        return []

    def fetch_beneficiary_categories(self) -> List[dict]:
        """
        Fetch 10 beneficiary sub-categories (who the procedures serve).
        Returns raw API dicts: {id, title}.
        """
        data = self._get_json(BENEFICIARY_API)
        if isinstance(data, list):
            log.info("[idarati] Loaded %d beneficiary categories", len(data))
            return data
        return []

    @staticmethod
    def _build_admin_body(raw: dict, proc_count: int = 0) -> str:
        """
        Build a text body for an administration entry.
        Combines title, contact info, and procedure count.
        """
        parts: List[str] = []

        title = (raw.get("title") or "").strip()
        if title:
            parts.append(f"الجهة الإدارية: {title}")

        emails   = ", ".join(raw.get("emails", []))
        phones   = ", ".join(raw.get("phones", []))
        websites = ", ".join(raw.get("websites", []))

        if emails:
            parts.append(f"البريد الإلكتروني: {emails}")
        if phones:
            parts.append(f"الهاتف: {phones}")
        if websites:
            parts.append(f"الموقع الإلكتروني: {websites}")
        if proc_count:
            parts.append(f"عدد الإجراءات: {proc_count}")

        return "\n".join(parts)

    def _build_admin_lookup(self, admins: List[dict]) -> Dict[str, dict]:
        """Build {admin_id: raw_admin_dict} for fast contact-info enrichment."""
        return {a["id"]: a for a in admins if a.get("id")}

    # ── Main entry point ──────────────────────────────────────────────────────

    def scrape(
        self,
        limit: Optional[int] = None,
        fetch_details: bool = True,
        already_scraped: Optional[List[dict]] = None,
    ) -> "ScrapeResult":
        """
        Full pipeline: procedures + administrations + categories + beneficiaries.

        Parameters
        ----------
        limit           : cap procedures at N (None = all 2,533)
        fetch_details   : if False, skip the per-procedure detail API calls
        already_scraped : dicts from a previous checkpoint to resume from

        Returns
        -------
        ScrapeResult  (procedures, administrations, admin_types, beneficiary_categories)
        """
        # ── Resume support ─────────────────────────────────────────────────
        done_ids: set = set()
        proc_results: List[IdaratiProcedure] = []
        if already_scraped:
            for item in already_scraped:
                done_ids.add(item.get("procedure_id", ""))
            proc_results = [IdaratiProcedure(**item) for item in already_scraped]
            log.info("[idarati] Resuming — %d procedures already scraped", len(proc_results))

        # ── Parallel reference data ────────────────────────────────────────
        log.info("[idarati] Fetching reference data…")
        categories    = self.fetch_categories()
        raw_admins    = self.fetch_all_administrations()
        admin_orgs    = self.fetch_admin_organisations()
        admin_types   = self.fetch_admin_types()
        beneficiaries = self.fetch_beneficiary_categories()

        # Build admin lookup for contact-info enrichment
        admin_lookup = self._build_admin_lookup(raw_admins)

        # Build procedure-count lookup from admin_orgs
        proc_count_by_admin = {a["id"]: a.get("proceduresCount", 0) for a in admin_orgs}

        # ── Build IdaratiAdministration objects ─────────────────────────────
        admin_results: List[IdaratiAdministration] = []
        for raw in raw_admins:
            aid   = raw.get("id", "")
            title = (raw.get("title") or "").strip()
            count = proc_count_by_admin.get(aid, 0)
            body  = self._build_admin_body(raw, count)
            admin_results.append(IdaratiAdministration(
                admin_id         = aid,
                title            = title,
                emails           = ", ".join(raw.get("emails", [])),
                phones           = ", ".join(raw.get("phones", [])),
                websites         = ", ".join(raw.get("websites", [])),
                is_active        = raw.get("isActive", True),
                procedures_count = count,
                body             = body,
            ))

        log.info("[idarati] %d administrations built", len(admin_results))

        # ── Procedures ─────────────────────────────────────────────────────
        metadata = self.fetch_all_metadata(limit=limit)
        to_do    = [m for m in metadata if str(m.get("id", "")) not in done_ids]
        log.info("[idarati] %d procedures to fetch (%d already done)",
                 len(to_do), len(done_ids))

        for idx, meta in enumerate(to_do, 1):
            pid         = str(meta.get("id", ""))
            thematic_id = str(meta.get("thematicId", ""))
            title_ar    = meta.get("title", "")
            admin       = meta.get("administrationTitle", "")
            cat_path    = categories.get(thematic_id, "")

            proc = IdaratiProcedure(
                procedure_id   = pid,
                title_ar       = title_ar,
                thematic_id    = thematic_id,
                administration = admin,
                category_path  = cat_path,
                url_ar = f"{BASE_URL}/informationnel/ar/thematique/{thematic_id}/{pid}",
                url_fr = f"{BASE_URL}/informationnel/fr/thematique/{thematic_id}/{pid}",
            )

            if fetch_details:
                detail    = self.fetch_detail_api(pid)
                documents = self.fetch_documents(pid)
                proc.body_ar = self._build_procedure_body(detail, documents, admin_lookup)
                time.sleep(self.delay_page)

            proc_results.append(proc)

            if idx % 10 == 0 or idx == len(to_do):
                log.info("[idarati] %d/%d procedures fetched", idx, len(to_do))

        return ScrapeResult(
            procedures             = proc_results,
            administrations        = admin_results,
            admin_types            = admin_types,
            beneficiary_categories = beneficiaries,
        )


@dataclass
class ScrapeResult:
    """Everything fetched from idarati.ma in a single scrape run."""
    procedures:             List[IdaratiProcedure]
    administrations:        List[IdaratiAdministration]
    admin_types:            List[dict]              # raw — 5 type categories
    beneficiary_categories: List[dict]              # raw — 10 beneficiary groups
