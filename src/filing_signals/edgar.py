"""Parsers for the three EDGAR sources the pipeline reads.

  full-index/<year>/QTR<n>/master.idx     every filing in a quarter: the universe
  data.sec.gov/submissions/CIK*.json      a filer's attributes and filing list
  <accession>.hdr.sgml                    the SGML header: the authoritative
                                          acceptance timestamp. (The friendlier
                                          -index-headers.html page 404s for most
                                          filings before mid-2014.)
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

ARCHIVES = "https://www.sec.gov/Archives/edgar"
SUBMISSIONS = "https://data.sec.gov/submissions"
EASTERN = ZoneInfo("America/New_York")

# The annual-report family. 10-K405 ran until 2002 and 10-KSB until 2009, so
# both still occur in older windows; amendments are kept as their own versions.
ANNUAL_FORMS = {"10-K", "10-K405", "10-KT", "10-KSB"}
ANNUAL_FORMS |= {f + "/A" for f in ANNUAL_FORMS}


def master_index_url(year: int, quarter: int) -> str:
    return f"{ARCHIVES}/full-index/{year}/QTR{quarter}/master.idx"


def submissions_url(cik: int, page: str | None = None) -> str:
    return f"{SUBMISSIONS}/{page}" if page else f"{SUBMISSIONS}/CIK{cik:010d}.json"


def filing_folder(cik: int, accession: str) -> str:
    return f"{ARCHIVES}/data/{cik}/{accession.replace('-', '')}"


def header_url(cik: int, accession: str) -> str:
    return f"{filing_folder(cik, accession)}/{accession}.hdr.sgml"


def document_url(cik: int, accession: str, primary_document: str | None) -> str:
    # very old filings have no primary document: fall back to the full submission
    name = primary_document or f"{accession}.txt"
    return f"{filing_folder(cik, accession)}/{name}"


# --- master.idx -------------------------------------------------------------

def parse_master_index(text: str) -> list[dict]:
    """Rows of a master.idx: CIK|Company Name|Form Type|Date Filed|Filename."""
    rows = []
    in_body = False
    for line in text.splitlines():
        if line.startswith("-----"):
            in_body = True
            continue
        if not in_body or not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, name, form, filed, path = parts
        rows.append({
            "cik": int(cik),
            "company_name": name.strip(),
            "form": form.strip(),
            "date_filed": date.fromisoformat(filed),
            "accession": path.rsplit("/", 1)[-1].removesuffix(".txt"),
        })
    return rows


# --- submissions JSON -------------------------------------------------------

def parse_filer(doc: dict) -> dict:
    return {
        "cik": int(doc["cik"]),
        "name": doc.get("name"),
        "entity_type": doc.get("entityType"),
        "sic": int(doc["sic"]) if str(doc.get("sic") or "").isdigit() else None,
        "sic_description": doc.get("sicDescription"),
        "state_of_incorporation": doc.get("stateOfIncorporation") or None,
        "fiscal_year_end": doc.get("fiscalYearEnd") or None,
        # current tickers only: using these to map history is survivorship bias
        "current_tickers": ",".join(doc.get("tickers") or []) or None,
        "former_names": [
            {"name": n["name"], "valid_from": _iso_date(n.get("from")), "valid_to": _iso_date(n.get("to"))}
            for n in doc.get("formerNames") or []
        ],
    }


def filing_pages(doc: dict) -> list[str]:
    """Extra pages holding a filer's older filings (beyond the newest ~1000)."""
    return [f["name"] for f in doc.get("filings", {}).get("files", [])]


def parse_filings(cik: int, block: dict) -> list[dict]:
    """Column-oriented filing arrays -> one dict per filing.

    `block` is doc['filings']['recent'] on the main document, or a whole extra
    page (pages carry the same columns at top level).
    """
    out = []
    for i, accession in enumerate(block["accessionNumber"]):
        out.append({
            "accession": accession,
            "cik": cik,
            "form": block["form"][i],
            "filing_date": _iso_date(block["filingDate"][i]),
            "report_date": _iso_date(block["reportDate"][i]),
            # labelled UTC but not reliably UTC: kept raw, never used for timing
            "accepted_api_raw": block["acceptanceDateTime"][i] or None,
            "primary_document": block["primaryDocument"][i] or None,
            "size_bytes": block["size"][i],
            "items": block.get("items", [""] * len(block["form"]))[i] or None,
        })
    return out


def load_json(raw: bytes) -> dict:
    return json.loads(raw)


# --- SGML header ------------------------------------------------------------

_ACCEPTED = re.compile(r"ACCEPTANCE-DATETIME>\s*(\d{14})")
# "<PERIOD>20240928" in .hdr.sgml; "CONFORMED PERIOD OF REPORT: 20240928" in
# the rendered header page and the full submission
_PERIOD = re.compile(r"(?:<PERIOD>|CONFORMED PERIOD OF REPORT:)\s*(\d{8})")


def parse_header(text: str) -> dict:
    """Acceptance time (Eastern, as EDGAR records it) and period of report."""
    m = _ACCEPTED.search(text)
    accepted_et = None
    if m:
        naive = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        accepted_et = naive.replace(tzinfo=EASTERN)
    p = _PERIOD.search(text)
    return {
        "accepted_at": accepted_et.astimezone(timezone.utc) if accepted_et else None,
        "header_period": datetime.strptime(p.group(1), "%Y%m%d").date() if p else None,
    }


def _iso_date(value: str | None) -> date | None:
    return date.fromisoformat(value[:10]) if value else None
