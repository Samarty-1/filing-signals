"""Phase 1 ingestion: index -> universe -> filers & filings -> headers & documents.

Every step is idempotent and resumable: raw responses are cached on disk,
the warehouse upserts, and work already done is skipped. A crashed or
interrupted run is finished by running the same command again.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone

import duckdb

from . import edgar
from .http import EdgarClient, NotFound
from .warehouse import RawStore, upsert

# Asset-backed issuers file 10-Ks but have no equity to price.
EXCLUDED_SIC = {6189}


@dataclass
class Params:
    first_year: int
    last_year: int
    n_firms: int
    seed: int
    workers: int = 4

    @property
    def window(self) -> tuple[date, date]:
        return date(self.first_year, 1, 1), date(self.last_year, 12, 31)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Pipeline:
    def __init__(self, con: duckdb.DuckDBPyConnection, client: EdgarClient, store: RawStore,
                 log=print):
        self.con, self.client, self.store, self.log = con, client, store, log

    # -- fetch with a disk cache ------------------------------------------
    def cached(self, url: str, *parts: str, refresh: bool = False) -> bytes:
        if not refresh and self.store.exists(*parts):
            return self.store.read(*parts)
        data = self.client.get(url)
        self.store.write(data, *parts)
        return data

    # -- step 1: the quarterly master index --------------------------------
    def load_index(self, p: Params) -> int:
        """Annual-report rows from every quarterly index in the window."""
        loaded = 0
        today = date.today()
        for year in range(p.first_year, p.last_year + 1):
            for q in range(1, 5):
                quarter_end = date(year, 3 * q, 1)
                if quarter_end > today:
                    continue
                key = f"{year}Q{q}"
                url = edgar.master_index_url(year, q).replace("master.idx", "master.gz")
                # a quarter still in progress keeps growing: refetch it
                refresh = (today - quarter_end).days < 120
                raw = self.cached(url, "index", f"{key}.idx.gz", refresh=refresh)
                text = _gunzip_if_needed(raw).decode("latin-1")
                rows = [
                    {**r, "index_quarter": key}
                    for r in edgar.parse_master_index(text)
                    if r["form"] in edgar.ANNUAL_FORMS
                ]
                loaded += upsert(self.con, "master_index", rows)
        self.log(f"index: {loaded:,} annual-report rows")
        return loaded

    # -- step 2: draw the sample -------------------------------------------
    def draw_universe(self, p: Params) -> list[int]:
        """Draw filers in a seeded random order until n_firms qualify.

        Candidates are every CIK with an annual report anywhere in the window,
        so firms that later died are as likely to be drawn as survivors.
        Every draw and its decision is recorded in `universe`.
        """
        start, end = p.window
        candidates = [r[0] for r in self.con.execute("""
            select distinct cik from master_index
            where date_filed between ? and ? and form in ('10-K', '10-K405', '10-KT', '10-KSB')
        """, [start, end]).fetchall()]
        candidates.sort(key=lambda cik: _seeded_rank(cik, p.seed))

        decided = dict(self.con.execute(
            "select cik, status from universe where seed = ?", [p.seed]).fetchall())
        included = [c for c, s in decided.items() if s == "included"]
        for rank, cik in enumerate(candidates, start=1):
            if len(included) >= p.n_firms:
                break
            if cik in decided:
                continue
            doc = self._submissions(cik)
            filer = edgar.parse_filer(doc)
            self._store_filer(filer)
            status, reason = _qualify(filer)
            upsert(self.con, "universe", [{
                "cik": cik, "draw_rank": rank, "seed": p.seed, "status": status,
                "reason": reason, "decided_at": now(),
            }])
            if status == "included":
                included.append(cik)
        self.log(f"universe: {len(included)} firms included "
                 f"from {len(candidates):,} candidates")
        return included[: p.n_firms]

    # -- step 3: filers and their filing lists -----------------------------
    def load_filings(self, ciks: list[int], p: Params) -> int:
        start, _ = p.window
        total = 0
        for cik in ciks:
            doc = self._submissions(cik)
            self._store_filer(edgar.parse_filer(doc))
            rows = edgar.parse_filings(cik, doc["filings"]["recent"])
            for page in doc["filings"].get("files", []):
                # older pages only matter if they reach into the window
                if page.get("filingTo") and date.fromisoformat(page["filingTo"]) < start:
                    continue
                raw = self.cached(edgar.submissions_url(cik, page["name"]),
                                  "submissions", page["name"] + ".gz")
                rows += edgar.parse_filings(cik, json.loads(raw))
            seen = now()
            total += upsert(self.con, "filings", [{**r, "first_seen_at": seen} for r in rows])
        self.log(f"filings: {total:,} filing records for {len(ciks)} firms")
        return total

    # -- step 4: headers and documents for annual reports ------------------
    def fetch_annual(self, ciks: list[int], p: Params) -> dict:
        start, end = p.window
        todo = self.con.execute("""
            select f.accession, f.cik, f.primary_document,
                   h.accession is null as need_header,
                   d.accession is null as need_document
            from filings f
            left join filing_headers h using (accession)
            left join documents d using (accession)
            where f.cik in (select unnest(?::bigint[]))
              and f.form in (select unnest(?::varchar[]))
              and f.filing_date between ? and ?
              and (h.accession is null or d.accession is null)
        """, [ciks, sorted(edgar.ANNUAL_FORMS), start, end]).fetchall()
        # a URL that already 404'd is not retried; a filing whose *source*
        # changed (e.g. a new header URL scheme) is
        self.failed = {u for (u,) in self.con.execute("select url from fetch_failures").fetchall()}
        self.log(f"documents: {len(todo)} annual reports to fetch")

        done = {"headers": 0, "documents": 0, "failures": 0}
        with ThreadPoolExecutor(max_workers=p.workers) as pool:
            futures = {pool.submit(self._fetch_one, *row): row for row in todo}
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                # all warehouse writes happen here, on one thread
                if result.get("header"):
                    upsert(self.con, "filing_headers", [result["header"]])
                    done["headers"] += 1
                if result.get("document"):
                    upsert(self.con, "documents", [result["document"]])
                    done["documents"] += 1
                for failure in result.get("failures", []):
                    upsert(self.con, "fetch_failures", [failure])
                    done["failures"] += 1
                if i % 250 == 0:
                    self.log(f"  {i}/{len(todo)}  ({self.client.stats.bytes / 1e9:.2f} GB)")
        self.log(f"documents: {done}")
        return done

    def _fetch_one(self, accession, cik, primary_document, need_header, need_document) -> dict:
        out: dict = {"failures": []}
        if need_header and (url := edgar.header_url(cik, accession)) not in self.failed:
            try:
                raw = self.cached(url, "filings", str(cik), accession, "header.sgml.gz")
                out["header"] = {"accession": accession,
                                 **edgar.parse_header(raw.decode("latin-1")),
                                 "fetched_at": now()}
            except NotFound as e:
                out["failures"].append(_failure(url, accession, e))
        if need_document and (url := edgar.document_url(cik, accession, primary_document)) not in self.failed:
            name = (primary_document or f"{accession}.txt") + ".gz"
            try:
                data = self.client.get(url)
                meta = self.store.write(data, "filings", str(cik), accession, name)
                out["document"] = {"accession": accession, "url": url, **meta,
                                   "fetched_at": now()}
            except NotFound as e:
                out["failures"].append(_failure(url, accession, e))
        return out

    # -- helpers -------------------------------------------------------------
    def _submissions(self, cik: int) -> dict:
        raw = self.cached(edgar.submissions_url(cik), "submissions", f"CIK{cik:010d}.json.gz")
        return json.loads(raw)

    def _store_filer(self, filer: dict) -> None:
        names = filer.pop("former_names")
        upsert(self.con, "filers", [{**filer, "fetched_at": now()}])
        upsert(self.con, "filer_former_names", [{"cik": filer["cik"], **n} for n in names])

    # -- the whole run -------------------------------------------------------
    def run(self, p: Params) -> None:
        run_id = uuid.uuid4().hex[:12]
        upsert(self.con, "ingest_runs", [{
            "run_id": run_id, "started_at": now(), "finished_at": None,
            "params": json.dumps(p.__dict__), "requests": None, "retries": None,
            "bytes": None, "status": "running"}])
        status = "failed"
        try:
            self.load_index(p)
            ciks = self.draw_universe(p)
            self.load_filings(ciks, p)
            self.fetch_annual(ciks, p)
            status = "ok"
        finally:
            s = self.client.stats
            self.con.execute("""
                update ingest_runs set finished_at = ?, requests = ?, retries = ?,
                       bytes = ?, status = ? where run_id = ?""",
                [now(), s.requests, s.retries, s.bytes, status, run_id])


def _qualify(filer: dict) -> tuple[str, str | None]:
    if filer["entity_type"] != "operating":
        return "excluded", f"entity_type={filer['entity_type']}"
    if filer["sic"] in EXCLUDED_SIC:
        return "excluded", f"sic={filer['sic']} (asset-backed)"
    return "included", None


def _seeded_rank(cik: int, seed: int) -> str:
    # stable across Python versions and platforms, unlike hash() or random
    return hashlib.sha256(f"{seed}:{cik}".encode()).hexdigest()


def _failure(url: str, accession: str, error: Exception) -> dict:
    return {"url": url, "accession": accession, "error": type(error).__name__,
            "failed_at": now()}


def _gunzip_if_needed(raw: bytes) -> bytes:
    import gzip
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
