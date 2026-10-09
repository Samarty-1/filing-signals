"""XBRL facts for the sampled companies (data.sec.gov companyfacts API).

Uses:
  * size: dei:EntityPublicFloat, the market value of non-affiliate equity a
    10-K reports on its cover page, keyed by that 10-K's accession. It is the
    control the text models must beat, because filing language mostly
    identifies small companies.
  * ground truth: reported annual revenue, the target for the extraction LLM.
  * Phase 4 ticker validation: dei:EntityCommonStockSharesOutstanding, so a
    price series can be checked against the float the same 10-K reports.

Only a whitelist of concepts is loaded. Revenue appears under several
concept names as US GAAP changed (ASC 606 introduced
RevenueFromContractWithCustomer... in 2018), so all of them are kept.
"""
from __future__ import annotations

import hashlib
import json

from .http import NotFound
from .pipeline import Pipeline, _failure, now
from .warehouse import upsert

CONCEPTS = {
    "dei": ["EntityPublicFloat", "EntityCommonStockSharesOutstanding"],
    "us-gaap": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "Assets",
        "NetIncomeLoss",
    ],
}

SCHEMA = """
create table if not exists xbrl_facts (
    fact_id     varchar primary key,    -- hash of the identifying fields below
    cik         bigint not null,
    taxonomy    varchar not null,
    concept     varchar not null,
    unit        varchar not null,
    value       double not null,
    period_start date,
    period_end  date not null,
    accession   varchar not null,
    fy          integer,
    fp          varchar,
    form        varchar,
    filed       date
);
"""


def companyfacts_url(cik: int) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def parse(doc: dict) -> list[dict]:
    cik = int(doc["cik"])
    rows = []
    for taxonomy, concepts in CONCEPTS.items():
        for concept in concepts:
            fact = doc.get("facts", {}).get(taxonomy, {}).get(concept)
            if not fact:
                continue
            for unit, values in fact["units"].items():
                for v in values:
                    rows.append({
                        "cik": cik, "taxonomy": taxonomy, "concept": concept, "unit": unit,
                        "value": float(v["val"]), "period_start": v.get("start"),
                        "period_end": v["end"], "accession": v["accn"], "fy": v.get("fy"),
                        "fp": v.get("fp"), "form": v.get("form"), "filed": v.get("filed"),
                    })
    # instant facts (assets, public float) have no period_start, so the key is
    # a hash rather than a composite primary key; repeats collapse to one row
    unique = {}
    for r in rows:
        key = "|".join(str(r[k]) for k in ("cik", "taxonomy", "concept", "unit", "period_start",
                                            "period_end", "accession"))
        unique[key] = {"fact_id": hashlib.sha1(key.encode()).hexdigest(), **r}
    return list(unique.values())


def load(pipe: Pipeline, log=print) -> int:
    pipe.con.execute(SCHEMA)
    ciks = [c for (c,) in pipe.con.execute(
        "select cik from universe where status = 'included' order by cik").fetchall()]
    # a company is done once every whitelisted concept it reports is loaded;
    # adding a concept re-parses the cached responses without new requests
    done = {c for (c,) in pipe.con.execute(
        "select cik from xbrl_facts group by cik having count(distinct concept) filter "
        "(where concept = 'EntityCommonStockSharesOutstanding') > 0").fetchall()}
    failed = {u for (u,) in pipe.con.execute("select url from fetch_failures").fetchall()}
    total = 0
    for i, cik in enumerate(ciks, start=1):
        url = companyfacts_url(cik)
        if cik in done or url in failed:
            continue
        try:
            raw = pipe.cached(url, "xbrl", f"CIK{cik:010d}.json.gz")
        except NotFound as e:       # companies that never filed XBRL
            upsert(pipe.con, "fetch_failures", [_failure(url, None, e)])
            continue
        total += upsert(pipe.con, "xbrl_facts", parse(json.loads(raw)))
        if i % 200 == 0:
            log(f"  xbrl {i}/{len(ciks)}")
    n = pipe.con.execute("select count(distinct cik) from xbrl_facts").fetchone()[0]
    log(f"xbrl: {total:,} facts loaded; {n} of {len(ciks)} companies have XBRL ({now():%H:%M})")
    return total
