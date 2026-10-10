"""Phase 5: export the warehouse and model outputs as static JSON for the
TypeScript explorer in web/.

    web/public/data/summary.json      pipeline counts + every phase's results
    web/public/data/firms.json        one row per firm, for search
    web/public/data/firms/<cik>.json  a firm's reports, change scores, 8-K
                                      events, risk scores, revenue answers

Everything is derived from public SEC data. Nothing here needs the 1.3 GB
raw store, so the site builds in CI from the committed JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pandas as pd

EVENT_ITEMS = {"1.03": "bankruptcy", "2.01": "acquisition/disposition",
               "4.01": "auditor change", "4.02": "restatement"}

REPORTS_SQL = """
with tick as (select accession, ticker from report_tickers),
flt as (
    select accession, max(value) as public_float from xbrl_facts
    where concept = 'EntityPublicFloat' group by 1
),
rev as (
    -- the report's own full-year revenue (same rule as the Phase 3b ground truth)
    select x.accession, arg_min(x.value, list_position(
        ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet',
         'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueGoodsNet'], x.concept)) as revenue
    from xbrl_facts x join annual_reports a using (accession)
    where x.unit = 'USD' and x.period_end = a.fiscal_period_end
      and date_diff('day', x.period_start, x.period_end) between 350 and 380
      and x.concept in ('RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet',
                        'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueGoodsNet')
    group by 1
),
chg as (
    select accession,
           max(sim_tfidf_pit) filter (where section = 'risk_factors') as sim_rf,
           max(sim_tfidf_pit) filter (where section = 'mdna') as sim_mdna,
           max(words) filter (where section = 'risk_factors') as words_rf,
           max(words) filter (where section = 'mdna') as words_mdna
    from section_changes group by 1
)
select a.cik, a.accession, a.form, a.is_amendment, a.version_no,
       year(a.fiscal_period_end) as fy, a.fiscal_period_end::varchar as period_end,
       strftime(a.accepted_at_et, '%Y-%m-%d %H:%M') as accepted_et,
       a.dated_next_business_day, t.ticker, f.public_float, r.revenue,
       c.sim_rf, c.sim_mdna, c.words_rf, c.words_mdna
from annual_reports a
left join tick t using (accession) left join flt f using (accession)
left join rev r using (accession) left join chg c using (accession)
where a.stored_path is not null
order by a.cik, a.fiscal_period_end, a.version_no
"""

EVENTS_SQL = """
select cik, filing_date::varchar as date, items
from filings
where form like '8-K%' and items is not null
  and (items like '%1.03%' or items like '%2.01%' or items like '%4.01%' or items like '%4.02%')
order by cik, filing_date
"""


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, float):
        return round(v, 4) if abs(v) < 10 else round(v)
    if hasattr(v, "item"):          # numpy scalars
        return _clean(v.item())
    return v


# Phase 3 outputs exist only for test-set reports. Outside it they are left
# out entirely, so the page can tell "not scored" from "answered unknown".
_OPTIONAL = {"risk_pct": ("risk_pct", "event_next_12m"),
             "answerable": ("answerable", "qlora_usd", "zero_shot_usd")}


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict("records"):
        rec = {k: _clean(v) for k, v in row.items()}
        for marker, keys in _OPTIONAL.items():
            if rec.get(marker) is None:
                for k in keys:
                    rec.pop(k, None)
        out.append(rec)
    return out


def _model_outputs(ml: Path) -> pd.DataFrame:
    """Per-accession Phase 3 outputs on the test years."""
    from .ml.llm_extract import parse_answer

    out = pd.DataFrame(columns=["accession"])
    base = ml / "results" / "baseline_test_scores.parquet"
    if base.exists():
        b = pd.read_parquet(base)[["accession", "label", "score_tfidf"]]
        # percentile among test reports reads better than a raw probability
        b["risk_pct"] = b.score_tfidf.rank(pct=True)
        out = b[["accession", "label", "risk_pct"]].rename(columns={"label": "event_next_12m"})
    rev = ml / "results" / "revenue_test_predictions.parquet"
    if rev.exists():
        r = pd.read_parquet(rev)
        r["qlora_usd"] = r.qlora.map(parse_answer)
        r["zero_shot_usd"] = r.zero_shot.map(parse_answer)
        out = out.merge(r[["accession", "answerable", "qlora_usd", "zero_shot_usd"]],
                        on="accession", how="outer")
    return out


def _results(ml: Path) -> dict:
    res = {}
    for name in ("baseline", "encoder", "llm_extract", "embed_change"):
        p = ml / "results" / f"{name}.json"
        if p.exists():
            res[name] = json.loads(p.read_text())
    p4 = ml.parent / "prices" / "phase4_results.json"
    res["phase4"] = json.loads(p4.read_text()) if p4.exists() else None
    return res


def run(con: duckdb.DuckDBPyConnection, data: Path, out: Path, log=print) -> dict:
    reports = con.execute(REPORTS_SQL).df()
    models = _model_outputs(data / "ml")
    if len(models):
        reports = reports.merge(models, on="accession", how="left")
    events = con.execute(EVENTS_SQL).df()
    events["kinds"] = events["items"].map(
        lambda s: [EVENT_ITEMS[i] for i in EVENT_ITEMS if i in s])
    filers = con.execute("""
        select cik, name, sic_description as industry, current_tickers
        from filers where cik in (select cik from annual_reports where stored_path is not null)
        order by name""").df()

    (out / "firms").mkdir(parents=True, exist_ok=True)
    index = []
    rep_by, ev_by = dict(tuple(reports.groupby("cik"))), dict(tuple(events.groupby("cik")))
    for f in filers.itertuples():
        reps = rep_by.get(f.cik, reports.iloc[0:0])
        evs = ev_by.get(f.cik, events.iloc[0:0])
        originals = reps[reps.version_no == 1]
        last_ticker = next((t for t in reversed(originals.ticker.tolist()) if isinstance(t, str)), None)
        firm = {"cik": int(f.cik), "name": f.name, "industry": f.industry,
                "current_tickers": f.current_tickers, "ticker": last_ticker,
                "reports": _records(reps.drop(columns="cik")),
                "events": [{"date": e.date, "items": e.items, "kinds": list(e.kinds)} for e in evs.itertuples()]}
        (out / "firms" / f"{int(f.cik)}.json").write_text(json.dumps(firm, separators=(",", ":")))
        rf = originals.sim_rf.dropna()
        index.append({"cik": int(f.cik), "name": f.name, "industry": f.industry, "ticker": last_ticker,
                      "first_fy": _clean(originals.fy.min()), "last_fy": _clean(originals.fy.max()),
                      "reports": int(len(originals)), "events": int(len(evs)),
                      "min_sim_rf": _clean(rf.min()) if len(rf) else None,
                      "has_current_ticker": bool(f.current_tickers is not None)})
    (out / "firms.json").write_text(json.dumps(index, separators=(",", ":")))

    counts = con.execute("""
        select (select count(*) from annual_reports where stored_path is not null) as reports,
               (select count(distinct cik) from annual_reports where stored_path is not null) as firms,
               (select count(*) from annual_reports where stored_path is not null and is_amendment) as amendments,
               (select count(*) from section_changes) as section_pairs,
               (select min(year(fiscal_period_end)) from annual_reports where stored_path is not null) as first_fy,
               (select max(year(fiscal_period_end)) from annual_reports where stored_path is not null) as last_fy
    """).df().iloc[0].to_dict()
    summary = {"counts": {k: _clean(v) for k, v in counts.items()}, "results": _results(data / "ml")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    size = sum(p.stat().st_size for p in out.rglob("*.json"))
    log(f"export: {len(index)} firms, {len(reports):,} reports, {len(events):,} events -> {out} "
        f"({size / 1e6:.1f} MB)")
    return summary
