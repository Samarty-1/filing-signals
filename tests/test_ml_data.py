"""Phase 3 data construction: labels, size join, passages, extractors.
No GPU needed; skipped when the ML extra isn't installed."""
import math
from datetime import datetime, timedelta, timezone

import pytest

pd = pytest.importorskip("pandas")

from filing_signals import xbrl  # noqa: E402
from filing_signals.ml import labels, revenue_data  # noqa: E402
from filing_signals.warehouse import connect, upsert  # noqa: E402


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def add_filing(con, acc, cik, form, accepted, period=None, items=None):
    upsert(con, "filings", [{
        "accession": acc, "cik": cik, "form": form, "filing_date": ts(accepted).date(),
        "report_date": ts(period).date() if period else None, "accepted_api_raw": None,
        "primary_document": None, "size_bytes": 1, "items": items, "first_seen_at": ts(accepted)}])
    if form.startswith("10-K"):
        upsert(con, "filing_headers", [{"accession": acc, "accepted_at": ts(accepted),
                                        "header_period": ts(period).date(), "fetched_at": ts(accepted)}])


@pytest.fixture
def dataset(tmp_path):
    con = connect(":memory:")
    con.execute(xbrl.SCHEMA)
    # filing lists downloaded on 2024-06-01: the label horizon
    upsert(con, "filers", [{"cik": c, "name": "x", "entity_type": "operating", "sic": 1,
                            "sic_description": None, "state_of_incorporation": None,
                            "fiscal_year_end": None, "current_tickers": None,
                            "fetched_at": ts("2024-06-01")} for c in (1, 2, 3)])
    # firm 1: 10-K 2020-03-01; restatement 8-K 200 days later -> positive;
    #         an auditor change the year before -> prior_event
    add_filing(con, "k1", 1, "10-K", "2020-03-01", "2019-12-31")
    add_filing(con, "e1", 1, "8-K", "2020-09-17", items="4.02")
    add_filing(con, "p1", 1, "8-K", "2019-06-01", items="4.01")
    # firm 2: same-day 8-K (announced with the 10-K) and one after 365 days -> negative
    add_filing(con, "k2", 2, "10-K", "2020-03-01", "2019-12-31")
    add_filing(con, "e2", 2, "8-K", "2020-03-01", items="4.02")
    add_filing(con, "e2b", 2, "8-K", "2021-03-05", items="1.03")
    # firm 3: 10-K whose 365-day window hasn't elapsed by 2024-06-01 -> excluded
    add_filing(con, "k3", 3, "10-K", "2023-09-01", "2023-06-30")
    con.execute("create table sections (accession varchar, section varchar, status varchar, text varchar)")
    con.executemany("insert into sections values (?, 'risk_factors', 'found', 'risk text')",
                    [["k1"], ["k2"], ["k3"]])
    upsert(con, "xbrl_facts", [{
        "fact_id": "f1", "cik": 1, "taxonomy": "dei", "concept": "EntityPublicFloat", "unit": "USD",
        "value": 5e7, "period_start": None, "period_end": "2019-06-30", "accession": "k1",
        "fy": 2019, "fp": "FY", "form": "10-K", "filed": "2020-03-01"}])
    out = tmp_path / "events.parquet"
    labels.build(con, out, log=lambda *_: None)
    return pd.read_parquet(out).set_index("accession")


def test_label_window_is_strictly_after_and_within_a_year(dataset):
    assert dataset.loc["k1", "label"] and dataset.loc["k1", "restatement"]
    assert not dataset.loc["k2", "label"]          # same-day and >365-day events don't count


def test_prior_event_uses_the_year_before(dataset):
    assert dataset.loc["k1", "prior_event"] and not dataset.loc["k2", "prior_event"]


def test_incomplete_label_window_is_excluded(dataset):
    assert "k3" not in dataset.index


def test_public_float_joins_on_the_filing(dataset):
    assert dataset.loc["k1", "has_float"] and math.isclose(dataset.loc["k1", "log_float"], math.log(1 + 5e7))
    assert not dataset.loc["k2", "has_float"] and dataset.loc["k2", "log_float"] == 0


def test_passage_keeps_table_numbers_on_following_lines():
    mdna = "\n".join(["Overview of the business " * 20, "Net sales", "(in millions)", "68,636",
                      "Unrelated text " * 40] + ["filler"] * 20)
    p = revenue_data.passage(mdna)
    assert "68,636" in p and "Unrelated" not in p


def test_answerable_recognises_common_written_forms():
    usd = 68_636_146_000
    assert revenue_data.is_answerable("Net sales were $68.6 billion", usd)
    assert revenue_data.is_answerable("Net sales 68,636 (in millions)", usd)
    assert revenue_data.is_answerable("Net sales 68,636,146 (in thousands)", usd)
    assert not revenue_data.is_answerable("Net sales 61,280", usd)


def test_extractors_parse_scales():
    from filing_signals.ml.llm_extract import parse_answer, regex_extract
    assert parse_answer(" 1234567000") == 1234567000
    assert parse_answer("unknown") is None
    assert math.isclose(parse_answer("68.6 billion"), 68.6e9)
    assert math.isclose(regex_extract("(in thousands)\nTotal revenues were $1,250 for the year"), 1.25e6)
    assert math.isclose(regex_extract("Net sales increased to $4.2 billion"), 4.2e9)
