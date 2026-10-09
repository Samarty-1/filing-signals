from collections import Counter
from datetime import datetime, timezone

import pytest

from filing_signals import features
from filing_signals.warehouse import connect, upsert


def test_measures():
    a, b = Counter("the cat sat".split()), Counter("the cat ran".split())
    assert features.jaccard(a, b) == pytest.approx(2 / 4)
    assert features.cosine(a, b) == pytest.approx(2 / 3)
    assert features.cosine(a, a) == pytest.approx(1.0)
    assert features.tokens("Item 1A: We're NOT co-operating, it's 2024.") == [
        "item", "a", "we're", "not", "co-operating", "it's"]


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@pytest.fixture
def con():
    """One firm: FY2019, FY2020, FY2021 originals; FY2019 amended *after* the
    FY2020 report was filed; FY2023 with no FY2022 (a gap)."""
    con = connect(":memory:")
    reports = [  # accession, form, period end, accepted, risk-factor text
        ("a19", "10-K", "2019-12-31", "2020-02-20", "risk alpha beta gamma"),
        ("a20", "10-K", "2020-12-31", "2021-02-20", "risk alpha beta delta"),
        ("a19x", "10-K/A", "2019-12-31", "2021-03-15", "risk totally rewritten words"),
        ("a21", "10-K", "2021-12-31", "2022-02-20", "risk alpha beta delta"),
        ("a23", "10-K", "2023-12-31", "2024-02-20", "risk new"),
    ]
    upsert(con, "filings", [{
        "accession": a, "cik": 1, "form": f, "filing_date": ts(acc).date(),
        "report_date": ts(p).date(), "accepted_api_raw": None, "primary_document": None,
        "size_bytes": 1, "items": None, "first_seen_at": ts(acc)} for a, f, p, acc, _ in reports])
    upsert(con, "filing_headers", [{
        "accession": a, "accepted_at": ts(acc), "header_period": ts(p).date(),
        "fetched_at": ts(acc)} for a, _, p, acc, _ in reports])
    con.execute("create table sections (accession varchar, section varchar, status varchar, text varchar)")
    con.executemany("insert into sections values (?, 'risk_factors', 'found', ?)",
                    [[a, text] for a, *_, text in reports])
    features.compute(con, log=lambda *_: None)
    return con


def test_pairs_are_consecutive_and_point_in_time(con):
    got = dict(con.execute("select accession, prior_accession from section_changes").fetchall())
    # FY2020 compares with the FY2019 *original*: the amendment came later.
    # FY2021 compares with FY2020. FY2023 has no FY2022, so no pair.
    assert got == {"a20": "a19", "a21": "a20"}


def test_identical_text_scores_one(con):
    sim = con.execute("select sim_cosine, sim_jaccard, sim_tfidf from section_changes "
                      "where accession = 'a21'").fetchone()
    assert sim == pytest.approx((1.0, 1.0, 1.0))
