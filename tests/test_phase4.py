"""Phase 4: point-in-time tickers, formation timing, delisting returns."""
from datetime import datetime, timezone

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from filing_signals import prices, returns  # noqa: E402
from filing_signals.warehouse import connect, upsert  # noqa: E402


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# --- tickers ----------------------------------------------------------------------

def test_ticker_from_inline_xbrl_tag_wins():
    raw = (b'<html><ix:nonNumeric name="dei:TradingSymbol" contextRef="c">ABCD</ix:nonNumeric>'
           b"<p>Our common stock trades under the symbol \xe2\x80\x9cWXYZ\xe2\x80\x9d.</p></html>")
    assert prices.extract_ticker(raw)[:2] == ("ABCD", "xbrl_tag")


def test_ticker_from_item5_text_ignores_exchange_names():
    raw = (b"<p>Our common stock is listed on the NYSE under the symbol \"KO\". The symbol KO "
           b"has been used since 1919.</p>")
    ticker, source, _ = prices.extract_ticker(raw)
    assert (ticker, source) == ("KO", "item5_text")


@pytest.fixture
def con():
    con = connect(":memory:")
    con.execute(prices.SCHEMA)
    return con


def add_report(con, acc, cik, accepted, period="2019-12-31"):
    upsert(con, "filings", [{
        "accession": acc, "cik": cik, "form": "10-K", "filing_date": ts(accepted).date(),
        "report_date": ts(period).date(), "accepted_api_raw": None, "primary_document": None,
        "size_bytes": 1, "items": None, "first_seen_at": ts(accepted)}])
    upsert(con, "filing_headers", [{"accession": acc, "accepted_at": ts(accepted),
                                    "header_period": ts(period).date(), "fetched_at": ts(accepted)}])


def test_resolution_prefers_own_ticker_then_q_then_current(con):
    upsert(con, "filers", [{"cik": c, "name": "x", "entity_type": "operating", "sic": 1,
                            "sic_description": None, "state_of_incorporation": None,
                            "fiscal_year_end": None, "current_tickers": t,
                            "fetched_at": ts("2024-06-01")} for c, t in ((1, None), (2, None), (3, "NEWT"))])
    for acc, cik in (("r1", 1), ("r2", 2), ("r3", 3)):
        add_report(con, acc, cik, "2020-03-01")
    con.executemany("insert into report_tickers values (?, ?, ?, 'item5_text', [])",
                    [["r1", 1, "AAA"], ["r2", 2, "BBB"], ["r3", 3, "OLDT"]])
    con.execute("""create table tiingo_tickers as select * from (values
        ('AAA', 'NYSE', 'Stock', 'USD', date '2016-01-04', date '2026-10-01'),
        ('BBB', 'NYSE', 'ETF',   'USD', date '2021-01-04', date '2026-10-01'),   -- recycled as an ETF
        ('BBBQ', 'PINK', 'Stock', 'USD', date '1999-01-04', date '2023-01-01'),  -- bankrupt successor
        ('NEWT', 'NASDAQ', 'Stock', 'USD', date '2016-01-04', date '2026-10-01'),
        ('OLDT', 'NASDAQ', 'Stock', 'USD', date '2001-01-04', date '2019-01-01') -- ended before filing
    ) t(ticker, exchange, asset_type, currency, start_date, end_date)""")
    got = dict(con.execute(f"select accession, symbol from ({prices.RESOLVED_SQL})").fetchall())
    # AAA: Tiingo's late start date doesn't matter; BBB: the ETF is skipped
    # for the Q ticker; OLDT had ended, so the same CIK's current ticker
    assert got == {"r1": "AAA", "r2": "BBBQ", "r3": "NEWT"}


def test_budget_stops_at_monthly_symbol_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "MONTHLY_SYMBOLS", 2)
    calls = []

    class Resp:
        status_code, text = 200, "[]"

        def raise_for_status(self):
            pass

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(a) or Resp())
    out = prices.fetch_prices(["A", "B", "C"], tmp_path, log=lambda *_: None, token="t")
    assert len(calls) == 2 and out["remaining"] == 1
    # a second run in the same month fetches nothing more
    out = prices.fetch_prices(["A", "B", "C"], tmp_path, log=lambda *_: None, token="t")
    assert len(calls) == 2 and out["cached_before"] == 2


# --- formation timing and returns ---------------------------------------------------

def price_table(con, rows):
    df = pd.DataFrame(rows, columns=["symbol", "day", "adj_close"])  # noqa: F841
    con.execute("""create or replace table daily_prices as
                   select symbol, day::date as day, adj_close, adj_close as close, 1 as volume from df""")


def test_filing_on_the_last_trading_day_waits_a_month(con):
    # January 2021 ends on a Sunday; its last close is Friday the 29th
    price_table(con, [("AAA", d, 10.0) for d in ("2021-01-28", "2021-01-29", "2021-02-26", "2021-03-31")])
    for acc, accepted in (("early", "2021-01-28T15:00:00"), ("late", "2021-01-29T21:30:00")):
        add_report(con, acc, 1 if acc == "early" else 2, accepted)
    con.execute("""create table section_changes as select * from (values
        (1, 'early', timestamptz '2021-01-28 15:00:00+00', 'risk_factors', 0.9),
        (2, 'late',  timestamptz '2021-01-29 21:30:00+00', 'risk_factors', 0.5)
    ) t(cik, accession, accepted_at, section, sim_tfidf_pit)""")
    con.execute("""create table report_symbols as select * from (values
        ('early', 'AAA', 5e8), ('late', 'AAA', 5e8)) t(accession, symbol, public_float)""")
    s = returns.signals(con)
    jan = s[(s.month == "2021-01-31") & (s.section == "risk_factors")]
    feb = s[(s.month == "2021-02-28") & (s.section == "risk_factors")]
    assert list(jan.accession) == ["early"]
    assert set(feb.accession) == {"early", "late"}


def test_delisting_month_return_and_shumway_adjustment(con):
    price_table(con, [("DEAD", "2021-01-29", 10.0), ("DEAD", "2021-02-10", 5.0),
                      ("LIVE", "2021-01-29", 10.0), ("LIVE", "2021-02-26", 11.0),
                      ("LIVE", "2021-06-30", 12.0)])
    r = returns.monthly_returns(con).set_index("symbol")
    assert r.loc["DEAD", "ret"] == pytest.approx(-0.5) and r.loc["DEAD", "delisted"]
    assert not r.loc["LIVE"].delisted.any()

    sig = pd.DataFrame({"month": pd.to_datetime(["2021-01-31"] * 2), "cik": [1, 2],
                        "section": "risk_factors", "accession": ["a", "b"], "sim": [0.5, 0.9],
                        "symbol": ["DEAD", "LIVE"], "public_float": [1e9, 1e9]})
    rets = returns.monthly_returns(con)
    bankrupt = pd.DataFrame({"cik": [1], "filing_date": ["2021-02-01"]})
    held = returns.holding_returns(sig, rets, bankrupt).set_index("symbol")
    assert held.loc["DEAD", "ret"] == pytest.approx(0.5 * 0.7 - 1)     # -50%, then -30%
    assert held.loc["LIVE", "ret"] == pytest.approx(0.1)
    unadjusted = returns.holding_returns(sig, rets, bankrupt, delist_adjust=False).set_index("symbol")
    assert unadjusted.loc["DEAD", "ret"] == pytest.approx(-0.5)


def test_long_short_is_most_similar_minus_most_changed(monkeypatch):
    monkeypatch.setattr(returns, "MIN_FIRMS", 10)
    held = pd.DataFrame({"section": "risk_factors", "hold": pd.Timestamp("2021-02-28"),
                         "sim": range(10), "ret": [0.0] * 8 + [0.05, 0.07],
                         "public_float": [1.0] * 9 + [3.0]})
    p = returns.portfolios(held).set_index("weighting")
    assert p.loc["ew", "ls"] == pytest.approx(0.06)               # (0.05 + 0.07) / 2 - 0
    assert p.loc["vw", "ls"] == pytest.approx((0.05 + 0.21) / 4)


def test_newey_west_constant_is_the_mean():
    import numpy as np
    y = np.array([0.01, -0.02, 0.03, 0.0, 0.02])
    beta, se = returns.newey_west_ols(y, np.ones((5, 1)), lags=0)
    assert beta[0] == pytest.approx(y.mean())
    assert se[0] == pytest.approx(y.std(ddof=1) / np.sqrt(5), rel=1e-9)


# --- the whole chain on synthetic prices ----------------------------------------------

def _synthetic_chain(drift_per_sim: float, seed: int = 0):
    """60 firms x 48 months: signals -> next-month returns -> quintile long-short.
    Firm i's text similarity is fixed; its monthly return is noise plus
    drift_per_sim x (similarity rank - 0.5)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-31", periods=48, freq="ME")
    firms = np.arange(60)
    sim = rng.uniform(0.8, 1.0, len(firms))
    rank = pd.Series(sim).rank(pct=True).values - 0.5
    sig = pd.DataFrame([{"month": m, "cik": int(f), "section": "risk_factors", "accession": f"a{f}",
                         "sim": sim[f], "symbol": f"S{f}", "public_float": 1e9}
                        for m in months[:-1] for f in firms])
    rets = pd.DataFrame([{"symbol": f"S{f}", "month": m, "delisted": False,
                          "ret": rng.normal(drift_per_sim * rank[f], 0.08)}
                         for m in months[1:] for f in firms])
    held = returns.holding_returns(sig, rets, pd.DataFrame(columns=["cik", "filing_date"]))
    return returns.portfolios(held)


def test_planted_effect_is_found_with_the_right_sign(monkeypatch):
    monkeypatch.setattr(returns, "MIN_FIRMS", 50)
    ls = _synthetic_chain(0.04).query("weighting == 'ew'").ls
    assert len(ls) == 47
    assert ls.mean() / (ls.std() / len(ls) ** 0.5) > 3         # Q5 - Q1 = most similar minus most changed


def test_pure_noise_is_not_significant(monkeypatch):
    monkeypatch.setattr(returns, "MIN_FIRMS", 50)
    ls = _synthetic_chain(0.0, seed=1).query("weighting == 'ew'").ls
    assert abs(ls.mean() / (ls.std() / len(ls) ** 0.5)) < 2.5


def test_backtest_refuses_a_second_run(tmp_path):
    (tmp_path / "prices").mkdir()
    (tmp_path / "prices" / "phase4_results.json").write_text("{}")
    with pytest.raises(returns.AlreadyRun):
        returns.run(connect(":memory:"), tmp_path)
