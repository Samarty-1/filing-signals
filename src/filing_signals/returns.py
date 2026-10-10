"""Phase 4: the pre-registered return test (see docs/PREREGISTRATION.md).

Do firms whose 10-K text barely changed from last year outperform firms that
rewrote it (Cohen, Malloy & Nguyen 2020, "Lazy Prices")? Calendar-time
quintile portfolios, rebalanced monthly:

  formation   at each month end t, every firm's most recent original 10-K
              accepted on an earlier calendar day and within 365 days, with a
              year-over-year pair and public float >= $100M on its cover page
  signal      point-in-time TF-IDF similarity to last year's section
              (features.sim_tfidf_pit): high = unchanged
  holding     month t+1, from the month-t close to the month t+1 close
  portfolio   Q5 (most similar) minus Q1 (most changed); equal-weighted
              primary, float-weighted secondary
  delisting   a price series that ends inside the holding month earns its
              return to the last close; if the firm filed an 8-K Item 1.03
              (bankruptcy) in the year before, -30% is compounded on top
              (Shumway 1997). Cash from a delisting earns zero.

Everything here is fixed by the pre-registration, committed before any price
was downloaded.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

MIN_FLOAT = 1e8
MIN_FIRMS = 50               # a month needs this many firms to form quintiles
QUINTILES = 5
DELIST_RETURN = -0.30
NW_LAGS = 3
SECTIONS = ("risk_factors", "mdna", "both")
WEIGHTINGS = ("ew", "vw")
PRIMARY = ("risk_factors", "ew")
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FACTOR_FILES = {"F-F_Research_Data_5_Factors_2x3": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
                "F-F_Momentum_Factor": ["Mom"]}


# --- ticker validation ------------------------------------------------------------

VALIDATE_SQL = """
-- a resolved symbol is accepted for a report when (1) it has a price within
-- 7 calendar days of the filing and (2) price x shares outstanding on the
-- cover is consistent with the public float the same cover reports. A
-- recycled or mismatched ticker usually fails (2) by orders of magnitude.
with px as (
    select r.accession, r.symbol, r.filed,
           arg_min(p.close, abs(date_diff('day', p.day, r.filed))) as close_near,
           min(abs(date_diff('day', p.day, r.filed))) as gap_days
    from resolved_symbols r join daily_prices p on p.symbol = r.symbol
    where p.day between r.filed - interval 7 day and r.filed + interval 7 day
    group by all
),
cover as (
    select accession,
           max(value) filter (where concept = 'EntityPublicFloat') as public_float,
           max(value) filter (where concept = 'EntityCommonStockSharesOutstanding') as shares
    from xbrl_facts where taxonomy = 'dei' group by 1
)
select r.accession, r.cik, r.symbol, px.close_near, c.public_float, c.shares,
       c.public_float / nullif(px.close_near * c.shares, 0) as float_to_mcap,
       px.close_near is not null as has_price,
       c.shares is null or c.public_float is null
         or (c.public_float / nullif(px.close_near * c.shares, 0) between 0.05 and 2.0) as size_ok
from resolved_symbols r left join px using (accession) left join cover c using (accession)
"""


def validate_matches(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    from .prices import RESOLVED_SQL
    con.execute(f"create or replace table resolved_symbols as {RESOLVED_SQL}")
    v = con.execute(VALIDATE_SQL).df()
    con.execute("create or replace table report_symbols as select * from v where has_price and size_ok")
    return v


# --- signals and returns -----------------------------------------------------------

def monthly_returns(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """(symbol, month, ret, delisted) for every symbol-month. `month` is the
    month-end date of the holding month; ret runs from the previous month's
    last close to this month's last close (or to the final close, for a
    series that ends inside the month)."""
    return con.execute("""
        with last_day as (select max(day) as d from daily_prices),
        m as (
            select symbol, last_day(day) as month, arg_max(adj_close, day) as px, max(day) as day
            from daily_prices where adj_close > 0 group by all
        ),
        r as (
            select symbol, month, day,
                   px / lag(px) over (partition by symbol order by month) - 1 as ret,
                   lag(month) over (partition by symbol order by month) as prev_month,
                   max(day) over (partition by symbol) as series_end
            from m
        )
        select symbol, month, ret,
               -- the series ends here, before the data does: a delisting
               day = series_end and day < (select d - interval 10 day from last_day) as delisted
        from r
        where ret is not null and prev_month = last_day(month - interval 1 month)
    """).df()


SIGNALS_SQL = """
with months as (
    -- formation happens at the month's last close, so a filing counts only
    -- if it was accepted on an earlier day than that close
    select last_day(day) as month, max(day) as last_close from daily_prices group by 1
),
sig as (
    select sc.cik, sc.accession, sc.accepted_at, sc.section, sc.sim_tfidf_pit as sim
    from section_changes sc where not isnan(sc.sim_tfidf_pit)
    union all
    -- 'both': the mean of the two sections' within-month percentile ranks
    -- is formed later; here, both sections must be present
    select cik, accession, any_value(accepted_at), 'both', null
    from section_changes where not isnan(sim_tfidf_pit)
    group by cik, accession having count(*) = 2
),
latest as (
    select m.month, s.cik, s.section, s.accession, s.sim
    from months m join sig s
      on s.accepted_at::date < m.last_close
     and s.accepted_at::date >= m.last_close - interval 365 day
    qualify row_number() over (partition by m.month, s.cik, s.section order by s.accepted_at desc) = 1
)
select l.month, l.cik, l.section, l.accession, l.sim, rs.symbol, rs.public_float
from latest l
join report_symbols rs using (accession)
where rs.public_float >= {min_float}
"""


def signals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    s = con.execute(SIGNALS_SQL.format(min_float=MIN_FLOAT)).df()
    # 'both' = average of the two sections' percentile ranks in that month
    parts = s[s.section != "both"].copy()
    parts["pct"] = parts.groupby(["month", "section"]).sim.rank(pct=True)
    avg = parts.groupby(["month", "cik"]).pct.mean().rename("sim_both")
    both = s[s.section == "both"].join(avg, on=["month", "cik"])
    both["sim"] = both.pop("sim_both")
    return pd.concat([parts.drop(columns="pct"), both], ignore_index=True)


def bankruptcy_filers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        select cik, filing_date from filings
        where form like '8-K%' and items like '%1.03%'""").df()


def holding_returns(sig: pd.DataFrame, rets: pd.DataFrame, bankrupt: pd.DataFrame,
                    delist_adjust: bool = True) -> pd.DataFrame:
    """Join each formation month's signals to the next month's returns."""
    sig = sig.copy()
    sig["hold"] = (sig.month + pd.offsets.MonthEnd(1))
    out = sig.merge(rets.rename(columns={"month": "hold"}), on=["symbol", "hold"], how="inner")
    if delist_adjust and len(out):
        b = bankrupt.copy()
        b["filing_date"] = pd.to_datetime(b.filing_date)
        hit = out[out.delisted][["cik", "hold"]].merge(b, on="cik")
        hit = hit[(hit.filing_date <= hit.hold) & (hit.filing_date > hit.hold - pd.Timedelta(days=365))]
        flag = out.set_index(["cik", "hold"]).index.isin(hit.set_index(["cik", "hold"]).index)
        out.loc[flag, "ret"] = (1 + out.loc[flag, "ret"]) * (1 + DELIST_RETURN) - 1
        out["shumway_adjusted"] = flag
    return out


def portfolios(held: pd.DataFrame) -> pd.DataFrame:
    """Monthly quintile returns -> long-short (Q5 - Q1) per section x weighting."""
    rows = []
    for (section, hold), g in held.groupby(["section", "hold"]):
        if len(g) < MIN_FIRMS:
            continue
        q = pd.qcut(g.sim.rank(method="first"), QUINTILES, labels=False) + 1
        for w in WEIGHTINGS:
            wt = pd.Series(1.0, index=g.index) if w == "ew" else g.public_float
            by_q = (g.ret * wt).groupby(q).sum() / wt.groupby(q).sum()
            rows.append({"section": section, "weighting": w, "month": hold, "n": len(g),
                         **{f"q{k}": by_q.get(k, np.nan) for k in range(1, QUINTILES + 1)},
                         "ls": by_q.get(QUINTILES, np.nan) - by_q.get(1, np.nan)})
    return pd.DataFrame(rows)


# --- inference -------------------------------------------------------------------

def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett kernel) standard errors."""
    n = len(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = xu[lag:].T @ xu[:-lag]
        S += w * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    cov = inv @ S @ inv * n / (n - X.shape[1])
    return beta, np.sqrt(np.diag(cov))


def load_factors(cache: Path) -> pd.DataFrame:
    """Monthly Fama-French 5 factors + momentum, in decimals, indexed by month end."""
    import requests

    frames = []
    for name, cols in FACTOR_FILES.items():
        path = cache / f"{name}.zip"
        if not path.exists():
            r = requests.get(FRENCH.format(name=name), timeout=60)
            r.raise_for_status()
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        # the monthly table comes first: rows whose first field is YYYYMM
        lines, started = [], False
        for line in text.splitlines():
            head = line.split(",")[0].strip()
            if len(head) == 6 and head.isdigit():
                lines.append(line)
                started = True
            elif started and lines:
                break
        df = pd.read_csv(io.StringIO("\n".join(lines)), header=None)
        df.columns = ["yyyymm"] + [c for c in cols]
        df["month"] = pd.to_datetime(df.yyyymm.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
        frames.append(df.set_index("month")[cols].astype(float) / 100)
    return frames[0].join(frames[1], how="inner")


def evaluate(ls: pd.Series, factors: pd.DataFrame) -> dict:
    """Mean, t-stat, Sharpe and FF5+UMD alpha (Newey-West) of a monthly series."""
    ls = ls.dropna()
    joined = pd.concat([ls.rename("ls"), factors], axis=1, join="inner").dropna()
    n = len(ls)
    mean, sd = ls.mean(), ls.std(ddof=1)
    _, se_mean = newey_west_ols(ls.values, np.ones((n, 1)))
    X = np.column_stack([np.ones(len(joined)),
                         joined[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].values])
    beta, se = newey_west_ols(joined.ls.values, X)
    return {
        "months": n, "first": str(ls.index.min().date()), "last": str(ls.index.max().date()),
        "mean_monthly": round(float(mean), 5), "t_mean_nw": round(float(mean / se_mean[0]), 2),
        "sharpe_annual": round(float(mean / sd * math.sqrt(12)), 3),
        "alpha_monthly": round(float(beta[0]), 5), "t_alpha_nw": round(float(beta[0] / se[0]), 2),
        "loadings": dict(zip(["mkt", "smb", "hml", "rmw", "cma", "mom"], np.round(beta[1:], 3).tolist())),
    }


def deflated_sharpe(trials: dict[str, pd.Series], chosen: str) -> dict:
    """Bailey & Lopez de Prado (2014) DSR of the pre-registered primary series,
    against the expected maximum Sharpe of all reported trials. Same formulas
    as Samarty-1/backtest-overfit-audit src/sharpe.py (per-period Sharpe,
    non-excess kurtosis)."""
    euler = 0.5772156649015329
    sr = {k: v.mean() / v.std(ddof=1) for k, v in trials.items()}
    var = float(np.var(list(sr.values()), ddof=1))
    n_trials = len(trials)
    q1, q2 = stats.norm.ppf(1 - 1 / n_trials), stats.norm.ppf(1 - 1 / (n_trials * math.e))
    sr_max = math.sqrt(var) * ((1 - euler) * q1 + euler * q2) if var > 0 else 0.0
    r = trials[chosen].dropna()
    n, skew, kurt = len(r), stats.skew(r, bias=False), stats.kurtosis(r, fisher=False, bias=False)
    s = sr[chosen]
    z = (s - sr_max) * math.sqrt(n - 1) / math.sqrt(1 - skew * s + (kurt - 1) / 4 * s * s)
    return {"n_trials": n_trials, "sharpe_monthly": round(float(s), 4),
            "expected_max_sharpe_null": round(float(sr_max), 4), "dsr": round(float(stats.norm.cdf(z)), 3)}


def minimum_detectable_effect(ls: pd.Series, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest true monthly mean the test would detect with this power."""
    ls = ls.dropna()
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * ls.std(ddof=1) / math.sqrt(len(ls)))


class AlreadyRun(RuntimeError):
    pass


def run(con: duckdb.DuckDBPyConnection, data: Path, log=print, force: bool = False) -> dict:
    """The pre-registered test runs once. A second run would let the result
    be re-rolled after seeing it, so it refuses unless forced, and a forced
    rerun is recorded as such in the results."""
    done = data / "prices" / "phase4_results.json"
    if done.exists() and not force:
        raise AlreadyRun(f"{done} exists: the pre-registered test has already run. "
                         "Rerunning is exploratory; pass --force and say so in the write-up.")
    v = validate_matches(con)
    log(f"symbols: {len(v)} reports resolved, {int(v.has_price.sum())} priced near filing, "
        f"{int((v.has_price & v.size_ok).sum())} pass the float/market-cap check")
    rets = monthly_returns(con)
    sig = signals(con)
    bankrupt = bankruptcy_filers(con)
    factors = load_factors(data / "prices" / "french")
    results = {"forced_rerun": done.exists(), "validation": {"resolved": len(v), "priced": int(v.has_price.sum()),
                              "accepted": int((v.has_price & v.size_ok).sum())}}
    for adjust in (True, False):
        held = holding_returns(sig, rets, bankrupt, delist_adjust=adjust)
        port = portfolios(held)
        key = "shumway" if adjust else "no_delist_adjustment"
        trials = {f"{s}/{w}": g.set_index("month").ls
                  for (s, w), g in port.groupby(["section", "weighting"])}
        block = {name: evaluate(series, factors) for name, series in trials.items()}
        primary = "/".join(PRIMARY)
        if primary in trials:
            block["deflated_sharpe"] = deflated_sharpe(trials, primary)
            block["mde_monthly"] = round(minimum_detectable_effect(trials[primary]), 5)
            p = block[primary]
            block["verdict"] = ("CONFIRMED" if p["alpha_monthly"] > 0 and p["t_alpha_nw"] >= 2.0
                                and block["deflated_sharpe"]["dsr"] >= 0.95 else "NOT CONFIRMED")
        if adjust:
            block["firms_per_month_median"] = int(port[port.weighting == "ew"].groupby("section").n.median().min())
            block["shumway_adjusted_obs"] = int(held.get("shumway_adjusted", pd.Series(dtype=bool)).sum())
            port.to_parquet(data / "prices" / "portfolios.parquet")
        results[key] = block
        summary = {k: (v.get("t_alpha_nw", v.get("dsr")) if isinstance(v, dict) else v) for k, v in block.items()}
        log(f"{key}: {json.dumps(summary)}")
    (data / "prices" / "phase4_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results
