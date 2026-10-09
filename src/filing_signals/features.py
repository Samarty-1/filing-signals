"""Phase 2b: year-over-year change in each section (the Lazy Prices signal).

Cohen, Malloy & Nguyen (2020, "Lazy Prices") find that firms which change
their 10-K language from one year to the next underperform. The change is
measured by similarity between this year's section and last year's:

  sim_cosine   cosine similarity of raw term-frequency vectors
  sim_jaccard  |words in both| / |words in either|
  sim_tfidf    cosine similarity of TF-IDF vectors (IDF fit on all sections),
               which down-weights boilerplate every filing shares

Pairs are point-in-time. "This year" is the original 10-K, as filed. "Last
year" is whichever version of the prior fiscal year's report was public at
that moment, so a later amendment cannot leak backwards. A pair needs fiscal
period ends 300-430 days apart; anything else is a gap or a fiscal-year change.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import duckdb
import pyarrow as pa

_WORD = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    small, large = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * large.get(k, 0) for k, v in small.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def jaccard(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    sa, sb = a.keys(), b.keys()
    return len(sa & sb) / len(sa | sb)


PAIRS_SQL = """
with cur as (
    -- this year's report: the original filing, with the section found
    select a.accession, a.cik, a.fiscal_period_end, a.accepted_at, s.section, s.text
    from annual_reports a
    join sections s using (accession)
    where a.version_no = 1 and s.status = 'found'
),
prior as (
    select a.accession, a.cik, a.fiscal_period_end, a.accepted_at, s.section, s.text
    from annual_reports a
    join sections s using (accession)
    where s.status = 'found'
)
select
    c.cik, c.section,
    c.accession                         as accession,
    c.fiscal_period_end,
    c.accepted_at,
    p.accession                         as prior_accession,
    p.fiscal_period_end                 as prior_period_end,
    c.text                              as text,
    p.text                              as prior_text
from cur c
join prior p
  on  p.cik = c.cik
  and p.section = c.section
  and date_diff('day', p.fiscal_period_end, c.fiscal_period_end) between 300 and 430
  and p.accepted_at < c.accepted_at
-- the latest prior version already public when this report was filed
qualify row_number() over (
    partition by c.accession, c.section order by p.accepted_at desc, p.accession desc
) = 1
"""


def compute(con: duckdb.DuckDBPyConnection, log=print) -> int:
    # timestamps stay in SQL (pairs table); Python only sees keys and text
    con.execute(f"create or replace temp table pairs as {PAIRS_SQL}")
    rows = [dict(zip(("accession", "prior_accession", "section", "text", "prior_text"), r))
            for r in con.execute("select accession, prior_accession, section, text, prior_text "
                                 "from pairs").fetchall()]
    log(f"features: {len(rows)} year-over-year section pairs")
    counts = {}
    for r in rows:
        for key in ("accession", "prior_accession"):
            ck = (r[key], r["section"])
            if ck not in counts:
                counts[ck] = Counter(tokens(r["text" if key == "accession" else "prior_text"]))

    # IDF over every distinct section document, per section type
    df: dict[str, Counter] = {}
    n_docs: Counter = Counter()
    for (acc, section), cnt in counts.items():
        df.setdefault(section, Counter()).update(cnt.keys())
        n_docs[section] += 1
    idf = {sec: {w: math.log((1 + n_docs[sec]) / (1 + d)) + 1 for w, d in dfs.items()}
           for sec, dfs in df.items()}

    def tfidf(cnt: Counter, section: str) -> Counter:
        w = idf[section]
        return Counter({k: v * w[k] for k, v in cnt.items()})

    out = []
    for r in rows:
        a = counts[(r["accession"], r["section"])]
        b = counts[(r["prior_accession"], r["section"])]
        out.append({
            "accession": r["accession"], "section": r["section"],
            "words": sum(a.values()), "prior_words": sum(b.values()),
            "sim_cosine": cosine(a, b), "sim_jaccard": jaccard(a, b),
            "sim_tfidf": cosine(tfidf(a, r["section"]), tfidf(b, r["section"])),
        })
    sims = pa.Table.from_pylist(out)  # noqa: F841 (read by name below)
    con.execute("""
        create or replace table section_changes as
        select p.cik, p.section, p.accession, p.fiscal_period_end, p.accepted_at,
               p.prior_accession, p.prior_period_end,
               s.words, s.prior_words, s.sim_cosine, s.sim_jaccard, s.sim_tfidf
        from pairs p join sims s using (accession, section)
    """)
    return len(out)
