"""End-to-end over a fake EDGAR: four filers, one amended 10-K, one excluded
asset-backed trust, one fund. Checks sampling, versions, point-in-time reads
and that a second run fetches nothing."""
import gzip
import json
from datetime import datetime, timezone

import pytest

from filing_signals import edgar
from filing_signals.http import NotFound, Stats
from filing_signals.pipeline import Params, Pipeline
from filing_signals.warehouse import RawStore, connect

INDEX_HEAD = "CIK|Company Name|Form Type|Date Filed|Filename\n" + "-" * 80 + "\n"


def index_line(cik, form, filed, acc):
    return f"{cik}|Co {cik}|{form}|{filed}|edgar/data/{cik}/{acc}.txt\n"


def submissions(cik, entity_type, sic, filings):
    cols = ["accessionNumber", "form", "filingDate", "reportDate", "acceptanceDateTime",
            "primaryDocument", "size", "items"]
    recent = {c: [] for c in cols}
    for f in filings:
        for c, v in zip(cols, f):
            recent[c].append(v)
    return json.dumps({"cik": str(cik), "name": f"Co {cik}", "entityType": entity_type,
                       "sic": str(sic), "tickers": [], "formerNames": [],
                       "filings": {"recent": recent, "files": []}}).encode()


def header(accepted_et, period):
    return f"<ACCEPTANCE-DATETIME>{accepted_et}\nCONFORMED PERIOD OF REPORT:\t{period}\n".encode()


class FakeEdgar:
    def __init__(self):
        self.stats = Stats()
        self.urls = {}
        q1 = (INDEX_HEAD
              + index_line(1, "10-K", "2015-03-02", "0000000001-15-000001")
              + index_line(1, "10-K/A", "2015-03-20", "0000000001-15-000002")
              + index_line(2, "10-K", "2015-03-03", "0000000002-15-000001")
              + index_line(3, "10-K", "2015-03-04", "0000000003-15-000001")
              + index_line(4, "10-K", "2015-03-05", "0000000004-15-000001"))
        self.urls[edgar.master_index_url(2015, 1).replace("master.idx", "master.gz")] = gzip.compress(q1.encode())
        for q in (2, 3, 4):
            self.urls[edgar.master_index_url(2015, q).replace("master.idx", "master.gz")] = gzip.compress(INDEX_HEAD.encode())
        # firm 1: a 10-K accepted 18:05 ET on 27 Feb (so dated the next business
        # day, 2 Mar), amended on 20 Mar. The API timestamp is 8 h off.
        self.urls[edgar.submissions_url(1)] = submissions(1, "operating", 3571, [
            ["0000000001-15-000001", "10-K", "2015-03-02", "2014-12-31",
             "2015-02-28T07:05:00.000Z", "k.htm", 100, ""],
            ["0000000001-15-000002", "10-K/A", "2015-03-20", "2014-12-31",
             "2015-03-20T14:00:00.000Z", "ka.htm", 50, ""],
            ["0000000001-15-000003", "8-K", "2015-04-01", "", "", "e.htm", 5, "2.02"]])
        self.urls[edgar.submissions_url(2)] = submissions(2, "operating", 6189, [])
        self.urls[edgar.submissions_url(3)] = submissions(3, "other", 6798, [])
        self.urls[edgar.submissions_url(4)] = submissions(4, "operating", 2834, [
            ["0000000004-15-000001", "10-K", "2015-03-05", "2014-12-31",
             "2015-03-05T21:00:00.000Z", "", 80, ""]])
        for acc, et, period in [("0000000001-15-000001", "20150227180500", "20141231"),
                                ("0000000001-15-000002", "20150320100000", "20141231"),
                                ("0000000004-15-000001", "20150305160000", "20141231")]:
            cik = int(acc[:10])
            self.urls[edgar.header_url(cik, acc)] = header(et, period)
        self.urls[edgar.document_url(1, "0000000001-15-000001", "k.htm")] = b"<html>original</html>"
        self.urls[edgar.document_url(1, "0000000001-15-000002", "ka.htm")] = b"<html>amended</html>"
        # firm 4's document is missing on the server: must be recorded, not crash
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        self.stats.add(requests=1)
        if url not in self.urls:
            raise NotFound(url)
        return self.urls[url]


@pytest.fixture
def run(tmp_path):
    fake = FakeEdgar()
    con = connect(tmp_path / "w.duckdb")
    pipe = Pipeline(con, fake, RawStore(tmp_path / "raw"), log=lambda *_: None)
    params = Params(2015, 2015, n_firms=5, seed=1, workers=2)
    pipe.run(params)
    return con, fake, pipe, params


def q(con, sql, *args):
    return con.execute(sql, list(args)).fetchall()


def test_universe_records_every_decision(run):
    con, *_ = run
    got = dict((c, (s, r)) for c, s, r in q(con, "select cik, status, reason from universe"))
    assert got == {1: ("included", None), 2: ("excluded", "sic=6189 (asset-backed)"),
                   3: ("excluded", "entity_type=other"), 4: ("included", None)}


def test_amendment_is_a_second_version(run):
    con, *_ = run
    got = q(con, """select accession, version_no, original_accession, is_amendment
                    from annual_reports where cik = 1 order by version_no""")
    assert got == [("0000000001-15-000001", 1, "0000000001-15-000001", False),
                   ("0000000001-15-000002", 2, "0000000001-15-000001", True)]


def test_point_in_time_read(run):
    con, *_ = run
    def as_of(ts):
        return q(con, "select accession from annual_reports_as_of(?::timestamptz) where cik = 1", ts)
    assert as_of("2015-02-27 23:00:00+00") == []                          # not yet accepted
    assert as_of("2015-03-01 00:00:00+00") == [("0000000001-15-000001",)]  # original
    assert as_of("2015-03-21 00:00:00+00") == [("0000000001-15-000002",)]  # amended


def test_after_hours_filing_is_dated_next_business_day(run):
    con, *_ = run
    got = q(con, """select filing_date, accepted_at_et, dated_next_business_day
                    from annual_reports where accession = '0000000001-15-000001'""")[0]
    assert str(got[0]) == "2015-03-02"
    assert got[1] == datetime(2015, 2, 27, 18, 5)
    assert got[2] is True


def test_api_timestamp_offset_is_measured(run):
    con, *_ = run
    got = dict(q(con, "select accession, api_minus_header_hours from api_timestamp_check"))
    # header 18:05 EST = 23:05 UTC; API says 07:05 "Z" the next day
    assert got["0000000001-15-000001"] == 8.0
    # 10:00 EDT (DST began 8 March 2015) = 14:00 UTC: this one is right
    assert got["0000000001-15-000002"] == 0.0


def test_missing_document_recorded_not_fatal(run):
    con, *_ = run
    assert q(con, "select accession, error from fetch_failures") == [
        ("0000000004-15-000001", "NotFound")]


def test_documents_stored_with_provenance(run):
    con, _, pipe, _ = run
    path, sha = q(con, "select stored_path, sha256 from documents "
                       "where accession = '0000000001-15-000002'")[0]
    assert pipe.store.read(*path.split("/")) == b"<html>amended</html>"
    assert len(sha) == 64


def test_second_run_fetches_nothing_new(run):
    con, fake, pipe, params = run
    before = len(fake.requested)
    pipe.run(params)
    # only the in-progress-quarter refresh rule could refetch; 2015 is long closed
    assert fake.requested[before:] == []
