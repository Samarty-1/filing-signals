from datetime import date, datetime, timezone

from filing_signals import edgar

MASTER = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 2015

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
1000045|NICHOLAS FINANCIAL INC|10-Q|2015-02-13|edgar/data/1000045/0001193125-15-048402.txt
1000180|SANDISK CORP|10-K|2015-02-11|edgar/data/1000180/0001000180-15-000010.txt
1000209|MEDALLION FINANCIAL CORP|10-K/A|2015-03-16|edgar/data/1000209/0001193125-15-091010.txt
"""


def test_master_index_rows():
    rows = edgar.parse_master_index(MASTER)
    assert len(rows) == 3
    assert rows[1] == {"cik": 1000180, "company_name": "SANDISK CORP", "form": "10-K",
                       "date_filed": date(2015, 2, 11), "accession": "0001000180-15-000010"}


def test_header_acceptance_is_eastern_and_respects_dst():
    # EDGAR records acceptance in Eastern time: UTC-5 in winter, UTC-4 in summer
    winter = edgar.parse_header("<ACCEPTANCE-DATETIME>20150211161502\n"
                                "CONFORMED PERIOD OF REPORT:\t20141228\n")
    summer = edgar.parse_header("<ACCEPTANCE-DATETIME>20240801060136\n")
    assert winter["accepted_at"] == datetime(2015, 2, 11, 21, 15, 2, tzinfo=timezone.utc)
    assert winter["header_period"] == date(2014, 12, 28)
    assert summer["accepted_at"] == datetime(2024, 8, 1, 10, 1, 36, tzinfo=timezone.utc)
    assert summer["header_period"] is None


def test_filings_are_column_arrays():
    block = {
        "accessionNumber": ["a1", "a2"], "form": ["10-K", "8-K"],
        "filingDate": ["2024-11-01", "2024-10-31"], "reportDate": ["2024-09-28", ""],
        "acceptanceDateTime": ["2024-11-01T14:01:36.000Z", ""],
        "primaryDocument": ["x.htm", ""], "size": [10, 20], "items": ["", "2.02"],
    }
    rows = edgar.parse_filings(320193, block)
    assert rows[0]["report_date"] == date(2024, 9, 28)
    assert rows[1]["report_date"] is None and rows[1]["primary_document"] is None
    assert rows[1]["items"] == "2.02"


def test_document_url_falls_back_to_full_submission():
    assert edgar.document_url(1, "0000000001-10-000001", None).endswith(
        "/data/1/000000000110000001/0000000001-10-000001.txt")


def test_filer_former_names():
    filer = edgar.parse_filer({
        "cik": "320193", "name": "Apple Inc.", "entityType": "operating", "sic": "3571",
        "tickers": ["AAPL"], "formerNames": [
            {"name": "APPLE COMPUTER INC", "from": "1994-01-26T00:00:00.000Z",
             "to": "2007-01-04T00:00:00.000Z"}]})
    assert filer["sic"] == 3571 and filer["current_tickers"] == "AAPL"
    assert filer["former_names"] == [{"name": "APPLE COMPUTER INC",
                                      "valid_from": date(1994, 1, 26),
                                      "valid_to": date(2007, 1, 4)}]


def test_hdr_sgml_period_tag():
    # .hdr.sgml writes the period as a tag, not the "CONFORMED PERIOD" line
    got = edgar.parse_header("<SEC-HEADER>x.hdr.sgml : 20120605\n"
                             "<ACCEPTANCE-DATETIME>20120605160807\n<TYPE>10-K\n<PERIOD>20120331\n")
    assert got["header_period"] == date(2012, 3, 31)
    assert got["accepted_at"] == datetime(2012, 6, 5, 20, 8, 7, tzinfo=timezone.utc)


def test_header_url_is_the_sgml_file():
    assert edgar.header_url(885322, "0001193125-12-260165").endswith(
        "/000119312512260165/0001193125-12-260165.hdr.sgml")
