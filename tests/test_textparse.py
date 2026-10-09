"""The section-extraction spec, as examples. Every test runs against both the
Python reference and the Rust parser (when it is built)."""
import pytest

from filing_signals import textparse

try:
    from filing_signals import _parser as rust
except ImportError:                     # pure-Python install: reference only
    rust = None

IMPLS = [pytest.param(textparse.parse_document, id="python")]
if rust is not None:
    IMPLS.append(pytest.param(rust.parse_document, id="rust"))

FILLER = "<p>" + "Our business faces many risks. " * 60 + "</p>"
MDNA = "<p>" + "Revenue grew because demand rose. " * 60 + "</p>"


def doc(body: str) -> bytes:
    return f"<html><head><title>10-K</title></head><body>{body}</body></html>".encode()


TOC = ("<table><tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>"
       "<tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>20</td></tr>"
       "<tr><td>Item 7.</td><td>Management&#8217;s Discussion and Analysis</td><td>30</td></tr>"
       "<tr><td>Item 8.</td><td>Financial Statements</td><td>40</td></tr></table>")


@pytest.mark.parametrize("parse", IMPLS)
def test_body_beats_table_of_contents(parse):
    out = parse(doc(TOC + "<p>Item 1A. Risk Factors</p>" + FILLER + "<p>Item 1B. Unresolved Staff Comments</p>"
                    + "<p>None.</p><p>ITEM 7. MANAGEMENT&#146;S DISCUSSION</p>" + MDNA
                    + "<p>Item 8. Financial Statements</p>"))
    assert out["risk_factors"]["status"] == "found"
    assert out["risk_factors"]["text"].startswith("Our business faces")
    assert "Unresolved" not in out["risk_factors"]["text"]
    assert out["mdna"]["status"] == "found" and out["mdna"]["text"].startswith("Revenue grew")


@pytest.mark.parametrize("parse", IMPLS)
def test_cross_reference_is_not_a_heading(parse):
    out = parse(doc("<p>See the discussion in Item 1A. Risk Factors for more.</p>" + FILLER))
    assert out["risk_factors"]["status"] == "not_found"


@pytest.mark.parametrize("parse", IMPLS)
def test_bare_number_heading_needs_the_title(parse):
    found = parse(doc(TOC + "<p>1A. RISK FACTORS</p>" + FILLER + "<p>1B. UNRESOLVED STAFF COMMENTS</p>"))
    assert found["risk_factors"]["status"] == "found"
    # "7." starting a numbered list item is not Item 7
    listed = parse(doc("<p>7. We may lose key customers.</p>" + MDNA + "<p>8. Rates may rise.</p>"))
    assert listed["mdna"]["status"] == "not_found"


@pytest.mark.parametrize("parse", IMPLS)
def test_smaller_company_omission(parse):
    out = parse(doc("<p>Item 1A. Risk Factors</p><p>Not required for smaller reporting companies.</p>"
                    "<p>Item 1B. Unresolved Staff Comments</p>"))
    assert out["risk_factors"]["status"] == "omitted"


@pytest.mark.parametrize("parse", IMPLS)
def test_incorporated_by_reference(parse):
    out = parse(doc("<p>Item 7. Management's Discussion and Analysis</p>"
                    "<p>Incorporated herein by reference to the Annual Report.</p><p>Item 8. Financial</p>"))
    assert out["mdna"]["status"] == "by_reference"


@pytest.mark.parametrize("parse", IMPLS)
def test_item_seven_a_variants_end_mdna(parse):
    out = parse(doc("<p>Item 7. Management's Discussion</p>" + MDNA + "<p>Item 7.A. Quantitative</p>"))
    assert out["mdna"]["status"] == "found" and "Quantitative" not in out["mdna"]["text"]


@pytest.mark.parametrize("parse", IMPLS)
def test_unterminated_section(parse):
    out = parse(doc("<p>Item 1A. Risk Factors</p>" + FILLER))
    assert out["risk_factors"]["status"] == "unterminated"


@pytest.mark.parametrize("parse", IMPLS)
def test_hidden_xbrl_scripts_and_comments_are_dropped(parse):
    out = parse(doc("<ix:header><ix:hidden>dei:Secret 123</ix:hidden></ix:header>"
                    "<script>var x = 'Item 1A. Risk Factors';</script><!-- Item 7. hidden -->"
                    "<p>Item 1A. Risk Factors</p>" + FILLER + "<p>Item 2. Properties</p>"))
    assert out["risk_factors"]["status"] == "found"
    assert out["n_chars"] < 2500 and "Secret" not in out["risk_factors"]["text"]


@pytest.mark.parametrize("parse", IMPLS)
def test_tags_split_inside_words_and_entities(parse):
    out = parse(doc("<p><span>It</span><span>em</span>&#160;1A.&nbsp;Risk&#160;Factors</p>"
                    "<p>Rates &amp; FX: the &#147;company&#148; can&#146;t hedge &#x2014; &bogus; "
                    + "x " * 600 + "</p><p>Item 1B. None</p>"))
    rf = out["risk_factors"]
    assert rf["status"] == "found"
    assert rf["text"].startswith("Rates & FX: the “company” can’t hedge — &bogus;")


@pytest.mark.parametrize("parse", IMPLS)
def test_windows_1252_bytes(parse):
    raw = (b"<p>Item 1A. Risk Factors</p><p>We can\x92t promise \x93anything\x94. "
           + b"x " * 600 + b"</p><p>Item 2. Properties</p>")
    rf = parse(raw)["risk_factors"]
    assert rf["status"] == "found" and rf["text"].startswith("We can’t promise “anything”.")
