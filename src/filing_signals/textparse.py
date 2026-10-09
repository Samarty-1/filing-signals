"""Reference implementation: 10-K document -> clean lines -> Item sections.

This is the specification. The Rust parser in `parser/` implements the same
algorithm and must produce identical output (tests/test_parser_parity.py runs
both over every stored filing). Python stays as the readable version and the
baseline for the speed comparison.

Algorithm
---------
1. Decode: UTF-8 if valid, else Windows-1252 (WHATWG mapping: the five
   undefined bytes map to the C1 code point with the same value). Older
   filings are cp1252; decoding them as UTF-8 would destroy every curly quote.
2. Drop comments and the contents of <script>, <style>, <head> and
   <ix:header> (inline-XBRL hidden facts, invisible on the rendered page).
3. Block-level tags become line breaks, table cells become spaces, and other
   tags vanish (so "<span>Item</span><span>1A</span>" reads as it renders).
4. Decode entities, including numeric references in the 128-159 range, which
   filings use as cp1252 code points (&#146; is a right quote).
5. Normalise whitespace within lines and drop empty lines.
6. Section boundaries: an Item heading counts only at the start of a short
   line, which excludes mid-sentence cross-references ("see Item 1A
   herein"). A section runs from its heading to the next heading of a *later*
   item. When the start heading occurs more than once (the table of contents
   and the body), the occurrence with the longest span wins: TOC entries are
   followed almost at once by the next item's line.
"""
from __future__ import annotations

import html.entities
import re

# --- 1. decoding -------------------------------------------------------------

_CP1252 = []
for b in range(256):
    try:
        _CP1252.append(bytes([b]).decode("cp1252"))
    except UnicodeDecodeError:
        _CP1252.append(chr(b))          # WHATWG: undefined bytes -> same code point
_CP1252_TABLE = str.maketrans({i: c for i, c in enumerate(_CP1252)})


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1").translate(_CP1252_TABLE)


# --- 2-5. html -> lines --------------------------------------------------------

BLOCK_TAGS = frozenset("""
    address article aside blockquote br center dd div dl dt fieldset figcaption figure
    footer form h1 h2 h3 h4 h5 h6 header hr li main nav ol p page pre section table
    tbody thead tfoot title tr ul
""".split())
CELL_TAGS = frozenset({"td", "th"})
DROP_CONTENT = ("script", "style", "head", "ix:header")

# Non-ASCII characters are written as code points: invisible characters typed
# literally into a pattern are unreviewable.
SPACE_CHARS = " \t\r\f\v" + "".join(map(chr, [0xA0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000]))
ZERO_WIDTH_CHARS = "".join(map(chr, [0x200B, 0x200C, 0x200D, 0xFEFF]))
DASHES = chr(0x2013) + chr(0x2014)
REPLACEMENT = chr(0xFFFD)

# Patterns spell out ASCII classes ([0-9], literal spaces) instead of \d and
# \s, which mean different Unicode sets in Python and Rust; the two parsers
# must agree byte for byte.
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_DROPS = [re.compile(rf"<{t}(?![A-Za-z0-9_:-]).*?</{t}[ \t\r\n]*>", re.S | re.I)
          for t in DROP_CONTENT]
_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9:_-]*)[^>]*>")
_ENTITY = re.compile(r"&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_SPACES = re.compile("[" + re.escape(SPACE_CHARS) + "]+")
_ZERO_WIDTH = re.compile("[" + ZERO_WIDTH_CHARS + "]")
_NAMED = html.entities.name2codepoint      # the HTML 4 set; Rust embeds the same table



def _tag(m: re.Match) -> str:
    name = m.group(2).lower()
    if name in BLOCK_TAGS:
        return "\n"
    if name in CELL_TAGS:
        return " "
    return ""


def _entity(m: re.Match) -> str:
    body = m.group(1)
    if body[0] == "#":
        n = int(body[2:], 16) if body[1] in "xX" else int(body[1:])
        if 128 <= n <= 159:
            return _CP1252[n]
        if n == 0 or n > 0x10FFFF or 0xD800 <= n <= 0xDFFF:
            return REPLACEMENT
        return chr(n)
    cp = _NAMED.get(body)
    return chr(cp) if cp is not None else m.group(0)


def to_lines(raw: bytes) -> list[str]:
    text = decode(raw)
    text = _COMMENT.sub("", text)
    for pattern in _DROPS:
        text = pattern.sub("", text)
    text = _TAG.sub(_tag, text)
    text = _ENTITY.sub(_entity, text)
    text = _ZERO_WIDTH.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = _SPACES.sub(" ", line).strip(" ")
        if line:
            lines.append(line)
    return lines


# --- 6. sections -----------------------------------------------------------------

ITEM_ORDER = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
              "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"]
RANK = {item: i for i, item in enumerate(ITEM_ORDER)}
SECTIONS = {"risk_factors": "1A", "mdna": "7"}
MAX_HEADING_LEN = 200

_HEADING = re.compile(
    r"^(?:part +i{1,3} *[,.:\-" + DASHES + r"]? *)?items? *([0-9]{1,2})(?:\.?([a-cA-C]))?(?![A-Za-z0-9])",
    re.I,
)
# Many filings drop the word "Item" from body headings ("1A. RISK FACTORS"),
# keeping it only in the table of contents. A bare number counts as a heading
# only when followed by that item's standard title, so numbered lists
# ("7. We may not...") never match.
_BARE = re.compile(
    r"^([0-9]{1,2})(?:\.?([a-cA-C]))?(?![A-Za-z0-9]) *[.:\-" + DASHES + r"]? *(.*)$", re.S)
_TITLES = {item: re.compile(title.replace(" ", r"[ \n]+"), re.I) for item, title in {
    "1": r"business", "1A": r"risk factors", "1B": r"unresolved staff",
    "1C": r"cybersecurity", "2": r"(?:description of )?propert", "3": r"legal proceedings",
    "4": r"mine safety|submission of matters|\(?removed and reserved|reserved",
    "5": r"market for", "6": r"selected (?:consolidated )?financial|\[?reserved",
    "7": r"management.{0,3}s discussion", "7A": r"quantitative and qualitative",
    "8": r"(?:consolidated )?financial statements", "9": r"changes in and disagreements",
    "9A": r"controls and procedures", "9B": r"other information",
    "9C": r"disclosure regarding foreign", "10": r"directors", "11": r"executive compensation",
    "12": r"security ownership", "13": r"certain relationships",
    "14": r"principal account", "15": r"exhibits", "16": r"form 10-k summary",
}.items()}

_OMITTED = re.compile(r"not[ \n]+(?:required|applicable)|smaller[ \n]+reporting[ \n]+compan|omitted", re.I)
_BY_REFERENCE = re.compile(r"incorporated[ \n]+(?:herein[ \n]+)?by[ \n]+reference", re.I)
MIN_BODY_CHARS = 1000


def headings(lines: list[str]) -> list[tuple[int, str]]:
    """(line index, item) for every line that starts with an Item heading."""
    out = []
    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_LEN:
            continue
        m = _HEADING.match(line)
        if m:
            item = m.group(1).lstrip("0") + (m.group(2) or "").upper()
            if item in RANK:
                out.append((i, item))
            continue
        m = _BARE.match(line)
        if m:
            item = m.group(1).lstrip("0") + (m.group(2) or "").upper()
            if item in _TITLES and _TITLES[item].match(m.group(3)):
                out.append((i, item))
    return out


def extract(lines: list[str], item: str) -> dict:
    """Locate one Item. status: found | omitted | by_reference | too_short |
    unterminated | not_found."""
    hs = headings(lines)
    best = None
    for k, (start, it) in enumerate(hs):
        if it != item:
            continue
        end = next((j for j, later in hs[k + 1:] if RANK[later] > RANK[item]), None)
        if end is None:
            if best is None:
                best = (start, None, -1)
            continue
        span = sum(len(x) for x in lines[start + 1:end])
        if best is None or span > best[2]:
            best = (start, end, span)
    if best is None:
        return {"status": "not_found", "start_line": None, "end_line": None, "text": ""}
    start, end, _ = best
    if end is None:
        return {"status": "unterminated", "start_line": start, "end_line": None, "text": ""}
    body = "\n".join(lines[start + 1:end])
    if len(body) < MIN_BODY_CHARS:
        status = ("omitted" if _OMITTED.search(lines[start] + " " + body)
                  else "by_reference" if _BY_REFERENCE.search(body)
                  else "too_short")
    else:
        status = "found"
    return {"status": status, "start_line": start, "end_line": end, "text": body}


def parse_document(raw: bytes) -> dict:
    lines = to_lines(raw)
    out = {"n_lines": len(lines), "n_chars": sum(len(x) for x in lines)}
    for name, item in SECTIONS.items():
        out[name] = extract(lines, item)
    return out
