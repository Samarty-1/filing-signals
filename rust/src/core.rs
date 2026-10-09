//! 10-K document -> clean lines -> Item sections.
//!
//! A line-for-line port of `filing_signals/textparse.py`, which is the
//! specification: read that module's docstring for the algorithm and the
//! reasons behind each step. tests/test_parser_parity.py checks the two give
//! identical output on every stored filing.

use std::collections::HashMap;
use std::sync::LazyLock;

use fancy_regex::Regex as FancyRegex;
use regex::Regex;

use crate::entities::NAMED;

// --- 1. decoding --------------------------------------------------------------

/// Windows-1252 for 0x80..=0x9F (WHATWG: undefined bytes map to the C1 code
/// point with the same value). Every other byte maps to the same code point.
const CP1252_HIGH: [u32; 32] = [
    0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
    0x2039, 0x0152, 0x008D, 0x017D, 0x008F, 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022,
    0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
];

fn cp1252_char(code: u32) -> char {
    if (0x80..=0x9F).contains(&code) {
        char::from_u32(CP1252_HIGH[(code - 0x80) as usize]).unwrap()
    } else {
        char::from_u32(code).unwrap()
    }
}

pub fn decode(raw: &[u8]) -> String {
    match std::str::from_utf8(raw) {
        Ok(s) => s.to_owned(),
        Err(_) => raw.iter().map(|&b| cp1252_char(b as u32)).collect(),
    }
}

// --- 2-5. html -> lines ----------------------------------------------------------
//
// The reference runs a sequence of regex substitutions over the whole
// document. Each pass here is a hand-written scanner with the same semantics
// (leftmost match, earliest close, identical length limits), in the same
// order, without per-match allocation. tests/test_parser_parity.py holds the
// two to identical output.

const BLOCK_TAGS: &[&str] = &[
    "address", "article", "aside", "blockquote", "br", "center", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "nav", "ol", "p", "page", "pre", "section", "table", "tbody",
    "thead", "tfoot", "title", "tr", "ul",
];
const DROP_CONTENT: &[&str] = &["script", "style", "head", "ix:header"];

static NAMED_MAP: LazyLock<HashMap<&'static str, u32>> =
    LazyLock::new(|| NAMED.iter().copied().collect());

/// First case-insensitive (ASCII) occurrence of `needle` (which starts with
/// '<') in `hay[from..]`.
fn find_ci(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let mut pos = from;
    while let Some(rel) = memchr::memchr(needle[0], &hay[pos..]) {
        let i = pos + rel;
        if hay.len() - i >= needle.len() && hay[i..i + needle.len()].eq_ignore_ascii_case(needle) {
            return Some(i);
        }
        pos = i + 1;
    }
    None
}

/// `<!--.*?-->` removed, as re.sub does it.
fn drop_comments(text: String) -> String {
    let b = text.as_bytes();
    let finder = memchr::memmem::Finder::new(b"<!--");
    let closer = memchr::memmem::Finder::new(b"-->");
    let mut out = String::with_capacity(text.len());
    let mut kept = 0;
    let mut pos = 0;
    while let Some(rel) = finder.find(&b[pos..]) {
        let start = pos + rel;
        match closer.find(&b[start + 4..]) {
            Some(r) => {
                let end = start + 4 + r + 3;
                out.push_str(&text[kept..start]);
                kept = end;
                pos = end;
            }
            None => break, // no closer anywhere later: no further match possible
        }
    }
    if kept == 0 {
        return text;
    }
    out.push_str(&text[kept..]);
    out
}

/// `<tag(?![A-Za-z0-9_:-]).*?</tag[ \t\r\n]*>` removed, case-insensitively.
fn drop_blocks(text: String, tag: &str) -> String {
    let b = text.as_bytes();
    let open = format!("<{tag}");
    let close = format!("</{tag}");
    let mut out = String::new();
    let mut kept = 0;
    let mut pos = 0;
    while let Some(start) = find_ci(b, open.as_bytes(), pos) {
        let after = start + open.len();
        if b.get(after).is_some_and(|&c| c.is_ascii_alphanumeric() || b"_:-".contains(&c)) {
            pos = start + 1;
            continue;
        }
        let mut cursor = after;
        let mut end = None;
        while let Some(c) = find_ci(b, close.as_bytes(), cursor) {
            let mut k = c + close.len();
            while k < b.len() && b" \t\r\n".contains(&b[k]) {
                k += 1;
            }
            if k < b.len() && b[k] == b'>' {
                end = Some(k + 1);
                break;
            }
            cursor = c + 1;
        }
        match end {
            Some(e) => {
                out.push_str(&text[kept..start]);
                kept = e;
                pos = e;
            }
            None => break,
        }
    }
    if kept == 0 {
        return text;
    }
    out.push_str(&text[kept..]);
    out
}

fn is_name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b':' || c == b'_' || c == b'-'
}

/// `<(/?)([A-Za-z][A-Za-z0-9:_-]*)[^>]*>` -> "\n" (block), " " (cell), "" (other).
fn replace_tags(text: String) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut kept = 0;
    let mut pos = 0;
    while let Some(rel) = memchr::memchr(b'<', &b[pos..]) {
        let lt = pos + rel;
        let mut k = lt + 1;
        if k < b.len() && b[k] == b'/' {
            k += 1;
        }
        if k >= b.len() || !b[k].is_ascii_alphabetic() {
            pos = lt + 1;
            continue;
        }
        let name_start = k;
        while k < b.len() && is_name_char(b[k]) {
            k += 1;
        }
        let name = &b[name_start..k];
        // `[^>]*>`: the first '>' after the name, or no match at this '<'
        let Some(gt) = memchr::memchr(b'>', &b[k..]) else {
            break; // no '>' anywhere later: no later '<' can match either
        };
        let end = k + gt + 1;
        out.push_str(&text[kept..lt]);
        if BLOCK_TAGS.iter().any(|t| t.as_bytes().eq_ignore_ascii_case(name)) {
            out.push('\n');
        } else if name.eq_ignore_ascii_case(b"td") || name.eq_ignore_ascii_case(b"th") {
            out.push(' ');
        }
        kept = end;
        pos = end;
    }
    out.push_str(&text[kept..]);
    out
}

fn push_cp(out: &mut String, n: u32) {
    if (128..=159).contains(&n) {
        out.push(cp1252_char(n));
    } else {
        match char::from_u32(n) {
            Some(c) if n != 0 => out.push(c),
            _ => out.push('\u{FFFD}'),
        }
    }
}

/// `&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});`
fn decode_entities(text: String) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut kept = 0;
    let mut pos = 0;
    while let Some(rel) = memchr::memchr(b'&', &b[pos..]) {
        let amp = pos + rel;
        let k = amp + 1;
        let run = |from: usize, ok: fn(&u8) -> bool| b[from..].iter().take_while(|c| ok(c)).count();
        // (end of the entity including ';', what it decodes to)
        let mut hit: Option<(usize, Option<u32>)> = None;
        if b.get(k) == Some(&b'#') {
            if matches!(b.get(k + 1), Some(b'x' | b'X')) {
                let n = run(k + 2, u8::is_ascii_hexdigit);
                if (1..=6).contains(&n) && b.get(k + 2 + n) == Some(&b';') {
                    let v = u32::from_str_radix(&text[k + 2..k + 2 + n], 16).unwrap();
                    hit = Some((k + 2 + n + 1, Some(v)));
                }
            } else {
                let n = run(k + 1, u8::is_ascii_digit);
                if (1..=7).contains(&n) && b.get(k + 1 + n) == Some(&b';') {
                    let v: u32 = text[k + 1..k + 1 + n].parse().unwrap();
                    hit = Some((k + 1 + n + 1, Some(v)));
                }
            }
        } else if b.get(k).is_some_and(u8::is_ascii_alphabetic) {
            let n = run(k, u8::is_ascii_alphanumeric);
            if (2..=32).contains(&n) && b.get(k + n) == Some(&b';') {
                hit = Some((k + n + 1, NAMED_MAP.get(&text[k..k + n]).copied()));
            }
        }
        match hit {
            Some((end, Some(cp))) => {
                out.push_str(&text[kept..amp]);
                push_cp(&mut out, cp);
                kept = end;
                pos = end;
            }
            // unknown named entity: the match is replaced by itself
            Some((end, None)) => pos = end,
            None => pos = amp + 1,
        }
    }
    out.push_str(&text[kept..]);
    out
}

fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\x0B' | '\x0C' | '\u{A0}' | '\u{2000}'..='\u{200A}'
        | '\u{202F}' | '\u{205F}' | '\u{3000}')
}

fn is_zero_width(c: char) -> bool {
    matches!(c, '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}')
}

/// Zero-width removal, split on '\n', space runs -> ' ', strip(' '), drop empties.
fn split_lines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut pending_space = false;
    for c in text.chars().chain(std::iter::once('\n')) {
        if is_zero_width(c) {
            continue;
        }
        if c == '\n' {
            if !line.is_empty() {
                lines.push(std::mem::take(&mut line));
            }
            pending_space = false;
        } else if is_space(c) {
            pending_space = !line.is_empty();
        } else {
            if pending_space {
                line.push(' ');
                pending_space = false;
            }
            line.push(c);
        }
    }
    lines
}

pub fn to_lines(raw: &[u8]) -> Vec<String> {
    let mut text = drop_comments(decode(raw));
    for tag in DROP_CONTENT {
        text = drop_blocks(text, tag);
    }
    let text = decode_entities(replace_tags(text));
    split_lines(&text)
}

// --- 6. sections -------------------------------------------------------------------

const ITEM_ORDER: &[&str] = &[
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16",
];
pub const SECTIONS: &[(&str, &str)] = &[("risk_factors", "1A"), ("mdna", "7")];
const MAX_HEADING_LEN: usize = 200;
const MIN_BODY_CHARS: usize = 1000;

fn rank(item: &str) -> Option<usize> {
    ITEM_ORDER.iter().position(|&x| x == item)
}

static HEADING: LazyLock<FancyRegex> = LazyLock::new(|| {
    FancyRegex::new(
        r"(?i)^(?:part +i{1,3} *[,.:\-\x{2013}\x{2014}]? *)?items? *([0-9]{1,2})(?:\.?([a-cA-C]))?(?![A-Za-z0-9])",
    )
    .unwrap()
});
static BARE: LazyLock<FancyRegex> = LazyLock::new(|| {
    FancyRegex::new(
        r"(?s)^([0-9]{1,2})(?:\.?([a-cA-C]))?(?![A-Za-z0-9]) *[.:\-\x{2013}\x{2014}]? *(.*)$",
    )
    .unwrap()
});
static TITLES: LazyLock<HashMap<&'static str, Regex>> = LazyLock::new(|| {
    [
        ("1", "business"), ("1A", "risk factors"), ("1B", "unresolved staff"),
        ("1C", "cybersecurity"), ("2", "(?:description of )?propert"), ("3", "legal proceedings"),
        ("4", r"mine safety|submission of matters|\(?removed and reserved|reserved"),
        ("5", "market for"), ("6", r"selected (?:consolidated )?financial|\[?reserved"),
        ("7", "management.{0,3}s discussion"), ("7A", "quantitative and qualitative"),
        ("8", "(?:consolidated )?financial statements"), ("9", "changes in and disagreements"),
        ("9A", "controls and procedures"), ("9B", "other information"),
        ("9C", "disclosure regarding foreign"), ("10", "directors"),
        ("11", "executive compensation"), ("12", "security ownership"),
        ("13", "certain relationships"), ("14", "principal account"), ("15", "exhibits"),
        ("16", "form 10-k summary"),
    ]
    .into_iter()
    .map(|(item, title)| {
        let pattern = format!("(?i)^(?:{})", title.replace(' ', "[ \n]+"));
        (item, Regex::new(&pattern).unwrap())
    })
    .collect()
});
static OMITTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)not[ \n]+(?:required|applicable)|smaller[ \n]+reporting[ \n]+compan|omitted")
        .unwrap()
});
static BY_REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)incorporated[ \n]+(?:herein[ \n]+)?by[ \n]+reference").unwrap()
});

fn item_key(number: &str, letter: Option<&str>) -> String {
    let mut key = number.trim_start_matches('0').to_string();
    if let Some(l) = letter {
        key.push_str(&l.to_ascii_uppercase());
    }
    key
}

pub fn headings(lines: &[String]) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        // Fast reject: a heading starts with "part", "item" or a digit. Only
        // ASCII first characters are rejected here; anything else still goes
        // through the (Unicode case-folding) regex, so output is unchanged.
        let first = line.as_bytes()[0];
        if first.is_ascii() && !matches!(first, b'p' | b'P' | b'i' | b'I' | b'0'..=b'9') {
            continue;
        }
        if line.chars().count() > MAX_HEADING_LEN {
            continue;
        }
        if let Ok(Some(m)) = HEADING.captures(line.as_str()) {
            let item = item_key(m.get(1).unwrap().as_str(), m.get(2).map(|g| g.as_str()));
            if rank(&item).is_some() {
                out.push((i, item));
            }
            continue;
        }
        if let Ok(Some(m)) = BARE.captures(line.as_str()) {
            let item = item_key(m.get(1).unwrap().as_str(), m.get(2).map(|g| g.as_str()));
            let titled = TITLES
                .get(item.as_str())
                .is_some_and(|title| title.is_match(m.get(3).unwrap().as_str()));
            if titled {
                out.push((i, item));
            }
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub status: &'static str,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub text: String,
}

pub fn extract(lines: &[String], hs: &[(usize, String)], item: &str) -> Section {
    let item_rank = rank(item).expect("known item");
    // (start, end, span); span -1 marks an unterminated candidate
    let mut best: Option<(usize, Option<usize>, i64)> = None;
    for (k, (start, it)) in hs.iter().enumerate() {
        if it != item {
            continue;
        }
        let end = hs[k + 1..]
            .iter()
            .find(|(_, later)| rank(later).unwrap() > item_rank)
            .map(|(j, _)| *j);
        match end {
            None => {
                if best.is_none() {
                    best = Some((*start, None, -1));
                }
            }
            Some(end) => {
                let span: i64 =
                    lines[start + 1..end].iter().map(|l| l.chars().count() as i64).sum();
                if best.is_none_or(|(_, _, s)| span > s) {
                    best = Some((*start, Some(end), span));
                }
            }
        }
    }
    let Some((start, end, _)) = best else {
        return Section { status: "not_found", start_line: None, end_line: None, text: String::new() };
    };
    let Some(end) = end else {
        return Section { status: "unterminated", start_line: Some(start), end_line: None, text: String::new() };
    };
    let body = lines[start + 1..end].join("\n");
    let status = if body.chars().count() < MIN_BODY_CHARS {
        if OMITTED.is_match(&format!("{} {}", lines[start], body)) {
            "omitted"
        } else if BY_REFERENCE.is_match(&body) {
            "by_reference"
        } else {
            "too_short"
        }
    } else {
        "found"
    };
    Section { status, start_line: Some(start), end_line: Some(end), text: body }
}

pub struct Document {
    pub n_lines: usize,
    pub n_chars: usize,
    pub sections: Vec<(&'static str, Section)>,
}

pub fn parse_document(raw: &[u8]) -> Document {
    let lines = to_lines(raw);
    let hs = headings(&lines);
    Document {
        n_lines: lines.len(),
        n_chars: lines.iter().map(|l| l.chars().count()).sum(),
        sections: SECTIONS.iter().map(|&(name, item)| (name, extract(&lines, &hs, item))).collect(),
    }
}
