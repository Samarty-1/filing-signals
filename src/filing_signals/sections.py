"""Phase 2a: annual-report documents -> Item sections in Parquet.

Uses the Rust parser (filing_signals._parser) when it is built, else the
Python reference; they give identical output. Sections land in
data/sections/part-*.parquet, one row per (document, section), and the
warehouse sees them through the `sections` view. Re-running parses only
documents not yet parsed by the current PARSER_VERSION.
"""
from __future__ import annotations

import gzip
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from . import textparse

try:
    from . import _parser as _rust
except ImportError:  # pragma: no cover - pure-Python install
    _rust = None

# Bump whenever extraction logic changes: stale parses are then redone.
PARSER_VERSION = "2.0"
BATCH = 256

SCHEMA = pa.schema([
    ("accession", pa.string()),
    ("parser_version", pa.string()),
    ("n_lines", pa.int64()),
    ("n_chars", pa.int64()),
    ("section", pa.string()),
    ("status", pa.string()),
    ("start_line", pa.int64()),
    ("end_line", pa.int64()),
    ("text", pa.large_string()),
])


def backend() -> str:
    return "rust" if _rust is not None else "python"


def _parse_batch(paths: list[str]) -> list[dict]:
    if _rust is not None:
        return _rust.parse_gz_files(paths)
    return [textparse.parse_document(gzip.decompress(Path(p).read_bytes())) for p in paths]


def section_glob(data: Path) -> str:
    return (data / "sections" / "part-*.parquet").as_posix()


def register_view(con: duckdb.DuckDBPyConnection, data: Path) -> bool:
    """(Re)create the `sections` view if any Parquet parts exist."""
    if not list((data / "sections").glob("part-*.parquet")):
        return False
    con.execute(f"""
        create or replace view sections as
        select * from read_parquet('{section_glob(data)}')
        where parser_version = '{PARSER_VERSION}'
    """)
    return True


def parse_all(con: duckdb.DuckDBPyConnection, data: Path, log=print) -> int:
    out_dir = data / "sections"
    out_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if register_view(con, data):
        done = {a for (a,) in con.execute("select distinct accession from sections").fetchall()}
    todo = [(a, p) for a, p in con.execute(
        "select accession, stored_path from documents order by accession").fetchall()
        if a not in done]
    log(f"parse: {len(todo)} documents to parse ({backend()} parser), {len(done)} already done")

    part = len(list(out_dir.glob("part-*.parquet")))
    t0 = time.perf_counter()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        parsed = _parse_batch([str(data / "raw" / p) for _, p in batch])
        rows = []
        for (accession, _), doc in zip(batch, parsed):
            for name in textparse.SECTIONS:
                s = doc[name]
                rows.append({
                    "accession": accession, "parser_version": PARSER_VERSION,
                    "n_lines": doc["n_lines"], "n_chars": doc["n_chars"],
                    "section": name, "status": s["status"],
                    "start_line": s["start_line"], "end_line": s["end_line"],
                    "text": s["text"] if s["status"] == "found" else None,
                })
        tmp = out_dir / f"part-{part:05d}.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression="zstd")
        tmp.replace(out_dir / f"part-{part:05d}.parquet")   # atomic: no half-written part
        part += 1
        log(f"  {min(i + BATCH, len(todo))}/{len(todo)}  {time.perf_counter() - t0:.0f}s")
    register_view(con, data)
    return len(todo)
