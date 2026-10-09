"""Parity and speed: Python reference vs Rust parser over every stored filing.

    python scripts/bench_parser.py            # all documents
    python scripts/bench_parser.py --limit 500

Parity: both implementations must return identical output (statuses, line
numbers, section text, counts) for every document; mismatches are listed.
Speed: single-threaded Python vs single-threaded Rust on the same in-memory
bytes, then Rust's parallel gzip-reading entry point on the files themselves.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

from filing_signals import _parser as rust
from filing_signals import textparse
from filing_signals.warehouse import connect

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    con = connect(args.data / "filings.duckdb")
    paths = [str(args.data / "raw" / p) for (p,) in con.execute(
        f"select stored_path from documents order by accession {f'limit {args.limit}' if args.limit else ''}"
    ).fetchall()]

    t = time.perf_counter()
    raws = [gzip.decompress(Path(p).read_bytes()) for p in paths]
    load_s = time.perf_counter() - t
    total_mb = sum(len(r) for r in raws) / 1e6
    print(f"{len(raws)} documents, {total_mb:,.0f} MB decompressed (load {load_s:.1f}s)")

    t = time.perf_counter()
    py_out = [textparse.parse_document(r) for r in raws]
    py_s = time.perf_counter() - t

    t = time.perf_counter()
    rs_out = [rust.parse_document(r) for r in raws]
    rs_s = time.perf_counter() - t

    t = time.perf_counter()
    par_out = rust.parse_gz_files(paths)
    par_s = time.perf_counter() - t

    mismatches = [(paths[i], _diff(a, b)) for i, (a, b) in enumerate(zip(py_out, rs_out)) if a != b]
    par_mismatch = sum(a != b for a, b in zip(rs_out, par_out))

    result = {
        "documents": len(raws),
        "decompressed_mb": round(total_mb),
        "python_s": round(py_s, 1),
        "rust_single_thread_s": round(rs_s, 1),
        "rust_parallel_incl_gunzip_s": round(par_s, 1),
        "threads": os.cpu_count(),
        "speedup_single_thread": round(py_s / rs_s, 1),
        "speedup_parallel": round(py_s / par_s, 1),
        "python_mb_per_s": round(total_mb / py_s, 1),
        "rust_mb_per_s": round(total_mb / rs_s, 1),
        "parity_mismatches": len(mismatches),
        "parallel_vs_single_mismatches": par_mismatch,
    }
    print(json.dumps(result, indent=2))
    for path, diff in mismatches[:10]:
        print("MISMATCH", path, diff)


def _diff(a: dict, b: dict) -> str:
    for key in a:
        if a[key] != b.get(key):
            if isinstance(a[key], dict):
                for k2 in a[key]:
                    if a[key][k2] != b[key].get(k2):
                        va, vb = a[key][k2], b[key][k2]
                        if isinstance(va, str):
                            i = next((j for j, (x, y) in enumerate(zip(va, vb)) if x != y), min(len(va), len(vb)))
                            return f"{key}.{k2} differs at char {i}: {va[i-20:i+20]!r} vs {vb[i-20:i+20]!r}"
                        return f"{key}.{k2}: {va!r} vs {vb!r}"
            return f"{key}: {a[key]!r} vs {b[key]!r}"
    return "?"


if __name__ == "__main__":
    main()
