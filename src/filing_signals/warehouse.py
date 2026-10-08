"""DuckDB warehouse and the gzip raw-file store behind it."""
from __future__ import annotations

import gzip
import hashlib
from importlib import resources
from pathlib import Path

import duckdb
import pyarrow as pa


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    # all timestamps are stored and compared in UTC; Eastern only for display
    con.execute("set TimeZone = 'UTC'")
    con.execute("set enable_progress_bar = false")
    for name in ("schema.sql", "views.sql"):
        con.execute(resources.files("filing_signals").joinpath("sql", name).read_text())
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> int:
    """Insert rows, replacing any existing row with the same primary key."""
    if not rows:
        return 0
    # one bulk insert through Arrow; executemany goes row by row and took
    # minutes for a quarter's index
    batch = pa.Table.from_pylist(rows)  # noqa: F841  (read by name below)
    con.execute(f"insert or replace into {table} by name select * from batch")
    return len(rows)


class RawStore:
    """Every fetched byte kept on disk, gzipped, so the warehouse can be rebuilt
    without re-downloading and every document's provenance is checkable."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def write(self, data: bytes, *parts: str) -> dict:
        dest = self.path(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        packed = gzip.compress(data, compresslevel=6)
        tmp.write_bytes(packed)
        tmp.replace(dest)           # atomic: a crash never leaves a half file
        return {
            "stored_path": dest.relative_to(self.root).as_posix(),
            "bytes_raw": len(data),
            "bytes_stored": len(packed),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def read(self, *parts: str) -> bytes:
        return gzip.decompress(self.path(*parts).read_bytes())
