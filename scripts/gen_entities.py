"""Regenerate rust/src/entities.rs from Python's HTML 4 entity table."""
import html.entities
from pathlib import Path

items = sorted(html.entities.name2codepoint.items())
lines = ["// Generated from Python's html.entities.name2codepoint (the HTML 4 set) so",
         "// the Rust parser decodes named entities exactly as the reference does.",
         "// Regenerate with scripts/gen_entities.py; do not edit by hand.",
         f"pub static NAMED: [(&str, u32); {len(items)}] = ["]
lines += [f'    ("{k}", 0x{v:04X}),' for k, v in items]
lines += ["];", ""]
out = Path(__file__).resolve().parents[1] / "rust" / "src" / "entities.rs"
out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print(f"{len(items)} entities -> {out}")
