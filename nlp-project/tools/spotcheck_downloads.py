"""Spot-check that each newly downloaded dataset is actually readable.

"Downloaded" is not the same as "usable": this opens every new dataset with the
reader the project will really use, streaming so memory stays low.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\Courses\nlp-project")
RAW = ROOT / "data" / "raw"
out: list[str] = []


def head(path: Path, lines: int = 3, encoding: str = "utf-8", limit: int = 160) -> list[str]:
    got = []
    with path.open("r", encoding=encoding, errors="replace") as handle:
        for i, line in enumerate(handle):
            if i >= lines:
                break
            got.append(line.rstrip("\n")[:limit])
    return got


def section(title: str) -> None:
    out.append("")
    out.append(f"=== {title} ===")


section("CUTE-Datasets (Tibetan subset)")
for rel in ["parallel-corpus/bo.txt", "non-parallel-corpus/n-bo.txt"]:
    path = RAW / "cute_v1" / rel
    if not path.exists():
        out.append(f"{rel}: MISSING")
        continue
    rows = head(path, 2, limit=110)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        tibetan = 0
        total = 0
        for _ in range(20000):
            line = handle.readline()
            if not line:
                break
            total += 1
            if any("\u0f00" <= ch <= "\u0fff" for ch in line):
                tibetan += 1
    out.append(f"{rel}: {path.stat().st_size / 1e9:.2f} GB, first-20k lines with Tibetan = {tibetan}/{total}")
    for row in rows:
        out.append(f"    {row}")

section("OpenPecha C0A2DD042")
op = RAW / "openpecha_c0a2dd042"
pairs = sorted((op / "text-pairs").glob("*-bo.txt")) if (op / "text-pairs").exists() else []
out.append(f"text-pairs/*-bo.txt count = {len(pairs)}")
if pairs:
    sample = pairs[0]
    out.append(f"sample {sample.name}: {sample.stat().st_size:,} bytes")
    for row in head(sample, 3, limit=110):
        out.append(f"    {row}")

section("Modern Tibetan Corpus")
mt = RAW / "modern_tibetan_corpus"
conllu = sorted((mt / "conllu").glob("*.conllu")) if (mt / "conllu").exists() else []
out.append(f"conllu files = {len(conllu)}")
if conllu:
    biggest = max(conllu, key=lambda p: p.stat().st_size)
    out.append(f"largest {biggest.name}: {biggest.stat().st_size:,} bytes")
    for row in head(biggest, 4, limit=110):
        out.append(f"    {row}")

section("Shajiu PublicCorpus")
sh = RAW / "shajiu_publiccorpus"
for rel in ["corpus/bilingual.txt", "corpus/phrase.txt", "TibetanData/token.vocab"]:
    path = sh / rel
    if path.exists():
        out.append(f"{rel}: {path.stat().st_size:,} bytes")
        for row in head(path, 2, limit=110):
            out.append(f"    {row}")
    else:
        out.append(f"{rel}: MISSING")

section("DongbaMIE (annotations)")
db = RAW / "dongbamie"
for rel in ["dongbaMIE_sentence_train.json", "dongbaMIE_sentence_dev.json", "dongbaMIE_paragraph_train.json"]:
    path = db / rel
    if not path.exists():
        out.append(f"{rel}: MISSING")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        out.append(f"{rel}: list, {len(data):,} records")
        if data:
            first = data[0]
            out.append(f"    record keys: {sorted(first.keys()) if isinstance(first, dict) else type(first).__name__}")
            if isinstance(first, dict):
                for key, value in list(first.items())[:6]:
                    preview = str(value)[:80].replace("\n", " ")
                    out.append(f"      {key}: {preview}")
    elif isinstance(data, dict):
        out.append(f"{rel}: dict, keys={list(data.keys())[:12]}")
    else:
        out.append(f"{rel}: {type(data).__name__}")

section("DongbaMIE images (zip contents)")
for rel in ["sentence.zip", "paragraph.zip"]:
    path = db / rel
    out.append(f"{rel}: {path.stat().st_size / 1e6:.1f} MB" if path.exists() else f"{rel}: MISSING")

section("Dongba1800")
d18 = RAW / "dongba1800"
files = [p for p in d18.rglob("*") if p.is_file()]
ext = Counter(p.suffix.lower() for p in files)
out.append(f"files={len(files):,}, extensions={dict(ext.most_common(6))}")
top = sorted(d18.iterdir())[:6]
out.append(f"top-level entries: {[p.name for p in top]}")

section("BDRC Tibetan OCR benchmark (parquet)")
try:
    import duckdb

    parquet = RAW / "bdrc_tibetan_ocr" / "data" / "test-00000-of-00001.parquet"
    connection = duckdb.connect()
    try:
        schema = connection.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet.as_posix()}')").fetchall()
        count = connection.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet.as_posix()}')").fetchone()[0]
        columns = [row[0] for row in schema]
        out.append(f"rows = {count:,}")
        out.append(f"columns = {columns}")
        preview = connection.execute(
            f"SELECT * FROM read_parquet('{parquet.as_posix()}') LIMIT 1"
        ).fetchall()
        if preview:
            for name, value in zip(columns, preview[0]):
                out.append(f"    {name}: {str(value)[:90]}")
    finally:
        connection.close()
except Exception as exc:  # noqa: BLE001
    out.append(f"parquet read FAILED: {type(exc).__name__}: {exc}")

(ROOT / "outputs" / "download_spotcheck.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
