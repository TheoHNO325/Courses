"""Snapshot the canonical data/ layout into data/manifests/expected_layout.json.

This file is the committed specification of what a correctly set-up checkout looks
like: paths, file counts, byte sizes, per-directory tree digests, and SHA-256 for
every file small enough to hash cheaply.  tools/verify_data_layout.py checks a tree
against it, so "did I reproduce the same data?" becomes a yes/no answer.

Run:
    python tools/snapshot_data_layout.py
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "manifests" / "expected_layout.json"

# Directories that are not part of the canonical tree (older, superseded downloads).
LEGACY = [
    "data/raw/flores200_bod_tibt",
    "data/raw/flores200_zho_hans",
    "data/raw/mitra_v2_probe",
]

# Hash file contents up to this size; larger files are pinned by size + tree digest.
HASH_LIMIT = 5 * 1024 * 1024
# Directories with more files than this are pinned by count + size + tree digest only.
# Keeping per-file hashes for every one of ~13.5k small corpus files would bloat this
# spec to megabytes for little extra diagnostic value.
PER_FILE_HASH_MAX_FILES = 1000
# The spec cannot describe itself.
IGNORE_NAMES = {OUT.name}

# Which step produces each subtree.
SOURCE_BY_PREFIX = {
    "data/raw/flores200_20260529": "download_datasets.py --only flores200",
    "data/raw/mitra_v2": "download_datasets.py --only mitra_v2_full",
    "data/raw/mitra_v2_eval": "download_datasets.py --only mitra_v2_eval",
    "data/raw/dongbamie": "download_datasets.py --only dongbamie",
    "data/raw/bdrc_tibetan_ocr": "download_datasets.py --only bdrc_tibetan_ocr_benchmark",
    "data/raw/cute_v1": "download_datasets.py --only cute_bo_subset",
    "data/raw/openpecha_c0a2dd042": "download_datasets.py --only openpecha_c0a2dd042",
    "data/raw/modern_tibetan_corpus": "download_datasets.py --only modern_tibetan_corpus",
    "data/raw/shajiu_publiccorpus": "download_datasets.py --only shajiu_publiccorpus_samples",
    "data/raw/dongbamie_code": "download_datasets.py --only dongbamie_code",
    "data/raw/dongba1800": "download_datasets.py --only dongba1800",
    "data/raw/_archives": "created automatically by the github_tarball downloads",
    "data/processed/mitra_v2_conservative": (
        "src/prepare_mitra.py --input data/raw/mitra_v2/bo-zh_matches.ndjson.gz "
        "--config configs/mitra_v2_filter.json --output-dir data/processed/mitra_v2_conservative "
        "--report reports/mitra_v2_filtering.json"
    ),
    "data/interim/fertility": "src/audit_fertility.py",
    "data/manifests": "tracked in git",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def tree_digest(rel_paths: list[tuple[str, int]]) -> str:
    """Stable digest over sorted 'relpath<TAB>size' lines (content-independent)."""
    digest = hashlib.sha256()
    for rel, size in sorted(rel_paths):
        digest.update(f"{rel}\t{size}\n".encode("utf-8"))
    return digest.hexdigest().upper()


def describe_dir(dir_path: Path, data_root: Path) -> dict:
    files: list[tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for name in filenames:
            if name in IGNORE_NAMES:
                continue
            full = Path(dirpath) / name
            try:
                size = full.stat().st_size
            except OSError:
                continue
            files.append((full.relative_to(dir_path).as_posix(), size))

    entry = {
        "path": dir_path.relative_to(ROOT).as_posix(),
        "kind": "dir",
        "file_count": len(files),
        "total_bytes": sum(s for _, s in files),
        "tree_sha256": tree_digest(files),
    }
    if len(files) <= PER_FILE_HASH_MAX_FILES:
        hashes = {}
        for rel, size in sorted(files):
            if size <= HASH_LIMIT:
                hashes[rel] = sha256_file(dir_path / rel)
        if hashes:
            entry["file_sha256"] = hashes
    else:
        entry["per_file_hashes_omitted"] = (
            f"{len(files)} files exceeds PER_FILE_HASH_MAX_FILES="
            f"{PER_FILE_HASH_MAX_FILES}; count, size and tree digest are pinned instead"
        )
    return entry


entries: list[dict] = []
legacy_present: list[dict] = []

# every directory that should exist, plus data/ itself's direct files
targets = [
    DATA / "raw" / name
    for name in sorted(os.listdir(DATA / "raw"))
    if (DATA / "raw" / name).is_dir()
]
targets += [DATA / "processed" / "mitra_v2_conservative", DATA / "interim" / "fertility", DATA / "manifests"]

for target in targets:
    if not target.exists():
        continue
    rel = target.relative_to(ROOT).as_posix()
    if rel in LEGACY:
        legacy_present.append(rel)
        continue
    entry = describe_dir(target, DATA)
    for prefix, source in SOURCE_BY_PREFIX.items():
        if rel == prefix:
            entry["produced_by"] = source
            break
    entries.append(entry)

# data/interim/duckdb_tmp is a scratch dir produced by duckdb; record but do not require it
spec = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "generated_from": "tools/snapshot_data_layout.py",
    "purpose": (
        "Canonical data/ layout. A fresh clone that runs the documented download and "
        "processing commands must match these paths, file counts, sizes and hashes."
    ),
    "hash_limit_bytes": HASH_LIMIT,
    "entries": entries,
    "scratch_dirs_not_required": ["data/interim/duckdb_tmp"],
    "legacy_non_canonical": LEGACY,
    "legacy_present_on_this_machine": legacy_present,
}

OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"wrote {OUT}")
print(f"entries: {len(entries)}")
total_files = sum(e["file_count"] for e in entries)
total_bytes = sum(e["total_bytes"] for e in entries)
print(f"covers {total_files:,} files, {total_bytes / 1e9:.2f} GB")
for entry in entries:
    print(f"  {entry['path']:48s} {entry['file_count']:6d} files {entry['total_bytes'] / 1e6:12.2f} MB")
if legacy_present:
    print("legacy dirs present (not part of the canonical tree):")
    for rel in legacy_present:
        print(f"  {rel}")
