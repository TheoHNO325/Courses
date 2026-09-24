"""Verify a data/ tree against the committed canonical layout spec.

Answers one question: does this checkout hold the same data, at the same paths, as
the machine the project was built on?

Checks, per spec entry: directory exists, file count, total bytes, tree digest
(a content-independent digest over relpath+size), and SHA-256 of every file the
spec pins.  Also reports files present but not described by the spec.

Usage
-----
    python tools/verify_data_layout.py                 # verify this checkout
    python tools/verify_data_layout.py --quick         # skip per-file hashing
    python tools/verify_data_layout.py --only data/raw/flores200_20260529,data/raw/mitra_v2
    python tools/verify_data_layout.py --root D:\\some\\other\\checkout

Exit code is 1 when anything fails, so it can gate CI or a setup script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "data" / "manifests" / "expected_layout.json"

# The spec cannot describe itself, so both the snapshot and the verifier ignore it.
IGNORE_NAMES = {SPEC_PATH.name}


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
    digest = hashlib.sha256()
    for rel, size in sorted(rel_paths):
        digest.update(f"{rel}\t{size}\n".encode("utf-8"))
    return digest.hexdigest().upper()


def scan(dir_path: Path) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for name in filenames:
            if name in IGNORE_NAMES:
                continue
            full = Path(dirpath) / name
            try:
                files.append((full.relative_to(dir_path).as_posix(), full.stat().st_size))
            except OSError:
                pass
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--only", default=None, help="comma separated path prefixes")
    parser.add_argument("--quick", action="store_true", help="skip per-file SHA-256")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    root = args.root.resolve()

    entries = spec["entries"]
    if args.only:
        wanted = tuple(p.strip() for p in args.only.split(",") if p.strip())
        entries = [e for e in entries if e["path"].startswith(wanted)]

    print(f"spec   : {args.spec}")
    print(f"root   : {root}")
    print(f"entries: {len(entries)}" + (" (filtered)" if args.only else ""))
    print("")

    failures: list[str] = []
    warnings: list[str] = []

    for entry in entries:
        rel = entry["path"]
        target = root / rel
        label = f"{rel:48s}"

        if not target.exists():
            print(f"{label} MISSING")
            failures.append(f"{rel}: directory missing")
            continue

        files = scan(target)
        count = len(files)
        total = sum(s for _, s in files)
        digest = tree_digest(files)

        problems = []
        if count != entry["file_count"]:
            problems.append(f"file count {count} != {entry['file_count']}")
        if total != entry["total_bytes"]:
            problems.append(f"bytes {total} != {entry['total_bytes']}")
        if digest != entry["tree_sha256"]:
            problems.append("tree digest differs")

        if not problems:
            print(f"{label} OK    {count:6d} files {total / 1e6:11.2f} MB")
        else:
            print(f"{label} FAIL  {'; '.join(problems)}")
            failures.extend(f"{rel}: {p}" for p in problems)

        # per-file hashes the spec pins
        if not args.quick:
            expected_hashes = entry.get("file_sha256") or {}
            checked = 0
            for file_rel, expected_hash in expected_hashes.items():
                file_path = target / file_rel
                if not file_path.exists():
                    failures.append(f"{rel}/{file_rel}: missing")
                    print(f"      MISSING {file_rel}")
                    continue
                actual = sha256_file(file_path)
                checked += 1
                if actual != expected_hash:
                    failures.append(f"{rel}/{file_rel}: sha256 mismatch")
                    print(f"      SHA256 MISMATCH {file_rel}")
            if checked:
                print(f"      verified {checked} file hashes")

        # extras not described by the spec
        if entry.get("file_sha256") and count > entry["file_count"]:
            described = set(entry["file_sha256"])
            actual_names = {rel_name for rel_name, _ in files}
            extra = sorted(actual_names - described)
            if extra:
                warnings.append(f"{rel}: {len(extra)} extra file(s), e.g. {extra[:3]}")

    print("")
    for warning in warnings:
        print(f"WARN  {warning}")

    legacy = spec.get("legacy_non_canonical", [])
    present_legacy = [p for p in legacy if (root / p).exists()]
    if present_legacy:
        print("")
        print("legacy (non-canonical) directories present, safe to ignore or delete:")
        for p in present_legacy:
            print(f"  {p}")

    print("")
    if failures:
        print(f"RESULT: {len(failures)} problem(s)")
        for problem in failures[:40]:
            print(f"  - {problem}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        return 1

    print("RESULT: data layout matches the canonical spec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
