"""Record the newly downloaded datasets in data/manifests/datasets.csv.

The project convention (data/raw/README.md) requires every download to be logged
with source, date, hash and local path.  This script only updates those fields for
the datasets that were just fetched; all other rows and columns are preserved
verbatim.

Run with --apply to write; without it, only prints what would change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "datasets.csv"

# dataset_id in datasets.csv  ->  (raw dir, license finding, status, note)
UPDATES: dict[str, dict] = {
    "openpecha_c0a2dd042": {
        "dir": "data/raw/openpecha_c0a2dd042",
        "archive": "data/raw/_archives/openpecha_c0a2dd042.tar.gz",
        "license": "CC0-1.0",
        "license_note": "GitHub API reports no LICENSE file in repo; CC0-1.0 claim not confirmed by the repository itself",
        "status": "downloaded",
    },
    "modern_tibetan_corpus": {
        "dir": "data/raw/modern_tibetan_corpus",
        "archive": "data/raw/_archives/modern_tibetan_corpus.tar.gz",
        "license": "MIT",
        "license_note": "GitHub API reports spdx_id=MIT; supersedes the previous TO_VERIFY entry",
        "status": "downloaded",
    },
    "shajiu_publiccorpus_samples": {
        "dir": "data/raw/shajiu_publiccorpus",
        "archive": "data/raw/_archives/shajiu_publiccorpus_samples.tar.gz",
        "license": "custom research terms",
        "license_note": "GitHub API reports no LICENSE file; keep sample_only use",
        "status": "downloaded",
    },
    "bdrc_tibetan_ocr_benchmark": {
        "dir": "data/raw/bdrc_tibetan_ocr",
        "license": "CC0-1.0",
        "license_note": "Confirmed via HuggingFace license tag",
        "status": "downloaded",
    },
    "dongbamie": {
        "dir": "data/raw/dongbamie",
        "license": "CC-BY-NC-SA-4.0",
        "license_note": "Confirmed via HuggingFace license tag; gated access used; never redistribute",
        "status": "downloaded",
    },
    "dongba1800": {
        "dir": "data/raw/dongba1800",
        "license": "LICENSE_TEXT_CONFLICT",
        "license_note": "figshare/ScienceDB article terms unresolved; paper states CC BY 4.0 but also a non-commercial restriction",
        "status": "downloaded",
    },
    "cute_v1": {
        "dir": "data/raw/cute_v1",
        "license": "CC-BY-4.0",
        "license_note": "Only the Tibetan subset was fetched (parallel-corpus/bo.txt, non-parallel-corpus/n-bo.txt)",
        "status": "downloaded_tibetan_subset_only",
    },
}

# Extra dataset rows that exist in the doc set but were deliberately not fetched.
DEFERRED = {
    "dongbamie_code": "data/raw/dongbamie_code",
}


def dir_summary(path: Path) -> tuple[int, int, str]:
    """Return (file_count, total_bytes, manifest_digest) for a directory tree.

    HuggingFace ``.cache`` bookkeeping is excluded so the digest describes the
    actual downloaded data rather than transfer metadata.
    """
    files = sorted(
        p for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts
    )
    digest = hashlib.sha256()
    total = 0
    for item in files:
        rel = item.relative_to(path).as_posix()
        size = item.stat().st_size
        total += size
        digest.update(f"{rel}\t{size}\n".encode("utf-8"))
    return len(files), total, digest.hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    today = date.today().isoformat()
    changes: list[str] = []

    for row in rows:
        did = row["dataset_id"]
        spec = UPDATES.get(did)
        if not spec:
            continue
        target = ROOT / spec["dir"]
        if not target.exists():
            changes.append(f"{did}: SKIPPED (not present: {spec['dir']})")
            continue

        count, total, digest = dir_summary(target)
        if count == 0:
            # still downloading (HuggingFace keeps in-flight files under .cache),
            # so do not claim this dataset is available yet
            changes.append(f"{did}: PENDING (no data files present yet; still downloading?)")
            continue
        row["local_path"] = spec["dir"]
        row["download_date"] = today
        row["license"] = spec["license"]
        row["status"] = spec["status"]
        row["sha256"] = f"tree={digest};files={count};bytes={total}"
        if spec.get("archive"):
            archive = ROOT / spec["archive"]
            if archive.exists():
                row["sha256"] += f";archive_sha256={file_sha256(archive)}"
        existing = row.get("notes") or ""
        note = spec["license_note"]
        if note not in existing:
            row["notes"] = (existing + " | " if existing else "") + note
        changes.append(
            f"{did}: status={spec['status']} files={count} bytes={total} digest={digest[:16]}…"
        )

    if not args.apply:
        print("DRY RUN (pass --apply to write)")
        for line in changes:
            print("  " + line)
        return 0

    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"updated {MANIFEST}")
    for line in changes:
        print("  " + line)

    # machine-readable companion record
    record = ROOT / "reports" / "dataset_download_manifest_update.json"
    record.write_text(
        json.dumps(
            {
                "updated": today,
                "manifest": "data/manifests/datasets.csv",
                "updates": changes,
                "not_fetched": DEFERRED,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
