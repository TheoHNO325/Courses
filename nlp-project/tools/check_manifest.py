"""Confirm datasets.csv still parses and the new fields landed correctly."""

import csv
from pathlib import Path

ROOT = Path(r"D:\Courses\nlp-project")
MANIFEST = ROOT / "data" / "manifests" / "datasets.csv"
out = []

with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = list(reader)

out.append(f"columns ({len(fields)}): {fields}")
out.append(f"rows: {len(rows)}")
out.append("")
out.append(f"{'dataset_id':32s} {'status':34s} {'download_date':14s} local_path")
for row in rows:
    out.append(
        f"{row['dataset_id']:32s} {row['status']:34s} {row['download_date']:14s} {row['local_path']}"
    )

out.append("")
out.append("### license column after update")
for row in rows:
    if row["status"].startswith("downloaded"):
        out.append(f"  {row['dataset_id']:32s} license={row['license']}")

out.append("")
out.append("### sha256 field sample (first downloaded row)")
for row in rows:
    if row["status"].startswith("downloaded"):
        out.append(f"  {row['dataset_id']}: {row['sha256']}")
        break

# every row must still have the same number of populated columns
bad = [r["dataset_id"] for r in rows if len(r) != len(fields) or None in r]
out.append("")
out.append("malformed rows: " + (", ".join(bad) if bad else "none"))

(ROOT / "outputs" / "manifest_validation.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
