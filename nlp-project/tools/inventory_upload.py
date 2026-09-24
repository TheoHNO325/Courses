"""Inventory what would be uploaded to GitHub, honouring the intended excludes.

Prunes excluded directories during traversal so it never walks .venv / .hf-cache /
the 2.4 GB model output tree.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(r"D:\Courses\nlp-project")

EXCLUDE_DIRS = {
    ".venv", ".hf-cache", ".git", "__pycache__", ".ipynb_checkpoints",
    "data", "outputs", "_vendor", ".locks",
}
EXCLUDE_SUFFIX = {".bin", ".safetensors", ".pt", ".pth", ".whl", ".tmp", ".pyc"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}

included: list[tuple[str, int]] = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for name in filenames:
        if name in EXCLUDE_NAMES or Path(name).suffix in EXCLUDE_SUFFIX:
            continue
        full = Path(dirpath) / name
        try:
            size = full.stat().st_size
        except OSError:
            continue
        included.append((full.relative_to(ROOT).as_posix(), size))

out = []
total = sum(s for _, s in included)
out.append("=== WOULD BE UPLOADED ===")
out.append(f"files: {len(included)}, total {total / 1e6:.2f} MB")
out.append("")

grouped: dict[str, list[tuple[str, int]]] = {}
for rel, size in included:
    grouped.setdefault(rel.split("/")[0], []).append((rel, size))

for top in sorted(grouped):
    entries = grouped[top]
    out.append(f"  {top:22s} {len(entries):5d} files  {sum(s for _, s in entries) / 1e3:11.1f} KB")
    for rel, size in sorted(entries, key=lambda x: -x[1])[:5]:
        out.append(f"        {size / 1e3:11.1f} KB  {rel}")

out.append("")
out.append("=== 20 largest files that WOULD be uploaded ===")
for rel, size in sorted(included, key=lambda x: -x[1])[:20]:
    out.append(f"  {size / 1e3:11.1f} KB  {rel}")

out.append("")
out.append("=== excluded from upload (datasets / models / caches) ===")
for name in [".venv", ".hf-cache", "data", "outputs", "tools/_vendor"]:
    target = ROOT / name
    if target.exists():
        count = 0
        size = 0
        for dirpath, dirnames, filenames in os.walk(target):
            for f in filenames:
                try:
                    size += (Path(dirpath) / f).stat().st_size
                    count += 1
                except OSError:
                    pass
        out.append(f"  {name:22s} {count:7d} files  {size / 1e9:8.3f} GB")

(ROOT / "outputs" / "github_upload_inventory.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
