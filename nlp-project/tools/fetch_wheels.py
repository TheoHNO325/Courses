"""Fetch wheels from PyPI with plain urllib, then let pip install them offline.

pip's own HTTP layer stalls in this environment (each request opens fine on a
fresh connection but pip's pooled connection hangs), while plain urllib requests
work reliably.  So we resolve and download the wheels ourselves with urllib, then
run `pip install --no-index --find-links=<dir>` which performs no network I/O.

Usage:
    python tools/fetch_wheels.py --out outputs/_wheels sacrebleu pandas openpyxl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PY_TAG = "cp39"
ABI_TAG = "cp39"
PLATFORM = "win_amd64"

TOO_NEW = re.compile(r">=\s*3\.(1[0-9])")
PRERELEASE = re.compile(r"(a|b|rc|dev)\d", re.IGNORECASE)


def fetch_json(url: str):
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "nlp-project-wheel-fetch")
    with DIRECT.open(request, timeout=60) as response:
        return json.loads(response.read())


def python_compatible(requires_python: str | None) -> bool:
    if not requires_python:
        return True
    match = TOO_NEW.search(requires_python)
    if match:
        return False
    return True


def pick_wheel(urls: list[dict]) -> dict | None:
    wheels = [u for u in urls if u["packagetype"] == "bdist_wheel"]
    for u in wheels:
        name = u["filename"]
        if name.endswith("-py3-none-any.whl") or name.endswith("-py2.py3-none-any.whl"):
            return u
    for u in wheels:
        name = u["filename"]
        if PY_TAG in name and PLATFORM in name:
            return u
    return None


def resolve(name: str) -> dict | None:
    """Return the download record for the newest compatible wheel of `name`."""
    meta = fetch_json(f"https://pypi.org/pypi/{name}/json")
    releases = meta["releases"]
    # newest version first, using the project's own ordering where possible
    versions = list(releases.keys())

    def sort_key(v: str):
        parts = re.split(r"[.\-+]", v)
        numeric = []
        for part in parts:
            numeric.append(int(part) if part.isdigit() else 0)
        return numeric + [0] * (5 - len(numeric))

    for version in sorted(versions, key=sort_key, reverse=True):
        # prefer stable releases; only fall back to a pre-release if nothing else has a wheel
        if PRERELEASE.search(version):
            continue
        files = releases[version]
        if not files:
            continue
        requires_python = files[0].get("requires_python")
        if not python_compatible(requires_python):
            continue
        wheel = pick_wheel(files)
        if wheel:
            return {"name": name, "version": version, "url": wheel["url"],
                    "filename": wheel["filename"], "size": wheel["size"]}
    return None


def download(url: str, dest: Path) -> int:
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "nlp-project-wheel-fetch")
    with DIRECT.open(request, timeout=120) as response, dest.open("wb") as handle:
        while True:
            block = response.read(1024 * 256)
            if not block:
                break
            handle.write(block)
    return dest.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "_wheels")
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    resolved = []
    failed = []

    for package in args.packages:
        if list(args.out.glob(f"{package.replace('-', '_')}-*.whl")) or list(
            args.out.glob(f"{package}-*.whl")
        ):
            print(f"{package:20s} already present")
            continue
        try:
            record = resolve(package)
        except Exception as exc:  # noqa: BLE001
            print(f"{package:20s} RESOLVE FAILED {type(exc).__name__}: {exc}")
            failed.append(package)
            continue
        if not record:
            print(f"{package:20s} NO COMPATIBLE WHEEL (cp39/win_amd64)")
            failed.append(package)
            continue
        dest = args.out / record["filename"]
        try:
            size = download(record["url"], dest)
            print(f"{package:20s} {record['version']:12s} {record['filename']} ({size:,} bytes)")
            resolved.append(record)
        except Exception as exc:  # noqa: BLE001
            print(f"{package:20s} DOWNLOAD FAILED {type(exc).__name__}: {exc}")
            failed.append(package)

    (args.out / "_resolved.json").write_text(
        json.dumps({"resolved": resolved, "failed": failed}, indent=2), encoding="utf-8"
    )
    print(f"\nresolved={len(resolved)} failed={len(failed)}")
    if failed:
        print("failed:", ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
