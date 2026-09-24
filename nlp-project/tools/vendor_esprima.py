"""Fetch the released esprima sdist from PyPI (master branch is inconsistent)."""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "tools" / "_vendor"
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROXY = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"})
)


def get_json(url: str):
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "nlp-project-vendor")
    with DIRECT.open(request, timeout=60) as response:
        return json.loads(response.read())


def main() -> int:
    meta = get_json("https://pypi.org/pypi/esprima/json")
    version = meta["info"]["version"]
    sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    print(f"esprima {version}: {sdist['filename']} ({sdist['size']} bytes)")

    archive = VENDOR / sdist["filename"]
    request = urllib.request.Request(sdist["url"])
    request.add_header("User-Agent", "nlp-project-vendor")
    with DIRECT.open(request, timeout=120) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"downloaded {archive.stat().st_size} bytes")

    # remove the broken master checkout first
    stale = VENDOR / "esprima"
    if stale.exists():
        shutil.rmtree(stale)

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        prefix = members[0].name.split("/")[0]
        for member in members:
            parts = member.name.split("/", 1)
            member.name = parts[1] if len(parts) > 1 else ""
            if not member.name:
                continue
            tar.extract(member, path=str(VENDOR))

    import sys

    sys.path.insert(0, str(VENDOR))
    import esprima

    print("esprima import OK; version attr:", getattr(esprima, "__version__", "n/a"))
    print("parse smoke test:", bool(esprima.parseScript("const a = (x) => x + 1;")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
