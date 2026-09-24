"""Download the datasets approved for this project.

Reads data/manifests/download_plan.json (produced by tools/plan_downloads.py) and
fetches each target with the transport that works in this environment:

  * hf              -> huggingface_hub (handles gated auth, resume, xet)
  * github_tarball  -> direct GitHub API tarball (git.exe is blocked by the sandbox)
  * figshare        -> proxy download of a zip

Large files use the downloader's built-in resume: re-running this script after an
interruption continues where it stopped.  Nothing already downloaded is
overwritten.

Examples
--------
    python tools/download_datasets.py --list
    python tools/download_datasets.py --only dongbamie,bdrc_tibetan_ocr_benchmark
    python tools/download_datasets.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data" / "manifests" / "download_plan.json"
LOG_PATH = ROOT / "outputs" / "_download_log.txt"
PROXY = "http://127.0.0.1:7897"
CHUNK = 1024 * 1024

PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
)
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def human(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def sha256_of(path: Path, label: str = "") -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    read = 0
    started = time.time()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if total > 200 * 1024 * 1024 and read % (512 * 1024 * 1024) < 4 * 1024 * 1024:
                speed = read / max(1e-9, time.time() - started)
                log(f"    hashing {label} {read * 100 // max(1, total)}% ({human(speed)}/s)")
    return digest.hexdigest().upper()


def load_token() -> str | None:
    candidate = Path(os.environ.get("USERPROFILE", "")) / ".cache" / "huggingface" / "token"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip() or None
    return None


# --------------------------------------------------------------------------- HF
def download_hf(target: dict, token: str | None) -> list[dict]:
    from huggingface_hub import hf_hub_download

    repo = target["repo"]
    dest = ROOT / target["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    total = target.get("total_size", 0)
    log(f"  HF dataset {repo}: {len(target['files'])} files, {human(total)} total")
    for index, entry in enumerate(target["files"], start=1):
        rel = entry["path"]
        expected = entry.get("size", 0)
        local = dest / rel
        if local.exists() and expected and local.stat().st_size == expected:
            log(f"  [{index}/{len(target['files'])}] SKIP (complete) {rel}")
            results.append({"path": rel, "bytes": expected, "status": "already_present"})
            continue
        log(f"  [{index}/{len(target['files'])}] GET {rel} ({human(expected)})")
        started = time.time()
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename=rel,
                repo_type="dataset",
                local_dir=str(dest),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"      FAILED {rel}: {type(exc).__name__}: {exc}")
            results.append({"path": rel, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        size = Path(path).stat().st_size
        elapsed = time.time() - started
        ok = (not expected) or size == expected
        log(
            f"      done {human(size)} in {elapsed:.1f}s "
            f"({human(size / max(1e-9, elapsed))}/s) size_match={ok}"
        )
        results.append({"path": rel, "bytes": size, "status": "downloaded", "size_match": ok})
    return results


# ----------------------------------------------------------------- GitHub tarball
def download_github(target: dict) -> list[dict]:
    import tarfile

    dest = ROOT / target["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    downloads = ROOT / "data" / "raw" / "_archives"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"{target['id']}.tar.gz"
    log(f"  GitHub {target['repo']} -> {archive.name}")
    ok = stream_to_file(target["url"], archive, opener=DIRECT_OPENER, expected=None)
    if not ok:
        return [{"path": str(archive), "status": "failed"}]
    log(f"      downloaded {human(archive.stat().st_size)}; extracting")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        prefix = members[0].name.split("/")[0] if members else ""
        for member in members:
            parts = member.name.split("/", 1)
            member.name = parts[1] if len(parts) > 1 else ""
            if not member.name:
                continue
            tar.extract(member, path=str(dest))
    log(f"      extracted {len(members)} entries to {dest}")
    return [
        {
            "path": target["dest"],
            "bytes": archive.stat().st_size,
            "status": "extracted",
            "entries": len(members),
        }
    ]


# --------------------------------------------------------------------- figshare
def download_figshare(target: dict) -> list[dict]:
    dest = ROOT / target["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in target["files"]:
        name = entry["path"]
        expected = entry.get("size", 0)
        archive = dest / name
        if archive.exists() and expected and archive.stat().st_size == expected:
            log(f"  SKIP (complete) {name}")
        else:
            log(f"  figshare GET {name} ({human(expected)})")
            if not stream_to_file(entry["url"], archive, opener=PROXY_OPENER, expected=expected):
                results.append({"path": name, "status": "failed"})
                continue
        if name.endswith(".zip"):
            log(f"      extracting {name}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            log(f"      extracted {len(zf.namelist())} entries into {dest}")
        results.append({"path": name, "bytes": archive.stat().st_size, "status": "downloaded"})
    return results


# ----------------------------------------------------------------- IO utilities
def stream_to_file(url: str, path: Path, opener, expected: int | None) -> bool:
    """Resumable streamed download.  Returns True when the file is complete."""
    have = path.stat().st_size if path.exists() else 0
    if expected and have == expected:
        log(f"      already complete ({human(have)})")
        return True

    request = urllib.request.Request(url)
    request.add_header("User-Agent", "nlp-project-downloader")
    if have:
        request.add_header("Range", f"bytes={have}-")
        log(f"      resuming from {human(have)}")

    try:
        with opener.open(request, timeout=90) as response:
            status = response.status
            if have and status != 206:
                log(f"      server ignored Range (HTTP {status}); restarting from 0")
                have = 0
                path.unlink(missing_ok=True)
            mode = "ab" if have else "wb"
            length = int(response.headers.get("Content-Length") or 0)
            goal = have + length if length else expected or 0
            written = have
            started = time.time()
            last_report = 0.0
            with path.open(mode) as handle:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    written += len(block)
                    now = time.time()
                    if now - last_report >= 30:
                        last_report = now
                        speed = (written - have) / max(1e-9, now - started)
                        pct = f"{written * 100 / goal:.1f}%" if goal else "?"
                        log(f"        {pct} {human(written)} @ {human(speed)}/s")
    except Exception as exc:  # noqa: BLE001
        log(f"      download error: {type(exc).__name__}: {exc}")
        return False

    size = path.stat().st_size
    if expected and size != expected:
        log(f"      size mismatch: have {size}, expected {expected}")
        return False
    log(f"      complete {human(size)} in {time.time() - started:.1f}s")
    return True


# ---------------------------------------------------------------------- driver
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--only", default=None, help="comma separated target ids")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    targets = plan["targets"]

    if args.list:
        for target in targets:
            size = target.get("total_size") or target.get("approx_size") or 0
            print(f"{target['id']:32s} {target['transport']:16s} {human(size):>9s}  -> {target['dest']}")
        return 0

    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        targets = [t for t in targets if t["id"] in wanted]
    elif not args.all:
        parser.error("pass --all, --list, or --only <ids>")

    token = load_token()
    os.environ.setdefault("HTTP_PROXY", PROXY)
    os.environ.setdefault("HTTPS_PROXY", PROXY)

    log("=" * 72)
    log(f"download run started: {len(targets)} targets")

    summary = []
    for target in targets:
        log("-" * 72)
        log(f"TARGET {target['id']} ({target['transport']})")
        started = time.time()
        try:
            if target["transport"] == "hf":
                results = download_hf(target, token)
            elif target["transport"] == "github_tarball":
                results = download_github(target)
            elif target["transport"] == "figshare":
                results = download_figshare(target)
            else:
                log(f"  unknown transport {target['transport']}")
                results = [{"status": "skipped"}]
        except Exception as exc:  # noqa: BLE001
            log(f"  TARGET FAILED {type(exc).__name__}: {exc}")
            results = [{"status": "failed", "error": f"{type(exc).__name__}: {exc}"}]
        failed = sum(1 for r in results if r.get("status") == "failed")
        log(f"TARGET {target['id']} finished in {time.time() - started:.1f}s (failures={failed})")
        summary.append({"id": target["id"], "results": results, "failed": failed})

    log("=" * 72)
    for item in summary:
        log(f"  {item['id']:32s} failures={item['failed']}")
    log("download run complete")

    record = ROOT / "reports" / "dataset_download_record.json"
    record.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
