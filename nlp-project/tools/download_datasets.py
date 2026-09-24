"""Download the datasets for this project, reproducing the original local layout.

Reads data/manifests/download_plan.json (produced by tools/plan_downloads.py) and
fetches each target with the transport that works in this environment:

  hf              huggingface_hub: gated auth, resume, xet.  Honours strip_prefix so
                  HF paths like data/language/bod_Tibt/x.parquet land at
                  bod_Tibt/x.parquet, matching the original tree exactly.
  github_tarball  GitHub API tarball (git.exe may be blocked in restricted sandboxes)
  url             plain HTTPS with resume and SHA-256 verification
  figshare        proxy download of a zip

Every file is verified against its pinned size, and against SHA-256 when the plan
provides one.  Re-running is safe: complete files are skipped, partial files resume.

Examples
--------
    python tools/download_datasets.py --list
    python tools/download_datasets.py --all
    python tools/download_datasets.py --only flores200,mitra_v2_full,mitra_v2_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data" / "manifests" / "download_plan.json"
LOG_PATH = ROOT / "outputs" / "dataset_download_log.txt"
PROXY = "http://127.0.0.1:7897"
CHUNK = 1024 * 1024
# Slow or resetting links are normal here; each retry resumes via HTTP Range.
MAX_URL_ATTEMPTS = 8

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
            if total > 400 * 1024 * 1024 and read % (512 * 1024 * 1024) < 4 * 1024 * 1024:
                speed = read / max(1e-9, time.time() - started)
                log(f"    hashing {label} {read * 100 // max(1, total)}% ({human(speed)}/s)")
    return digest.hexdigest().upper()


def load_token() -> str | None:
    candidate = Path(os.environ.get("USERPROFILE", "")) / ".cache" / "huggingface" / "token"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip() or None
    return None


def is_complete(path: Path, expected_size: int, expected_sha: str | None) -> bool:
    if not path.exists():
        return False
    if expected_size and path.stat().st_size != expected_size:
        return False
    if expected_sha and sha256_of(path) != expected_sha:
        log(f"      sha256 mismatch on existing {path.name}; re-downloading")
        return False
    return True


# --------------------------------------------------------------------------- HF
def download_hf(target: dict, token: str | None) -> list[dict]:
    from huggingface_hub import hf_hub_download

    repo = target["repo"]
    dest = ROOT / target["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    strip_prefix = target.get("strip_prefix")
    results = []
    files = target["files"]
    log(f"  HF dataset {repo}: {len(files)} files, {human(target.get('total_size', 0))} total")
    if strip_prefix:
        log(f"      strip_prefix='{strip_prefix}' (staged then moved into place)")

    for index, entry in enumerate(files, start=1):
        rel = entry["path"]
        expected = entry.get("size", 0)
        expected_sha = entry.get("sha256")
        dest_rel = entry.get("dest") or (
            rel[len(strip_prefix):] if strip_prefix and rel.startswith(strip_prefix) else rel
        )
        local = dest / dest_rel

        if is_complete(local, expected, expected_sha):
            log(f"  [{index}/{len(files)}] SKIP (verified) {dest_rel}")
            results.append({"path": dest_rel, "bytes": local.stat().st_size, "status": "already_present"})
            continue

        log(f"  [{index}/{len(files)}] GET {rel} ({human(expected)}) -> {dest_rel}")
        started = time.time()
        # Stage under dest/_hf_stage when the remote path must be rewritten, so HF's
        # own resume metadata stays valid; download straight in otherwise.
        target_dir = dest / "_hf_stage" if strip_prefix else dest
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename=rel,
                repo_type="dataset",
                local_dir=str(target_dir),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"      FAILED {rel}: {type(exc).__name__}: {exc}")
            results.append({"path": dest_rel, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue

        downloaded = Path(path)
        if strip_prefix and downloaded != local:
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists():
                local.unlink()
            shutil.move(str(downloaded), str(local))

        size = local.stat().st_size
        elapsed = time.time() - started
        size_ok = (not expected) or size == expected
        sha_value = None
        sha_ok = True
        if expected_sha:
            sha_value = sha256_of(local)
            sha_ok = sha_value == expected_sha
        log(
            f"      done {human(size)} in {elapsed:.1f}s size_match={size_ok} sha_match={sha_ok}"
        )
        if not (size_ok and sha_ok):
            results.append({"path": dest_rel, "bytes": size, "status": "failed", "size_match": size_ok, "sha_match": sha_ok})
            continue
        # clean up the staging area for rewritten paths
        if strip_prefix:
            shutil.rmtree(dest / "_hf_stage", ignore_errors=True)
        results.append(
            {"path": dest_rel, "bytes": size, "status": "downloaded",
             "size_match": size_ok, "sha_match": sha_ok, "sha256": sha_value}
        )
    return results


# ------------------------------------------------------------------- plain URL
def download_url(target: dict) -> list[dict]:
    dest = ROOT / target["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    files = target["files"]
    log(f"  URL target: {len(files)} files, {human(target.get('total_size', 0))} total")
    for index, entry in enumerate(files, start=1):
        rel = entry["path"]
        expected = entry.get("size", 0)
        expected_sha = entry.get("sha256")
        local = dest / rel
        if is_complete(local, expected, expected_sha):
            log(f"  [{index}/{len(files)}] SKIP (verified) {rel}")
            results.append({"path": rel, "bytes": local.stat().st_size, "status": "already_present",
                            "sha256": expected_sha})
            continue
        log(f"  [{index}/{len(files)}] GET {rel} ({human(expected)})")
        ok = False
        for attempt in range(1, MAX_URL_ATTEMPTS + 1):
            if stream_to_file(entry["url"], local, opener=DIRECT_OPENER, expected=expected):
                ok = True
                break
            if attempt < MAX_URL_ATTEMPTS:
                log(f"      attempt {attempt}/{MAX_URL_ATTEMPTS} incomplete; resuming in {2 * attempt}s")
                time.sleep(2 * attempt)
        if not ok:
            log(f"      giving up on {rel} after {MAX_URL_ATTEMPTS} attempts")
            results.append({"path": rel, "status": "failed"})
            continue
        size = local.stat().st_size
        sha_value = sha256_of(local) if expected_sha else None
        sha_ok = (not expected_sha) or sha_value == expected_sha
        log(f"      size_match={size == expected if expected else 'n/a'} sha_match={sha_ok}")
        results.append(
            {"path": rel, "bytes": size, "status": "downloaded" if sha_ok else "failed",
             "sha_match": sha_ok, "sha256": sha_value}
        )
    return results


# ----------------------------------------------------------------- GitHub tarball
def download_github(target: dict) -> list[dict]:
    dest = ROOT / target["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    downloads = ROOT / "data" / "raw" / "_archives"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"{target['id']}.tar.gz"
    log(f"  GitHub {target['repo']} -> {archive.name}")
    if not stream_to_file(target["url"], archive, opener=DIRECT_OPENER, expected=None):
        return [{"path": str(archive), "status": "failed"}]
    log(f"      downloaded {human(archive.stat().st_size)}; extracting")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            parts = member.name.split("/", 1)
            member.name = parts[1] if len(parts) > 1 else ""
            if not member.name:
                continue
            tar.extract(member, path=str(dest))
    log(f"      extracted {len(members)} entries to {dest}")
    return [
        {"path": target["dest"], "bytes": archive.stat().st_size, "status": "extracted",
         "entries": len(members), "archive_sha256": sha256_of(archive)}
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
        if is_complete(archive, expected, None):
            log(f"  SKIP (size ok) {name}")
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
        results.append({"path": name, "bytes": archive.stat().st_size, "status": "downloaded",
                        "sha256": sha256_of(archive)})
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

    started = time.time()
    try:
        with opener.open(request, timeout=240) as response:
            status = response.status
            if have and status != 206:
                log(f"      server ignored Range (HTTP {status}); restarting from 0")
                have = 0
                path.unlink(missing_ok=True)
            mode = "ab" if have else "wb"
            length = int(response.headers.get("Content-Length") or 0)
            goal = have + length if length else expected or 0
            written = have
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
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    targets = plan["targets"]

    if args.list:
        for target in targets:
            size = target.get("total_size") or target.get("approx_size") or 0
            flag = " [GATED]" if target.get("gated") else ""
            print(f"{target['id']:32s} {target['transport']:16s} {human(size):>10s}  -> {target['dest']}{flag}")
        return 0

    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = wanted - {t["id"] for t in targets}
        if unknown:
            parser.error(f"unknown target id(s): {', '.join(sorted(unknown))}")
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
            elif target["transport"] == "url":
                results = download_url(target)
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
            {"created_at_utc": datetime.now(timezone.utc).isoformat(), "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
