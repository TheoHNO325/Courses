"""Reconnaissance for the dataset downloads.

Produces data/manifests/download_plan.json describing exactly which remote files
to fetch, their sizes, and the transport to use (proxy vs direct).

Nothing here downloads bulk data; it only queries APIs and probes the gated
DongbaMIE access with the locally stored HuggingFace token.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "_download_recon.txt"
PLAN = ROOT / "data" / "manifests" / "download_plan.json"

PROXY = "http://127.0.0.1:7897"
PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
)
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

lines: list[str] = []


def log(message: str = "") -> None:
    lines.append(message)


def human(num: int | float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def fetch_json(url: str, proxied: bool, token: str | None = None):
    opener = PROXY_OPENER if proxied else DIRECT_OPENER
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "nlp-project-download-planner")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with opener.open(request, timeout=40) as response:
        return response.status, json.loads(response.read())


# ---------------------------------------------------------------- HF token
token_path = Path(os.environ.get("USERPROFILE", "")) / ".cache" / "huggingface" / "token"
HF_TOKEN = None
if token_path.exists():
    HF_TOKEN = token_path.read_text(encoding="utf-8").strip() or None
log(f"HF token present: {bool(HF_TOKEN)} (len={len(HF_TOKEN) if HF_TOKEN else 0})")

plan: dict[str, object] = {"generated_from": "tools/plan_downloads.py", "targets": []}


def hf_files(repo: str) -> list[dict]:
    status, info = fetch_json(
        f"https://huggingface.co/api/datasets/{repo}?blobs=true", proxied=True, token=HF_TOKEN
    )
    return [
        {"path": s["rfilename"], "size": s.get("size") or 0}
        for s in info.get("siblings", [])
    ]


def probe_gated_file(repo: str, path: str) -> str:
    """HEAD-like probe of the resolve URL to confirm the token unlocks the file."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "nlp-project-download-planner")
    request.add_header("Range", "bytes=0-64")
    if HF_TOKEN:
        request.add_header("Authorization", f"Bearer {HF_TOKEN}")
    try:
        with PROXY_OPENER.open(request, timeout=40) as response:
            return f"HTTP {response.status} ({len(response.read())} bytes)"
    except urllib.error.HTTPError as exc:
        return f"HTTPError {exc.code} {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- DongbaMIE
log("\n=== DongbaMIE (gated) ===")
try:
    files = hf_files("thinklis/DongbaMIE")
    total = sum(f["size"] for f in files)
    log(f"files={len(files)} total={human(total)}")
    for f in sorted(files, key=lambda x: -x["size"]):
        log(f"   {f['path']:52s} {human(f['size'])}")
    small = [f for f in files if f["path"].endswith(".json")]
    probe = probe_gated_file("thinklis/DongbaMIE", small[0]["path"]) if small else "n/a"
    log(f"gated probe on {small[0]['path'] if small else '?'} -> {probe}")
    plan["targets"].append(
        {
            "id": "dongbamie",
            "transport": "hf",
            "repo": "thinklis/DongbaMIE",
            "type": "dataset",
            "gated": True,
            "dest": "data/raw/dongbamie",
            "files": files,
            "total_size": total,
        }
    )
except Exception as exc:  # noqa: BLE001
    log(f"ERROR {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- BDRC OCR
log("\n=== BDRC/tibetan-ocr-benchmark ===")
try:
    files = hf_files("BDRC/tibetan-ocr-benchmark")
    total = sum(f["size"] for f in files)
    log(f"files={len(files)} total={human(total)}")
    for f in sorted(files, key=lambda x: -x["size"])[:8]:
        log(f"   {f['path']:60s} {human(f['size'])}")
    plan["targets"].append(
        {
            "id": "bdrc_tibetan_ocr_benchmark",
            "transport": "hf",
            "repo": "BDRC/tibetan-ocr-benchmark",
            "type": "dataset",
            "gated": False,
            "dest": "data/raw/bdrc_tibetan_ocr",
            "files": files,
            "total_size": total,
        }
    )
except Exception as exc:  # noqa: BLE001
    log(f"ERROR {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- CUTE (Tibetan only)
log("\n=== CMLI-NLP/CUTE-Datasets (Tibetan subset) ===")
try:
    files = hf_files("CMLI-NLP/CUTE-Datasets")
    log(f"repo files={len(files)} total={human(sum(f['size'] for f in files))}")
    wanted = [
        f
        for f in files
        if f["path"] in ("parallel-corpus/bo.txt", "non-parallel-corpus/n-bo.txt")
    ]
    for f in files:
        log(f"   {f['path']:46s} {human(f['size'])}")
    log(f"selected {len(wanted)} files, total {human(sum(f['size'] for f in wanted))}")
    plan["targets"].append(
        {
            "id": "cute_bo_subset",
            "transport": "hf",
            "repo": "CMLI-NLP/CUTE-Datasets",
            "type": "dataset",
            "gated": False,
            "dest": "data/raw/cute_v1",
            "files": wanted,
            "total_size": sum(f["size"] for f in wanted),
        }
    )
except Exception as exc:  # noqa: BLE001
    log(f"ERROR {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- GitHub repos (tarball, direct)
log("\n=== GitHub repos ===")
GH = [
    ("openpecha_c0a2dd042", "OpenPecha-Data/C0A2DD042", "main", "data/raw/openpecha_c0a2dd042"),
    (
        "modern_tibetan_corpus",
        "tibetan-nlp/modern-tibetan-corpus",
        "main",
        "data/raw/modern_tibetan_corpus",
    ),
    ("shajiu_publiccorpus_samples", "Shajiu/PublicCorpus", "master", "data/raw/shajiu_publiccorpus"),
    ("dongbamie_code", "thinklis/DongbaMIE", "main", "data/raw/dongbamie_code"),
]
for target_id, repo, branch, dest in GH:
    try:
        status, info = fetch_json(f"https://api.github.com/repos/{repo}", proxied=False)
        size = (info.get("size") or 0) * 1024
        log(f"{repo}: HTTP {status} ~{human(size)} license={(info.get('license') or {}).get('spdx_id')}")
        plan["targets"].append(
            {
                "id": target_id,
                "transport": "github_tarball",
                "repo": repo,
                "branch": branch,
                "type": "repo",
                "dest": dest,
                "url": f"https://api.github.com/repos/{repo}/tarball/{branch}",
                "approx_size": size,
            }
        )
    except Exception as exc:  # noqa: BLE001
        log(f"{repo}: ERROR {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- Dongba1800
log("\n=== Dongba1800 sources ===")
for label, url, proxied in [
    ("figshare api#26969755", "https://api.figshare.com/v2/articles/26969755", True),
    ("figshare api#26969755 direct", "https://api.figshare.com/v2/articles/26969755", False),
    ("sciencedb doi", "https://api.figshare.com/v2/articles?institution=0", True),
]:
    try:
        status, info = fetch_json(url, proxied=proxied)
        if isinstance(info, dict) and "files" in info:
            log(f"{label}: HTTP {status} title={info.get('title', '')[:60]}")
            for f in info.get("files", []):
                log(f"     {f.get('name')} {human(f.get('size', 0))} {f.get('download_url')}")
            plan["targets"].append(
                {
                    "id": "dongba1800",
                    "transport": "figshare",
                    "article_id": 26969755,
                    "type": "dataset",
                    "dest": "data/raw/dongba1800",
                    "files": [
                        {"path": f.get("name"), "size": f.get("size", 0), "url": f.get("download_url")}
                        for f in info.get("files", [])
                    ],
                    "total_size": sum(f.get("size", 0) for f in info.get("files", [])),
                }
            )
        else:
            log(f"{label}: HTTP {status} (list response, {len(info) if isinstance(info, list) else '?'} items)")
    except Exception as exc:  # noqa: BLE001
        log(f"{label}: {type(exc).__name__}: {exc}")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
