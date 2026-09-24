"""Build data/manifests/download_plan.json: exactly which remote files to fetch.

Covers EVERY dataset present in our local data/raw/ so that a fresh clone running
`tools/download_datasets.py --all` reproduces the same tree, with the same paths,
as the original machine.

Transports
----------
hf              huggingface_hub (gated auth, resume, xet); supports strip_prefix
                so an HF path like data/language/bod_Tibt/x.parquet can land at
                bod_Tibt/x.parquet, matching our local layout
github_tarball  GitHub API tarball (git.exe is unavailable in restricted sandboxes)
url             plain HTTPS file with resume and SHA-256 verification
figshare        proxy download of a zip

Run:
    python tools/plan_downloads.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dataset_download_recon.txt"
PLAN = ROOT / "data" / "manifests" / "download_plan.json"

PROXY = "http://127.0.0.1:7897"
PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
)
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

MITRA_COMMIT = "bf7b340cd4f75bda3089479a6367ad94a8cbee10"
MITRA_RAW = f"https://raw.githubusercontent.com/dharmamitra/mitra-parallel/{MITRA_COMMIT}"

lines: list[str] = []


def log(message: str = "") -> None:
    lines.append(message)


def human(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def fetch_json(url: str, proxied: bool, token: str | None = None, attempts: int = 4):
    """JSON GET with retries.

    The local proxy intermittently raises URLError(Errno 2) or resets the
    connection.  Without retries a single glitch silently drops a whole target
    from the plan, which would give a fresh clone a different data tree.
    """
    opener = PROXY_OPENER if proxied else DIRECT_OPENER
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url)
        request.add_header("User-Agent", "nlp-project-download-planner")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with opener.open(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"      JSON attempt {attempt}/{attempts} failed for {url}: {type(exc).__name__}: {exc}")
            time.sleep(2 * attempt)
    raise last if last else RuntimeError("fetch_json failed")


def remote_size(url: str, fallback: int = 0, attempts: int = 3) -> int:
    """Content-Length via HEAD, retried, with a pinned fallback.

    raw.githubusercontent.com occasionally resets the connection.  A failure here
    must not silently yield size 0, because that would disable size verification
    for that file.
    """
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, method="HEAD")
        request.add_header("User-Agent", "nlp-project-download-planner")
        try:
            with DIRECT_OPENER.open(request, timeout=60) as response:
                size = int(response.headers.get("Content-Length") or 0)
                if size:
                    return size
        except Exception as exc:  # noqa: BLE001
            log(f"      HEAD attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}")
    if fallback:
        log(f"      using pinned fallback size {fallback}")
    return fallback


token_path = Path(os.environ.get("USERPROFILE", "")) / ".cache" / "huggingface" / "token"
HF_TOKEN = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else None
log(f"HF token present: {bool(HF_TOKEN)}")

# Every dataset under data/raw/ must appear here.  A missing id means a transient
# API failure, and the plan is rejected rather than silently written short.
REQUIRED_TARGETS = {
    "flores200",
    "mitra_v2_full",
    "mitra_v2_eval",
    "dongbamie",
    "bdrc_tibetan_ocr_benchmark",
    "cute_bo_subset",
    "openpecha_c0a2dd042",
    "modern_tibetan_corpus",
    "shajiu_publiccorpus_samples",
    "dongbamie_code",
    "dongba1800",
}

plan: dict[str, object] = {
    "generated_from": "tools/plan_downloads.py",
    "note": (
        "Targets cover every dataset under data/raw/. Derived data "
        "(data/processed/, data/interim/) is produced by the src/ scripts; "
        "see README sections 3 and 6."
    ),
    "legacy_non_canonical": [
        "data/raw/flores200_bod_tibt",
        "data/raw/flores200_zho_hans",
        "data/raw/mitra_v2_probe",
    ],
    "targets": [],
}


def hf_files(repo: str) -> list[dict]:
    status, info = fetch_json(
        f"https://huggingface.co/api/datasets/{repo}?blobs=true", proxied=True, token=HF_TOKEN
    )
    return [{"path": s["rfilename"], "size": s.get("size") or 0} for s in info.get("siblings", [])]


def add_hf_target(target_id, repo, dest, files, gated=False, strip_prefix=None, note=""):
    total = sum(f["size"] for f in files)
    plan["targets"].append(
        {
            "id": target_id,
            "transport": "hf",
            "repo": repo,
            "type": "dataset",
            "gated": gated,
            "dest": dest,
            "strip_prefix": strip_prefix,
            "files": files,
            "total_size": total,
            "note": note,
        }
    )
    log(f"  {target_id}: {len(files)} files, {human(total)}")


def add_url_target(target_id, dest, entries, note=""):
    files = []
    for rel, url, sha256, fallback in entries:
        size = remote_size(url, fallback=fallback)
        files.append({"path": rel, "url": url, "size": size, "sha256": sha256})
        log(f"  {target_id}: {rel} {human(size)}")
    plan["targets"].append(
        {
            "id": target_id,
            "transport": "url",
            "type": "dataset",
            "dest": dest,
            "files": files,
            "total_size": sum(f["size"] for f in files),
            "note": note,
        }
    )


# ------------------------------------------------------------------ FLORES-200
# Gated.  HF stores these under data/language/<lang>/; our local layout dropped that
# prefix, so strip_prefix reproduces data/raw/flores200_20260529/<lang>/<split>.parquet.
log("\n=== FLORES-200 (gated) ===")
# remote path -> (local path relative to dest, SHA-256 verified against our copy)
FLORES_WANTED = {
    "data/language/bod_Tibt/dev-00000-of-00001.parquet": (
        "bod_Tibt/dev-00000-of-00001.parquet",
        "5F7949E1302C5322EE132368B3374215BBD8FFAA24B30E4756777AFD8EE6E06F",
    ),
    "data/language/bod_Tibt/devtest-00000-of-00001.parquet": (
        "bod_Tibt/devtest-00000-of-00001.parquet",
        "D3A3F62EF18A5FEB99AA619332379BCA65D47F713F65B85EF8138DF36486D622",
    ),
    "data/language/zho_Hans/dev-00000-of-00001.parquet": (
        "zho_Hans/dev-00000-of-00001.parquet",
        "F9151A874D14B3DD9086C7F788425618EB99E12246BC0FAE99EA33FD363D19E8",
    ),
    "data/language/zho_Hans/devtest-00000-of-00001.parquet": (
        "zho_Hans/devtest-00000-of-00001.parquet",
        "10E0038BEF15832B981F50FECE056F7F6ECDE8CCFFA837A08999BB2B5575D973",
    ),
}
try:
    all_files = {f["path"]: f for f in hf_files("facebook/flores")}
    flores_files = []
    for remote, (local, sha) in FLORES_WANTED.items():
        entry = all_files.get(remote)
        if entry is None:
            log(f"  MISSING on HF: {remote}")
            continue
        flores_files.append(
            {"path": remote, "size": entry["size"], "dest": local, "sha256": sha}
        )
    add_hf_target(
        "flores200",
        "facebook/flores",
        "data/raw/flores200_20260529",
        flores_files,
        gated=True,
        strip_prefix="data/language/",
        note="Evaluation only. Never train or tune on FLORES. Requires accepting the gated terms.",
    )
except Exception as exc:  # noqa: BLE001
    log(f"  ERROR {type(exc).__name__}: {exc}")

# --------------------------------------------------- MITRA v2 full + official eval
log("\n=== MITRA Parallel v2 (pinned commit, plain HTTPS) ===")
add_url_target(
    "mitra_v2_full",
    "data/raw/mitra_v2",
    [
        (
            "bo-zh_matches.ndjson.gz",
            f"{MITRA_RAW}/v2/bo-zh_matches.ndjson.gz",
            "9179AD2F491A77FCEB06A52FDD93E1D2AECE3AD8623EF0BAA18A08E6373DF897",
            80459833,
        )
    ],
    note="836,559 records; Wylie source text, convert with pyewts.",
)
add_url_target(
    "mitra_v2_eval",
    "data/raw/mitra_v2_eval",
    [
        (
            "bo2zh.tsv",
            f"{MITRA_RAW}/v2/evaluation/bo-zh/bo2zh.tsv",
            "324138D98644402C6231F67AE709B9FA16247DC5C1DA240117FB524E4BAA9D58",
            897700,
        ),
        (
            "lotsawahouse.tsv",
            f"{MITRA_RAW}/v2/evaluation/bo-zh/lotsawahouse.tsv",
            "F4FF4C0A889178C1D534B66D926EF6B2B19EE84C84652C327F34B2B47B812189",
            469575,
        ),
    ],
    note="Fixed official held-out benchmarks. Never train or tune on these.",
)

# ------------------------------------------------------------------- DongbaMIE
log("\n=== DongbaMIE (gated) ===")
try:
    files = hf_files("thinklis/DongbaMIE")
    for f in sorted(files, key=lambda x: -x["size"]):
        log(f"   {f['path']:52s} {human(f['size'])}")
    add_hf_target(
        "dongbamie",
        "thinklis/DongbaMIE",
        "data/raw/dongbamie",
        files,
        gated=True,
        note="CC-BY-NC-SA-4.0. Never redistribute outside the team.",
    )
except Exception as exc:  # noqa: BLE001
    log(f"  ERROR {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------- BDRC
log("\n=== BDRC/tibetan-ocr-benchmark ===")
try:
    files = hf_files("BDRC/tibetan-ocr-benchmark")
    add_hf_target(
        "bdrc_tibetan_ocr_benchmark",
        "BDRC/tibetan-ocr-benchmark",
        "data/raw/bdrc_tibetan_ocr",
        files,
        note="CC0-1.0, 472 rows. OCR evaluation only.",
    )
except Exception as exc:  # noqa: BLE001
    log(f"  ERROR {type(exc).__name__}: {exc}")

# ----------------------------------------------------------------- CUTE subset
log("\n=== CMLI-NLP/CUTE-Datasets (Tibetan subset only) ===")
try:
    files = hf_files("CMLI-NLP/CUTE-Datasets")
    wanted = [
        f for f in files
        if f["path"] in ("parallel-corpus/bo.txt", "non-parallel-corpus/n-bo.txt")
    ]
    add_hf_target(
        "cute_bo_subset",
        "CMLI-NLP/CUTE-Datasets",
        "data/raw/cute_v1",
        wanted,
        note="Machine-translated synthetic data: ablation only, never a gold benchmark.",
    )
except Exception as exc:  # noqa: BLE001
    log(f"  ERROR {type(exc).__name__}: {exc}")

# ----------------------------------------------------------------- GitHub repos
log("\n=== GitHub repos ===")
GH = [
    ("openpecha_c0a2dd042", "OpenPecha-Data/C0A2DD042", "main", "data/raw/openpecha_c0a2dd042",
     "CC0-1.0 claimed; the repository itself contains no LICENSE file."),
    ("modern_tibetan_corpus", "tibetan-nlp/modern-tibetan-corpus", "main",
     "data/raw/modern_tibetan_corpus", "MIT."),
    ("shajiu_publiccorpus_samples", "Shajiu/PublicCorpus", "master",
     "data/raw/shajiu_publiccorpus", "Custom research terms; sample-only use."),
    ("dongbamie_code", "thinklis/DongbaMIE", "main", "data/raw/dongbamie_code",
     "Paper code and figures (not the dataset)."),
]
for target_id, repo, branch, dest, note in GH:
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
                "note": note,
            }
        )
    except Exception as exc:  # noqa: BLE001
        log(f"{repo}: ERROR {type(exc).__name__}: {exc}")

# ------------------------------------------------------------------ Dongba1800
log("\n=== Dongba1800 (figshare) ===")
try:
    status, info = fetch_json("https://api.figshare.com/v2/articles/26969755", proxied=True)
    files = [
        {"path": f.get("name"), "size": f.get("size", 0), "url": f.get("download_url")}
        for f in info.get("files", [])
    ]
    if files:
        log(f"  {info.get('title', '')[:70]}")
        for f in files:
            log(f"     {f['path']} {human(f['size'])}")
        plan["targets"].append(
            {
                "id": "dongba1800",
                "transport": "figshare",
                "article_id": 26969755,
                "type": "dataset",
                "dest": "data/raw/dongba1800",
                "files": files,
                "total_size": sum(f["size"] for f in files),
                "note": "Licence text conflicts: paper says CC BY 4.0 but also states non-commercial.",
            }
        )
except Exception as exc:  # noqa: BLE001
    log(f"  ERROR {type(exc).__name__}: {exc}")

present = {t["id"] for t in plan["targets"]}
missing = sorted(REQUIRED_TARGETS - present)
extra = sorted(present - REQUIRED_TARGETS)

if missing:
    log("")
    log("ERROR: plan is INCOMPLETE, refusing to overwrite the existing plan.")
    log("       missing targets: " + ", ".join(missing))
    log("       This is almost always a transient API/proxy failure; re-running is safe.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(1)

plan["target_count"] = len(plan["targets"])
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\n".join(lines))
if extra:
    print(f"\nnote: unexpected extra targets: {', '.join(extra)}")
print(f"\nwrote {PLAN} ({len(plan['targets'])} targets, all required present)")
