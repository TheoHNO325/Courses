"""Safety gate before pushing to a PUBLIC repo: scan everything to be uploaded.

Checks both generic credential patterns and the literal value of the local
HuggingFace token, so a leaked token cannot slip into the commit.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(r"D:\Courses\nlp-project")

EXCLUDE_DIRS = {
    ".venv", ".hf-cache", ".git", "__pycache__", ".ipynb_checkpoints",
    "data", "outputs", "_vendor", ".locks",
}
EXCLUDE_SUFFIX = {".bin", ".safetensors", ".pt", ".pth", ".whl", ".tmp", ".pyc", ".gz", ".zip", ".jpg", ".png"}

PATTERNS = {
    "HF token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "GH oauth": re.compile(r"gho_[A-Za-z0-9]{30,}"),
    "GH pat": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    "GH fine-grained": re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    "AWS key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer": re.compile(r"[Bb]earer\s+[A-Za-z0-9_\-\.]{25,}"),
    "password assign": re.compile(r"(?i)(password|passwd|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
}

# the real token value, loaded so we can search for it verbatim
token_path = Path(os.environ.get("USERPROFILE", "")) / ".cache" / "huggingface" / "token"
literal = None
if token_path.exists():
    literal = token_path.read_text(encoding="utf-8").strip()

findings: list[str] = []
scanned = 0
text_files = 0

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for name in filenames:
        if Path(name).suffix in EXCLUDE_SUFFIX:
            continue
        full = Path(dirpath) / name
        rel = full.relative_to(ROOT).as_posix()
        scanned += 1
        try:
            raw = full.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue  # binary
        text_files += 1
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue

        if literal and literal in text:
            findings.append(f"!! LITERAL HF TOKEN in {rel}")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                snippet = match.group(0)
                masked = snippet[:8] + "..." + snippet[-4:] if len(snippet) > 16 else snippet
                findings.append(f"[{label}] {rel}: {masked}")

print(f"scanned {scanned} files ({text_files} text) under {ROOT}")
print(f"token literal loaded: {bool(literal)} (len={len(literal) if literal else 0})")
print("")
if findings:
    print("FINDINGS:")
    for f in findings:
        print("  " + f)
else:
    print("RESULT: no secrets found in the upload set")

out = ROOT / "outputs" / "github_secret_scan.txt"
out.write_text(
    "\n".join(
        [f"scanned={scanned} text={text_files} token_literal_loaded={bool(literal)}"]
        + (findings or ["no secrets found"])
    )
    + "\n",
    encoding="utf-8",
)
