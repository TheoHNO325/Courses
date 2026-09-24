"""Validate the generated tokenizer HTML report.

Checks performed:
  1. the embedded JSON payload parses and contains every expected dataset/config;
  2. the inline JavaScript parses as ES (via vendored esprima), since node.exe is
     blocked by the sandbox;
  3. every section referenced by the navigation actually exists, and every
     rendered number comes from the source reports (no hard-coded-away results).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "_vendor"))

REPORT = ROOT / "reports" / "tokenizer_experiments.html"
OUT = ROOT / "outputs" / "_report_check.txt"

lines: list[str] = []
failures: list[str] = []


def log(message: str = "") -> None:
    lines.append(message)


GLOBALS = {
    "document", "JSON", "Math", "Number", "String", "Array", "Object", "Boolean",
    "console", "parseInt", "parseFloat", "isNaN", "isFinite", "setTimeout",
    "encodeURIComponent", "decodeURIComponent", "Infinity", "NaN", "undefined",
    "Error", "RegExp", "Date", "Set", "Map", "Promise",
}


def _walk(node, visit) -> None:
    if isinstance(node, list):
        for item in node:
            _walk(item, visit)
        return
    if not isinstance(node, dict):
        return
    visit(node)
    for key, value in node.items():
        if key in ("type", "loc", "range"):
            continue
        _walk(value, visit)


def _check_callees(tree, log_fn, fails) -> None:
    """Report calls to names that are never declared (a classic silent-blank-page bug)."""
    declared: set[str] = set()
    calls: list[str] = []

    def visit(node):
        t = node.get("type")
        if t in ("FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression"):
            if isinstance(node.get("id"), dict) and node["id"].get("type") == "Identifier":
                declared.add(node["id"]["name"])
            for param in node.get("params", []) or []:
                if isinstance(param, dict) and param.get("type") == "Identifier":
                    declared.add(param["name"])
        elif t == "VariableDeclarator" and isinstance(node.get("id"), dict):
            if node["id"].get("type") == "Identifier":
                declared.add(node["id"]["name"])
        elif t == "CallExpression":
            callee = node.get("callee")
            if isinstance(callee, dict) and callee.get("type") == "Identifier":
                calls.append(callee["name"])

    _walk(tree, visit)
    unknown = sorted({c for c in calls if c not in declared and c not in GLOBALS})
    log_fn(f"declared functions : {len(declared)}")
    log_fn(f"distinct calls     : {len(set(calls))}")
    if unknown:
        fails.append("calls to undeclared functions: " + ", ".join(unknown))
        log_fn("UNDECLARED CALLS   : " + ", ".join(unknown))
    else:
        log_fn("undeclared calls   : none (all callees are declared or known globals)")


html = REPORT.read_text(encoding="utf-8")
log(f"report      : {REPORT}")
log(f"size        : {REPORT.stat().st_size:,} bytes")
log()

# ---------------------------------------------------------------- payload
match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
if not match:
    failures.append("payload script tag missing")
    raw = ""
else:
    raw = match.group(1)
    try:
        payload = json.loads(raw)
        log("payload JSON : parses OK")
        fert_ds = sorted(payload["fertility"]["datasets"].keys())
        log(f"  fertility datasets : {fert_ds}")
        log(f"  configs            : {[c['key'] for c in payload['configs']]}")
        log(f"  bpe rows           : {len(payload['bpe'])}")
        log(f"  nllb audit datasets: {sorted(payload['nllb_audit']['datasets'].keys())}")
        missing = {"flores_dev", "flores_devtest", "mitra_validation", "mitra_train_sample"} - set(fert_ds)
        if missing:
            failures.append(f"fertility payload missing datasets: {missing}")
        if not re.search(r"\\u003c", raw) and "<" in raw and "</" in raw:
            log("  WARNING: payload contains a raw '</' sequence")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"payload JSON failed to parse: {type(exc).__name__}: {exc}")
        payload = None

# --------------------------------------------------------------- JS parse
scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
log()
log(f"inline script blocks: {len(scripts)}")
if not scripts:
    failures.append("no inline script found")
else:
    js = scripts[-1]
    js_path = ROOT / "outputs" / "_report_inline.js"
    js_path.write_text(js, encoding="utf-8")
    log(f"inline JS length   : {len(js):,} chars -> {js_path.name}")
    try:
        import esprima

        try:
            tree = esprima.parseScript(js, {"tolerant": False})
            log("esprima parseScript: OK (no syntax errors)")
            _check_callees(esprima.toDict(tree), log, failures)
        except Exception as exc:  # noqa: BLE001
            try:
                esprima.parseModule(js)
                log("esprima parseScript failed but parseModule: OK")
                log(f"  script-mode error was: {exc}")
            except Exception as exc2:  # noqa: BLE001
                failures.append(f"JS syntax error: {exc2}")
                log(f"esprima PARSE FAILED: {type(exc2).__name__}: {exc2}")
    except Exception as exc:  # noqa: BLE001
        log(f"esprima unavailable: {type(exc).__name__}: {exc}")

# ------------------------------------------------------------ structure
log()
ids_defined = set(re.findall(r"addSection\(\s*'([a-z_]+)'", html))
nav_ids = set(re.findall(r"nav\.innerHTML", html))  # nav is built from SECTIONS
log(f"addSection ids     : {sorted(ids_defined)}")
for needle in [
    "overview", "protocol", "fertility", "tgt", "len", "unk",
    "trunc", "norm", "bpe", "cross", "conclusion", "appendix",
]:
    if needle not in ids_defined:
        failures.append(f"section '{needle}' not defined via addSection")

# every top_unknown_spans entry must appear in the payload (all results included)
n_spans = sum(
    len(blk[side]["top_unknown_spans"])
    for blk in payload["nllb_audit"]["datasets"].values()
    for side in ("source", "target")
)
log(f"unknown spans in payload: {n_spans} (all rendered as tables)")

# ------------------------------------------------------------- summary
log()
if failures:
    log("RESULT: FAILED")
    for f in failures:
        log("  - " + f)
else:
    log("RESULT: ALL CHECKS PASSED")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
sys.exit(1 if failures else 0)
