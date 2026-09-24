"""Verify the environment after installing new packages.

Confirms both that the new tooling works and that the existing NLLB pipeline is
intact (torch / transformers / numpy unchanged).
"""

from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path

out: list[str] = []


def version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "NOT INSTALLED"


out.append("### versions")
for name in [
    "torch", "transformers", "tokenizers", "sentencepiece", "numpy", "duckdb",
    "pyewts", "huggingface_hub", "sacrebleu", "pandas", "matplotlib",
    "openpyxl", "tabulate", "lxml", "regex",
]:
    out.append(f"  {name:20s} {version(name)}")

out.append("")
out.append("### imports")

try:
    import torch

    out.append(f"  torch              OK  {torch.__version__}")
    out.append(f"  cuda available     {torch.cuda.is_available()}")
except Exception as exc:  # noqa: BLE001
    out.append(f"  torch              FAILED {type(exc).__name__}: {exc}")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    out.append(f"  transformers       OK  {version('transformers')}")
except Exception as exc:  # noqa: BLE001
    out.append(f"  transformers       FAILED {type(exc).__name__}: {exc}")

try:
    import sacrebleu

    out.append(f"  sacrebleu          OK  {sacrebleu.__version__}")
except Exception as exc:  # noqa: BLE001
    out.append(f"  sacrebleu          FAILED {type(exc).__name__}: {exc}")

for mod in ["pandas", "matplotlib", "openpyxl", "tabulate", "lxml", "duckdb", "pyewts"]:
    try:
        module = __import__(mod)
        out.append(f"  {mod:18s} OK  {getattr(module, '__version__', 'n/a')}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"  {mod:18s} FAILED {type(exc).__name__}: {exc}")

# ---- functional check: the evaluation protocol must actually run -------------
out.append("")
out.append("### sacrebleu chrF++ / BLEU functional check")
try:
    import sacrebleu

    hypotheses = ["周一，斯坦福大学医学院的科学家宣布，他们发明了一种新型诊断工具。"]
    references = [["周一，斯坦福大学医学院的科学家宣布，他们发明了一种可以将细胞按类型分类的新型诊断工具。"]]
    chrf = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    out.append(f"  chrF++  = {chrf.score:.4f}")
    out.append(f"  chrF++ signature = {chrf.format(width=2).replace(chr(10), ' | ')}")
    out.append(f"  BLEU    = {bleu.score:.4f}")
    out.append(f"  BLEU signature   = {bleu.get_signature()}")
except Exception as exc:  # noqa: BLE001
    out.append(f"  FAILED {type(exc).__name__}: {exc}")

# ---- the existing NLLB checkpoint must still load ---------------------------
out.append("")
out.append("### existing NLLB artifacts still usable")
model_dir = Path(r"D:\Courses\nlp-project\outputs\nllb-200-distilled-600M")
weights = model_dir / "pytorch_model.bin"
out.append(f"  checkpoint present : {weights.exists()} ({weights.stat().st_size:,} bytes)" if weights.exists() else "  checkpoint MISSING")
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    encoded = tokenizer("བོད་ཡིག", return_tensors=None)
    decoded = tokenizer.convert_tokens_to_ids("zho_Hans")
    out.append(f"  tokenizer load    : OK (bod token ids={encoded['input_ids']})")
    out.append(f"  zho_Hans id       : {decoded}")
except Exception as exc:  # noqa: BLE001
    out.append(f"  tokenizer load    : FAILED {type(exc).__name__}: {exc}")

Path(r"D:\Courses\nlp-project\outputs\_env_verify.txt").write_text(
    "\n".join(out) + "\n", encoding="utf-8"
)
Path(r"D:\Courses\nlp-project\outputs\_env_verify.json").write_text(
    json.dumps({"lines": out}, ensure_ascii=False, indent=2), encoding="utf-8"
)
