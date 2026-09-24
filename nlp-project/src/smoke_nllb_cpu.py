"""Run a small local NLLB CPU translation smoke test.

This script intentionally performs inference only.  It loads a complete local
NLLB checkpoint, translates a small FLORES sample, and writes JSONL examples.
It never trains, modifies raw data, or silently falls back to an API.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_flores(root: Path, split: str, limit: int) -> list[tuple[int, str, str]]:
    source = root / "bod_Tibt" / f"{split}-00000-of-00001.parquet"
    target = root / "zho_Hans" / f"{split}-00000-of-00001.parquet"
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT source.id, source.sentence, target.sentence "
            "FROM read_parquet(?) AS source JOIN read_parquet(?) AS target USING (id) "
            "ORDER BY source.id LIMIT ?",
            [str(source), str(target), limit],
        ).fetchall()
    finally:
        connection.close()
    return [(int(row_id), str(source_text), str(target_text)) for row_id, source_text, target_text in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "devtest"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    root = args.project_root.resolve()
    model_dir = (args.model_dir or root / "outputs" / "nllb-200-distilled-600M").resolve()
    required = [model_dir / "config.json", model_dir / "tokenizer.json"]
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not all(path.exists() for path in required) or not weight_files:
        raise FileNotFoundError(
            f"完整 NLLB checkpoint not found in {model_dir}. "
            "Download the model weights first; tokenizer-only files are insufficient."
        )

    torch.set_num_threads(max(1, args.threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer.src_lang = "bod_Tibt"
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    load_seconds = time.perf_counter() - started

    rows = load_flores(root / "data" / "raw" / "flores200_20260529", args.split, args.limit)
    sources = [row[1] for row in rows]
    targets = [row[2] for row in rows]
    inputs = tokenizer(sources, return_tensors="pt", padding=True, truncation=False)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("zho_Hans")
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
        )
    generation_seconds = time.perf_counter() - generation_started
    translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

    output_rows = [
        {
            "id": row_id,
            "source_tibetan": source,
            "reference_chinese": reference,
            "hypothesis_chinese": hypothesis,
        }
        for (row_id, source, reference), hypothesis in zip(rows, translations)
    ]
    output = root / "reports" / "nllb_cpu_smoke.json"
    output.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_dir": str(model_dir),
                "split": args.split,
                "examples": len(output_rows),
                "torch_version": torch.__version__,
                "threads": torch.get_num_threads(),
                "load_seconds": round(load_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "seconds_per_example": round(generation_seconds / max(1, len(output_rows)), 3),
                "rows": output_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(f"Loaded in {load_seconds:.2f}s; generated {len(output_rows)} examples in {generation_seconds:.2f}s")


if __name__ == "__main__":
    main()
