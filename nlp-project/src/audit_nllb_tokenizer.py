"""Audit the selected NLLB tokenizer without downloading model weights.

This script is deliberately CPU-only.  It loads ``AutoTokenizer`` and streams
the local corpus, reporting token-length and unknown-token statistics.  It does
not train a model and does not perform semantic evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import duckdb
from transformers import AutoTokenizer


DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
FLORES_LANGUAGES = {"source": "bod_Tibt", "target": "zho_Hans"}
MITRA_LANGUAGES = {"source": "bod_Tibt", "target": "zho_Hant"}


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return float(ordered[index])


def describe_unknown_spans(spans: Counter, limit: int = 30) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for text, count in spans.most_common(limit):
        result.append(
            {
                "text": text,
                "count": count,
                "characters": [
                    {
                        "character": character,
                        "codepoint": f"U+{ord(character):04X}",
                        "unicode_name": unicodedata.name(character, "UNNAMED"),
                    }
                    for character in text
                ],
            }
        )
    return result


def summary(
    values: Sequence[int],
    unk_counts: Sequence[int],
    unknown_spans: Counter,
    trained_limit: int,
) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(mean(values), 3),
        "median": median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "over_256": sum(value > 256 for value in values),
        "over_512_trained_limit": sum(value > trained_limit for value in values),
        "unknown_token_total": sum(unk_counts),
        "examples_with_unknown_token": sum(value > 0 for value in unk_counts),
        "top_unknown_spans": describe_unknown_spans(unknown_spans),
    }


def reservoir_sample(rows: Iterable[Tuple[str, str]], count: int, seed: int) -> List[Tuple[str, str]]:
    """Deterministically sample pairs while keeping bounded memory."""
    rng = random.Random(seed)
    sample: List[Tuple[str, str]] = []
    for index, row in enumerate(rows):
        if index < count:
            sample.append(row)
            continue
        replacement = rng.randint(0, index)
        if replacement < count:
            sample[replacement] = row
    return sample


def load_mitra_rows(path: Path, count: int, seed: int) -> List[Tuple[str, str]]:
    def rows() -> Iterable[Tuple[str, str]]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                yield record["source_tibetan"], record["target_chinese"]

    return reservoir_sample(rows(), count, seed)


def load_mitra_validation(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append((record["source_tibetan"], record["target_chinese"]))
    return rows


def load_flores(root: Path, split: str) -> List[Tuple[str, str]]:
    source = root / "bod_Tibt" / f"{split}-00000-of-00001.parquet"
    target = root / "zho_Hans" / f"{split}-00000-of-00001.parquet"
    connection = duckdb.connect()
    try:
        result = connection.execute(
            "SELECT source.sentence, target.sentence "
            "FROM read_parquet(?) AS source "
            "JOIN read_parquet(?) AS target USING (id) "
            "ORDER BY id",
            [str(source), str(target)],
        ).fetchall()
    finally:
        connection.close()
    return [(str(source_text), str(target_text)) for source_text, target_text in result]


def audit_pairs(
    tokenizer: Any,
    pairs: Sequence[Tuple[str, str]],
    source_lang: str,
    target_lang: str,
    batch_size: int,
    trained_limit: int,
) -> Dict[str, Any]:
    source_lengths: List[int] = []
    target_lengths: List[int] = []
    source_unknowns: List[int] = []
    target_unknowns: List[int] = []
    source_unknown_spans: Counter = Counter()
    target_unknown_spans: Counter = Counter()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        sources = [row[0] for row in batch]
        targets = [row[1] for row in batch]
        tokenizer.src_lang = source_lang
        source_encoded = tokenizer(
            sources,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        tokenizer.tgt_lang = target_lang
        target_encoded = tokenizer(
            text_target=targets,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        unk_id = tokenizer.unk_token_id
        source_ids = source_encoded["input_ids"]
        target_ids = target_encoded["input_ids"]
        for text, ids, offsets in zip(sources, source_ids, source_encoded["offset_mapping"]):
            for token, (start, end) in zip(ids, offsets):
                if token == unk_id and end > start:
                    source_unknown_spans[text[start:end]] += 1
        for text, ids, offsets in zip(targets, target_ids, target_encoded["offset_mapping"]):
            for token, (start, end) in zip(ids, offsets):
                if token == unk_id and end > start:
                    target_unknown_spans[text[start:end]] += 1
        source_lengths.extend(len(ids) for ids in source_ids)
        target_lengths.extend(len(ids) for ids in target_ids)
        source_unknowns.extend(sum(token == unk_id for token in ids) for ids in source_ids)
        target_unknowns.extend(sum(token == unk_id for token in ids) for ids in target_ids)
    return {
        "pairs": len(pairs),
        "source_language": source_lang,
        "target_language": target_lang,
        "source": summary(source_lengths, source_unknowns, source_unknown_spans, trained_limit),
        "target": summary(target_lengths, target_unknowns, target_unknown_spans, trained_limit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--mitra-train-sample", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--trained-limit", type=int, default=512)
    args = parser.parse_args()

    root = args.project_root.resolve()
    tokenizer_kwargs: Dict[str, Any] = {}
    if args.cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = str(args.cache_dir.resolve())
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tokenizer_kwargs)

    flores_root = root / "data" / "raw" / "flores200_20260529"
    mitra_root = root / "data" / "processed" / "mitra_v2_conservative"
    datasets = {
        "flores_dev": (load_flores(flores_root, "dev"), FLORES_LANGUAGES),
        "flores_devtest": (load_flores(flores_root, "devtest"), FLORES_LANGUAGES),
        "mitra_validation": (load_mitra_validation(mitra_root / "validation.jsonl.gz"), MITRA_LANGUAGES),
        "mitra_train_sample": (
            load_mitra_rows(mitra_root / "train.jsonl.gz", args.mitra_train_sample, args.seed),
            MITRA_LANGUAGES,
        ),
    }

    report: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "trained_input_limit_used_for_flagging": args.trained_limit,
        "seed": args.seed,
        "datasets": {},
    }
    for name, (pairs, languages) in datasets.items():
        report["datasets"][name] = audit_pairs(
            tokenizer,
            pairs,
            languages["source"],
            languages["target"],
            args.batch_size,
            args.trained_limit,
        )

    output = root / "reports" / "nllb_tokenizer_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
