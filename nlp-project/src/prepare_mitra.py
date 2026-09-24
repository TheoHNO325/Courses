"""Build a conservative, document-split MITRA v2 training corpus.

This script performs only structural filtering. It does not claim that the
remaining Tibetan-Chinese pairs are semantically correct.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import io
import json
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

import pyewts


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def has_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff" for char in text)


def has_tibetan_letter(text: str) -> bool:
    return any("\u0f40" <= char <= "\u0f6c" for char in text)


def has_ascii_digit(text: str) -> bool:
    return any(char.isascii() and char.isdigit() for char in text)


def filter_record(
    record: dict[str, Any], converter: pyewts.pyewts, config: dict[str, Any]
) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if config["require_one_to_one"] and (
        len(record["root_segnr"]) != 1 or len(record["par_segnr"]) != 1
    ):
        return None, ["not_one_to_one"]

    source = str(record["root_string"])
    target = str(record["par_string"])
    source_length = len(source)
    target_length = len(target)
    ratio = target_length / max(1, source_length)

    if not config["min_source_wylie_chars"] <= source_length <= config["max_source_wylie_chars"]:
        reasons.append("source_length_outside_bounds")
    if not config["min_target_chinese_chars"] <= target_length <= config["max_target_chinese_chars"]:
        reasons.append("target_length_outside_bounds")
    if not config["min_target_to_source_char_ratio"] <= ratio <= config["max_target_to_source_char_ratio"]:
        reasons.append("length_ratio_outside_bounds")
    if config["require_cjk"] and not has_cjk(target):
        reasons.append("target_has_no_cjk")
    if config["exclude_ascii_digits"] and has_ascii_digit(target):
        reasons.append("target_has_ascii_digit")

    warnings: list[str] = []
    converted = converter.toUnicode(source, warnings)
    if config["reject_ewts_warnings"] and warnings:
        reasons.append("ewts_conversion_warning")
    if config["require_tibetan_letter_after_conversion"] and not has_tibetan_letter(converted):
        reasons.append("converted_source_has_no_tibetan_letter")
    return converted, reasons


def document_group(filename: str) -> str:
    if filename.startswith("BO_K"):
        return "Kanjur"
    if filename.startswith("BO_T"):
        return "Tengyur"
    return "Other"


def document_hash(seed: str, filename: str) -> str:
    return hashlib.sha256(f"{seed}\0{filename}".encode("utf-8")).hexdigest()


def select_validation_documents(
    document_counts: dict[str, int], validation_fraction: float, seed: str
) -> set[str]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for filename, count in document_counts.items():
        grouped[document_group(filename)].append((filename, count))

    selected: set[str] = set()
    for group, documents in grouped.items():
        target = round(sum(count for _, count in documents) * validation_fraction)
        current = 0
        for filename, count in sorted(documents, key=lambda item: document_hash(seed, item[0])):
            if abs(current + count - target) < abs(current - target):
                selected.add(filename)
                current += count
        if target and not any(filename in selected for filename, _ in documents):
            filename, _ = min(documents, key=lambda item: abs(item[1] - target))
            selected.add(filename)
    return selected


def quantiles(values: list[float], fractions: tuple[float, ...]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        f"p{round(fraction * 100)}": ordered[round((len(ordered) - 1) * fraction)]
        for fraction in fractions
    }


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(zipped, encoding="utf-8", newline="\n")
    try:
        yield text
    finally:
        text.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    actual_hash = sha256(args.input)
    if actual_hash != config["input_sha256"]:
        raise RuntimeError(f"Input SHA-256 mismatch: {actual_hash}")

    converter = pyewts.pyewts()
    raw_records = 0
    accepted_records = 0
    reason_counts: Counter[str] = Counter()
    first_failure_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    accepted_scores: list[float] = []

    for record in iter_records(args.input):
        raw_records += 1
        _, reasons = filter_record(record, converter, config)
        if reasons:
            reason_counts.update(reasons)
            first_failure_counts[reasons[0]] += 1
            continue
        accepted_records += 1
        document_counts[str(record["filename"])] += 1
        accepted_scores.append(float(record["score"]))

    validation_documents = select_validation_documents(
        dict(document_counts), config["validation_fraction"], config["split_seed"]
    )
    score_bounds = list(quantiles(accepted_scores, (0.2, 0.4, 0.6, 0.8)).values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_tmp = args.output_dir / "train.jsonl.gz.tmp"
    validation_tmp = args.output_dir / "validation.jsonl.gz.tmp"
    train_final = args.output_dir / "train.jsonl.gz"
    validation_final = args.output_dir / "validation.jsonl.gz"
    split_counts: Counter[str] = Counter()
    split_documents: dict[str, set[str]] = {"train": set(), "validation": set()}

    with deterministic_gzip_text(train_tmp) as train_handle, deterministic_gzip_text(
        validation_tmp
    ) as validation_handle:
        for record in iter_records(args.input):
            converted, reasons = filter_record(record, converter, config)
            if reasons:
                continue
            filename = str(record["filename"])
            split = "validation" if filename in validation_documents else "train"
            score = float(record["score"])
            output_record = {
                "id": record["id"],
                "source_tibetan": unicodedata.normalize("NFC", str(converted).strip()),
                "source_wylie": str(record["root_string"]),
                "target_chinese": unicodedata.normalize("NFC", str(record["par_string"]).strip()),
                "score": score,
                "score_stratum": f"Q{bisect.bisect_right(score_bounds, score) + 1}",
                "filename": filename,
                "source_segment_id": str(record["root_segnr"][0]),
                "target_segment_id": str(record["par_segnr"][0]),
            }
            handle = validation_handle if split == "validation" else train_handle
            handle.write(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            split_counts[split] += 1
            split_documents[split].add(filename)

    if sum(split_counts.values()) != accepted_records:
        raise RuntimeError("Second-pass record count does not match first pass")

    train_tmp.replace(train_final)
    validation_tmp.replace(validation_final)
    manifest = {
        "input": str(args.input),
        "input_sha256": actual_hash,
        "config": str(args.config),
        "split_unit": "filename",
        "split_seed": config["split_seed"],
        "validation_documents": sorted(validation_documents),
        "outputs": {
            "train": {
                "path": str(train_final),
                "records": split_counts["train"],
                "documents": len(split_documents["train"]),
                "sha256": sha256(train_final),
            },
            "validation": {
                "path": str(validation_final),
                "records": split_counts["validation"],
                "documents": len(split_documents["validation"]),
                "sha256": sha256(validation_final),
            },
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "raw_records": raw_records,
        "accepted_records": accepted_records,
        "accepted_fraction": accepted_records / raw_records,
        "independent_rejection_reason_counts": dict(sorted(reason_counts.items())),
        "exclusive_first_failure_counts": dict(sorted(first_failure_counts.items())),
        "accepted_documents": len(document_counts),
        "accepted_score_quantiles": quantiles(
            accepted_scores, (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 1.0)
        ),
        "split": manifest["outputs"],
        "config": config,
        "limitation": "Structural filtering only; semantic alignment and translation fidelity were not human-verified.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
