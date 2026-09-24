"""Stream-scan MITRA v2 and create a deterministic stratified review sample."""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import random
import statistics
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyewts


REQUIRED_FIELDS = {
    "id",
    "score",
    "root_segnr",
    "par_segnr",
    "root_string",
    "par_string",
    "root_length",
    "par_length",
    "src_lang",
    "tgt_lang",
    "filename",
}


def iter_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def quantile(sorted_values: list[float] | list[int], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    index = round((len(sorted_values) - 1) * fraction)
    return float(sorted_values[index])


def first_pass(path: Path) -> tuple[dict[str, Any], list[float], float]:
    scores = array("d")
    lengths = array("I")
    counts: Counter[str] = Counter()
    source_segment_counts: Counter[str] = Counter()
    target_segment_counts: Counter[str] = Counter()

    for _, record in iter_records(path):
        counts["records"] += 1
        missing = REQUIRED_FIELDS.difference(record)
        if missing:
            counts["records_missing_required_fields"] += 1
            continue

        source = str(record["root_string"])
        target = str(record["par_string"])
        if not source.strip():
            counts["blank_source"] += 1
        if not target.strip():
            counts["blank_target"] += 1
        if any("\u0f00" <= char <= "\u0fff" for char in source):
            counts["source_with_tibetan_unicode"] += 1
        if record.get("src_lang") != "bo":
            counts["unexpected_src_lang"] += 1
        if record.get("tgt_lang") != "zh":
            counts["unexpected_tgt_lang"] += 1

        score = float(record["score"])
        length = max(int(record["root_length"]), int(record["par_length"]))
        if score > 1.0:
            counts["score_above_one"] += 1
        if any(char.isascii() and char.isdigit() for char in target):
            counts["target_with_ascii_digit"] += 1
        if record.get("gemini_score") is not None:
            counts["gemini_score_non_null"] += 1
        scores.append(score)
        lengths.append(max(0, length))
        source_segment_counts[str(len(record["root_segnr"]))] += 1
        target_segment_counts[str(len(record["par_segnr"]))] += 1

    sorted_scores = sorted(scores)
    score_bounds = [quantile(sorted_scores, q) for q in (0.2, 0.4, 0.6, 0.8)]
    score_summary = {
        "min": sorted_scores[0],
        "p20": score_bounds[0],
        "p40": score_bounds[1],
        "median": quantile(sorted_scores, 0.5),
        "p60": score_bounds[2],
        "p80": score_bounds[3],
        "p95": quantile(sorted_scores, 0.95),
        "max": sorted_scores[-1],
        "mean": statistics.fmean(scores),
    }
    del sorted_scores

    sorted_lengths = sorted(lengths)
    length_median = quantile(sorted_lengths, 0.5)
    length_summary = {
        "min": sorted_lengths[0],
        "median": length_median,
        "p95": quantile(sorted_lengths, 0.95),
        "max": sorted_lengths[-1],
    }

    for key in (
        "records_missing_required_fields",
        "blank_source",
        "blank_target",
        "source_with_tibetan_unicode",
        "unexpected_src_lang",
        "unexpected_tgt_lang",
        "score_above_one",
        "target_with_ascii_digit",
        "gemini_score_non_null",
    ):
        counts.setdefault(key, 0)

    stats = {
        "input": str(path),
        "counts": dict(sorted(counts.items())),
        "score": score_summary,
        "max_source_or_target_length": length_summary,
        "source_segment_count_distribution": dict(sorted(source_segment_counts.items(), key=lambda x: int(x[0]))),
        "target_segment_count_distribution": dict(sorted(target_segment_counts.items(), key=lambda x: int(x[0]))),
    }
    return stats, score_bounds, length_median


def sample_records(
    path: Path,
    score_bounds: list[float],
    length_median: float,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    strata_count = 10
    if sample_size % strata_count:
        raise ValueError("sample_size must be divisible by 10")
    quota = sample_size // strata_count
    rng = random.Random(seed)
    seen: Counter[tuple[int, int]] = Counter()
    reservoirs: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for line_number, record in iter_records(path):
        if REQUIRED_FIELDS.difference(record):
            continue
        score = float(record["score"])
        max_length = max(int(record["root_length"]), int(record["par_length"]))
        score_bin = bisect.bisect_right(score_bounds, score)
        length_bin = int(max_length > length_median)
        key = (score_bin, length_bin)
        seen[key] += 1

        reservoir = reservoirs[key]
        if len(reservoir) < quota:
            reservoir.append({"line_number": line_number, **record})
        else:
            replacement = rng.randrange(seen[key])
            if replacement < quota:
                reservoir[replacement] = {"line_number": line_number, **record}

    missing = {str(key): quota - len(reservoirs[key]) for key in ((s, l) for s in range(5) for l in range(2)) if len(reservoirs[key]) < quota}
    if missing:
        raise RuntimeError(f"Not enough records in strata: {missing}")

    converter = pyewts.pyewts()
    output: list[dict[str, Any]] = []
    for score_bin in range(5):
        for length_bin in range(2):
            selected = sorted(reservoirs[(score_bin, length_bin)], key=lambda r: (float(r["score"]), r["id"]))
            for record in selected:
                warnings: list[str] = []
                source_wylie = str(record["root_string"])
                source_unicode = converter.toUnicode(source_wylie, warnings)
                flags: list[str] = []
                if len(record["root_segnr"]) != 1:
                    flags.append("multi_source_segment")
                if len(record["par_segnr"]) != 1:
                    flags.append("multi_target_segment")
                if any(char.isascii() and char.isdigit() for char in str(record["par_string"])):
                    flags.append("target_has_ascii_digit")
                if warnings:
                    flags.append("ewts_conversion_warning")

                output.append(
                    {
                        "sample_id": len(output) + 1,
                        "score_stratum": f"Q{score_bin + 1}",
                        "length_stratum": "long" if length_bin else "short",
                        "line_number": record["line_number"],
                        "id": record["id"],
                        "score": float(record["score"]),
                        "root_length": int(record["root_length"]),
                        "par_length": int(record["par_length"]),
                        "source_segment_count": len(record["root_segnr"]),
                        "target_segment_count": len(record["par_segnr"]),
                        "source_wylie": source_wylie,
                        "source_tibetan_unicode": source_unicode,
                        "target_chinese": str(record["par_string"]),
                        "source_segment_ids": " | ".join(map(str, record["root_segnr"])),
                        "target_segment_ids": " | ".join(map(str, record["par_segnr"])),
                        "filename": str(record["filename"]),
                        "co_occ": record.get("co_occ"),
                        "gemini_score": record.get("gemini_score"),
                        "automatic_flags": " | ".join(flags),
                        "ewts_warnings": " | ".join(warnings[:5]),
                        "alignment_ok": "",
                        "fidelity_1_5": "",
                        "chinese_fluency_1_5": "",
                        "style_match_1_5": "",
                        "ewts_conversion_ok": "",
                        "notes": "",
                    }
                )

    stratum_sizes = {f"Q{s + 1}_{'long' if l else 'short'}": seen[(s, l)] for s in range(5) for l in range(2)}
    return output, stratum_sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    stats, score_bounds, length_median = first_pass(args.input)
    sample, stratum_sizes = sample_records(
        args.input,
        score_bounds=score_bounds,
        length_median=length_median,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    stats["sample"] = {
        "size": len(sample),
        "seed": args.seed,
        "design": "five score quantile bins crossed with two median-length bins",
        "population_by_stratum": stratum_sizes,
    }

    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sample_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.stats_output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
