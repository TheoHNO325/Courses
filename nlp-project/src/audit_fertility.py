"""Unified tokenizer/fertility audit for the Tibetan--Chinese project.

This CPU-only experiment compares the content-token statistics of the local
NLLB tokenizer, the HY-MT2 tokenizer, and two CS336 assignment-1 BPE models.
It deliberately measures raw text without translation-model weights.  The
CS336 models are trained only on a deterministic 10k Tibetan sample from the
already-filtered MITRA training split.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
import random
import statistics
import sys
import types
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import duckdb
from transformers import AutoTokenizer


ASSIGNMENT1 = Path(r"D:\Courses\cs336\assignment1-basics")
if str(ASSIGNMENT1) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT1))


RAW_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MARK_AWARE_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def load_cs336_bpe_compat() -> tuple[Any, Any, Any]:
    """Load the assignment-1 implementation under the project's Python 3.9.

    The course files use the Python 3.10 ``X | Y`` annotation syntax.  We do
    not edit those files; this small in-memory compatibility shim only removes
    that annotation before importing the same implementation.
    """
    module = types.ModuleType("train_bpe")
    module.__file__ = str(ASSIGNMENT1 / "train_bpe.py")
    source = (ASSIGNMENT1 / "train_bpe.py").read_text(encoding="utf-8")
    source = source.replace("input_path: str | os.PathLike", "input_path")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module._raw_pretokenize = module.pretokenize

    # The student implementation assumes that every requested merge exists.
    # A small Tibetan-only corpus can exhaust its pair table first, so retain
    # the same merge rule but stop cleanly at the largest attainable vocabulary.
    def train_safe(freq_table: Any, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab = module.init_vocab()
        merges: list[tuple[bytes, bytes]] = []
        pair_to_word: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        pair_count: Counter = Counter()
        for word, count in freq_table.items():
            for index in range(len(word) - 1):
                pair = (word[index], word[index + 1])
                pair_count[pair] += count
                pair_to_word.setdefault(pair, set()).add(word)
        for iteration in range(max(0, vocab_size - len(vocab))):
            if not pair_count:
                break
            best_pair = max(pair_count, key=lambda pair: (pair_count[pair], (vocab[pair[0]], vocab[pair[1]])))
            merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
            new_id = 257 + iteration
            vocab[new_id] = merges[-1][0] + merges[-1][1]
            affected = list(pair_to_word.get(best_pair, set()))
            for word in affected:
                for index in range(len(word) - 1):
                    pair = (word[index], word[index + 1])
                    pair_count[pair] -= freq_table[word]
                    if pair_count[pair] <= 0:
                        pair_count.pop(pair, None)
                    pair_to_word.get(pair, set()).discard(word)
            for word in affected:
                merged: list[int] = []
                index = 0
                while index < len(word):
                    if index < len(word) - 1 and (word[index], word[index + 1]) == best_pair:
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(word[index])
                        index += 1
                merged_word = tuple(merged)
                count = freq_table[word]
                freq_table[merged_word] += count
                for index in range(len(merged_word) - 1):
                    pair = (merged_word[index], merged_word[index + 1])
                    pair_count[pair] += count
                    pair_to_word.setdefault(pair, set()).add(merged_word)
                freq_table.pop(word, None)
        return vocab, merges

    module.train = train_safe
    sys.modules["train_bpe"] = module
    tokenizer_module = importlib.import_module("bpe_tokenizer")
    return tokenizer_module.BPEtokenizer, module.train_bpe, module


BPEtokenizer, train_bpe, cs336_train_module = load_cs336_bpe_compat()


NLLB_TOKENIZER = "facebook/nllb-200-distilled-600M"
NLLB_LANGS = {"source": "bod_Tibt", "target": "zho_Hant"}
HY_TOKENIZER_DIR = "outputs/hy-mt2-tokenizer"


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))])


def describe(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p90": round(percentile(values, 0.90), 4),
        "p95": round(percentile(values, 0.95), 4),
        "p99": round(percentile(values, 0.99), 4),
        "max": max(values),
    }


def tibetan_syllable_count(text: str) -> int:
    """Approximate syllables using tsheg boundaries, not lexical words."""
    pieces = []
    current: list[str] = []
    for char in text:
        if char in {"\u0f0b", "\u0f0c"}:
            if current:
                pieces.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        pieces.append("".join(current))
    return sum(any("\u0f40" <= char <= "\u0f6c" for char in piece) for piece in pieces)


def reservoir(rows: Iterable[tuple[str, str]], count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    result: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        if index < count:
            result.append(row)
        else:
            replacement = rng.randint(0, index)
            if replacement < count:
                result[replacement] = row
    return result


def load_flores(root: Path, split: str) -> list[tuple[str, str]]:
    source = root / "bod_Tibt" / f"{split}-00000-of-00001.parquet"
    target = root / "zho_Hans" / f"{split}-00000-of-00001.parquet"
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT source.sentence, target.sentence "
            "FROM read_parquet(?) AS source JOIN read_parquet(?) AS target USING (id) "
            "ORDER BY id",
            [str(source), str(target)],
        ).fetchall()
    finally:
        connection.close()
    return [(str(source), str(target)) for source, target in rows]


def load_jsonl(path: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            result.append((record["source_tibetan"], record["target_chinese"]))
    return result


def load_train_sample(path: Path, count: int, seed: int) -> list[tuple[str, str]]:
    def rows() -> Iterator[tuple[str, str]]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                yield record["source_tibetan"], record["target_chinese"]

    return reservoir(rows(), count, seed)


def encode_lengths(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    nllb_lang: str | None = None,
    target: bool = False,
) -> tuple[list[int], list[int], int]:
    if nllb_lang is not None and target:
        tokenizer.tgt_lang = nllb_lang
        encoded = tokenizer(text_target=list(texts), add_special_tokens=False, truncation=False)
    elif nllb_lang is not None:
        tokenizer.src_lang = nllb_lang
        encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False)
    else:
        encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unknowns = [sum(token == unk_id for token in row) for row in ids] if unk_id is not None else [0] * len(ids)
    return [len(row) for row in ids], unknowns, sum(unknowns)


def bpe_lengths(
    tokenizer: BPEtokenizer,
    texts: Sequence[str],
    *,
    syllable_aware: bool = False,
) -> tuple[list[int], list[int], int]:
    prepared = list(texts)
    if syllable_aware:
        prepared = [text.replace("\u0f0b", "\u0f0b ").replace("\u0f0c", "\u0f0c ") for text in texts]
    lengths = [len(tokenizer.encode(text)) for text in prepared]
    return lengths, [0] * len(texts), 0


def metric_summary(texts: Sequence[str], lengths: Sequence[int], unknowns: Sequence[int]) -> dict[str, Any]:
    codepoints = [len(text) for text in texts]
    syllables = [tibetan_syllable_count(text) for text in texts]
    ratios_codepoints = [length / max(1, chars) for length, chars in zip(lengths, codepoints)]
    ratios_syllables = [length / syllable for length, syllable in zip(lengths, syllables) if syllable]
    return {
        "length_tokens": describe(lengths),
        "unicode_codepoints": describe(codepoints),
        "fertility_tokens_per_codepoint": {
            "count": len(ratios_codepoints),
            "mean": round(statistics.mean(ratios_codepoints), 6) if ratios_codepoints else 0.0,
            "median": round(statistics.median(ratios_codepoints), 6) if ratios_codepoints else 0.0,
            "p95": round(percentile(ratios_codepoints, 0.95), 6) if ratios_codepoints else 0.0,
        },
        "tibetan_syllables": describe(syllables),
        "fertility_tokens_per_tibetan_syllable": {
            "count": len(ratios_syllables),
            "mean": round(statistics.mean(ratios_syllables), 6) if ratios_syllables else 0.0,
            "median": round(statistics.median(ratios_syllables), 6) if ratios_syllables else 0.0,
            "p95": round(percentile(ratios_syllables, 0.95), 6) if ratios_syllables else 0.0,
        },
        "unknown_token_total": sum(unknowns),
        "examples_with_unknown_token": sum(value > 0 for value in unknowns),
        "unknown_rate_over_tokens": round(sum(unknowns) / max(1, sum(lengths)), 8),
        "over_256_tokens": sum(value > 256 for value in lengths),
        "over_512_tokens": sum(value > 512 for value in lengths),
    }


def configure_cs336_pretokenizer(variant: str) -> None:
    """Select the assignment pretokenizer for the current BPE training run."""
    if variant == "mark_aware":
        import regex as re

        def pretokenize_mark_aware(chunk: str, special_tokens: list[str]) -> Counter:
            local_counter: Counter = Counter()
            escaped_tokens = [re.escape(token) for token in special_tokens]
            escaped_tokens.sort(key=len, reverse=True)
            split_pattern = "|".join(escaped_tokens)
            segments = re.split(split_pattern, chunk)
            for segment in segments:
                if not segment:
                    continue
                for match in re.finditer(MARK_AWARE_PAT, segment):
                    local_counter[match.group()] += 1
            return local_counter

        cs336_train_module.pretokenize = pretokenize_mark_aware
    else:
        cs336_train_module.pretokenize = cs336_train_module.__dict__["_raw_pretokenize"]


def build_cs336_tokenizer(corpus: Path, vocab_size: int, output_dir: Path, variant: str) -> BPEtokenizer:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    if vocab_path.exists() and merges_path.exists():
        tokenizer = BPEtokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=[])
        tokenizer.PAT = MARK_AWARE_PAT if variant == "mark_aware" else RAW_PAT
        return tokenizer
    configure_cs336_pretokenizer(variant)
    vocab, merges = train_bpe(str(corpus), vocab_size, ["<|endoftext|>"], use_mp=False)
    vocab_json = {str(index): value.decode("latin-1") for index, value in vocab.items()}
    merges_json = [[left.decode("latin-1"), right.decode("latin-1")] for left, right in merges]
    vocab_path.write_text(json.dumps(vocab_json, ensure_ascii=False, indent=2), encoding="utf-8")
    merges_path.write_text(json.dumps(merges_json, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer = BPEtokenizer(vocab, merges, special_tokens=[])
    tokenizer.PAT = MARK_AWARE_PAT if variant == "mark_aware" else RAW_PAT
    return tokenizer


def make_corpus(rows: Sequence[tuple[str, str]], path: Path, syllable_aware: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source, _ in rows:
            text = source
            if syllable_aware:
                text = text.replace("\u0f0b", "\u0f0b ").replace("\u0f0c", "\u0f0c ")
            handle.write(text.replace("\n", " ").strip() + "\n")


def normalization_summary(tokenizer: Any, texts: Sequence[str], lang: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, transform in {
        "NFC": lambda text: unicodedata.normalize("NFC", text),
        "NFD": lambda text: unicodedata.normalize("NFD", text),
    }.items():
        transformed = [transform(text) for text in texts]
        lengths, _, _ = encode_lengths(tokenizer, transformed, nllb_lang=lang)
        result[label] = {
            "changed_examples": sum(a != b for a, b in zip(texts, transformed)),
            "token_length": describe(lengths),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--nllb-tokenizer-dir", type=Path, default=None)
    parser.add_argument("--hy-tokenizer-dir", type=Path, default=None)
    parser.add_argument("--train-sample", type=int, default=10000)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()
    root = args.project_root.resolve()

    nllb_kwargs: dict[str, Any] = {"local_files_only": True}
    local_nllb_dir = root / "outputs" / "nllb-tokenizer"
    if args.nllb_tokenizer_dir:
        nllb_kwargs["pretrained_model_name_or_path"] = str(args.nllb_tokenizer_dir.resolve())
    elif local_nllb_dir.exists():
        nllb_kwargs["pretrained_model_name_or_path"] = str(local_nllb_dir)
    else:
        nllb_kwargs["pretrained_model_name_or_path"] = NLLB_TOKENIZER
    nllb = AutoTokenizer.from_pretrained(**nllb_kwargs)

    hy_dir = (args.hy_tokenizer_dir or (root / HY_TOKENIZER_DIR)).resolve()
    hy = AutoTokenizer.from_pretrained(str(hy_dir), local_files_only=True)

    processed = root / "data" / "processed" / "mitra_v2_conservative"
    datasets: dict[str, list[tuple[str, str]]] = {
        "flores_dev": load_flores(root / "data" / "raw" / "flores200_20260529", "dev"),
        "flores_devtest": load_flores(root / "data" / "raw" / "flores200_20260529", "devtest"),
        "mitra_validation": load_jsonl(processed / "validation.jsonl.gz"),
        "mitra_train_sample": load_train_sample(processed / "train.jsonl.gz", args.train_sample, args.seed),
    }
    train_rows = datasets["mitra_train_sample"]
    corpus_dir = root / "data" / "interim" / "fertility"
    raw_corpus = corpus_dir / "mitra_train_10000_tibetan_raw.txt"
    syllable_corpus = corpus_dir / "mitra_train_10000_tibetan_syllable_aware.txt"
    make_corpus(train_rows, raw_corpus, False)
    make_corpus(train_rows, syllable_corpus, True)

    cs336: dict[str, dict[str, BPEtokenizer]] = {"raw": {}, "syllable_aware": {}, "mark_aware": {}}
    for vocab_size in args.vocab_sizes:
        for variant, corpus in [
            ("raw", raw_corpus),
            ("syllable_aware", syllable_corpus),
            ("mark_aware", raw_corpus),
        ]:
            out = root / "outputs" / "cs336-bpe-fertility" / variant / f"vocab_{vocab_size}"
            cs336[variant][str(vocab_size)] = build_cs336_tokenizer(corpus, vocab_size, out, variant)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "train_sample_size": args.train_sample,
        "datasets": {},
        "tokenizers": {
            "nllb": {"id": str(nllb_kwargs["pretrained_model_name_or_path"]), "class": nllb.__class__.__name__},
            "hy_mt2": {"id": str(hy_dir), "class": hy.__class__.__name__, "vocab_size": hy.vocab_size},
            "cs336": {"variants": list(cs336), "vocab_sizes": args.vocab_sizes},
        },
    }

    for name, pairs in datasets.items():
        sources = [pair[0] for pair in pairs]
        targets = [pair[1] for pair in pairs]
        target_lang = "zho_Hans" if name.startswith("flores_") else "zho_Hant"
        dataset_report: dict[str, Any] = {"pairs": len(pairs), "models": {}}
        for model_name, tokenizer in [("nllb", nllb), ("hy_mt2", hy)]:
            src_lengths, src_unknowns, _ = encode_lengths(tokenizer, sources, nllb_lang="bod_Tibt" if model_name == "nllb" else None)
            tgt_lengths, tgt_unknowns, _ = encode_lengths(
                tokenizer,
                targets,
                nllb_lang=target_lang if model_name == "nllb" else None,
                target=model_name == "nllb",
            )
            dataset_report["models"][model_name] = {
                "source": metric_summary(sources, src_lengths, src_unknowns),
                "target": metric_summary(targets, tgt_lengths, tgt_unknowns),
            }
        cs_report: dict[str, Any] = {}
        for variant, tokenizers in cs336.items():
            cs_report[variant] = {}
            for vocab_size, tokenizer in tokenizers.items():
                lengths, unknowns, _ = bpe_lengths(tokenizer, sources, syllable_aware=variant == "syllable_aware")
                cs_report[variant][vocab_size] = {"source": metric_summary(sources, lengths, unknowns)}
        dataset_report["models"]["cs336"] = cs_report
        dataset_report["normalization_source_nllb"] = normalization_summary(nllb, sources, "bod_Tibt")
        report["datasets"][name] = dataset_report

    output = root / "reports" / "fertility_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for name, dataset_report in report["datasets"].items():
        nllb_src = dataset_report["models"]["nllb"]["source"]["fertility_tokens_per_tibetan_syllable"]["mean"]
        hy_src = dataset_report["models"]["hy_mt2"]["source"]["fertility_tokens_per_tibetan_syllable"]["mean"]
        print(f"{name}: pairs={dataset_report['pairs']} NLLB_toks/syll={nllb_src} HY_toks/syll={hy_src}")


if __name__ == "__main__":
    main()
