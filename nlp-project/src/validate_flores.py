"""Low-memory validation for aligned FLORES Tibetan and Chinese files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import duckdb


REVISION = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"
EXPECTED_ROWS = {"dev": 997, "devtest": 1012}
LANGUAGES = {"tibetan": "bod_Tibt", "chinese": "zho_Hans"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def duplicate_summary(connection: duckdb.DuckDBPyConnection, view: str) -> Dict[str, int]:
    repeated, extras = connection.execute(
        f"""
        SELECT count(*), coalesce(sum(n - 1), 0)
        FROM (
            SELECT sentence, count(*) AS n
            FROM {view}
            GROUP BY sentence
            HAVING count(*) > 1
        )
        """
    ).fetchone()
    return {
        "distinct_values_repeated": int(repeated),
        "extra_occurrences": int(extras),
    }


def pair_duplicate_summary(connection: duckdb.DuckDBPyConnection) -> Dict[str, int]:
    repeated, extras = connection.execute(
        """
        SELECT count(*), coalesce(sum(n - 1), 0)
        FROM (
            SELECT b.sentence AS tibetan, z.sentence AS chinese, count(*) AS n
            FROM bo_all AS b
            JOIN zh_all AS z ON b.split = z.split AND b.id = z.id
            GROUP BY b.sentence, z.sentence
            HAVING count(*) > 1
        )
        """
    ).fetchone()
    return {
        "distinct_values_repeated": int(repeated),
        "extra_occurrences": int(extras),
    }


def unicode_profile(sentences: Iterable[str], language: str) -> Dict[str, Any]:
    sentence_count = 0
    total_chars = 0
    nonspace_chars = 0
    target_chars = 0
    nfc_changed = 0
    replacement_chars = 0
    zero_width_chars = 0
    tsheg = 0
    nonbreaking_tsheg = 0

    for sentence in sentences:
        sentence = sentence or ""
        sentence_count += 1
        total_chars += len(sentence)
        nfc_changed += unicodedata.normalize("NFC", sentence) != sentence
        replacement_chars += sentence.count("\uFFFD")
        zero_width_chars += sum(sentence.count(char) for char in ("\u200B", "\u200C", "\u200D", "\uFEFF"))
        for char in sentence:
            if char.isspace():
                continue
            nonspace_chars += 1
            codepoint = ord(char)
            if language == "tibetan":
                target_chars += 0x0F00 <= codepoint <= 0x0FFF
            else:
                target_chars += (
                    0x3400 <= codepoint <= 0x4DBF
                    or 0x4E00 <= codepoint <= 0x9FFF
                    or 0x20000 <= codepoint <= 0x2EBEF
                    or 0x30000 <= codepoint <= 0x323AF
                )
        if language == "tibetan":
            tsheg += sentence.count("\u0F0B")
            nonbreaking_tsheg += sentence.count("\u0F0C")

    profile: Dict[str, Any] = {
        "sentences": sentence_count,
        "total_characters": total_chars,
        "nonspace_characters": nonspace_chars,
        "target_script_characters": target_chars,
        "target_script_ratio_of_nonspace": round(target_chars / nonspace_chars, 6)
        if nonspace_chars
        else 0.0,
        "sentences_changed_by_nfc": int(nfc_changed),
        "replacement_characters": replacement_chars,
        "zero_width_characters": zero_width_chars,
    }
    if language == "tibetan":
        profile.update({"tsheg_u_0f0b": tsheg, "nonbreaking_tsheg_u_0f0c": nonbreaking_tsheg})
    return profile


def stream_sentences(
    connection: duckdb.DuckDBPyConnection, view: str, batch_size: int = 128
) -> Iterable[str]:
    cursor = connection.execute(f"SELECT sentence FROM {view}")
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            return
        for (sentence,) in batch:
            yield sentence


def evenly_spaced_indices(row_count: int, sample_count: int = 10) -> List[int]:
    return sorted({int((index + 0.5) * row_count / sample_count) for index in range(sample_count)})


def configure_connection(
    project_root: Path, memory_limit: str, threads: int
) -> duckdb.DuckDBPyConnection:
    if not re.fullmatch(r"[1-9][0-9]*(MB|GB)", memory_limit.upper()):
        raise ValueError("memory_limit must look like 256MB or 1GB")
    if not 1 <= threads <= 4:
        raise ValueError("threads must be between 1 and 4 for this CPU workflow")

    temp_dir = project_root / "data" / "interim" / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{memory_limit.upper()}'")
    connection.execute(f"SET threads={threads}")
    escaped_temp_dir = str(temp_dir).replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped_temp_dir}'")
    return connection


def register_views(connection: duckdb.DuckDBPyConnection, data_root: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for label, config in LANGUAGES.items():
        prefix = "bo" if label == "tibetan" else "zh"
        for split in EXPECTED_ROWS:
            path = data_root / config / f"{split}-00000-of-00001.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            connection.read_parquet(str(path)).create_view(f"{prefix}_{split}")
            files[f"{config}_{split}"] = path

    connection.execute(
        "CREATE TEMP VIEW bo_all AS "
        "SELECT 'dev' AS split, * FROM bo_dev UNION ALL "
        "SELECT 'devtest' AS split, * FROM bo_devtest"
    )
    connection.execute(
        "CREATE TEMP VIEW zh_all AS "
        "SELECT 'dev' AS split, * FROM zh_dev UNION ALL "
        "SELECT 'devtest' AS split, * FROM zh_devtest"
    )
    return files


def split_check(connection: duckdb.DuckDBPyConnection, split: str) -> Dict[str, int]:
    row = connection.execute(
        f"""
        SELECT
            (SELECT count(*) FROM bo_{split}) AS tibetan_rows,
            (SELECT count(*) FROM zh_{split}) AS chinese_rows,
            (SELECT count(DISTINCT id) FROM bo_{split}) AS tibetan_unique_ids,
            (SELECT count(DISTINCT id) FROM zh_{split}) AS chinese_unique_ids,
            count(*) FILTER (WHERE b.id IS NULL) AS missing_tibetan_ids,
            count(*) FILTER (WHERE z.id IS NULL) AS missing_chinese_ids,
            count(*) FILTER (
                WHERE b.id IS NOT NULL AND z.id IS NOT NULL AND b.URL IS DISTINCT FROM z.URL
            ) AS url_mismatch,
            count(*) FILTER (
                WHERE b.id IS NOT NULL AND z.id IS NOT NULL AND b.domain IS DISTINCT FROM z.domain
            ) AS domain_mismatch,
            count(*) FILTER (
                WHERE b.id IS NOT NULL AND z.id IS NOT NULL AND b.topic IS DISTINCT FROM z.topic
            ) AS topic_mismatch,
            count(*) FILTER (
                WHERE b.id IS NOT NULL AND z.id IS NOT NULL
                  AND b.has_image IS DISTINCT FROM z.has_image
            ) AS has_image_mismatch,
            count(*) FILTER (
                WHERE b.id IS NOT NULL AND z.id IS NOT NULL
                  AND b.has_hyperlink IS DISTINCT FROM z.has_hyperlink
            ) AS has_hyperlink_mismatch,
            count(*) FILTER (WHERE b.sentence IS NULL OR trim(b.sentence) = '') AS blank_tibetan,
            count(*) FILTER (WHERE z.sentence IS NULL OR trim(z.sentence) = '') AS blank_chinese
        FROM bo_{split} AS b
        FULL OUTER JOIN zh_{split} AS z ON b.id = z.id
        """
    ).fetchone()
    names = [description[0] for description in connection.description]
    return {name: int(value) for name, value in zip(names, row)}


def exact_cross_split_overlap(connection: duckdb.DuckDBPyConnection) -> Dict[str, int]:
    tibetan = connection.execute(
        "SELECT count(*) FROM (SELECT sentence FROM bo_dev INTERSECT SELECT sentence FROM bo_devtest)"
    ).fetchone()[0]
    chinese = connection.execute(
        "SELECT count(*) FROM (SELECT sentence FROM zh_dev INTERSECT SELECT sentence FROM zh_devtest)"
    ).fetchone()[0]
    pairs = connection.execute(
        """
        SELECT count(*) FROM (
            SELECT b.sentence AS tibetan, z.sentence AS chinese
            FROM bo_dev AS b JOIN zh_dev AS z USING (id)
            INTERSECT
            SELECT b.sentence AS tibetan, z.sentence AS chinese
            FROM bo_devtest AS b JOIN zh_devtest AS z USING (id)
        )
        """
    ).fetchone()[0]
    return {"tibetan": int(tibetan), "chinese": int(chinese), "aligned_pair": int(pairs)}


def distribution(
    connection: duckdb.DuckDBPyConnection, column: str, limit: int | None = None
) -> List[Dict[str, Any]]:
    if column not in {"domain", "topic"}:
        raise ValueError(column)
    query = f"SELECT {column}, count(*) AS n FROM bo_all GROUP BY {column} ORDER BY n DESC, {column}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return [{"name": name, "count": int(count)} for name, count in connection.execute(query).fetchall()]


def manual_sample(connection: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, count in EXPECTED_ROWS.items():
        indices = evenly_spaced_indices(count)
        index_sql = ", ".join(str(index) for index in indices)
        query = f"""
            WITH aligned AS (
                SELECT row_number() OVER (ORDER BY b.id) - 1 AS row_index,
                       b.id, b.domain, b.topic,
                       b.sentence AS tibetan, z.sentence AS chinese
                FROM bo_{split} AS b
                JOIN zh_{split} AS z ON b.id = z.id
            )
            SELECT row_index, id, domain, topic, tibetan, chinese
            FROM aligned
            WHERE row_index IN ({index_sql})
            ORDER BY row_index
        """
        for row_index, row_id, domain, topic, tibetan, chinese in connection.execute(query).fetchall():
            rows.append(
                {
                    "split": split,
                    "row_index": int(row_index),
                    "id": int(row_id),
                    "domain": domain,
                    "topic": topic,
                    "tibetan": tibetan,
                    "chinese": chinese,
                    "alignment_ok": "",
                    "fidelity_1_5": "",
                    "chinese_fluency_1_5": "",
                    "style_match_1_5": "",
                    "notes": "",
                }
            )
    return rows


def validate(
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    memory_limit: str = "256MB",
    threads: int = 2,
) -> Tuple[Dict[str, Any], Path, Path]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = configure_connection(project_root, memory_limit, threads)
    try:
        files = register_views(connection, data_root)
        split_checks = {split: split_check(connection, split) for split in EXPECTED_ROWS}
        critical_fields = (
            "missing_tibetan_ids",
            "missing_chinese_ids",
            "url_mismatch",
            "domain_mismatch",
            "topic_mismatch",
            "has_image_mismatch",
            "has_hyperlink_mismatch",
            "blank_tibetan",
            "blank_chinese",
        )
        passed = all(
            check["tibetan_rows"] == EXPECTED_ROWS[split]
            and check["chinese_rows"] == EXPECTED_ROWS[split]
            and check["tibetan_unique_ids"] == EXPECTED_ROWS[split]
            and check["chinese_unique_ids"] == EXPECTED_ROWS[split]
            and all(check[field] == 0 for field in critical_fields)
            for split, check in split_checks.items()
        )

        summary: Dict[str, Any] = {
            "dataset": "facebook/flores",
            "revision": REVISION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed_critical_alignment_checks": passed,
            "runtime": {
                "duckdb": duckdb.__version__,
                "memory_limit": memory_limit.upper(),
                "threads": threads,
            },
            "files": {
                name: {
                    "path": str(path.relative_to(project_root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for name, path in files.items()
            },
            "split_checks": split_checks,
            "exact_duplicates": {
                "tibetan": duplicate_summary(connection, "bo_all"),
                "chinese": duplicate_summary(connection, "zh_all"),
                "aligned_pair": pair_duplicate_summary(connection),
            },
            "exact_cross_split_overlap": exact_cross_split_overlap(connection),
            "unicode": {
                "tibetan": unicode_profile(stream_sentences(connection, "bo_all"), "tibetan"),
                "chinese": unicode_profile(stream_sentences(connection, "zh_all"), "chinese"),
            },
            "domains": distribution(connection, "domain"),
            "top_topics": distribution(connection, "topic", limit=15),
        }
        sample_rows = manual_sample(connection)
    finally:
        connection.close()

    summary_path = output_dir / "flores200_local_full_check.json"
    sample_path = output_dir / "flores200_manual_check_20.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with sample_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    return summary, summary_path, sample_path


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root / "data" / "raw" / "flores200_20260529",
    )
    parser.add_argument("--output-dir", type=Path, default=default_root / "reports")
    parser.add_argument("--memory-limit", default="256MB")
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, summary_path, sample_path = validate(
        project_root=args.project_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "passed": summary["passed_critical_alignment_checks"],
                "summary": str(summary_path),
                "manual_sample": str(sample_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["passed_critical_alignment_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
