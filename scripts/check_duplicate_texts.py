#!/usr/bin/env python3
"""Detect duplicate samples in JSONL corpora used for pretraining.

This script scans one or more JSONL files (optionally plain-text lines),
normalizes the `text` field (same logic as training pipeline), hashes each
sample, and reports duplicate counts.

Usage example:
    python scripts/check_duplicate_texts.py \
        --paths /path/to/chinanews_pretrain.jsonl \
        --top-k 20 --report-file logs/duplicates.txt

Outputs:
    - Summary on stdout (total samples, unique samples, duplicates).
    - Optional detailed report written to `--report-file`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 为可选依赖
    tqdm = None  # type: ignore[assignment]


EXTRACT_KEYS = ("text", "content", "completion", "response")


def extract_text(record: object) -> Optional[str]:
    if record is None:
        return None
    if isinstance(record, str):
        return record.strip() or None
    if isinstance(record, dict):
        for key in EXTRACT_KEYS:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        iterator: Iterable[str] = handle
        progress = None
        if tqdm is not None:
            progress = tqdm(handle, desc=f"Scanning {path.name}", unit="lines")
            iterator = progress
        for line in iterator:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                record = stripped
            text = extract_text(record)
            if text:
                yield normalize_text(text)
        if progress is not None:
            progress.close()


def analyze_paths(paths: Iterable[Path], limit: Optional[int]) -> Tuple[int, Counter]:
    processed = 0
    counter: Counter = Counter()
    for path in paths:
        for text in iter_jsonl(path):
            counter[hash_text(text)] += 1
            processed += 1
            if limit is not None and processed >= limit:
                return processed, counter
    return processed, counter


def load_sample_texts(paths: Iterable[Path], hashes: Iterable[str], limit: int = 3) -> Dict[str, list[str]]:
    target = set(hashes)
    samples: Dict[str, list[str]] = {h: [] for h in target}
    for path in paths:
        for text in iter_jsonl(path):
            h = hash_text(text)
            if h in target and len(samples[h]) < limit:
                samples[h].append(text)
            if all(len(v) >= limit for v in samples.values()):
                return samples
    return samples


def write_report(report_path: Path, summary: str, details: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(summary)
        handle.write("\n\n")
        handle.write(details)


def format_summary(total: int, counter: Counter) -> str:
    unique = sum(1 for _ in counter)
    duplicates = sum(count - 1 for count in counter.values() if count > 1)
    summary = [
        "Duplicate Detection Summary",
        "===========================",
        f"Total samples scanned : {total}",
        f"Unique samples        : {unique}",
        f"Duplicate instances   : {duplicates}",
        f"Duplicate rate        : {duplicates / total:.6f}" if total else "Duplicate rate        : N/A",
    ]
    return "\n".join(summary)


def format_details(counter: Counter, top_k: int, sample_texts: Dict[str, list[str]]) -> str:
    lines = ["Top duplicate hashes (count, hash, sample text)", "-----------------------------------------------"]
    for hash_value, count in counter.most_common(top_k):
        if count < 2:
            break
        samples = sample_texts.get(hash_value, [])
        preview = samples[0][:120].replace("\n", " ") + ("..." if len(samples[0]) > 120 else "") if samples else "<no sample>"
        lines.append(f"{count:7d} | {hash_value} | {preview}")
        for extra in samples[1:]:
            snippet = extra[:120].replace("\n", " ") + ("..." if len(extra) > 120 else "")
            lines.append(f"         | {'':64} | {snippet}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect duplicate samples in JSONL corpora")
    parser.add_argument("--paths", nargs="+", required=True, help="One or more JSONL files")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples to scan")
    parser.add_argument("--top-k", type=int, default=20, help="Number of duplicate clusters to display")
    parser.add_argument("--report-file", type=str, default=None, help="Optional path to save detailed report")
    parser.add_argument("--sample-per-duplicate", type=int, default=3, help="Saved examples per duplicate hash")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p).expanduser().resolve() for p in args.paths]
    for path in paths:
        if not path.exists():
            print(f"❌ File not found: {path}", file=sys.stderr)
            return

    total, counter = analyze_paths(paths, args.max_samples)
    summary = format_summary(total, counter)
    print(summary)

    duplicates_only = Counter({k: v for k, v in counter.items() if v > 1})
    if not duplicates_only:
        print("✅ No duplicates found (within scanned range).")
        return

    sample_hashes = [hash_value for hash_value, _ in duplicates_only.most_common(args.top_k)]
    sample_texts = load_sample_texts(paths, sample_hashes, limit=args.sample_per_duplicate)
    details = format_details(duplicates_only, args.top_k, sample_texts)
    print("\n" + details)

    if args.report_file:
        write_report(Path(args.report_file), summary, details)
        print(f"\n📝 Detailed report saved to {args.report_file}")


if __name__ == "__main__":
    main()


"""

python scripts/check_duplicate_texts.py \
    --paths /home/modelenv/chentianxuan/projects/open_source_data_process/data/chinanews_pretrain.jsonl \
    --max-samples 200000 --top-k 20 \
    --report-file logs/duplicate_report.txt
"""