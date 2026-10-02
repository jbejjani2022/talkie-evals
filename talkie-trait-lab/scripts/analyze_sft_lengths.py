#!/usr/bin/env python
"""Measure rendered SFT lengths under both Talkie tokenizers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from talkie_base_experiments.tokenization_talkie import TalkieTokenizer
from talkie_base_experiments.training.common import ATOMIC_CHAT_TEMPLATE, ROLE_TOKENS


def _load_tokenizer(model_path: str) -> TalkieTokenizer:
    tokenizer = TalkieTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": list(ROLE_TOKENS)})
    tokenizer.chat_template = ATOMIC_CHAT_TEMPLATE
    return tokenizer


def _render(messages: list[dict[str, str]], eos_token: str) -> str:
    role_tokens = dict(zip(("system", "user", "assistant"), ROLE_TOKENS, strict=True))
    pieces: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in role_tokens:
            continue
        pieces.extend((role_tokens[role], "\n", str(message["content"])))
        if role == "assistant":
            pieces.append(eos_token)
        pieces.append("\n")
    return "".join(pieces)


def _rendered_length(messages: list[dict[str, str]], tokenizer: TalkieTokenizer) -> int:
    """Count the exact atomic rendering through tiktoken's native encoder."""
    length = 0
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            continue
        length += 1  # Atomic role token.
        content = "\n" + str(message["content"])
        if role == "assistant":
            length += len(tokenizer._encoding.encode_ordinary(content))
            length += 1  # Existing atomic EOS.
            length += len(tokenizer._encoding.encode_ordinary("\n"))
        else:
            length += len(tokenizer._encoding.encode_ordinary(content + "\n"))
    return length


def _analyze_file(args: tuple[str, str, str, int]) -> dict[str, Any]:
    parquet_path, first_model, second_model, limit = args
    first = _load_tokenizer(first_model)
    second = _load_tokenizer(second_model)
    length_pairs: list[tuple[int, int]] = []
    by_source: dict[str, Counter[str]] = defaultdict(Counter)

    for batch in pq.ParquetFile(parquet_path).iter_batches(columns=["messages", "source"]):
        rows = batch.to_pylist()
        for row in rows:
            first_length = _rendered_length(row["messages"], first)
            second_length = _rendered_length(row["messages"], second)
            length_pairs.append((first_length, second_length))
            source = row["source"]
            counts = by_source[source]
            counts["total"] += 1
            counts["first_over"] += first_length > limit
            counts["second_over"] += second_length > limit
            counts["either_over"] += max(first_length, second_length) > limit
            counts["both_over"] += min(first_length, second_length) > limit

    return {
        "length_pairs": length_pairs,
        "by_source": {source: dict(counts) for source, counts in by_source.items()},
    }


def _percentile(sorted_values: list[int], percentile: float) -> int:
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--first-model", required=True)
    parser.add_argument("--second-model", required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.data.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {args.data}")
    tasks = [(str(path), args.first_model, args.second_model, args.limit) for path in files]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        parts = list(executor.map(_analyze_file, tasks))

    pairs = [pair for part in parts for pair in part["length_pairs"]]
    first_lengths = sorted(pair[0] for pair in pairs)
    second_lengths = sorted(pair[1] for pair in pairs)
    total = len(pairs)
    first_only = sum(a > args.limit and b <= args.limit for a, b in pairs)
    second_only = sum(a <= args.limit and b > args.limit for a, b in pairs)
    both_over = sum(a > args.limit and b > args.limit for a, b in pairs)
    either_over = first_only + second_only + both_over

    combined_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for part in parts:
        for source, counts in part["by_source"].items():
            combined_sources[source].update(counts)

    def count_and_percent(count: int) -> dict[str, float | int]:
        return {"count": count, "percent": 100 * count / total}

    report = {
        "examples": total,
        "context_limit": args.limit,
        "models": {
            args.first_name: {
                "path": args.first_model,
                "percentiles": {
                    f"p{int(p * 1000) / 10:g}": _percentile(first_lengths, p)
                    for p in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
                },
                "over_limit": count_and_percent(sum(x > args.limit for x in first_lengths)),
            },
            args.second_name: {
                "path": args.second_model,
                "percentiles": {
                    f"p{int(p * 1000) / 10:g}": _percentile(second_lengths, p)
                    for p in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
                },
                "over_limit": count_and_percent(sum(x > args.limit for x in second_lengths)),
            },
        },
        "joint": {
            "fits_both": count_and_percent(total - either_over),
            f"only_{args.first_name}_over": count_and_percent(first_only),
            f"only_{args.second_name}_over": count_and_percent(second_only),
            "both_over": count_and_percent(both_over),
            "either_over": count_and_percent(either_over),
        },
        "length_difference": {
            f"{args.first_name}_minus_{args.second_name}_mean": sum(a - b for a, b in pairs) / total,
            "max_absolute": max(abs(a - b) for a, b in pairs),
        },
        "sources": {
            source: {
                **dict(counts),
                "either_over_percent": 100 * counts["either_over"] / counts["total"],
            }
            for source, counts in sorted(combined_sources.items())
        },
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
