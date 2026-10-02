#!/usr/bin/env python
"""Create one SFT snapshot containing only rows that fit both tokenizers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from analyze_sft_lengths import _load_tokenizer, _rendered_length
from talkie_base_experiments.training.common import ROLE_TOKEN_IDS, ROLE_TOKENS


def _filter_file(args: tuple[str, str, str, str, int]) -> dict[str, Any]:
    input_path, output_path, first_model, second_model, limit = args
    first = _load_tokenizer(first_model)
    second = _load_tokenizer(second_model)
    for tokenizer in (first, second):
        actual = tuple(tokenizer.convert_tokens_to_ids(token) for token in ROLE_TOKENS)
        if actual != ROLE_TOKEN_IDS:
            raise ValueError(f"Atomic role-token IDs are {actual}, expected {ROLE_TOKEN_IDS}")

    parquet = pq.ParquetFile(input_path)
    writer = pq.ParquetWriter(output_path, parquet.schema_arrow, compression="zstd")
    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        for batch in parquet.iter_batches():
            rows = batch.to_pylist()
            keep: list[bool] = []
            for row in rows:
                first_length = _rendered_length(row["messages"], first)
                second_length = _rendered_length(row["messages"], second)
                fits = first_length <= limit and second_length <= limit
                keep.append(fits)
                counts["input"] += 1
                counts["kept"] += fits
                counts["dropped"] += not fits
                source_counts = by_source[row["source"]]
                source_counts["input"] += 1
                source_counts["kept"] += fits
                source_counts["dropped"] += not fits
            table = pa.Table.from_batches([batch]).filter(pa.array(keep))
            if len(table):
                writer.write_table(table)
    finally:
        writer.close()
    return {
        "input_file": input_path,
        "output_file": output_path,
        "counts": dict(counts),
        "sources": {source: dict(values) for source, values in by_source.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-model", required=True)
    parser.add_argument("--second-model", required=True)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    files = sorted(args.input.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {args.input}")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)

    tasks = [
        (str(path), str(args.output / path.name), args.first_model, args.second_model, args.limit)
        for path in files
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        shards = list(executor.map(_filter_file, tasks))

    totals: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    for shard in shards:
        totals.update(shard["counts"])
        for source, counts in shard["sources"].items():
            sources[source].update(counts)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "first_model": args.first_model,
        "second_model": args.second_model,
        "chat_template": "talkie_atomic",
        "role_token_ids": dict(zip(ROLE_TOKENS, ROLE_TOKEN_IDS, strict=True)),
        "context_limit": args.limit,
        "rule": "keep iff rendered length <= context_limit under both tokenizers",
        "counts": dict(totals),
        "sources": {source: dict(counts) for source, counts in sorted(sources.items())},
        "shards": shards,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
