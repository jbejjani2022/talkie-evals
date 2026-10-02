#!/usr/bin/env python
"""Create a deterministic held-out SFT split and conditional-BPB examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from talkie_base_experiments.training.common import ROLE_TOKENS

ROLE_PREFIX = dict(zip(("system", "user", "assistant"), ROLE_TOKENS, strict=True))


def _conditional_examples(dataset: Dataset, eos_token: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for conversation_index, row in enumerate(dataset):
        pieces: list[str] = []
        for turn_index, message in enumerate(row["messages"]):
            role = message["role"]
            if role not in ROLE_PREFIX:
                continue
            pieces.extend((ROLE_PREFIX[role], "\n"))
            content = str(message["content"])
            if role == "assistant" and content:
                examples.append(
                    {
                        "prompt": "".join(pieces),
                        "continuation": content,
                        "conversation_index": conversation_index,
                        "turn_index": turn_index,
                        "source": row.get("source"),
                    }
                )
            pieces.append(content)
            if role == "assistant":
                pieces.append(eos_token)
            pieces.append("\n")
    return examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--train-shards", type=int, default=6)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    files = sorted(args.input.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train-*.parquet files found in {args.input}")
    if args.validation_samples <= 0 or args.train_shards <= 0:
        raise ValueError("validation-samples and train-shards must be positive")

    dataset = load_dataset("parquet", data_files=[str(path) for path in files], split="train")
    if args.validation_samples >= len(dataset):
        raise ValueError("validation-samples must be smaller than the dataset")
    split = dataset.train_test_split(
        test_size=args.validation_samples,
        seed=args.seed,
        shuffle=True,
    )
    train = split["train"]
    validation = split["test"]

    args.output.mkdir(parents=True)
    for index in range(args.train_shards):
        shard = train.shard(args.train_shards, index, contiguous=True)
        shard.to_parquet(args.output / f"train-{index:05d}-of-{args.train_shards:05d}.parquet")
    validation_path = args.output / "validation.parquet"
    validation.to_parquet(validation_path)
    conditional = Dataset.from_list(_conditional_examples(validation, args.eos_token))
    conditional_path = args.output / "validation-assistant-turns.parquet"
    conditional.to_parquet(conditional_path)

    input_manifest = args.input / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "input_manifest_sha256": _sha256(input_manifest) if input_manifest.is_file() else None,
        "rule": "datasets.Dataset.train_test_split with a fixed integer test size",
        "seed": args.seed,
        "rows": {
            "input": len(dataset),
            "train": len(train),
            "validation_conversations": len(validation),
            "validation_assistant_turns": len(conditional),
        },
        "fingerprints": {
            "input": dataset._fingerprint,
            "train": train._fingerprint,
            "validation": validation._fingerprint,
            "validation_assistant_turns": conditional._fingerprint,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(args.output.glob("*.parquet"))
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["rows"], indent=2))


if __name__ == "__main__":
    main()
