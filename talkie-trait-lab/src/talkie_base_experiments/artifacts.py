"""Download and conversion operations for official Talkie checkpoints."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict, snapshot_download
from safetensors import safe_open

from .configuration_talkie import TalkieConfig
from .registry import ModelSpec
from .tokenization_talkie import TalkieTokenizer

VOCAB_FILENAME = "vocab.txt"
REMOTE_CODE_FILES = (
    "configuration_talkie.py",
    "modeling_talkie.py",
    "tokenization_talkie.py",
)

VLLM_AUTO_MAP = {
    "AutoConfig": "configuration_talkie.TalkieConfig",
    "AutoModel": "modeling_talkie.TalkieModel",
    "AutoModelForCausalLM": "modeling_talkie.TalkieForCausalLM",
    "AutoTokenizer": ["tokenization_talkie.TalkieTokenizer", None],
}


def download_original(spec: ModelSpec, destination_root: Path) -> Path:
    """Download only the official raw checkpoint and vocabulary."""
    destination = destination_root / spec.name
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=destination,
        allow_patterns=[spec.checkpoint_filename, VOCAB_FILENAME, "README.md"],
    )
    return destination


def extract_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Read supported upstream checkpoint envelopes and remove compile prefixes."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"No Talkie state dict found in {checkpoint_path}")
    normalized = {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    if "embed.weight" not in normalized:
        raise ValueError(
            f"No Talkie state dict found in {checkpoint_path}; "
            f"state keys: {list(state_dict)[:20]}"
        )
    return normalized


def convert_original(
    spec: ModelSpec,
    source_dir: Path,
    destination_root: Path,
    max_shard_size: str = "5GB",
) -> Path:
    """Convert an official checkpoint to a self-contained Transformers repo."""
    checkpoint_path = source_dir / spec.checkpoint_filename
    vocab_path = source_dir / VOCAB_FILENAME
    for required in (checkpoint_path, vocab_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing source artifact: {required}")

    state_dict = extract_state_dict(checkpoint_path)
    vocab_size = int(state_dict["embed.weight"].shape[0])
    config = TalkieConfig(vocab_size=vocab_size)
    _validate_state_dict(config, state_dict)

    # Upstream inference converts every parameter to bfloat16 before execution.
    # Persist that exact representation while leaving integer tensors unchanged.
    converted = {
        key: (tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor)
        for key, tensor in state_dict.items()
    }
    # vLLM puts the causal head outside AutoModel and maps this standard alias to it.
    converted["model.lm_head.weight"] = converted["lm_head"].clone()

    destination = destination_root / spec.name
    destination.mkdir(parents=True, exist_ok=True)
    configure_vllm_metadata(config, state_dict["lm_head_gain.w_g"])
    config.source_repo = spec.repo_id
    config.source_checkpoint = spec.checkpoint_filename
    config.source_revision = spec.revision
    config.torch_dtype = "bfloat16"
    config.save_pretrained(destination)

    tokenizer = TalkieTokenizer(str(vocab_path))
    tokenizer.save_pretrained(destination)
    package_dir = Path(__file__).resolve().parent
    for filename in REMOTE_CODE_FILES:
        shutil.copy2(package_dir / filename, destination / filename)

    save_torch_state_dict(
        converted,
        destination,
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    (destination / "conversion.json").write_text(
        json.dumps(
            {
                "source_repo": spec.repo_id,
                "source_checkpoint": spec.checkpoint_filename,
                "source_revision": spec.revision,
                "dtype": "bfloat16",
                "converter": "talkie-base-experiments",
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def configure_vllm_metadata(
    config: TalkieConfig,
    lm_head_gain: torch.Tensor,
) -> None:
    """Preserve the semantics vLLM's external causal head needs."""
    config.architectures = ["TalkieForCausalLM"]
    config.auto_map = dict(VLLM_AUTO_MAP)
    config.logit_scale = float(lm_head_gain.detach().float().cpu().item())


def read_lm_head_gain(model: Path) -> float:
    """Read and validate Talkie's scalar output-head gain from safetensors."""
    key = "lm_head_gain.w_g"
    index_path = model / "model.safetensors.index.json"
    single_path = model / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        try:
            weights_path = model / index["weight_map"][key]
        except KeyError as error:
            raise ValueError(f"{index_path} does not map {key}") from error
    elif single_path.is_file():
        weights_path = single_path
    else:
        raise FileNotFoundError(
            f"Expected {index_path.name} or {single_path.name} under {model}"
        )
    with safe_open(weights_path, framework="pt", device="cpu") as tensors:
        if key not in tensors.keys():  # noqa: SIM118 -- safe_open is not iterable
            raise ValueError(f"{weights_path} does not contain {key}")
        gain = tensors.get_tensor(key)
    if gain.numel() != 1:
        raise ValueError(f"{key} must contain one scalar, got shape {tuple(gain.shape)}")
    value = float(gain.float().item())
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def configure_vllm_directory_metadata(model: Path) -> float | None:
    """Atomically add vLLM metadata to a local Talkie model directory."""
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "talkie":
        return None
    gain = read_lm_head_gain(model)
    config["architectures"] = ["TalkieForCausalLM"]
    config["auto_map"] = dict(VLLM_AUTO_MAP)
    config["logit_scale"] = gain
    temporary = config_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, config_path)
    return gain


def validate_vllm_directory_metadata(model: Path) -> float | None:
    """Fail closed when local Talkie metadata changes its output semantics."""
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "talkie":
        return None
    auto_map = config.get("auto_map", {})
    if auto_map.get("AutoModel") != VLLM_AUTO_MAP["AutoModel"]:
        raise ValueError(f"Talkie vLLM model lacks validated AutoModel metadata: {model}")
    if "logit_scale" not in config:
        raise ValueError(f"Talkie vLLM model lacks logit_scale metadata: {model}")
    expected = read_lm_head_gain(model)
    actual = float(config["logit_scale"])
    if actual != expected:
        raise ValueError(
            f"Talkie logit_scale mismatch under {model}: config={actual}, weights={expected}"
        )
    return actual


def _validate_state_dict(config: TalkieConfig, state_dict: dict[str, torch.Tensor]) -> None:
    from .modeling_talkie import TalkieForCausalLM

    with torch.device("meta"):
        expected = TalkieForCausalLM(config).state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    shape_mismatches = sorted(
        key
        for key in set(expected) & set(state_dict)
        if expected[key].shape != state_dict[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "Checkpoint does not exactly match the Talkie architecture: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )
