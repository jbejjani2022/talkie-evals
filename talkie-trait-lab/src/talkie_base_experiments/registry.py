"""Authoritative metadata for the two original Talkie base checkpoints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    checkpoint_filename: str
    revision: str


MODELS = {
    "talkie-1930-13b-base": ModelSpec(
        "talkie-1930-13b-base",
        "talkie-lm/talkie-1930-13b-base",
        "final.ckpt",
        "b7c97680791f7fca4262c3c80b36ff7d666faab0",
    ),
    "talkie-web-13b-base": ModelSpec(
        "talkie-web-13b-base",
        "talkie-lm/talkie-web-13b-base",
        "base.ckpt",
        "1e5b771c9d38d44f54d35e722c5c0d73da418dd8",
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError as error:
        choices = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model {name!r}; choose one of: {choices}") from error


def model_names(value: str) -> list[str]:
    if value == "all":
        return list(MODELS)
    get_model_spec(value)
    return [value]
