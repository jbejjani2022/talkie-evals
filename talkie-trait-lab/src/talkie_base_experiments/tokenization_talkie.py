"""Transformers tokenizer wrapper preserving Talkie's original tiktoken IDs."""

import re
import shutil
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import PreTrainedTokenizer

PATTERN = "|".join([
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
])
VOCAB_SIZE = 65536
END_OF_TEXT = "<|endoftext|>"
TOKEN_RE = re.compile(r"^<\|talkie:(\d+)\|>$")


class TalkieTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, **kwargs):
        self.vocab_file = str(vocab_file)
        ranks = load_tiktoken_bpe(self.vocab_file)
        ranks = {token: rank for token, rank in ranks.items() if rank < VOCAB_SIZE - 1}
        self._encoding = tiktoken.Encoding(
            name=f"talkie-{id(self)}", pat_str=PATTERN, mergeable_ranks=ranks,
            special_tokens={END_OF_TEXT: VOCAB_SIZE - 1},
        )
        kwargs.setdefault("eos_token", END_OF_TEXT)
        kwargs.setdefault("bos_token", None)
        kwargs.setdefault("pad_token", None)
        kwargs.setdefault("model_max_length", 4096)
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def get_vocab(self) -> dict[str, int]:
        vocab = {self._token_name(i): i for i in range(VOCAB_SIZE - 1)}
        vocab[END_OF_TEXT] = VOCAB_SIZE - 1
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return [self._token_name(i) for i in self._encoding.encode_ordinary(text)]

    def _convert_token_to_id(self, token: str) -> int:
        if token == END_OF_TEXT:
            return VOCAB_SIZE - 1
        match = TOKEN_RE.match(token)
        return int(match.group(1)) if match else 0

    def _convert_id_to_token(self, index: int) -> str:
        return END_OF_TEXT if index == VOCAB_SIZE - 1 else self._token_name(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._encoding.decode([self._convert_token_to_id(t) for t in tokens], errors="replace")

    def _decode(self, token_ids: list[int], skip_special_tokens: bool = False, **kwargs) -> str:
        # Added chat tokens are outside tiktoken's base vocabulary. Decode ordinary
        # runs with tiktoken and splice added tokens back without inserting spaces.
        pieces: list[str] = []
        ordinary_ids: list[int] = []
        special_ids = set(self.all_special_ids)

        def flush_ordinary() -> None:
            if ordinary_ids:
                pieces.append(self._encoding.decode(ordinary_ids, errors="replace"))
                ordinary_ids.clear()

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            added_token = self.added_tokens_decoder.get(token_id)
            if added_token is None:
                ordinary_ids.append(token_id)
            else:
                flush_ordinary()
                pieces.append(str(added_token))
        flush_ordinary()
        return "".join(pieces)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        destination = Path(save_directory) / (
            f"{filename_prefix}-vocab.txt" if filename_prefix else "vocab.txt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if Path(self.vocab_file).resolve() != destination.resolve():
            shutil.copyfile(self.vocab_file, destination)
        return (str(destination),)

    @staticmethod
    def _token_name(index: int) -> str:
        return f"<|talkie:{index}|>"


TalkieTokenizer.register_for_auto_class("AutoTokenizer")
