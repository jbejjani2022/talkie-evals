from __future__ import annotations
from dataclasses import dataclass
import torch
@dataclass
class SequenceScore:
    log_probability: float
    token_count: int
    byte_count: int
    token_ids: list[int]
    token_log_probabilities: list[float]

class TransformersBackend:

    def loglikelihood(self, prefixes: list[str], continuations: list[str]) -> list[SequenceScore]:
        if len(prefixes) != len(continuations):
            raise ValueError('prefixes and continuations must have equal length')
        results: list[SequenceScore] = []
        device = next(self.model.parameters()).device
        for start in range(0, len(prefixes), self.config.batch_size):
            prefix_batch = prefixes[start:start + self.config.batch_size]
            continuation_batch = continuations[start:start + self.config.batch_size]
            sequences: list[list[int]] = []
            target_ids: list[list[int]] = []
            prefix_lengths: list[int] = []
            for prefix, continuation in zip(prefix_batch, continuation_batch, strict=True):
                prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                if not prefix_ids:
                    if self.tokenizer.eos_token_id is None:
                        raise ValueError('An empty likelihood prefix requires an EOS token')
                    prefix_ids = [self.tokenizer.eos_token_id]
                continuation_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
                if not continuation_ids:
                    raise ValueError('Likelihood continuations cannot be empty')
                sequence = prefix_ids + continuation_ids
                if self.config.max_model_len is not None and len(sequence) > self.config.max_model_len:
                    raise ValueError(f'Likelihood sequence has {len(sequence)} tokens, exceeding max_model_len={self.config.max_model_len}')
                sequences.append(sequence)
                target_ids.append(continuation_ids)
                prefix_lengths.append(len(prefix_ids))
            inputs = self.tokenizer.pad({'input_ids': sequences}, return_tensors='pt').to(device)
            with torch.inference_mode():
                logits = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False).logits
            padded_length = inputs.input_ids.shape[1]
            for row, (sequence, targets, prefix_length, continuation) in enumerate(zip(sequences, target_ids, prefix_lengths, continuation_batch, strict=True)):
                padding = padded_length - len(sequence)
                token_log_probabilities: list[float] = []
                for offset, token_id in enumerate(targets):
                    token_position = padding + prefix_length + offset
                    token_logits = logits[row, token_position - 1]
                    token_log_probabilities.append(float(torch.log_softmax(token_logits, dim=-1)[token_id].item()))
                results.append(SequenceScore(log_probability=sum(token_log_probabilities), token_count=len(targets), byte_count=len(continuation.encode('utf-8')), token_ids=list(targets), token_log_probabilities=token_log_probabilities))
        return results
