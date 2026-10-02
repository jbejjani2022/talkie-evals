"""Small explicit LoRA implementation for Talkie's linear projections.

Adapter tensors are FP32; original weights retain their loaded dtype. No
embedding/head updates or model-sized checkpoint copies are made.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

TARGETS = ('attn_query', 'attn_key', 'attn_value', 'attn_resid',
           'mlp_gate', 'mlp_linear', 'mlp_resid')


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError('rank and alpha must be positive')
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,
                                              device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                              device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.base(x)
        update = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
        return original + (update * self.scale).to(original.dtype)


def inject_lora(model: nn.Module, rank: int, alpha: float) -> list[str]:
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('Model already has LoRA adapters')
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if name.split('.')[-1] in TARGETS and isinstance(mod, nn.Linear)]
    if not candidates:
        raise ValueError('No Talkie projection modules found')
    model.requires_grad_(False)
    for name, module in candidates:
        parent, _, key = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, key,
                LoRALinear(module, rank, alpha))
    return [name for name, _ in candidates]


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.endswith(('.lora_A', '.lora_B'))}


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = {n: p for n, p in model.named_parameters()
              if n.endswith(('.lora_A', '.lora_B'))}
    if params.keys() != state.keys():
        raise ValueError('Adapter parameter names do not match')
    with torch.no_grad():
        for name, p in params.items():
            if p.shape != state[name].shape or not torch.isfinite(state[name]).all():
                raise ValueError(f'Invalid adapter tensor: {name}')
            p.copy_(state[name])


def collate_rows(rows: list[dict[str, Any]], pad_id: int, device: str = 'cpu') -> dict[str, torch.Tensor]:
    width = max(len(r['input_ids']) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        n = len(row['input_ids'])
        ids[i, :n] = torch.tensor(row['input_ids'], device=device)
        attention[i, :n] = 1
        mask = torch.tensor(row['assistant_masks'], dtype=torch.bool, device=device)
        labels[i, :n] = torch.where(mask, ids[i, :n], -100)
    return {'input_ids': ids, 'attention_mask': attention, 'labels': labels}


def batch_indices(rows: list[dict[str, Any]], max_padded_tokens: int = 4096,
                  window: int = 128) -> list[list[int]]:
    """Length-sort small shuffled windows without dropping or repeating rows."""
    batches = []
    for start in range(0, len(rows), window):
        ordered = sorted(range(start, min(start + window, len(rows))),
                         key=lambda i: len(rows[i]['input_ids']))
        current: list[int] = []
        width = 0
        for i in ordered:
            length = len(rows[i]['input_ids'])
            if length > max_padded_tokens:
                raise ValueError('A row exceeds the padded-token budget')
            if current and max(width, length) * (len(current) + 1) > max_padded_tokens:
                batches.append(current)
                current, width = [], 0
            current.append(i)
            width = max(width, length)
        if current:
            batches.append(current)
    assert sorted(i for b in batches for i in b) == list(range(len(rows)))
    return batches
