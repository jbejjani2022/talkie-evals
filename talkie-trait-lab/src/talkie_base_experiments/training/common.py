from typing import Any
import torch
from torch import nn
ConfigError = ValueError
ROLE_TOKENS = ('<|system|>', '<|user|>', '<|assistant|>')
ROLE_TOKEN_IDS = (65536, 65537, 65538)
ATOMIC_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
LEGACY_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}\n{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"

def _resize_vocabulary_randomly(model: Any, new_size: int) -> None:
    """Grow input and Talkie's parameter-based output rows with random initialization."""
    input_embeddings = model.get_input_embeddings()
    old_size = input_embeddings.num_embeddings
    if old_size == new_size:
        return
    if old_size > new_size:
        raise ConfigError(f'Refusing to shrink model vocabulary from {old_size} to {new_size}')
    raw_lm_head = getattr(model, 'lm_head', None)
    old_lm_head = raw_lm_head if isinstance(raw_lm_head, nn.Parameter) else None
    model.resize_token_embeddings(new_size, mean_resizing=False)
    if old_lm_head is not None and old_lm_head.shape[0] == old_size:
        values = old_lm_head.new_empty((new_size, *old_lm_head.shape[1:]))
        with torch.no_grad():
            values.normal_(mean=0.0, std=model.config.initializer_range)
            values[:old_size].copy_(old_lm_head)
        model.lm_head = nn.Parameter(values, requires_grad=old_lm_head.requires_grad)
        model.config.vocab_size = new_size

def _tokenize_messages_with_assistant_mask(example: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Render the built-in template and derive exact masks for Talkie's slow tokenizer."""
    pieces: list[tuple[str, bool]] = []
    for message in example['messages']:
        role = message['role']
        content = message['content']
        if tokenizer.chat_template == ATOMIC_CHAT_TEMPLATE:
            role_prefix = {name: token for name, token in zip(('system', 'user', 'assistant'), ROLE_TOKENS, strict=True)}.get(role)
            if role_prefix is None:
                continue
            pieces.append((f'{role_prefix}\n', False))
            pieces.append((str(content), role == 'assistant'))
            if role == 'assistant':
                pieces.append((tokenizer.eos_token, True))
            pieces.append(('\n', False))
        elif role == 'system':
            pieces.append((f'System: {content}\n', False))
        elif role == 'user':
            pieces.append((f'User: {content}\n', False))
        elif role == 'assistant':
            pieces.extend([('Assistant: ', False), (str(content), True), (tokenizer.eos_token, True), ('\n', False)])
    text = ''.join((piece for piece, _ in pieces))
    assistant_spans: list[tuple[int, int]] = []
    byte_offset = 0
    for piece, is_assistant in pieces:
        next_offset = byte_offset + len(piece.encode())
        if is_assistant:
            assistant_spans.append((byte_offset, next_offset))
        byte_offset = next_offset
    encoding = getattr(tokenizer, '_encoding', None)
    if encoding is None:
        raise ConfigError("assistant_only_loss with a slow tokenizer requires TalkieTokenizer's tiktoken encoding")
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    added_id_to_bytes = {token_id: token.encode() for token, token_id in tokenizer.added_tokens_encoder.items()}
    assistant_mask: list[int] = []
    byte_offset = 0
    for token_id in input_ids:
        token_bytes = added_id_to_bytes.get(token_id)
        if token_bytes is None:
            token_bytes = encoding.decode_single_token_bytes(token_id)
        next_offset = byte_offset + len(token_bytes)
        assistant_mask.append(int(any((byte_offset < end and next_offset > start for start, end in assistant_spans))))
        byte_offset = next_offset
    if 1 not in assistant_mask:
        raise ConfigError('SFT example has no assistant tokens')
    return {'input_ids': input_ids, 'assistant_masks': assistant_mask}
