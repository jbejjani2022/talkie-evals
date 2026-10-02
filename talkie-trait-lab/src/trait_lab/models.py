"""Pinned upstream downloads, BF16 conversion, and exact atomic interface rows."""
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file
from talkie_base_experiments.artifacts import download_original, convert_original, configure_vllm_metadata
from talkie_base_experiments.registry import get_model_spec
from talkie_base_experiments.modeling_talkie import TalkieForCausalLM
from talkie_base_experiments.tokenization_talkie import TalkieTokenizer
from .paths import ROOT, BUNDLE, model_path
from .io import verify_bundle, write, sha

NAMES = {'vintage': 'talkie-1930-13b-base', 'web': 'talkie-web-13b-base'}

def validate_identity(path, family):
    from .io import read
    spec = get_model_spec(NAMES[family])
    config = read(Path(path) / 'config.json')
    if config.get('source_repo') != spec.repo_id or config.get('source_revision') != spec.revision:
        raise ValueError('Model/family or pinned base revision mismatch')
    if config.get('vocab_size') != 65539:
        raise ValueError('Use an atomic-interface model with 65539 vocabulary rows')
    if sha(Path(path) / 'vocab.txt') != sha(BUNDLE / 'interfaces' / family / 'vocab.txt'):
        raise ValueError('Model/tokenizer family mismatch')

def prepare(family, source_dir=None):
    verify_bundle()
    output = model_path(family)
    if output.exists():
        raise FileExistsError(output)
    spec = get_model_spec(NAMES[family])
    original = Path(source_dir) if source_dir else download_original(spec, ROOT / 'original')
    converted = convert_original(spec, original, ROOT / 'converted')
    model = TalkieForCausalLM.from_pretrained(converted, dtype=torch.bfloat16)
    rows = load_file(str(BUNDLE / 'interfaces' / family / 'rows.safetensors'))
    if model.embed.weight.shape[0] != 65536:
        raise ValueError('Expected original 65536-row model')
    # Preserve every original row and append the exact 3 experimental rows.
    model.embed = torch.nn.Embedding.from_pretrained(
        torch.cat([model.embed.weight.detach(), rows['embed.weight']]), freeze=False)
    model.lm_head = torch.nn.Parameter(torch.cat([model.lm_head.detach(), rows['lm_head']]))
    model.config.vocab_size = 65539
    configure_vllm_metadata(model.config, model.lm_head_gain.w_g)
    tok = TalkieTokenizer.from_pretrained(BUNDLE / 'interfaces' / family)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model.config.pad_token_id = tok.eos_token_id
    stage = output.with_name(output.name + '.partial')
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    model.save_pretrained(stage, safe_serialization=True, max_shard_size='5GB')
    tok.save_pretrained(stage)
    for name in ('configuration_talkie.py', 'modeling_talkie.py', 'tokenization_talkie.py'):
        shutil.copyfile(Path(__import__('talkie_base_experiments').__file__).parent / name, stage / name)
    write(stage / 'reproduction.json', {
        'family': family, 'source_repo': spec.repo_id, 'revision': spec.revision,
        'interface_sha256': sha(BUNDLE / 'interfaces' / family / 'rows.safetensors'),
        'source_checkpoint_sha256': sha(original / spec.checkpoint_filename),
        'vocab_sha256': sha(original / 'vocab.txt'), 'vocab_size': 65539})
    stage.rename(output)
    return output

def load(family, path=None, adapter=None):
    from .paths import require_gpu
    from .io import read
    from talkie_base_experiments.training.lora import inject_lora, load_adapter_state
    require_gpu()
    base = Path(path) if path else model_path(family)
    validate_identity(base, family)
    tok = TalkieTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = TalkieForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.config.use_cache = False
    if adapter:
        adapter = Path(adapter)
        meta = read(adapter / 'adapter.json')
        if meta['family'] != family or meta['base_revision'] != get_model_spec(NAMES[family]).revision:
            raise ValueError('Adapter/base identity mismatch')
        if sha(base / 'vocab.txt') != meta['vocab_sha256']:
            raise ValueError('Adapter/tokenizer mismatch')
        expected = load_file(str(BUNDLE / 'interfaces' / family / 'rows.safetensors'))
        for name, actual in [('embed.weight', model.embed.weight[-3:]), ('lm_head', model.lm_head[-3:])]:
            if not torch.equal(actual.cpu(), expected[name]):
                raise ValueError('Adapter requires the exact atomic interface rows')
        if sha(adapter / 'adapter.safetensors') != meta['sha256']:
            raise ValueError('Adapter checksum mismatch')
        inject_lora(model, meta['rank'], meta['alpha'])
        load_adapter_state(model, load_file(str(adapter / 'adapter.safetensors')))
    return model.eval(), tok
