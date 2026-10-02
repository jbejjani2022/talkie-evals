"""Model-only export in the custom experimental LoRA format (not PEFT)."""
from pathlib import Path
from safetensors.torch import save_file, load_file
from talkie_base_experiments.training.lora import adapter_state
from talkie_base_experiments.registry import get_model_spec
from .models import NAMES
from .io import sha, write

def export_run(output, model, family, rank, alpha, state, base):
    dest = Path(output) / 'terminal-adapter'; dest.mkdir(exist_ok=True)
    temporary = dest / 'adapter.safetensors.partial'
    save_file(adapter_state(model), str(temporary)); temporary.replace(dest / 'adapter.safetensors')
    write(dest / 'adapter.json', {'family': family, 'rank': rank, 'alpha': alpha,
        'base_revision': get_model_spec(NAMES[family]).revision,
        'vocab_sha256': sha(Path(base) / 'vocab.txt'), 'step': state['step'],
        'dataset': state['dataset'], 'loss_tokens': state['loss_tokens'],
        'sha256': sha(dest / 'adapter.safetensors')})
