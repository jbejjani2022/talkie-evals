import base64
from pathlib import Path
import torch
import pytest
from transformers import AutoModelForCausalLM
from talkie_base_experiments.configuration_talkie import TalkieConfig
from talkie_base_experiments.modeling_talkie import TalkieForCausalLM
from talkie_base_experiments.registry import get_model_spec
from talkie_base_experiments import artifacts

def test_download_is_pinned_and_only_fetches_required_files(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(artifacts,'snapshot_download',lambda **kw:seen.append(kw))
    spec=get_model_spec('talkie-1930-13b-base')
    artifacts.download_original(spec,tmp_path)
    assert seen[0]['revision']==spec.revision
    assert set(seen[0]['allow_patterns'])=={'final.ckpt','vocab.txt','README.md'}

def test_converter_and_remote_transformers_round_trip_preserve_bf16_weights(monkeypatch,tmp_path):
    real=TalkieConfig
    def small(vocab_size):
        return real(vocab_size=vocab_size,n_layer=1,n_head=2,n_embd=128,head_dim=64,
            max_position_embeddings=32,eos_token_id=vocab_size-1)
    monkeypatch.setattr(artifacts,'TalkieConfig',small)
    torch.manual_seed(99)
    model=TalkieForCausalLM(small(256)).eval()
    with torch.no_grad():model.lm_head.normal_(std=.02);model.lm_head_gain.w_g.fill_(3.5)
    source=tmp_path/'source';source.mkdir()
    torch.save({'model':{'_orig_mod.'+k:v for k,v in model.state_dict().items()}},source/'final.ckpt')
    (source/'vocab.txt').write_text(''.join(f'{base64.b64encode(bytes([i])).decode()} {i}\n' for i in range(256)))
    output=artifacts.convert_original(get_model_spec('talkie-1930-13b-base'),source,tmp_path/'converted')
    loaded=AutoModelForCausalLM.from_pretrained(output,trust_remote_code=True,dtype=torch.bfloat16,
        attn_implementation='sdpa').eval()
    for name,value in model.state_dict().items():
        assert torch.equal(value.to(torch.bfloat16),loaded.state_dict()[name])
    assert loaded.config.logit_scale==3.5
    assert loaded.config.source_revision==get_model_spec('talkie-1930-13b-base').revision
    ids=torch.tensor([[2,4,8]])
    with torch.inference_mode():
        assert torch.equal(model.to(torch.bfloat16)(ids,use_cache=False).logits,loaded(ids,use_cache=False).logits)

def test_malformed_raw_checkpoint_fails_closed(tmp_path):
    checkpoint=tmp_path/'bad.ckpt';torch.save({'other':torch.ones(2)},checkpoint)
    with pytest.raises(ValueError,match='No Talkie state dict'):
        artifacts.extract_state_dict(checkpoint)

def test_model_family_and_tokenizer_identity_are_checked():
    from trait_lab.models import validate_identity
    from trait_lab.paths import BUNDLE
    for family in ('vintage','web'):
        validate_identity(BUNDLE/'interfaces'/family,family)
    with pytest.raises(ValueError,match='mismatch'):
        validate_identity(BUNDLE/'interfaces/vintage','web')
