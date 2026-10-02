import copy
import io

import pytest
import torch
from torch import nn

from talkie_base_experiments.training.lora import (
    LoRALinear, inject_lora, adapter_state, load_adapter_state,
    batch_indices, collate_rows,
)

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_query=nn.Linear(4,4,bias=False)
        self.other=nn.Linear(4,4,bias=False)
    def forward(self,x):
        return self.other(torch.tanh(self.attn_query(x)))


def test_zero_adapter_and_only_adapter_updates():
    torch.manual_seed(4)
    model=Tiny()
    x=torch.randn(3,4)
    before=model(x).detach().clone()
    frozen={n:p.clone() for n,p in model.named_parameters()}
    assert inject_lora(model,2,4)==['attn_query']
    assert torch.equal(model(x),before)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    model(x).square().sum().backward(); opt.step()
    assert not torch.equal(model(x),before)
    assert torch.equal(model.attn_query.base.weight,frozen['attn_query.weight'])
    assert torch.equal(model.other.weight,frozen['other.weight'])
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)


def test_adapter_and_optimizer_resume_exact():
    torch.manual_seed(3)
    model=Tiny(); inject_lora(model,2,4)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    x=torch.randn(3,4)
    def step(m,o):
        o.zero_grad(); m(x).square().sum().backward(); o.step()
    step(model,opt)
    restored=copy.deepcopy(model)
    state=adapter_state(model)
    other=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.01)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    with torch.no_grad(): restored.attn_query.lora_B.zero_()
    load_adapter_state(restored,state)
    step(model,opt); step(restored,other)
    for a,b in zip(model.parameters(),restored.parameters()): assert torch.equal(a,b)
    with pytest.raises(ValueError): load_adapter_state(restored,{})


def test_padding_loss_counts_and_batch_coverage():
    rows=[{'input_ids':list(range(n)), 'assistant_masks':[0]+[1]*(n-1)} for n in [3,7,2,5,8]]
    batches=batch_indices(rows,16,window=3)
    assert sorted(i for batch in batches for i in batch)==list(range(len(rows)))
    for group in batches:
        selected=[rows[i] for i in group]
        batch=collate_rows(selected,0)
        assert batch['input_ids'].numel()<=16
        assert int((batch['labels'][:,1:]!=-100).sum())==sum(len(r['input_ids'])-1 for r in selected)
        assert (batch['labels'][batch['attention_mask']==0]==-100).all()
    with pytest.raises(ValueError): batch_indices(rows,4)


def test_selected_assistant_loss_matches_full_model_and_gradients():
    import importlib.util
    from pathlib import Path
    from talkie_base_experiments.configuration_talkie import TalkieConfig
    from talkie_base_experiments.modeling_talkie import TalkieForCausalLM
    from trait_lab import train as module
    torch.manual_seed(9)
    model=TalkieForCausalLM(TalkieConfig(vocab_size=64,n_layer=1,n_head=2,n_embd=128,
                                      head_dim=64,max_position_embeddings=32,eos_token_id=63)).float()
    with torch.no_grad(): model.lm_head.normal_(std=.02)
    inject_lora(model,2,4)
    other=copy.deepcopy(model)
    rows=[{'input_ids':[1,2,3,4,5],'assistant_masks':[0,0,1,1,1]},
          {'input_ids':[3,4,5],'assistant_masks':[0,1,1]}]
    batch=collate_rows(rows,63)
    reference=other(**batch,use_cache=False).loss
    actual,count=module.loss_sum(model,batch)
    assert count==5
    torch.testing.assert_close(actual/count,reference,rtol=1e-6,atol=1e-6)
    (actual/count).backward(); reference.backward()
    for (n,p),(m,q) in zip(model.named_parameters(),other.named_parameters()):
        if p.requires_grad: torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)


def test_trait_subset_preserves_scenarios_and_polarity_balance():
    import importlib.util
    from pathlib import Path
    from collections import Counter
    from trait_lab import train as module
    rows=[{'category':trait,'polarity':polarity,'source_scenario_id':str(i),'pair':pair}
          for trait in ['A','B'] for polarity in ['good','bad']
          for i in range(10) for pair in [1,2]]
    selected=module.select_trait_subset(rows,3)
    assert selected==module.select_trait_subset(rows,3)
    assert len(selected)==24
    assert set(Counter((r['category'],r['polarity'],r['source_scenario_id']) for r in selected).values())=={2}
    assert set(Counter((r['category'],r['polarity']) for r in selected).values())=={6}
