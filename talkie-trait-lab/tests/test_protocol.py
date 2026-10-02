import json
import math
from pathlib import Path
import pytest
import torch
from trait_lab.eval import score_pair, TEMPLATES
from trait_lab.fewshot_protocol import conditions, selected_examples, render_demos
from trait_lab.train import candidate_scores
from trait_lab.reference import TransformersBackend
from talkie_base_experiments.configuration_talkie import TalkieConfig
from talkie_base_experiments.modeling_talkie import TalkieForCausalLM

def test_bpb_is_raw_utf8_including_space_and_ties_are_first():
    answers = [' é', ' long ASCII answer']
    value = score_pair([-2., -5.], answers, 0)
    assert value['answer_bytes'] == [3, 18]
    assert value['selected_index'] == 1  # A raw-logprob comparison would choose 0.
    assert value['bits_per_byte'][0] == pytest.approx(2/(math.log(2)*3))
    assert score_pair([-1., -1.], [' x', ' y'], 1)['selected_index'] == 0
    with pytest.raises(ValueError): score_pair([float('nan'), -1.], answers, 0)

def test_conditions_and_complementary_controls():
    config = dict(n_values=[0,1,2,4,8,16,32],banks_per_trait=5,balanced_controls=True,reverse_order_n=8)
    cs=conditions(config)
    assert len(cs)==961
    rows=[dict(prompt=f'Q{i}',choices=['hi','lo'],answer_index=i%2) for i in range(8)]
    banks=[dict(trait='Openness',bank=0,rows=rows)]
    base=dict(n=8,target='Openness',bank=0,reverse=False)
    a=selected_examples(base|dict(arm='balanced-a'),banks)
    b=selected_examples(base|dict(arm='balanced-b'),banks)
    assert all(x[0]==y[0] and x[1]!=y[1] for x,y in zip(a,b))
    task={'likelihood_prompt_templates':TEMPLATES,'choice_continuation_prefix':' '}
    assert render_demos(dict(n=0),banks,task)==''

class TinyTokenizer:
    eos_token_id=31
    padding_side='left'
    def encode(self,s,add_special_tokens=False): return [ord(c)%30+1 for c in s]
    def pad(self,values,return_tensors):
        from transformers import BatchEncoding
        seqs=values['input_ids']; width=max(map(len,seqs))
        return BatchEncoding({'input_ids':torch.tensor([[31]*(width-len(s))+s for s in seqs]),
            'attention_mask':torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in seqs])})

def test_canonical_candidate_scores_equal_independent_reference():
    from types import SimpleNamespace
    torch.manual_seed(8)
    model=TalkieForCausalLM(TalkieConfig(vocab_size=32,n_layer=1,n_head=2,n_embd=128,head_dim=64,
        max_position_embeddings=32,eos_token_id=31)).float().eval()
    with torch.no_grad(): model.lm_head.normal_(std=.02)
    tok=TinyTokenizer();pairs=[('Q:',' hi'),('Q:',' no thanks')]
    actual=candidate_scores(model,tok,pairs)
    ref=TransformersBackend();ref.model=model;ref.tokenizer=tok
    ref.config=SimpleNamespace(batch_size=2,max_model_len=32)
    expected=ref.loglikelihood([p for p,a in pairs],[a for p,a in pairs])
    assert actual==pytest.approx([r.log_probability for r in expected],abs=1e-6)

def test_bundled_fewshot_split_and_banks_are_disjoint():
    from trait_lab.fewshot import inputs
    config,banks,rows=inputs()
    assert len(rows)==3200
    groups={r['group_id'] for r in rows}
    import csv
    from trait_lab.paths import BUNDLE
    with (BUNDLE/'fewshot/scenarios.csv').open() as f:
        splits={r['scenario_id']:r for r in csv.DictReader(f)}
    assert all(splits[str(r['source_scenario_id'])]['group_id'] not in groups for b in banks for r in b['rows'][:32])
