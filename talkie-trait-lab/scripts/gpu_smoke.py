"""One-GPU validation. Supply existing canonical atomic models explicitly."""
import argparse
import json
import os
from pathlib import Path
import pyarrow.parquet as pq
import torch
from trait_lab.paths import require_gpu, BUNDLE
from trait_lab.models import load
from trait_lab.train import candidate_scores, parity_check, select_trait_subset
from trait_lab.eval import TEMPLATES, score_pair
from trait_lab.io import write, sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vintage-model',type=Path,required=True);p.add_argument('--web-model',type=Path,required=True)
    p.add_argument('--adapters',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require_gpu()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    report={'job_id':os.environ.get('SLURM_JOB_ID'),'torch':torch.__version__,'models':{}}
    rows=select_trait_subset(pq.read_table(BUNDLE/'trait.parquet').to_pylist(),1)
    for family, path in [('vintage',a.vintage_model),('web',a.web_model)]:
        model,tok=load(family,path)
        baseline={}
        for interface in ('bare','chat'):
            candidates=[(TEMPLATES[interface].format(**r),' '+str(ans)) for r in rows for ans in r['choices']]
            scores=candidate_scores(model,tok,candidates)
            assert all(__import__('math').isfinite(s) for s in scores)
            baseline[interface]=scores
        identity=parity_check(model,tok)
        del model;torch.cuda.empty_cache()
        model,tok=load(family,path,a.adapters/f'{family}-tulu')
        adapter_parity=parity_check(model,tok)
        candidates=[(TEMPLATES['bare'].format(**r),' '+str(ans)) for r in rows for ans in r['choices']]
        actual=candidate_scores(model,tok,candidates)
        assert actual != baseline['bare']
        report['models'][family]={'base_parity':identity,'adapter_parity':adapter_parity,
            'base_scores':baseline,'adapter_scores':actual,'adapter_changed_scores':True,
            'adapter_sha256':sha(a.adapters/f'{family}-tulu/adapter.safetensors')}
        del model;torch.cuda.empty_cache()
    write(a.output,report|{'status':'complete'})

if __name__=='__main__':main()
