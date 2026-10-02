import argparse
import json
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(prog='trait-lab')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('verify-bundle')
    sub.add_parser('list-data')
    models = sub.add_parser('prepare-models'); models.add_argument('--family', choices=['vintage','web','both'], default='both')
    models.add_argument('--source-dir', type=Path, help='Original checkpoint+vocab directory; single family only')
    d = sub.add_parser('download-data'); d.add_argument('which', choices=['tulu','trait','all'])
    d = sub.add_parser('prepare-data'); d.add_argument('--family', choices=['vintage','web','both'], default='both')
    d.add_argument('--dataset', action='append', required=True, help='vintage, tulu, source slug, or all')
    d.add_argument('--model', type=Path, help='Existing atomic model; single family only')
    d = sub.add_parser('prepare-custom')
    d.add_argument('--family',choices=['vintage','web'],required=True)
    d.add_argument('--name',required=True); d.add_argument('--messages',type=Path,required=True)
    d.add_argument('--loss-token-cap',type=int,default=0); d.add_argument('--seed',type=int,default=20260915)
    d.add_argument('--model',type=Path)
    sub.add_parser('train', add_help=False)
    for name in ('eval','fewshot','fewshot-check'):
        d = sub.add_parser(name)
        d.add_argument('--family', choices=['vintage','web'], required=True)
        if name != 'fewshot-check':
            d.add_argument('--output',type=Path,required=True); d.add_argument('--model',type=Path)
            d.add_argument('--adapter',type=Path); d.add_argument('--smoke-per-group',type=int,default=0)
        if name == 'eval': d.add_argument('--interface', choices=['bare','chat'], default='bare')
        else:
            d.add_argument('--condition', action='append'); d.add_argument('--config', type=Path)
    d=sub.add_parser('contrast'); d.add_argument('--left',type=Path,required=True); d.add_argument('--right',type=Path,required=True)
    d.add_argument('--output',type=Path,required=True);d.add_argument('--replicates',type=int,default=2000);d.add_argument('--seed',type=int,default=20260925)
    if len(sys.argv)>1 and sys.argv[1]=='train':
        from .train import main as train
        sys.argv.pop(1); return train()
    a=p.parse_args()
    if a.command=='verify-bundle':
        from .io import verify_bundle
        m=verify_bundle(); print(f"Verified {len(m['files'])} bundled files")
    elif a.command=='list-data':
        from .paths import BUNDLE
        from .io import read
        for key, r in read(BUNDLE/'selections.json.gz').items():
            print(f"{key}: {len(r['rows'])} rows, {r['loss_tokens']} assistant-loss tokens")
    elif a.command=='prepare-models':
        from .models import prepare
        if a.source_dir and a.family=='both': p.error('--source-dir needs one family')
        for family in ['vintage','web'] if a.family=='both' else [a.family]: print(prepare(family,a.source_dir))
    elif a.command=='download-data':
        from .data import download
        download(a.which)
    elif a.command=='prepare-data':
        from .data import prepare_data
        if a.model and a.family=='both': p.error('--model needs one family')
        print(prepare_data(['vintage','web'] if a.family=='both' else [a.family],a.dataset,{a.family:a.model} if a.model else None))
    elif a.command=='eval':
        from .eval import run
        print(json.dumps(run(a.family,a.output,a.interface,a.model,a.adapter,a.smoke_per_group),indent=2))
    elif a.command=='prepare-custom':
        from .data import prepare_custom
        print(prepare_custom(a.family,a.name,a.messages,a.loss_token_cap,a.seed,a.model))
    elif a.command=='fewshot-check':
        from .fewshot import check
        print(json.dumps(check(a.family,a.condition,a.config),indent=2))
    elif a.command=='fewshot':
        from .fewshot import run
        run(a.family,a.output,a.condition,a.config,a.model,a.adapter,a.smoke_per_group)
    elif a.command=='contrast':
        from .analyze import contrast
        print(json.dumps(contrast(a.left,a.right,a.output,a.replicates,a.seed),indent=2))

if __name__ == '__main__':
    main()
