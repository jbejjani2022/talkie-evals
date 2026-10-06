import argparse
import json
from pathlib import Path
from trait_lab.paths import ROOT

ITEMS = ROOT / 'behavior' / 'items'
SCORES = ROOT / 'behavior' / 'scores'
DOWNLOADS = ROOT / 'downloads' / 'behavior'


def main():
    from .items import EVALS, TEXT_MODE
    all_evals = list(EVALS) + [f'{n}_text' for n in TEXT_MODE]
    p = argparse.ArgumentParser(prog='behavior-evals')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('fetch')
    sub.add_parser('materialize')
    sub.add_parser('materialize-text')
    sub.add_parser('check-tokens')
    for name in ('validate', 'score'):
        d = sub.add_parser(name)
        d.add_argument('--family', choices=['vintage', 'web'], required=True)
        d.add_argument('--arm', required=True, help='Output name, e.g. vintage-tulu')
        d.add_argument('--adapter', type=Path); d.add_argument('--model', type=Path)
        d.add_argument('--items', type=Path, default=ITEMS)
        if name == 'score':
            d.add_argument('--interface', action='append', choices=['bare', 'chat'])
            d.add_argument('--eval', action='append', choices=all_evals)
            d.add_argument('--output', type=Path, default=SCORES)
        else:
            d.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'fetch':
        from .sources import fetch
        print(fetch(DOWNLOADS))
    elif a.command == 'materialize':
        from .items import materialize
        print(json.dumps(materialize(DOWNLOADS, ITEMS), indent=2))
    elif a.command == 'materialize-text':
        from .items import materialize_text
        print(json.dumps(materialize_text(ITEMS), indent=2))
    elif a.command == 'check-tokens':
        print(json.dumps(check_tokens(ITEMS), indent=2))
    elif a.command == 'validate':
        from .score import validate
        print(json.dumps(validate(a.family, a.arm, a.items, a.output, a.model, a.adapter), indent=2))
    elif a.command == 'score':
        from .score import run
        print(json.dumps(run(a.family, a.arm, a.items, a.output, a.interface or ['bare', 'chat'],
                             a.eval or list(EVALS), a.model, a.adapter), indent=2))


def check_tokens(items_root):
    """CPU check that every candidate is one token for both tokenizers; prompt-length statistics."""
    from talkie_base_experiments.tokenization_talkie import TalkieTokenizer
    from trait_lab.paths import BUNDLE
    from .items import EVALS, jsonl, render
    from .score import candidate_ids
    report = {}
    for family in ('vintage', 'web'):
        tok = TalkieTokenizer.from_pretrained(BUNDLE / 'interfaces' / family)
        for name in EVALS:
            items = jsonl(Path(items_root) / f'{name}.jsonl')
            for candidates in {tuple(r['candidates']) for r in items}:
                candidate_ids(tok, list(candidates))
            for interface in ('bare', 'chat'):
                lengths = sorted(len(tok.encode(render(r, interface), add_special_tokens=False)) for r in items)
                report[f'{family}/{interface}/{name}'] = {'items': len(items), 'tokens': sum(lengths),
                    'median': lengths[len(lengths) // 2], 'max': lengths[-1]}
    return report


if __name__ == '__main__':
    main()
