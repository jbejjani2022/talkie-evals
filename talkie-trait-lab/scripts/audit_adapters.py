"""CPU-only complete tensor round-trip audit of the optional adapter pack."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file
from trait_lab.io import read, sha, write
from trait_lab.paths import BUNDLE

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapters',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-originals',action='store_true',help='For packager; requires original local checkpoint paths')
    a=p.parse_args();result=[]
    for meta in read(BUNDLE/'adapter-inventory.json'):
        path=a.adapters/meta['key'];current=read(path/'adapter.json')
        if current!=meta or sha(path/'adapter.safetensors')!=meta['sha256']:
            raise ValueError(f'Adapter identity/hash mismatch: {path}')
        tensors=load_file(str(path/'adapter.safetensors'))
        assert len(tensors)==560 and sum(x.numel() for x in tensors.values())==62341120
        for name,value in tensors.items():
            assert value.dtype==torch.float32 and torch.isfinite(value).all()
            if name.endswith('.lora_A'):assert value.shape[0]==meta['rank']
            elif name.endswith('.lora_B'):assert value.shape[1]==meta['rank']
            else:raise ValueError(f'Unexpected tensor: {name}')
        if a.compare_originals:
            original=Path(meta['source_checkpoint'])
            assert sha(original)==meta['source_checkpoint_sha256']
            payload=torch.load(original,map_location='cpu',weights_only=False)
            assert payload['adapter'].keys()==tensors.keys()
            assert all(torch.equal(tensors[k],v) for k,v in payload['adapter'].items())
            del payload
        result.append({'key':meta['key'],'tensors':len(tensors),'sha256':meta['sha256'],'exact_original_match':a.compare_originals})
    write(a.output,{'status':'complete','count':len(result),'adapters':result})

if __name__=='__main__':main()
