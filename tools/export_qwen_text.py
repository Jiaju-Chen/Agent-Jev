"""Portable exact extraction of a local Qwen3.5 snapshot's text parameters."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sha(file):
    digest=hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new output directory')
    config=AutoConfig.from_pretrained(str(args.source)).get_text_config()
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16)
    expected=model.state_dict();state={};ignored=[]
    for file in sorted(args.source.glob('*.safetensors')):
        with safe_open(file,framework='pt',device='cpu') as weights:
            for key in weights.keys():
                if key.startswith('model.language_model.'):
                    target='model.'+key.removeprefix('model.language_model.')
                elif key.startswith('language_model.'):
                    target=key.removeprefix('language_model.')
                else:
                    target=key
                if target in expected:
                    assert target not in state
                    state[target]=weights.get_tensor(key)
                    assert state[target].shape==expected[target].shape,target
                else:
                    ignored.append(key)
    if 'lm_head.weight' not in state and config.tie_word_embeddings:
        state['lm_head.weight']=state['model.embed_tokens.weight']
    assert set(state)==set(expected),sorted(set(expected)-set(state))
    model.load_state_dict(state,strict=True,assign=True);model.tie_weights()
    args.out.mkdir(parents=True)
    model.save_pretrained(str(args.out),safe_serialization=True)
    AutoTokenizer.from_pretrained(str(args.source)).save_pretrained(str(args.out))
    (args.out/'text_extraction.json').write_text(json.dumps(dict(
        source_hashes={p.name:sha(p) for p in args.source.glob('*.safetensors')},
        parameters=sum(p.numel() for p in model.parameters()),
        tensor_dtypes=dict(Counter(str(v.dtype) for v in state.values())),
        original_text_tensor_dtypes_preserved=True,skipped_keys=ignored),indent=2)+'\n')
    print(json.dumps({'status':'complete','output':str(args.out)}))


if __name__=='__main__':
    main()
