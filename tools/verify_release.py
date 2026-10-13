"""CPU-only portable trainer and ALFWorld parity check; retains test artifacts."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from decoder_decision.prompts import single_token_labels, encode_alfworld


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-encoder',type=Path)
    p.add_argument('--alfworld-data',type=Path)
    p.add_argument('--tokenizer',type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent.parent
    spec=importlib.util.spec_from_file_location('test_core',root/'tests/test_core.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.CoreTest.setUpClass()
    base=args.out/'base';fixture.CoreTest.model.save_pretrained(base)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
    common=[sys.executable,'-m','decoder_decision.train','--model',str(base),
            '--train',str(root/'examples/train_demo.jsonl'),'--dev',str(root/'examples/dev_demo.jsonl'),
            '--device','cpu','--epochs','2','--accum','1','--warmup-steps','1','--decay-steps','2',
            '--patience','99']
    jobs=[('full',args.out/'full',[]),('staged',args.out/'staged',['--stop-after-epoch','1']),
          ('resumed',args.out/'staged',['--resume'])]
    for name,out,extra in jobs:
        with (args.out/f'{name}.log').open('x') as log:
            subprocess.run(common+['--out',str(out)]+extra,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    full=torch.load(args.out/'full/latest/optimizer.pt',map_location='cpu',weights_only=False)
    restored=torch.load(args.out/'staged/latest/optimizer.pt',map_location='cpu',weights_only=False)
    assert full['step']==restored['step']==4
    assert full['serial']==restored['serial']==4
    for a,b in zip(full['masters'],restored['masters']):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key in full['optimizer']['state']:
        for field in ['exp_avg','exp_avg_sq','step']:
            torch.testing.assert_close(full['optimizer']['state'][key][field],restored['optimizer']['state'][key][field],rtol=0,atol=0)
        assert full['optimizer']['state'][key]['exp_avg'].dtype==torch.float32
    assert full['selection']==restored['selection']
    report=dict(cpu_portable_sft='passed',optimizer_resume='bitwise_equal_to_uninterrupted',
                full_updates=4,fp32_master_and_moments=True,only_model_slots=['best','latest'])
    if args.reference_encoder:
        assert args.alfworld_data and args.tokenizer
        spec=importlib.util.spec_from_file_location('reference_encoder',args.reference_encoder)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        tok=AutoTokenizer.from_pretrained(str(args.tokenizer));table=single_token_labels(tok)
        assert table==old.labels(tok)
        rows=[json.loads(s) for s in args.alfworld_data.read_text().splitlines()][:100]
        compared=0
        for row in rows:
            for seed in [0,777,20261004]:
                previous=old.encode(row,tok,table,2048,seed)
                current=encode_alfworld(row,tok,table,2048,seed,labeled=True)
                assert current.ids==previous['ids']
                assert current.gold==previous['gold']
                assert current.keys==[row['legal_actions'][i] for i in previous['permutation']]
                assert current.history_tokens_removed==previous['history_tokens_removed']
                compared+=1
        report['alfworld_encoder_parity']=dict(status='exact_token_ids_gold_and_candidate_mapping',cases=compared)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
