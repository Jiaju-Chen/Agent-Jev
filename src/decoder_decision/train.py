"""Portable supervised choice training with only best/latest checkpoints."""
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from .engine import DecisionModel
from .prompts import ChoiceQuestion, encode_alfworld
from .controls import learning_rate, evaluation_due, update_selection


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,help='Text causal LM; Qwen3.5 requires text export')
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--data-format',choices=['general','alfworld'],default='general')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--accum',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--warmup-steps',type=int)
    p.add_argument('--decay-steps',type=int)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--min-delta',type=float,default=.005)
    p.add_argument('--seed',type=int,default=20261004)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epoch',type=int,help='Stage boundary; global schedule/epoch budget unchanged')
    args=p.parse_args()
    assert args.epochs>0 and args.accum>0 and args.eval_every>0
    assert args.stop_after_epoch is None or 1<=args.stop_after_epoch<=args.epochs
    train,dev=rows(args.train),rows(args.dev)
    if not train or not dev:
        raise ValueError('Train and dev must both be nonempty')
    # Require task/episode grouping when trajectory data are used. Also reject
    # exact input leakage (even if its labels differ) for all dataset types.
    fields=['state','question','options'] if args.data_format=='general' else ['task','observation','history','legal_actions']
    identity=lambda r:json.dumps({k:r.get(k,[]) if k=='history' else r[k] for k in fields},sort_keys=True,ensure_ascii=False)
    if {identity(r) for r in train}&{identity(r) for r in dev}:
        raise ValueError('Exact state/question/options overlap between train and dev')
    if any('task_id' in r for r in train+dev):
        if not all('task_id' in r for r in train+dev):
            raise ValueError('Supply task_id on every trajectory row')
        if {r['task_id'] for r in train}&{r['task_id'] for r in dev}:
            raise ValueError('Split trajectories by task, not individual turns')
    spe=math.ceil(len(train)/args.accum)
    warmup=args.warmup_steps if args.warmup_steps is not None else min(20,max(1,spe//10))
    decay=args.decay_steps if args.decay_steps is not None else max(spe,warmup+1)
    assert decay>warmup>0
    args.out.mkdir(parents=True,exist_ok=args.resume)
    latest=args.out/'latest';best=args.out/'best'
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(train_sha256=fingerprint(args.train),dev_sha256=fingerprint(args.dev),
                  warmup_steps=warmup,decay_steps=decay,steps_per_epoch=spe,
                  optimizer='FP32 AdamW masters and moments',selection='dev restricted CE')
    torch.manual_seed(args.seed);random.seed(args.seed)
    model=DecisionModel.from_pretrained(str(latest) if args.resume else args.model,
                                        device=args.device,max_tokens=args.max_tokens)
    device=model.label_ids.device
    model.lm.config.use_cache=False
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    named=list(model.lm.named_parameters());params=[p for _,p in named]
    masters=[torch.nn.Parameter(p.detach().float().clone()) for p in params]
    optimizer=torch.optim.AdamW(masters,lr=args.lr,betas=(.9,.95),weight_decay=.01,foreach=False)
    step=serial=0;selection={}
    if args.resume:
        saved=torch.load(latest/'optimizer.pt',map_location='cpu',weights_only=False)
        for key in ['train_sha256','dev_sha256','warmup_steps','decay_steps','epochs','accum','lr',
                    'min_lr','eval_every','patience','min_delta','seed','max_tokens','data_format']:
            assert saved['config'][key]==config[key],f'Resume changes {key}'
        assert saved['names']==[n for n,_ in named]
        with torch.no_grad():
            for p_,master,v in zip(params,masters,saved['masters']):
                master.copy_(v);p_.copy_(master)
        optimizer.load_state_dict(saved['optimizer'])
        step,serial,selection=saved['step'],saved['serial'],saved['selection']
        if selection.get('stop'):
            raise ValueError('Run was early-stopped; start a new explicit experiment to extend it')
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
        if device.type=='cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'],device=device)
        del saved
    batches=[]
    for epoch in range(args.epochs):
        shuffled=list(train);random.Random(args.seed+epoch).shuffle(shuffled)
        batches.extend(shuffled[i:i+args.accum] for i in range(0,len(shuffled),args.accum))
    assert serial==sum(len(b) for b in batches[:step])

    def encode(r,seed):
        if args.data_format=='alfworld':
            return encode_alfworld(r,model.tokenizer,model.label_table,args.max_tokens,seed,labeled=True)
        return model.encode(r['state'],ChoiceQuestion(r['question'],r['options']),seed,r['target'])

    fixed=[encode(r,777+i) for i,r in enumerate(dev)]
    metrics=(args.out/'metrics.jsonl').open('a' if args.resume else 'x')
    def emit(value):
        metrics.write(json.dumps(value)+'\n');metrics.flush();print(json.dumps(value),flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval();loss=correct=0
        for item in fixed:
            scores=model(item)
            v=F.cross_entropy(scores[None,:],torch.tensor([item.gold],device=scores.device))
            if not bool(torch.isfinite(v)):
                raise FloatingPointError('Nonfinite dev loss')
            loss+=v.item();correct+=int(scores.argmax().item()==item.gold)
        emit(dict(event='dev',step=step,loss=loss/len(fixed),accuracy=correct/len(fixed),examples=len(fixed)))
        model.train()
        return loss/len(fixed)

    def select(loss):
        nonlocal selection
        selection,improved=update_selection(selection,loss,step,spe,args.min_delta,args.patience)
        if improved:
            model.save_pretrained(best)
            (best/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        emit(dict(event='selection',step=step,**selection))

    def save_latest():
        model.save_pretrained(latest)
        state=dict(masters=[m.detach() for m in masters],optimizer=optimizer.state_dict(),
                   names=[n for n,_ in named],step=step,serial=serial,selection=selection,config=config,
                   torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        torch.save(state,latest/'optimizer.pending.pt')
        os.replace(latest/'optimizer.pending.pt',latest/'optimizer.pt')
        (latest/'selection.json').write_text(json.dumps(dict(step=step,selection=selection),indent=2)+'\n')

    if not args.resume:
        select(evaluate())
    model.train()
    for index,batch in enumerate(batches):
        if index<step:
            continue
        optimizer.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
        loss_sum=correct=0
        for r in batch:
            item=encode(r,args.seed+serial);serial+=1
            scores=model(item)
            loss=F.cross_entropy(scores[None,:],torch.tensor([item.gold],device=scores.device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            loss_sum+=loss.item();correct+=int(scores.argmax().item()==item.gold)
            (loss/len(batch)).backward()
        for original,master in zip(params,masters):
            master.grad=None if original.grad is None else original.grad.float()
        norm=torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True)
        step=index+1
        for group in optimizer.param_groups:
            group['lr']=learning_rate(step,args.lr,args.min_lr,warmup,decay)
        optimizer.step()
        with torch.no_grad():
            for original,master in zip(params,masters):
                original.copy_(master)
        emit(dict(event='train_update',step=step,loss=loss_sum/len(batch),accuracy=correct/len(batch),
                  lr=optimizer.param_groups[0]['lr'],gradient_norm=norm.item(),examples=len(batch)))
        if evaluation_due(step,spe,args.eval_every):
            select(evaluate())
        if step%spe==0 or selection.get('stop') or step==len(batches):
            save_latest()
        if selection.get('stop') or (args.stop_after_epoch and step==args.stop_after_epoch*spe):
            break
    config.update(status='early_stopped' if selection.get('stop') else
                  'complete' if step==len(batches) else 'epoch_stage_complete',step=step,selection=selection)
    (args.out/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
    metrics.close()


if __name__=='__main__':
    main()
