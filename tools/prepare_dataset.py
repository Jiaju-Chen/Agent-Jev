"""Convert local CSV/JSON/JSONL decision data; optionally split by task/person ID."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def parse_value(value):
    if isinstance(value,str) and value.lstrip().startswith(('[','{')):
        return json.loads(value)
    return value


def options(value):
    value=parse_value(value)
    if isinstance(value,list):
        result={str(i):v for i,v in enumerate(value)}
    elif isinstance(value,dict) and set(value)=={'label','text'}:
        if len(value['label'])!=len(value['text']):
            raise ValueError('Choice label/text lists must have equal lengths')
        result={str(k):v for k,v in zip(value['label'],value['text'])}
        if len(result)!=len(value['label']):
            raise ValueError('Choice labels must be unique')
    elif isinstance(value,dict):
        result={str(k):v for k,v in value.items()}
    else:
        raise ValueError('Options must be a mapping, list, or {label:[], text:[]}')
    if not result or any(not isinstance(v,str) for v in result.values()):
        raise ValueError('Option descriptions must be strings')
    if any(not k for k in result):
        raise ValueError('Option keys cannot be empty')
    return result


def convert(row,state_field,question_field,options_field,target_field,
            target_format='key',question=None,group_field=None):
    choices=options(row[options_field]);target=row[target_field]
    if target_format=='index':
        if isinstance(target,bool) or str(target).strip()!=str(int(target)):
            raise ValueError('Index labels must be integers')
        index=int(target)
        if not 0<=index<len(choices):
            raise ValueError('Target index is out of range (zero-based)')
        target=list(choices)[index]
    elif target_format=='text':
        hits=[k for k,v in choices.items() if v==target]
        if len(hits)!=1:
            raise ValueError('Text target must match exactly one option')
        target=hits[0]
    else:
        target=str(target)
    if target not in choices:
        raise ValueError('Target is not an option key')
    text=question if question is not None else row[question_field]
    if not isinstance(text,str) or not text.strip():
        raise ValueError('Question must be a nonempty string')
    result=dict(state=parse_value(row[state_field]),question=text,options=choices,target=target)
    if group_field:
        if group_field not in row:
            raise ValueError('Missing group field: '+group_field)
        result['task_id']=str(row[group_field])
    return result


def split_groups(rows,fraction,seed):
    if not 0<fraction<1:
        raise ValueError('dev-fraction must be between 0 and 1')
    groups={}
    has_ids=[('task_id' in r) for r in rows]
    if any(has_ids) and not all(has_ids):
        raise ValueError('All rows must supply grouping IDs or none must')
    for row in rows:
        # Exact duplicate inputs remain together when explicit groups are absent.
        identity=json.dumps({k:row[k] for k in ['state','question','options']},sort_keys=True)
        key=row['task_id'] if all(has_ids) else hashlib.sha256(identity.encode()).hexdigest()
        groups.setdefault(key,[]).append(row)
    if len(groups)<2:
        raise ValueError('Need at least two independent groups for train/dev')
    keys=sorted(groups);random.Random(seed).shuffle(keys)
    n=max(1,min(len(keys)-1,round(len(keys)*fraction)))
    dev_keys=set(keys[:n])
    train=[r for k in keys if k not in dev_keys for r in groups[k]]
    dev=[r for k in keys if k in dev_keys for r in groups[k]]
    return train,dev


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='New directory containing train.jsonl/dev.jsonl')
    p.add_argument('--state-field',default='state')
    p.add_argument('--question-field',default='question')
    p.add_argument('--question',help='Same question for every row')
    p.add_argument('--options-field',default='options')
    p.add_argument('--target-field',default='target')
    p.add_argument('--target-format',choices=['key','index','text'],default='key')
    p.add_argument('--group-field',help='Episode/person/scenario ID; kept within one split')
    p.add_argument('--dev-fraction',type=float,default=.1)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.input.suffix.lower()=='.csv':
        with args.input.open(newline='') as stream:
            source=list(csv.DictReader(stream))
    elif args.input.suffix.lower()=='.json':
        source=json.loads(args.input.read_text())
        if not isinstance(source,list):
            raise ValueError('JSON input must be a list of rows')
    else:
        source=[json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    converted=[convert(r,args.state_field,args.question_field,args.options_field,args.target_field,
                       args.target_format,args.question,args.group_field) for r in source]
    train,dev=split_groups(converted,args.dev_fraction,args.seed)
    args.out.mkdir(parents=True,exist_ok=False)
    for name,values in [('train',train),('dev',dev)]:
        with (args.out/f'{name}.jsonl').open('x') as stream:
            for row in values:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    report=dict(train_rows=len(train),dev_rows=len(dev),seed=args.seed,
                grouping=args.group_field or 'exact input identity',
                input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                target_format=args.target_format)
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
