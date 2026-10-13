"""Token-level choice labels; option keys are application IDs, not model labels."""
import json
import random
import string
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class EncodedChoice:
    ids: list[int]
    keys: list[str]  # in model-logit order after permutation
    gold: int | None = None
    history_tokens_removed: int = 0


@dataclass
class ChoiceQuestion:
    text: str
    options: Mapping[str, str]

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError('Question text must be nonempty')
        if not self.options or any(not isinstance(k,str) or not k for k in self.options):
            raise ValueError('Options need nonempty string keys')
        if any(not isinstance(v,str) for v in self.options.values()):
            raise ValueError('Each option description must be a string')


def single_token_labels(tokenizer, capacity=255):
    names = list(string.ascii_uppercase)+[a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    table, used = [], set()
    for name in names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if len(ids)==1 and ids[0] not in used:
            table.append((name,ids[0])); used.add(ids[0])
        if len(table)==capacity:
            break
    if not table:
        raise ValueError('Tokenizer has no supported single-token labels')
    return table


def state_text(state: Any):
    return state if isinstance(state,str) else json.dumps(state,ensure_ascii=False,sort_keys=True,allow_nan=False)


def option_suffix(question, tokenizer, table, seed, target=None):
    keys = list(question.options)
    if not 1 <= len(keys) <= len(table):
        raise ValueError(f'{len(keys)} options exceed tokenizer/checkpoint capacity {len(table)}')
    if target is not None and target not in question.options:
        raise ValueError('Target must be an option key')
    random.Random(seed).shuffle(keys)
    ids = tokenizer.encode('\n\nQuestion: '+question.text+'\nOptions:',add_special_tokens=False)
    opening = tokenizer.encode('\n(',add_special_tokens=False)
    for i,key in enumerate(keys):
        ids += opening+[table[i][1]]+tokenizer.encode(') '+question.options[key],add_special_tokens=False)
    ids += tokenizer.encode('\nAnswer: (',add_special_tokens=False)
    return ids,keys,keys.index(target) if target is not None else None


def encode_choice(state, question, tokenizer, table, max_tokens=2048, seed=0, target=None):
    suffix, keys, gold = option_suffix(question,tokenizer,table,seed,target)
    ids = tokenizer.encode('Context:\n'+state_text(state),add_special_tokens=False)+suffix
    if len(ids)>max_tokens:
        raise ValueError(f'Input has {len(ids)} tokens, limit {max_tokens}; shorten state explicitly')
    return EncodedChoice(ids,keys,gold)


def encode_alfworld(row, tokenizer, table, max_tokens=2048, seed=0, labeled=False):
    """Exact current training contract, including historical step representation.

    Pass history as supplied by the existing training/evaluation pipeline.
    No environment import or global dataset/server path is required.
    """
    actions = row['legal_actions']
    if len(set(actions)) != len(actions):
        raise ValueError('Legal actions must be distinct')
    question = ChoiceQuestion('Choose the next legal action that advances the task.',
                              {a:a for a in actions})
    suffix,keys,gold = option_suffix(question,tokenizer,table,seed,row['action'] if labeled else None)
    essential = tokenizer.encode('Context:\nTask: '+row['task']+'\nCurrent observation: '+row['observation'],
                                 add_special_tokens=False)
    room = max_tokens-len(essential)-len(suffix)
    if room < 0:
        raise ValueError('Task/current observation/options exceed input limit')
    recent = row.get('history',[])[-2:]
    history = '\nRecent actions and observations:\n'+'\n'.join(
        f"Observation {h['step']}: {h['observation']}\nAction {h['step']}: {h['action']}" for h in recent) if recent else ''
    hist_ids = tokenizer.encode(history,add_special_tokens=False)
    retained = hist_ids[-room:] if room and len(hist_ids)>room else hist_ids if len(hist_ids)<=room else []
    return EncodedChoice(essential+retained+suffix,keys,gold,len(hist_ids)-len(retained))
