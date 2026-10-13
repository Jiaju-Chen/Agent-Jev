"""No-download installation smoke test with a tiny random model (not a trained agent)."""
import argparse
import json
import string
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
from decoder_decision import DecisionModel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True,help='New directory for a tiny checkpoint')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42)
    names=list(string.ascii_uppercase)+[a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    vocab={name:i for i,name in enumerate(['[UNK]','[PAD]','[EOS]']+names)}
    raw=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    lm=GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab),n_positions=512,n_embd=16,n_layer=1,n_head=2,
                                 bos_token_id=2,eos_token_id=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
    model=DecisionModel(lm,tok).eval();model.save_pretrained(args.out)
    loaded=DecisionModel.from_pretrained(str(args.out),device='cpu')
    result=loaded.decide({'resident':'A'},{'participation':{
        'question':'Choose an option.','options':{'contribute':'Contribute.','wait':'Wait.','decline':'Decline.'}}})
    print(json.dumps({'model':'tiny random model; schema and installation demo only','answers':result},indent=2))


if __name__=='__main__':
    main()
