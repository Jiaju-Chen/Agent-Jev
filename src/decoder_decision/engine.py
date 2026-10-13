"""Read only label rows of the existing head; never call generate()."""
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .prompts import ChoiceQuestion, encode_choice, encode_alfworld, single_token_labels


class DecisionModel(nn.Module):
    def __init__(self, lm, tokenizer, max_tokens=2048, temperature=1., label_table=None):
        super().__init__()
        if not math.isfinite(temperature) or temperature<=0:
            raise ValueError('Temperature must be finite and positive')
        if max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        self.lm, self.tokenizer = lm, tokenizer
        self.max_tokens, self.temperature = max_tokens, temperature
        self.label_table = label_table or single_token_labels(tokenizer)
        if len({idx for _,idx in self.label_table}) != len(self.label_table):
            raise ValueError('Every choice label must have a distinct token ID')
        head=lm.get_output_embeddings()
        if not isinstance(head,nn.Linear):
            raise ValueError('This backend requires a linear LM output head')
        for name,idx in self.label_table:
            if tokenizer.encode(name,add_special_tokens=False)!=[idx]:
                raise ValueError('Label table does not match tokenizer')
        self.register_buffer('label_ids',torch.tensor([idx for _,idx in self.label_table]),persistent=False)

    @classmethod
    def from_pretrained(cls, model, device='auto', dtype=None, max_tokens=None, temperature=None,
                        revision=None):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        folder = Path(model)
        if not folder.exists():
            folder = Path(snapshot_download(str(model),revision=revision))
        if device=='auto':
            device='cuda' if torch.cuda.is_available() else 'cpu'
        config = AutoConfig.from_pretrained(str(folder))
        if config.model_type=='qwen3_5':
            raise ValueError('Use tools/export_qwen_text.py first for the multimodal Qwen3.5 snapshot')
        dtype = dtype or (torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
        lm = AutoModelForCausalLM.from_pretrained(str(folder),dtype=dtype,attn_implementation='sdpa')
        # Preserve native small FP32 exceptions when using a BF16 bulk load.
        if dtype==torch.bfloat16:
            for file in sorted(folder.glob('*.safetensors')):
                with safe_open(file,framework='pt',device='cpu') as weights:
                    for key in weights.keys():
                        if weights.get_slice(key).get_dtype()=='F32':
                            lm.get_parameter(key).data = weights.get_tensor(key)
        tok = AutoTokenizer.from_pretrained(str(folder))
        cfgfile=folder/'decision_config.json'
        cfg=json.loads(cfgfile.read_text()) if cfgfile.exists() else {}
        table=[(r['label'],r['token_id']) for r in cfg['label_table']] if 'label_table' in cfg else None
        instance=cls(lm,tok,max_tokens=max_tokens if max_tokens is not None else cfg.get('max_total_tokens',2048),
                     temperature=temperature if temperature is not None else cfg.get('temperature',1.),
                     label_table=table)
        return instance.to(device).eval()

    def forward(self, item):
        """Differentiable raw candidate logits, also usable for restricted CE SFT."""
        if not item.ids or not 1<=len(item.keys)<=len(self.label_table):
            raise ValueError('Encoded choice needs input tokens and supported candidates')
        ids=torch.tensor([item.ids],device=self.label_ids.device,dtype=torch.long)
        output=self.lm.base_model(input_ids=ids,attention_mask=torch.ones_like(ids),
                                  use_cache=False,return_dict=True)
        h=output.last_hidden_state[0,-1]
        head=self.lm.get_output_embeddings()
        chosen=self.label_ids[:len(item.keys)]
        bias=head.bias[chosen] if getattr(head,'bias',None) is not None else None
        logits=F.linear(h,head.weight[chosen],bias)
        # Gemma2 applies an elementwise final logit softcap after the LM head.
        # Apply it to selected rows in the model's dtype before float conversion.
        text_config=self.lm.config.get_text_config()
        softcap=getattr(text_config,'final_logit_softcapping',None)
        if softcap is not None:
            if softcap<=0:
                raise ValueError('final_logit_softcapping must be positive')
            logits=softcap*torch.tanh(logits/softcap)
        return logits.float()

    def encode(self,state,question,seed=0,target=None):
        return encode_choice(state,question,self.tokenizer,self.label_table,self.max_tokens,seed,target)

    @torch.inference_mode()
    def predict_encoded(self,item):
        self.eval()
        if self.label_ids.device.type=='cuda':
            torch.cuda.synchronize(self.label_ids.device)
        started=time.perf_counter()
        logits=self(item)
        if self.label_ids.device.type=='cuda':
            torch.cuda.synchronize(self.label_ids.device)
        forward_readout_seconds=time.perf_counter()-started
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite candidate logits')
        probs=(logits/self.temperature).softmax(-1).cpu().tolist()
        index=max(range(len(probs)),key=probs.__getitem__)
        k=len(probs)
        concentration=(max(probs)-1/k)/(1-1/k) if k>1 else 1.
        return dict(choice=item.keys[index],probabilities=dict(zip(item.keys,probs)),
                    concentration=max(0.,min(1.,concentration)),
                    entropy_nats=-sum(p*math.log(p) for p in probs if p>0),
                    input_tokens=len(item.ids),forward_readout_seconds=forward_readout_seconds)

    def decide(self,state,questions,seed=0):
        """One independent forward per question; question IDs never enter input.

        This version evaluates questions serially. It does not implement shared
        prefix caching, tree masks or Jev's parallel multi-question sampler.
        Noul/score are deterministic adapters over the same choice distribution.
        """
        result={}
        for key,q in questions.items():
            kind='choice' if isinstance(q,ChoiceQuestion) else q.get('type','choice')
            if kind=='choice':
                question=q if isinstance(q,ChoiceQuestion) else ChoiceQuestion(q['question'],q['options'])
            elif kind=='noul':
                question=ChoiceQuestion(q['question'],{
                    'yes':q.get('true_description','Yes, the statement is true.'),
                    'no':q.get('false_description','No, the statement is false.')})
            elif kind=='score':
                levels=q['levels']
                if not 2<=len(levels)<=10 or any(not isinstance(v,str) for v in levels):
                    raise ValueError('Score requires 2..10 ordered string level descriptions')
                question=ChoiceQuestion(q['question'],{str(i):f'Level {i}: {v}' for i,v in enumerate(levels)})
            else:
                raise ValueError('Supported types: choice, noul, score')
            started=time.perf_counter()
            answer=self.predict_encoded(self.encode(state,question,seed))
            answer['decision_seconds']=time.perf_counter()-started
            answer['type']=kind
            if kind=='noul':
                answer['noul']=answer['probabilities']['yes']
            elif kind=='score':
                answer['score']=sum(int(k)*p for k,p in answer['probabilities'].items())
            result[key]=answer
        return result

    def decide_alfworld(self,row,seed=0):
        item=encode_alfworld(row,self.tokenizer,self.label_table,self.max_tokens,seed)
        return self.predict_encoded(item)

    def save_pretrained(self,out):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        self.lm.save_pretrained(str(out),safe_serialization=True)
        self.tokenizer.save_pretrained(str(out))
        (out/'decision_config.json').write_text(json.dumps(dict(
            format='decoder_decision_v1',temperature=self.temperature,max_total_tokens=self.max_tokens,
            label_table=[dict(label=n,token_id=i) for n,i in self.label_table],
            supported_types=['choice','noul_adapter','score_adapter'],multi_question='independent_serial',
            probability_calibration='not_established'),indent=2)+'\n')
