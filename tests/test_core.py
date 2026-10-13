"""CPU tests for one-pass readout, candidate mapping and portable checkpoints."""
import json
import random
import string
import tempfile
import unittest
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from decoder_decision import DecisionModel, ChoiceQuestion
from decoder_decision.sampling import sample_choice
from decoder_decision.prompts import encode_choice


class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        names=list(string.ascii_uppercase)+[a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        vocab={name:i for i,name in enumerate(['[UNK]','[PAD]','[EOS]']+names)}
        raw=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));raw.pre_tokenizer=Whitespace()
        cls.tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
        torch.manual_seed(42)
        cls.lm=GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab),n_positions=512,n_embd=16,n_layer=1,n_head=2,
                                         resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,bos_token_id=2,eos_token_id=2)).eval()
        cls.model=DecisionModel(cls.lm,cls.tok).eval()

    def test_readout_matches_full_vocabulary_logits(self):
        q=ChoiceQuestion('Select one.',{str(i):'description '+str(i) for i in range(40)})
        item=self.model.encode({'state':'example'},q,seed=27)
        ids=torch.tensor([item.ids])
        with torch.no_grad():
            direct=self.lm(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1,self.model.label_ids[:40]]
            selective=self.model(item)
        torch.testing.assert_close(selective,direct,rtol=1e-5,atol=1e-6)

    def test_one_backbone_call_and_external_key_mapping(self):
        count=[0]
        hook=self.lm.base_model.register_forward_hook(lambda *a:count.__setitem__(0,count[0]+1))
        try:
            answer=self.model.decide('state',{'q':{'question':'Which one?',
                'options':{'stay_home':'Stay home.','visit_friend':'Visit a friend.','work':'Go to work.'}}})['q']
        finally:
            hook.remove()
        self.assertEqual(count[0],1)
        self.assertEqual(set(answer['probabilities']),{'stay_home','visit_friend','work'})
        self.assertIn(answer['choice'],answer['probabilities'])
        self.assertAlmostEqual(sum(answer['probabilities'].values()),1,places=6)
        a=random.Random(5);b=random.Random(5)
        self.assertEqual([sample_choice(answer,a) for _ in range(20)],
                         [sample_choice(answer,b) for _ in range(20)])

    def test_target_remapping_and_variable_cardinality(self):
        for k in [3,5,40]:
            q=ChoiceQuestion('Which?',{str(i):f'option {i}' for i in range(k)})
            item=self.model.encode('state',q,seed=44,target=str(k-1))
            self.assertEqual(item.keys[item.gold],str(k-1))
            self.assertEqual(len(self.model(item)),k)
        with self.assertRaises(ValueError):
            encode_choice('large state',q,self.tok,self.model.label_table,max_tokens=1)

    def test_question_ids_are_metadata_and_typed_adapters(self):
        definition={'question':'Which?','options':{'x':'one','y':'two'}}
        a=self.model.decide('state',{'alpha':definition})['alpha']
        b=self.model.decide('state',{'beta':definition})['beta']
        self.assertEqual(a['probabilities'],b['probabilities'])
        output=self.model.decide('state',{
            'yesno':{'type':'noul','question':'Is it true?'},
            'rating':{'type':'score','question':'Which level?','levels':['low','medium','high']}})
        self.assertEqual(output['yesno']['noul'],output['yesno']['probabilities']['yes'])
        self.assertAlmostEqual(output['rating']['score'],sum(int(k)*p for k,p in output['rating']['probabilities'].items()))

    def test_real_hf_checkpoint_roundtrip(self):
        # Keep test artifacts: workspace policy does not authorize cleanup.
        folder=Path(tempfile.mkdtemp(prefix='decoder-decision-checkpoint-'))
        self.model.save_pretrained(folder)
        restored=DecisionModel.from_pretrained(str(folder),device='cpu')
        item=self.model.encode('state',ChoiceQuestion('Which?',{'a':'one','b':'two'}))
        with torch.no_grad():
            torch.testing.assert_close(restored(item),self.model(item),rtol=0,atol=0)
        self.assertEqual(restored.label_table,self.model.label_table)


if __name__=='__main__':
    unittest.main()
