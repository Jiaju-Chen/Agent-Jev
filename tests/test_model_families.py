"""Compare selective logits to full-model logits for several real HF families."""
import unittest
import torch
from transformers import (LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,
                          Gemma2Config,Gemma2ForCausalLM)
from decoder_decision import DecisionModel,ChoiceQuestion
import test_core


class ModelFamiliesTest(unittest.TestCase):
    def test_selective_readout_matches_full_models(self):
        test_core.CoreTest.setUpClass();tok=test_core.CoreTest.tok
        configs=[
            (LlamaForCausalLM,LlamaConfig(vocab_size=len(tok),hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)),
            (Qwen2ForCausalLM,Qwen2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)),
            (Gemma2ForCausalLM,Gemma2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                final_logit_softcapping=2.0,bos_token_id=2,eos_token_id=2,pad_token_id=1)),
        ]
        for cls,config in configs:
            with self.subTest(family=config.model_type):
                lm=cls(config).eval();model=DecisionModel(lm,tok).eval()
                for count in [3,40]:
                    item=model.encode('state',ChoiceQuestion('Choose.',{str(i):str(i) for i in range(count)}))
                    ids=torch.tensor([item.ids])
                    with torch.no_grad():
                        direct=lm(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1,model.label_ids[:count]]
                        selected=model(item)
                    torch.testing.assert_close(selected,direct,rtol=1e-5,atol=1e-6)
