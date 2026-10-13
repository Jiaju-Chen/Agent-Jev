# What is reproduced?

Agent-Jev implements a **single-question, single-forward decoder readout**:

1. Encode state, question and a closed list of option descriptions.
2. End the sequence at an answer slot.
3. Read the final hidden state h.
4. Project through W[label_token_ids] of the existing LM head (with model-specific
   elementwise logit processing where supported).
5. Softmax over the valid K choices and map probabilities back to application IDs.

No label text or JSON is autoregressively generated. JSON formatting is ordinary
application code. The same differentiable logits train with restricted CE.
Changing K changes the number of head rows read, not the parameter shapes.
The decoder can see the entire supplied option list at its answer position.

## Input and label contract

The default plain renderer is:

```text
Context:
<text or serialized JSON>

Question: <question>
Options:
(A) <description>
(B) <description>
...
Answer: (
```

Labels are inserted as validated token IDs. They are output-slot identifiers,
not permanent semantic classes. Training permutations remap targets correctly.
Question IDs remain outside the prompt. Arbitrary option IDs remain application
metadata; descriptions carry the semantics supplied to the model.

For the ALFWorld adapter, task/current observation/history are separately encoded
to preserve the original token contract. The release test checks exact token IDs,
gold positions and candidate mappings against100 real rows at3 seeds (300 cases).

## Typed answers

- Choice: candidate argmax and full probability mapping.
- Noul: a two-choice yes/no distribution; return p(yes).
- Score: a choice over2..10 ordered descriptions; return the expected level index.
- Concentration: (p_max - 1/K)/(1 - 1/K), with1 for a singleton.
- Entropy: -sum p log p.

Concentration is not a calibrated accuracy estimate. CE is a proper scoring rule,
but finite-data training does not by itself establish calibration. The toolkit
currently has no RLCD or temperature-fitting algorithm.

## Multiple questions

Each question is evaluated independently and serially. Q questions require Q
backbone forwards. No branch sees another question's tokens. There is currently
no shared-state cache or tree-shaped mask, and this release does not claim a
parallel multi-question speed advantage. For Qwen3.5, a future branch cache would
also need to fork DeltaNet recurrent state, not only ordinary attention K/V.

## Relationship to Jev and Decider

Jev's public typed API inspires the interface. Its full backbone/training details
are undisclosed; this code does not claim to recover them. Decider is an existing
open implementation of the established label-logit decision approach with its
own inference, training, temperatures and serving optimizations. We acknowledge
that shared foundation rather than claiming the readout or CE algorithm is new.

This repository provides a compact user-data training path, exact ALFWorld
adapter, reproducible checkpoint/selection controls and simulation examples.
Source code and examples can be useful independently of a research novelty claim.

## Timing

`forward_readout_seconds`: synchronized device input creation, backbone and
selected-head projection; excludes tokenization and probability/key formatting.
`decision_seconds` in decide(): encoding, readout, posterior and key mapping;
excludes simulator/environment execution. Temperature division affects probabilities,
not which option is the argmax when T>0 (apart from numerical ties).
