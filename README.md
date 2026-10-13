# Agent-Jev

**Train your own single-forward decision model. Use it in an agent, simulator, or ordinary program.**

[中文说明](README.zh-CN.md) · [Training & data](docs/training.md) · [Model support](docs/models.md) · [ALFWorld](docs/alfworld.md) · [Architecture](docs/architecture.md)

Agent-Jev turns a text causal language model into a closed-choice decision function:

```text
state + question + options
          ↓
one backbone forward → selected LM-head rows → option probabilities
          ↓
application-defined choice / yes-no probability / expected rating
```

No text decoding loop or extra classification head is needed. Fine-tune the existing
parameters with candidate cross entropy using your own labelled decisions.

This is an independent **Jev-style decision toolkit**. It does not reproduce Jev's
undisclosed backbone, RLCD, or parallel sampler. Its readout shares the established
single-pass approach used by [Decider](https://github.com/Mapika/decider).

## Quick start

### 1. Install

Python 3.10+; CPU works, CUDA is recommended for full-size models.

```bash
git clone https://github.com/Jiaju-Chen/Agent-Jev.git
cd Agent-Jev
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

The repository/package name is **Agent-Jev**; the Python import is
`decoder_decision`. No checkpoints or research datasets are bundled.

### 2. Verify the installation without downloading a model

```bash
python examples/cpu_smoke.py --out ./runs/tiny-smoke
```

This creates a tiny **random** model and prints a valid decision response. It tests
installation and the interface, not decision quality. Re-run with a new output path.

### 3. Load a text model

```python
from decoder_decision import DecisionModel

# Local text checkpoint or a Hugging Face text causal-LM repository.
model = DecisionModel.from_pretrained("Qwen/Qwen2.5-0.5B", device="auto")
state = {"resident": {"budget": 120, "trust_in_neighbors": 0.7},
         "event": "A community repair project needs another 80 units."}

choice_answers = model.decide(
    state=state,
    questions={
        "participation": {
            "question": "Which decision would this resident most likely make?",
            "options": {
                "contribute": "Contribute 40 units.",
                "wait": "Wait for more information.",
                "decline": "Decline to contribute.",
            },
        },
    },
)
print(choice_answers["participation"]["choice"])
print(choice_answers["participation"]["probabilities"])
```

The example base model is small enough to try easily; its decision probabilities
are not a trained or validated social-behavior model. Train on relevant data for
your application. Use a local fine-tuned checkpoint in the same call when available.

**Qwen3.5:** its upstream base repository includes vision components. First use
[the text-export instructions](docs/models.md#qwen35-text-export), then pass the
export directory. Existing text-only SFT checkpoints load directly.

## Use your own data

One JSONL row contains a state, a question, described options, and a target **key**:

```json
{"task_id":"scenario-001","state":{"resident":"A","budget":120},"question":"Which decision would this resident make?","options":{"contribute":"Contribute 40 units.","wait":"Wait for information.","decline":"Decline."},"target":"contribute"}
```

Supply separate train/dev files, or convert and split your CSV/JSON/JSONL:

```bash
python tools/prepare_dataset.py \
  --input ./my_data.csv --out ./data/decisions \
  --state-field context --question-field question \
  --options-field choices --target-field label --target-format index \
  --group-field person_id --dev-fraction 0.1

agent-jev-train \
  --model Qwen/Qwen2.5-0.5B \
  --train ./data/decisions/train.jsonl --dev ./data/decisions/dev.jsonl \
  --out ./runs/my-model --epochs 10 --accum 32
```

`index` labels are zero-based; `key` and `text` labels are also supported.
The group field keeps all decisions of a person/task/scenario within one split.
The converter accepts option maps, lists, and `{label: [...], text: [...]}`.
See [training documentation](docs/training.md) for the full schema, schedule,
checkpoint selection, exact resume, and original ALFWorld data format.

Only **best** and **latest** model slots are retained. `best` minimizes fixed dev CE;
`latest` includes model, FP32 master weights, AdamW state, RNG and global step.

## More than action selection

The application defines its question. It can ask about participation, cooperation,
resource allocation, sentiment, routing, judgments, or a simulator's next action.

```python
answers = model.decide(state, {
    "support": {"type": "noul", "question": "Does the resident support the project?"},
    "trust": {"type": "score", "question": "How much does the resident trust the organizer?",
              "levels": ["Low trust", "Moderate trust", "High trust"]},
})
# answers["support"]["noul"]: probability of yes
# answers["trust"]["score"]: expected level index in [0, 2]
```

Choice, Noul and Score use the same choice readout; Noul/Score are arithmetic
adapters. Multiple questions currently run **independently and serially**, with
one forward per question. They do not share a cached prefix in this release.
Question IDs are metadata and do not enter the prompt.

For stochastic simulation:

```python
import random
from decoder_decision.sampling import sample_choice

rng = random.Random(42)
action = sample_choice(choice_answers["participation"], rng)
# Your simulator applies action and produces the next state.
```

Run the complete query example with:

```bash
python examples/social_simulation.py --model ./runs/my-model/best
```

Probabilities reflect the model/training target. They are not automatically real
population frequencies. Human-behavior claims require relevant held-out validation.

## Model support

| Family | Current evidence |
|---|---|
| Qwen3.5 text-only | Original 2B full-size research run; exact ALFWorld encoder parity |
| Qwen2 / Qwen2.5 | Tiny real HF Qwen2 architecture: selected logits match full-model logits |
| Llama | Tiny real HF Llama architecture: selected logits match full-model logits |
| Gemma2 | Tiny real HF Gemma2 architecture: readout includes final logit softcapping |
| GPT-2 | CPU end-to-end inference, checkpoint, SFT and resume fixtures |
| Other text causal LMs | Need a linear head and a full-logit equivalence check; not all are validated |

These are compatibility checks, **not accuracy benchmarks for pretrained weights**.
Other multimodal models, remote custom code, GGUF, quantization, model sharding,
and PEFT adapters are outside the current loader. See [details](docs/models.md).

## ALFWorld lineage

The toolkit grew from full Qwen3.5-2B decision SFT on successful ALFWorld teacher
trajectories. The adapter keeps that original task/observation/two-history/all-options
representation and candidate remapping, while the general API has no environment dependency.

```bash
python examples/alfworld_choice.py --model ./your-alfworld-checkpoint
```

The example is an ALFWorld-shaped input; it does not launch an environment.
[The ALFWorld guide](docs/alfworld.md) shows the adapter, native trajectory data,
and a rollout integration sketch.

## Tests and reproducibility

```bash
python -m unittest discover -s tests -v
python tools/verify_release.py --out ./runs/trainer-verification
```

CPU tests cover real HF model readouts, variable options beyond Z, target/key
mapping, typed adapters, seeded sampling, checkpoint roundtrip, grouped datasets,
and resumed-versus-uninterrupted FP32 training. Optional verification checks the
ALFWorld encoder against an original reference. Observed checks and their limits
are recorded in [evidence/VERIFICATION.json](evidence/VERIFICATION.json).

## Project structure

```text
src/decoder_decision/  inference, prompts, SFT, schedule, sampling
examples/             no-download CPU smoke, social query, ALFWorld input
tools/                CSV/JSON conversion, Qwen3.5 export, verification
tests/                CPU checks
docs/                 models, training/data, ALFWorld, architecture
```

## References and license

- [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): typed decision interface inspiration.
- [Mapika/Decider](https://github.com/Mapika/decider): prior single-pass decision readout and training work.
- [Qwen](https://huggingface.co/Qwen): foundation models used in the research workflow.
- [ALFWorld](https://github.com/alfworld/alfworld): original environment/data setting.

Source code is [Apache-2.0](LICENSE). Model weights and datasets use their own
upstream terms. This repository is independent of those projects.
