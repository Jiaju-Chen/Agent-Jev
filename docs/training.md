# Data and supervised training

## Canonical row

```json
{"task_id":"episode-0001","state":{"resident":"A","budget":120},"question":"Which decision would the resident make?","options":{"contribute":"Contribute 40 units.","wait":"Wait.","decline":"Decline."},"target":"contribute"}
```

- `state`: text or JSON; JSON is rendered deterministically.
- `question`: nonempty text.
- `options`: nonempty mapping of application ID to description string.
- `target`: an existing option ID, not A/B/C or an output-token ID.
- `task_id`: optional group ID; required on every row when supplied on any row.

All real options are retained. During training their order changes with a
reproducible seed; `target` is remapped to the resulting label position.
Dev uses fixed permutations at every evaluation.

## Convert an existing dataset

Download/export your chosen dataset using its own tools, then convert a local
CSV, JSON list or JSONL. No external dataset is downloaded by the converter.

```bash
python tools/prepare_dataset.py \
  --input ./decisions.csv --out ./data/decisions \
  --state-field context --question-field instruction \
  --options-field choices --target-field answer --target-format key \
  --group-field episode_id --dev-fraction 0.1 --seed 42
```

Supported choices:

```json
{"stay":"Stay home.","visit":"Visit a friend."}
```

```json
["Stay home.","Visit a friend."]
```

```json
{"label":["A","B"],"text":["Stay home.","Visit a friend."]}
```

For lists, generated keys are `"0"`, `"1"`, etc. Target modes:

| Mode | Meaning |
|---|---|
| `key` | Target equals the choice key |
| `index` | Zero-based integer into the supplied option order |
| `text` | Target description matches exactly one option |

CSV state/options cells may contain JSON. `--question "Choose the next action."`
supplies one common question when the dataset lacks a question column.

`--group-field` maps a person/task/episode/scenario field to task_id and keeps the
whole group within one split. Without it, exact duplicate inputs stay together,
but related turns/personas are not recognized: choose the grouping relevant to
your claimed generalization. The output directory must be new. A manifest records
the input hash, seed and grouping. Existing train/dev files can be used directly.

The trainer rejects exact input overlap and task_id overlap between train/dev.
If contradictory labels occur for the same input, resolve their label provenance
or use a distribution-learning extension; this trainer currently expects one target.

## Train

```bash
agent-jev-train \
  --model ./models/qwen35-text \
  --train ./data/decisions/train.jsonl --dev ./data/decisions/dev.jsonl \
  --out ./runs/my-model --epochs 10 --accum 32
```

Equivalent: `python -m decoder_decision.train ...`.

All original LM parameters are trainable; no extra head is introduced.
The objective is restricted candidate CE:

\[
L=-\log\frac{\exp z_{\mathrm{target}}}{\sum_{k=1}^{K}\exp z_k}.
\]

One forward/backward per example, gradient accumulation between optimizer updates,
BF16 bulk on CUDA (FP32 CPU), FP32 master weights and AdamW moments, betas(.9,.95),
weight decay.01, clip1, gradient checkpointing. Optimizer states require substantial
memory; keep checkpoint temporary-file headroom in addition to model space.

## Schedule

`--lr 1e-5`, `--min-lr 1e-6`, explicit `--warmup-steps` and `--decay-steps`.
These are **optimizer updates**, not examples or microbatches. Warmup rises linearly;
cosine decay reaches the floor at decay_steps, then stays there even across epochs.
The global update index is restored without restarting warmup.

If omitted, warmup is at most20 updates (reduced for very small datasets), and
decay defaults to one epoch, at least warmup+1. Explicit research example:

```bash
agent-jev-train \
  --model ./models/qwen35-text \
  --train ./data/train.jsonl --dev ./data/dev.jsonl \
  --out ./runs/alfworld-adaptation --data-format alfworld \
  --epochs 10 --accum 32 --warmup-steps 20 --decay-steps 814
```

814 corresponds to ceil(26035/32) in our expanded ALFWorld corpus. Recompute
for your dataset or choose a fixed budget; the numeric setting is not universal.

## Evaluation and checkpoints

Default: fixed dev CE/accuracy every approximately200 updates **and epoch end**.
Periodic evaluations within50 updates of an epoch boundary merge into that
boundary. On tiny datasets there may be only epoch-end evaluations.

- `best/`: absolute minimum dev CE model/tokenizer/label table.
- `latest/`: current model and FP32 masters, AdamW, RNG, global step/example
  position and early-stop state in `optimizer.pt`.
- `metrics.jsonl`: every update and dev/selection event.
- `training_config.json`: final configuration and termination status.

After at least one epoch, default patience is3 dev evaluation windows without a
CE improvement of.005 nats. Absolute best and significant improvement are separate.
`--epochs 10` is a maximum, not a promise to override early stopping.
Test scores never select a checkpoint. This portable trainer does not automatically
run an environment; evaluate its best/last checkpoints in your simulator separately.

Resume using identical data/schedule/budget arguments plus `--resume`:

```bash
agent-jev-train \
  --model ./models/qwen35-text \
  --train ./data/decisions/train.jsonl --dev ./data/decisions/dev.jsonl \
  --out ./runs/my-model --epochs 10 --accum 32 --resume
```

Hashes and core hyperparameters are checked. `latest` is written at epoch/final/
early-stop boundaries, so an interruption can lose work since its last write.
The saved FP32 masters are authoritative on recovery. An already early-stopped
run needs a new explicit experiment to change its stopping policy.

`--stop-after-epoch N` stages a run at an epoch boundary without changing the
planned LR schedule, useful for separate environment evaluation. Resume with the
same total epochs and without that stage limit.

## Tiny end-to-end exercise

```bash
python examples/cpu_smoke.py --out ./runs/tiny-base
agent-jev-train \
  --model ./runs/tiny-base \
  --train examples/train_demo.jsonl --dev examples/dev_demo.jsonl \
  --out ./runs/tiny-sft --device cpu --epochs 2 --accum 1 \
  --warmup-steps 1 --decay-steps 2
```

The tiny model and synthetic labels are a software exercise. They provide no
scientific evidence about decision accuracy or human behavior.
