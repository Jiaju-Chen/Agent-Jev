# Models and runtime

## Loader contract

`DecisionModel.from_pretrained(path_or_hf_repo, device="auto", revision=None)`
loads a standard Hugging Face **text causal LM** and tokenizer. It calls the
model's base transformer and projects the final answer-slot state through only
the selected rows of a linear LM head. Input/output embeddings may be tied or
untied. Gemma2's configured final logit softcap is applied to those selected rows.

Validated readout families: GPT-2, Qwen2 (including Qwen2.5's architecture),
Llama and Gemma2 using tiny real HF model fixtures; the Qwen3.5 text path was used
in the full 2B ALFWorld research run. Tiny tests establish computation equivalence,
not task quality or full-size performance. New model families should be checked
against the full model's last-position logits before claiming compatibility.

The default renderer is **plain text**, not a model-specific chat template.
Training and inference use the same renderer. An instruct model can load but may
need adaptation to this prompt format; pretrained decision quality is not guaranteed.

## Devices, dtype and memory

- `device="auto"`: CUDA if available, otherwise CPU.
- CUDA defaults to BF16; CPU defaults to FP32. Pass `dtype=torch.float32` or
  `dtype=torch.float16` explicitly when your hardware/model needs it.
- Native small FP32 tensors are restored from local safetensors during BF16 load.
  This preserves Qwen3.5's DeltaNet precision exceptions.
- Loading does not install optimized kernels, enable quantization, shard weights,
  or use a third-party custom model implementation.
- Full SFT uses FP32 masters and optimizer moments in addition to inference weights;
  its memory requirement is considerably larger than inference alone.

Optional Qwen3.5 kernels/runtime can materially change speed. Our research runtime
used torch2.6.0, transformers5.17.0, Triton3.2.0 and flash-linear-attention0.5.2.
Those versions are a measured environment, not a request to upgrade dependencies
in an existing experiment. Install a separate environment for deployment.

## Qwen3.5 text export

The original Qwen3.5 base config is multimodal. Download a local snapshot, then
extract its text parameters strictly, preserving every native tensor dtype:

```bash
hf download Qwen/Qwen3.5-2B-Base --local-dir ./models/qwen35-source
python tools/export_qwen_text.py \
  --source ./models/qwen35-source --out ./models/qwen35-text
python examples/alfworld_choice.py --model ./models/qwen35-text
```

The exporter checks every text key and shape, retains the original native dtypes,
handles tied embeddings and records input hashes. It has no internal server paths.
It uses CPU RAM during conversion; no GPU is required. The public exporter derives
from the validated research exporter; the release verification does not rerun a
full-size export/download.

## Label capacity

Labels are A..Z followed by two-letter labels that encode as **one token**.
Multi-token labels are skipped and duplicate token IDs rejected. Qwen3.5 supports
our capacity of255; other tokenizers may support fewer. Capacity is checked before
inference. A checkpoint stores its actual label table in `decision_config.json`.
A change in option count selects a different number of rows of the same head;
no parameter matrix is resized or initialized at inference time.

## Existing Decider checkpoints

Decider shares the letter-row readout, but its prompt layouts, score handling,
per-type temperatures and serving optimizations are its own API contract.
This loader does **not** promise exact Decider inference compatibility. For direct
use of released Decider weights, use its upstream interface; use this toolkit to
train and serve checkpoints with the documented Agent-Jev rendering contract.

## Currently outside scope

Remote custom code, arbitrary multimodal architectures, GGUF, quantized output
heads, multi-device sharding, PEFT wrappers, and automatic temperature calibration.
The loader needs an ordinary linear output head. A new model with further logit
scaling/normalization needs an explicit adapter and equivalence tests.
