# Agent-Jev

**用自己的数据，训练一个单次前向的决策模型。** 可以接入 agent、社会模拟，或者普通软件流程。

[English](README.md) · [数据与训练](docs/training.md) · [模型支持](docs/models.md) · [ALFWorld](docs/alfworld.md) · [原理](docs/architecture.md)

输入是 **state＋question＋options**，输出是用户定义的选项 ID 和每个选项的概率。
模型只做一次骨干前向，再读取原有 LM head 中的标签权重；不需要生成答案文本或添加分类 head。
多个 question 当前分别计算，每题一次前向。

这是独立的 Jev 风格工具包，复现的是单次前向决策范式。它没有复现 Jev 未公开的完整架构、RLCD 和并行采样器。
这种读出方式已有 [Decider](https://github.com/Mapika/decider) 等先例；本项目主要提供用户自己的数据训练、轨迹适配和仿真上手流程。

## 安装与第一次运行

```bash
git clone https://github.com/Jiaju-Chen/Agent-Jev.git
cd Agent-Jev
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/cpu_smoke.py --out ./runs/tiny-smoke
```

Windows PowerShell 激活命令：`.venv\Scripts\Activate.ps1`。

CPU smoke 不下载模型，创建一个小型随机模型来检查安装和接口，不代表模型效果。
代码包叫 `agent-jev`，Python 导入仍是 `decoder_decision`。

```python
from decoder_decision import DecisionModel

model = DecisionModel.from_pretrained("Qwen/Qwen2.5-0.5B")
result = model.decide(
    state={"居民": {"预算": 120, "邻里信任": 0.7}, "事件": "社区正在筹款维修公共设施"},
    questions={"参与方式": {
        "question": "这个居民更可能采取哪种决定？",
        "options": {"捐款": "捐款40元", "等待": "等待更多信息", "拒绝": "拒绝参与"},
    }},
)
print(result)
```

这里的小型 base 模型仅方便上手。用于社会模拟时，需要用相关行为数据或明确来源的合成决策数据训练、验证；
模型概率不自动等于真实人群的行为频率。Qwen3.5 的原始仓库是多模态模型，先按[文档](docs/models.md#qwen35-text-export)导出文本权重。

## 用自己的数据训练

JSONL 每行：

```json
{"task_id":"scenario-001","state":{"预算":120},"question":"这个居民会怎么决定？","options":{"捐款":"捐款40元","等待":"等待信息","拒绝":"拒绝参与"},"target":"捐款"}
```

`target` 是选项 ID，不是内部 A/B/C 标签。候选顺序打乱和正确答案映射由工具完成。

```bash
agent-jev-train \
  --model Qwen/Qwen2.5-0.5B \
  --train ./train.jsonl --dev ./dev.jsonl \
  --out ./runs/my-model --epochs 10 --accum 32
```

支持 CSV、JSON、JSONL 转换及按人／任务／情境分组划分，见 [tools/prepare_dataset.py](tools/prepare_dataset.py) 和[训练文档](docs/training.md)。
不要把同一条轨迹的不同 turn 随机分到 train 和 dev。

训练使用原模型全量参数、候选交叉熵、FP32 master weights 与 AdamW 状态。支持独立设置 warmup、衰减终点和最低学习率，
每约200次更新及每个 epoch 末评估 dev，只保存 best/latest；恢复训练保留全局 step、随机数和早停状态。
10 epoch 是预算上限，可能因 dev CE 早停提前结束。

## 示例

```bash
python examples/social_simulation.py --model ./runs/my-model/best
python examples/alfworld_choice.py --model ./your-alfworld-checkpoint
```

社会模拟可以按概率采样，也可以取 argmax，状态更新由模拟器负责。
支持 choice、yes/no 概率和等级期望值。当前没有 RL 或自动概率校准。
ALFWorld 示例保留 task、observation、最近两条历史和全部合法动作的输入格式；不安装环境也能测试该适配器。

## 支持范围与检查

- Qwen3.5 文本模型：来自原2B训练流程，并检查原 ALFWorld 编码完全一致。
- Qwen2／Qwen2.5、Llama、Gemma2、GPT-2：使用真实 HF 小模型检查候选 logits 与完整模型输出一致。
- 其他模型：需要线性 LM head；额外 logits 变换需适配并检查。
- 模型架构兼容性不等于在某个数据集上效果已验证。当前不支持 GGUF、量化 head、PEFT、模型分片或任意多模态模型。

```bash
python -m unittest discover -s tests -v
python tools/verify_release.py --out ./runs/trainer-verification
```

源码采用 [Apache-2.0](LICENSE)，仓库不包含模型权重或研究数据集。
