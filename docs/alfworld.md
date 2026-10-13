# ALFWorld quick start and original input contract

ALFWorld is the toolkit's original research setting. The **general** decision API
has no ALFWorld dependency; use this adapter when you have an ALFWorld-trained
checkpoint or want to fine-tune native trajectory rows.

## A decision without installing an environment

```bash
python examples/alfworld_choice.py --model ./your-text-checkpoint
```

This passes an ALFWorld-shaped observation and legal action list to the model.
It verifies the adapter, not an episode success rate. You can supply `--row row.json`.

```python
row = {
    "task": "put a clean apple in the fridge",
    "observation": "You are in a kitchen. You see a sink and a fridge.",
    "history": [],
    "legal_actions": ["go to sinkbasin 1", "go to fridge 1", "inventory", "look"],
}
answer = model.decide_alfworld(row, seed=0)
action = answer["choice"]  # actual legal command, not A/B/C
```

## Native trajectory rows

```json
{"task_id":"task/trial/game.tw-pddl","task":"put a clean apple in the fridge","observation":"You are in a kitchen.","history":[],"legal_actions":["go to sinkbasin 1","go to fridge 1","look"],"action":"go to sinkbasin 1"}
```

Fine-tune with `--data-format alfworld`. Retain successful teacher episodes as one
possible BC dataset, split **tasks/episodes before extracting turns**, and keep
an independent dev set. Success-only BC does not teach all failure/recovery states.
The toolkit does not bundle VETTA weights, a teacher collector, or ALFWorld data.

The adapter retains the original representation:

1. `Context:`, task and current observation.
2. At most two historical observation/action entries.
3. The question "Choose the next legal action that advances the task."
4. Every legal action, permuted reproducibly, with a single-token label.
5. A final `Answer: (` slot, read without generating text.

Task/current observation/options are never cropped. Old history tokens may be
cropped from the left when the context limit is exceeded. Other overflows fail
explicitly. History step fields are rendered exactly as supplied: preserve your
collector/evaluator's convention rather than silently renumbering them.

## Optional environment integration

Install `python -m pip install -e '.[alfworld]'` and obtain data using the
[ALFWorld project's setup guide](https://github.com/alfworld/alfworld).
Native TextWorld/planner dependencies depend on your OS and need separate setup.
The following sketch uses a local compiled `game.tw-pddl`; it is not a downloadable
full benchmark harness:

```python
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
from decoder_decision import DecisionModel

model = DecisionModel.from_pretrained("./your-alfworld-checkpoint")
env_id = textworld.gym.register_game(
    "/path/to/game.tw-pddl",
    textworld.EnvInfos(won=True, admissible_commands=True),
    max_episode_steps=50,
    wrappers=[AlfredDemangler()],
)
env = textworld.gym.make(env_id)
try:
    env.seed(0)
    observation, info = env.reset()
    task = observation.split("Your task is to:", 1)[-1].strip()
    history = []
    for step in range(50):
        row = {"task": task, "observation": observation,
               "history": history[-2:], "legal_actions": info["admissible_commands"]}
        action = model.decide_alfworld(row, seed=step)["choice"]
        next_obs, reward, done, info = env.step(action)
        history.append({"step": step, "observation": observation, "action": action})
        observation = next_obs
        if done:
            break
    print("won:", info.get("won", False))
finally:
    env.close()
```

A controlled performance comparison should fix task lists, environment/action
budget, history format, candidate permutation, device and warmup. Record success
and mean/median episode, decision and environment times separately. This release
makes no new success or latency claim for that sketch.
