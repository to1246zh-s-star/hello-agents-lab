import json
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

sft_result = rl_tool.run({"action": "load_dataset", "format": "sft", "max_samples": 5})
sft_data = json.loads(sft_result)
print(f"[SFT格式] 数据集大小: {sft_data['dataset_size']}, 字段: {sft_data['sample_keys']}")

rl_result = rl_tool.run({"action": "load_dataset", "format": "rl", "max_samples": 5})
rl_data = json.loads(rl_result)
print(f"[RL格式] 数据集大小: {rl_data['dataset_size']}, 字段: {rl_data['sample_keys']}")

for reward_type, extra in [
    ("accuracy", {}),
    ("length_penalty", {"max_length": 1024, "penalty_weight": 0.001}),
    ("step", {"step_bonus": 0.1}),
]:
    payload = {"action": "create_reward", "reward_type": reward_type}
    payload.update(extra)
    r = json.loads(rl_tool.run(payload))
    print(f"[{reward_type}] {r['description']}")
