import json
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

print("=" * 50)
print("Step 1: SFT快速测试(10个样本, 1 epoch)")
print("=" * 50)
sft_result_str = rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/quick_test_sft",
    "max_samples": 10,
    "num_epochs": 1,
    "batch_size": 2,
    "use_lora": True
})
sft_result = json.loads(sft_result_str)
print(f"SFT完成: {sft_result['output_dir']}")

print("=" * 50)
print("Step 2: GRPO快速测试(5个样本, 1 epoch)")
print("=" * 50)
grpo_result_str = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/quick_test_grpo",
    "max_samples": 5,
    "num_epochs": 1,
    "batch_size": 2,
    "use_lora": True
})
grpo_result = json.loads(grpo_result_str)
print(f"GRPO完成: {grpo_result['output_dir']}")

print("=" * 50)
print("Step 3: 评估")
print("=" * 50)
eval_result_str = rl_tool.run({
    "action": "evaluate",
    "model_path": "./models/quick_test_grpo",
    "max_samples": 10,
    "use_lora": True
})
eval_result = json.loads(eval_result_str)
print(f"准确率: {eval_result['accuracy']}")
print(f"平均奖励: {eval_result['average_reward']}")
