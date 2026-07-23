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

print("原始返回类型:", type(sft_result_str))
print("原始返回内容:")
print(sft_result_str)

sft_result = json.loads(sft_result_str)
print("解析后的字段名:", list(sft_result.keys()))
