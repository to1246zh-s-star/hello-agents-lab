import json
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

result_str = rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "/root/autodl-tmp/hf_cache/Qwen3-0.6B",
    "output_dir": "./models/sft_full",
    "max_samples": 500,
    "num_epochs": 3,
    "batch_size": 4,
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
})

print("=" * 50)
print("原始返回内容:")
print(result_str)
print("=" * 50)

result = json.loads(result_str)
print("解析后的字段名:", list(result.keys()))
