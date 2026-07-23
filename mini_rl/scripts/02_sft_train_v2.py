import json
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()
result_str = rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "/root/autodl-tmp/hf_cache/Qwen3-0.6B",
    "output_dir": "./models/sft_full_v2",
    "max_samples": 2000,      # 500 -> 2000,给模型更充分的机会学会格式
    "num_epochs": 3,
    "batch_size": 4,
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
})
print(result_str)
