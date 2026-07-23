import json
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()
result_str = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full_v2",
    "output_dir": "./models/grpo_full_v2",
    "max_samples": 300,
    "num_epochs": 1,
    "batch_size": 4,
    "learning_rate": 2e-6,        # 1e-5 -> 2e-6, 大幅调低,避免调度峰值把有效LR冲得过高
    "warmup_ratio": 0.03,         # 0.1 -> 0.03, 缩短warmup阶段,减少LR震荡的窗口
    "num_generations": 4,
    "max_new_tokens": 400,
    "temperature": 0.7,           # 0.8 -> 0.7, 略微降低采样随机性,减少输出发散的风险
    "kl_coef": 0.1,               # 0.05 -> 0.1, 加强KL约束,防止策略跑偏
    "clip_range": 0.2,
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
    "use_fp16": False,
    "use_bf16": False,
    "reward_type": "combined",
    "reward_config": {
        "components": [
            {"type": "accuracy", "weight": 1.0},
            {"type": "length_penalty", "weight": 0.5, "target_length": 200},
            {"type": "step", "weight": 0.3, "step_bonus": 0.1}
        ]
    },
    "save_steps": 30,             # 100 -> 30, 更频繁保存,方便回退到崩溃前的checkpoint
    "logging_steps": 10,          # 20 -> 10, 更密集地观察训练动态,及时发现异常
})
print(result_str)
