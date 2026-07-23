import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from hello_agents.rl import create_rl_dataset
from hello_agents.rl.rewards import (
    create_accuracy_reward,
    create_length_penalty_reward,
    create_step_reward,
)

MODEL_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full_v2"
OUTPUT_DIR = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/grpo_full_custom"

# ---------- 1. 加载tokenizer + 模型 ----------
print("加载tokenizer和模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

# ---------- 2. 手动应用LoRA(库的GRPOTrainerWrapper没有做这一步) ----------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------- 3. 数据集 ----------
print("加载数据集...")
dataset = create_rl_dataset(max_samples=300, model_name=MODEL_PATH)

# ---------- 4. 组合奖励函数(链式包装:accuracy -> length_penalty -> step) ----------
base_reward = create_accuracy_reward(tolerance=1e-4)
reward_with_length = create_length_penalty_reward(
    base_reward, max_length=800, penalty_weight=0.1
)
combined_reward = create_step_reward(reward_with_length, step_bonus=0.05)

# ---------- 5. 真正生效的GRPOConfig,这次所有关键参数都手动传入 ----------
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-6,          # 真正生效
    warmup_steps=5,
    logging_steps=5,
    save_steps=50,
    fp16=False,
    bf16=False,
    num_generations=4,           # 真正生效
    max_completion_length=400,   # 真正生效(GRPOConfig里对应参数名)
    temperature=0.7,             # 真正生效
    beta=0.1,                    # 真正生效,这是TRL里KL系数的参数名(不是kl_coef)
    report_to=["tensorboard"],
    remove_unused_columns=False,
)

# ---------- 6. 训练 ----------
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=combined_reward,
    processing_class=tokenizer,
)

print("开始训练...")
trainer.train()

print(f"保存模型到 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("完成")
