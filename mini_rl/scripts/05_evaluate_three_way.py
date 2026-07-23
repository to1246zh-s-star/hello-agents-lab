import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_PATH = "/root/autodl-tmp/hf_cache/Qwen3-0.6B"
SFT_MODEL_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full_v2"
GRPO_ADAPTER_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/grpo_full_custom"

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="test")
eval_samples = dataset.select(range(30))  # 先用30条快速评估

def extract_final_answer(text):
    match = re.search(r"Final Answer:\s*([^\n]+)", text)
    if match:
        num_match = re.search(r"-?\d+\.?\d*", match.group(1))
        if num_match:
            return num_match.group()
    return None

def extract_ground_truth(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return None

def evaluate_model(model, tokenizer, name):
    correct = 0
    total = len(eval_samples)
    format_correct = 0
    total_length = 0

    for sample in eval_samples:
        question = sample["question"]
        gt = extract_ground_truth(sample["answer"])

        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs, max_new_tokens=400, do_sample=False
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        completion = text[len(prompt):]

        total_length += len(completion)

        pred = extract_final_answer(completion)
        if pred is not None:
            format_correct += 1
            try:
                if abs(float(pred) - float(gt)) < 1e-4:
                    correct += 1
            except ValueError:
                pass

    print(f"\n===== {name} =====")
    print(f"准确率: {correct}/{total} = {correct/total:.2%}")
    print(f"格式正确率: {format_correct}/{total} = {format_correct/total:.2%}")
    print(f"平均输出长度(字符): {total_length/total:.1f}")
    return {"accuracy": correct/total, "format_correctness": format_correct/total, "avg_length": total_length/total}

results = {}

print("加载预训练模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
results["预训练模型"] = evaluate_model(base_model, tokenizer, "预训练模型")
del base_model

print("加载SFT模型...")
sft_tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH)
results["SFT模型"] = evaluate_model(sft_model, sft_tokenizer, "SFT模型")

print("加载GRPO模型(SFT基座 + LoRA adapter)...")
grpo_model = PeftModel.from_pretrained(sft_model, GRPO_ADAPTER_PATH)
results["GRPO模型"] = evaluate_model(grpo_model, sft_tokenizer, "GRPO模型")

print("\n" + "=" * 50)
print("三方对比总结")
print("=" * 50)
print(f"{'模型':<15}{'准确率':<12}{'格式正确率':<12}{'平均长度':<12}")
for name, r in results.items():
    print(f"{name:<15}{r['accuracy']:<12.2%}{r['format_correctness']:<12.2%}{r['avg_length']:<12.1f}")
