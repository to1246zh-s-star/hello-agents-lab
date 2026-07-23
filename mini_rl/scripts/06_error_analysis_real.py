import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

SFT_MODEL_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full_v2"
GRPO_ADAPTER_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/grpo_full_custom"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

NUM_SAMPLES = 80

dataset = load_dataset("openai/gsm8k", "main", split="test")
eval_samples = dataset.select(range(NUM_SAMPLES))

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

def count_gt_steps(answer_text):
    reasoning = answer_text.split("####")[0]
    lines = [l for l in reasoning.strip().split("\n") if l.strip()]
    return len(lines)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def classify_error(question, prediction, ground_truth, has_format):
    if not has_format:
        return "格式错误"
    numbers_in_pred = re.findall(r"-?\d+\.?\d*", prediction)
    try:
        gt_val = float(ground_truth)
    except ValueError:
        return "理解错误"
    if any(abs(float(n) - gt_val) < 1e-4 for n in numbers_in_pred if is_number(n)):
        return "计算错误"
    reasoning_lines = len([l for l in prediction.split("\n") if l.strip()])
    if reasoning_lines <= 1:
        return "理解错误"
    return "推理错误"

def evaluate_and_analyze(model, tokenizer, name):
    error_types = {"计算错误": 0, "推理错误": 0, "理解错误": 0, "格式错误": 0}
    difficulty_groups = {"简单(1-2步)": [], "中等(3-4步)": [], "困难(5+步)": []}
    correct = 0
    errors_detail = []

    for idx, sample in enumerate(eval_samples):
        question = sample["question"]
        gt = extract_ground_truth(sample["answer"])
        gt_steps = count_gt_steps(sample["answer"])

        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        completion = text[len(prompt):]

        pred = extract_final_answer(completion)
        has_format = pred is not None
        is_correct = False
        if has_format:
            try:
                is_correct = abs(float(pred) - float(gt)) < 1e-4
            except ValueError:
                is_correct = False

        if is_correct:
            correct += 1
        else:
            err_type = classify_error(question, completion, gt, has_format)
            error_types[err_type] += 1
            errors_detail.append({
                "question": question[:80], "prediction": pred,
                "ground_truth": gt, "error_type": err_type,
            })

        if gt_steps <= 2:
            difficulty_groups["简单(1-2步)"].append(is_correct)
        elif gt_steps <= 4:
            difficulty_groups["中等(3-4步)"].append(is_correct)
        else:
            difficulty_groups["困难(5+步)"].append(is_correct)

        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx+1}/{NUM_SAMPLES}")

    total = len(eval_samples)
    total_errors = total - correct

    print(f"\n{'='*60}\n{name} —— 完整评估报告\n{'='*60}")
    print(f"总体准确率: {correct}/{total} = {correct/total:.2%}")
    print(f"总错误数: {total_errors}")
    print(f"\n错误类型分布:")
    for etype, count in error_types.items():
        pct = count / total_errors * 100 if total_errors > 0 else 0
        print(f"  {etype}: {count} ({pct:.1f}%)")
    print(f"\n不同难度的准确率:")
    for group_name, results in difficulty_groups.items():
        if len(results) > 0:
            acc = sum(results) / len(results)
            print(f"  {group_name}: {acc:.2%} ({len(results)}个样本)")

    return {"accuracy": correct/total, "error_types": error_types,
            "difficulty": {k: (sum(v)/len(v) if v else None, len(v)) for k, v in difficulty_groups.items()},
            "errors_sample": errors_detail[:5]}

print("加载SFT模型...")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(DEVICE)
sft_report = evaluate_and_analyze(sft_model, tokenizer, "SFT模型")

print("\n加载GRPO模型...")
grpo_model = PeftModel.from_pretrained(sft_model, GRPO_ADAPTER_PATH).to(DEVICE)
grpo_report = evaluate_and_analyze(grpo_model, tokenizer, "GRPO模型")

with open("error_analysis_report.json", "w", encoding="utf-8") as f:
    json.dump({"SFT": sft_report, "GRPO": grpo_report}, f, ensure_ascii=False, indent=2)
print("\n完整报告已保存到 error_analysis_report.json")
