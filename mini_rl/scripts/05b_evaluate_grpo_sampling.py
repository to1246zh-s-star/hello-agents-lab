import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

SFT_MODEL_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full_v2"
GRPO_ADAPTER_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/grpo_full_custom"

dataset = load_dataset("openai/gsm8k", "main", split="test")
eval_samples = dataset.select(range(30))

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

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH)
grpo_model = PeftModel.from_pretrained(base_model, GRPO_ADAPTER_PATH)

# 每题采样4次,只要有一次对就算Accuracy@4
correct_at_1 = 0
correct_at_4 = 0
total = len(eval_samples)

for sample in eval_samples:
    question = sample["question"]
    gt = extract_ground_truth(sample["answer"])
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    any_correct = False
    for i in range(4):
        outputs = grpo_model.generate(
            **inputs, max_new_tokens=400, do_sample=True, temperature=0.7
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        completion = text[len(prompt):]
        pred = extract_final_answer(completion)
        if pred is not None:
            try:
                if abs(float(pred) - float(gt)) < 1e-4:
                    if i == 0:
                        correct_at_1 += 1
                    any_correct = True
            except ValueError:
                pass
    if any_correct:
        correct_at_4 += 1

print(f"Accuracy@1(采样): {correct_at_1}/{total} = {correct_at_1/total:.2%}")
print(f"Accuracy@4(采样): {correct_at_4}/{total} = {correct_at_4/total:.2%}")
