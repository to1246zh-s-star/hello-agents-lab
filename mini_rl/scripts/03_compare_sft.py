from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_PATH = "/root/autodl-tmp/hf_cache/Qwen3-0.6B"
SFT_MODEL_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full"

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

print("=" * 50)
print("预训练模型输出:")
print("=" * 50)
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
inputs = base_tokenizer(prompt, return_tensors="pt")
outputs = base_model.generate(**inputs, max_new_tokens=200)
print(base_tokenizer.decode(outputs[0], skip_special_tokens=False))

print("=" * 50)
print("SFT模型输出:")
print("=" * 50)
sft_tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH)
inputs = sft_tokenizer(prompt, return_tensors="pt")
outputs = sft_model.generate(**inputs, max_new_tokens=200)
print(sft_tokenizer.decode(outputs[0], skip_special_tokens=False))
