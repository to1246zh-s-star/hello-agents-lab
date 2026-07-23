from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.8)
text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(text)
print("=" * 30)
print("是否包含Final Answer:", "Final Answer" in text)
print("生成的token数:", outputs.shape[1] - inputs['input_ids'].shape[1])
