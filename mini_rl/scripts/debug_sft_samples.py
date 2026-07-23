from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/hello-agents-lab/mini_rl/models/sft_full_v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

questions = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
]

for q in questions:
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.8)
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("=" * 50)
    print(text)
    print("包含Final Answer:", "Final Answer" in text)
