from dotenv import load_dotenv
load_dotenv()

from my_llm import MyLLM

llm = MyLLM(provider="my_custom_provider")

response = llm.invoke([{"role": "user", "content": "用一句话介绍你自己"}])
print("模型回复:", response)
