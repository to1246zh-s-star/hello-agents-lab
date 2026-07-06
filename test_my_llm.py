from dotenv import load_dotenv
load_dotenv()

from my_llm import MyLLM

print("=== 测试自定义分支 ===")
llm1 = MyLLM(provider="my_custom_provider")
print("llm1 初始化成功:", type(llm1))

print()
print("=== 测试默认分支（走父类逻辑）===")
llm2 = MyLLM()
print("llm2 初始化成功:", type(llm2))
