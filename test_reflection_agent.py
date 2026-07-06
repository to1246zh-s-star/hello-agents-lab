from dotenv import load_dotenv
load_dotenv()

from hello_agents import HelloAgentsLLM
from reflection_agent import ReflectionAgent

llm = HelloAgentsLLM()
agent = ReflectionAgent(name="反思助手", llm=llm, max_rounds=2)

result = agent.run("用80字左右写一段介绍杭州西湖的文案")
print("\n=== 最终答案 ===")
print(result)
