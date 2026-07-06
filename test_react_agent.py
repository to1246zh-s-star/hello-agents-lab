from dotenv import load_dotenv
load_dotenv()

from hello_agents import HelloAgentsLLM
from react_agent import ReActAgent
from calculator_tool import calculator

llm = HelloAgentsLLM()
agent = ReActAgent(name="计算助手", llm=llm, tools={"calculator": calculator}, max_steps=5)

result = agent.run("请计算 (23+7)*3 再除以5")
print("\n最终答案:", result)
print("Python直接验算:", (23+7)*3/5)
