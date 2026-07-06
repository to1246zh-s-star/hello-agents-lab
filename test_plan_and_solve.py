from dotenv import load_dotenv
load_dotenv()

from hello_agents import HelloAgentsLLM
from plan_and_solve_agent import PlanAndSolveAgent

llm = HelloAgentsLLM()
agent = PlanAndSolveAgent(name="规划助手", llm=llm)

result = agent.run("小明有20元，买了3支笔每支2元，又买了一本书花了8元，请问小明还剩多少钱？")
print("\n=== 最终答案 ===")
print(result)
