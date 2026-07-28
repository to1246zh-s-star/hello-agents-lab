"""GAIA 分级评估 + Drop Rate（真实键名版）"""
from dotenv import load_dotenv; load_dotenv(override=True)
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import GAIAEvaluationTool

GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

llm = HelloAgentsLLM()
agent = SimpleAgent(name="GAIA-Agent", llm=llm, system_prompt=GAIA_SYSTEM_PROMPT)
tool = GAIAEvaluationTool()

acc = {}
for level in [1, 2, 3]:
    res = tool.run(agent=agent, level=level, max_samples=15, export_results=True)
    em = res['exact_match_rate']
    correct = res['exact_matches']
    total = res['total_samples']
    acc[level] = em
    print(f"Level {level}: 精确匹配={em:.2%}  部分匹配={res['partial_match_rate']:.2%}  ({correct}/{total})")

print("\n" + "="*50)
if acc.get(1) and acc.get(2):
    print(f"Drop 1->2: {(acc[1]-acc[2])/acc[1]:.1%}" if acc[1] else "Level1为0，Drop无意义")
if acc.get(2) and acc.get(3):
    print(f"Drop 2->3: {(acc[2]-acc[3])/acc[2]:.1%}" if acc[2] else "Level2为0，Drop无意义")
