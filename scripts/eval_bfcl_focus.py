"""对可疑类别放量验证：multiple(疑似真短板) + parallel(疑似噪声高)"""
from dotenv import load_dotenv; load_dotenv(override=True)
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import BFCLEvaluationTool
from statsmodels.stats.proportion import proportion_confint

llm = HelloAgentsLLM()
agent = SimpleAgent(name="BFCL-Agent", llm=llm)
tool = BFCLEvaluationTool()

print("\n放量验证 (50 样本 + Wilson 95%CI)\n" + "="*50)
for cat in ["multiple", "parallel"]:
    res = tool.run(agent=agent, category=cat, max_samples=50)
    c, t = res['correct_samples'], res['total_samples']
    lo, hi = proportion_confint(c, t, method='wilson')
    print(f"{cat:<14} {c}/{t} = {c/t:.1%}  95%CI [{lo:.1%}, {hi:.1%}]")
