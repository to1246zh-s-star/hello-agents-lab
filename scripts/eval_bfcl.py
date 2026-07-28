"""BFCL 渐进式 + 多类别评估"""
from dotenv import load_dotenv; load_dotenv()
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import BFCLEvaluationTool

llm = HelloAgentsLLM()
agent = SimpleAgent(name="BFCL-Agent", llm=llm)
tool = BFCLEvaluationTool()

print("\n" + "="*60 + "\n渐进式评估: simple_python\n" + "="*60)
for n in [5, 50]:
    res = tool.run(agent=agent, category="simple_python", max_samples=n)
    print(f"  样本={n:>4}  准确率={res['overall_accuracy']:.2%}  "
          f"({res['correct_samples']}/{res['total_samples']})")

# ---- 多类别横扫（每类 10 样本快速摸底）----
print("\n" + "="*60 + "\n多类别评估\n" + "="*60)
categories = ["simple_python", "multiple", "parallel", "irrelevance"]
summary = {}
for cat in categories:
    try:
        res = tool.run(agent=agent, category=cat, max_samples=10)
        summary[cat] = res['overall_accuracy']
        print(f"  {cat:<16} {res['overall_accuracy']:.2%}")
    except Exception as e:
        print(f"  {cat:<16} 失败: {e}")
        summary[cat] = None

print("\n分类准确率汇总:")
for cat, acc in summary.items():
    bar = "█" * int((acc or 0) * 20)
    print(f"  {cat:<16} {bar:<20} {acc:.2%}" if acc is not None else f"  {cat:<16} N/A")
