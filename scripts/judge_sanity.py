"""毒丸测试：给裁判几道明显错误的题，看它能否识别"""
from dotenv import load_dotenv; load_dotenv()
from hello_agents import HelloAgentsLLM
import json

llm = HelloAgentsLLM()

# 一道故意错的：答案与解答矛盾 + 逻辑错误
POISON = """请评估这道AIME题目的正确性(1-5分)，只返回JSON {"correctness": N, "reason": "..."}
题目：设 x + y = 10, x - y = 4，求 x*y 的值。
答案：58
解答：由方程组解得 x=7, y=3，所以 x*y = 7*3 = 58。"""  # 7*3=21，不是58；且x=7,y=3代入x-y=4错误

for i in range(3):   # 跑3次看稳定性
    out = "".join(llm.think([{"role":"user","content":POISON}]))
    print(f"[{i}] {out.strip()[:200]}")
