import time
import re
from agent import Agent

class PlanAndSolveAgent(Agent):
    def __init__(self, name, llm):
        super().__init__(name, llm)

    def run(self, user_input: str) -> str:
        self.add_message(user_input, "user")

        # 阶段1：制定计划
        plan_prompt = f"""请把下面这个问题拆解成若干个有序的子步骤（用数字编号列出），
不要直接给出最终答案，只给出解题步骤。

问题：{user_input}
"""
        plan_text = self._invoke_with_delay([{"role": "user", "content": plan_prompt}])
        print(f"\n[生成的计划]\n{plan_text}")

        steps = re.findall(r"\d+[\.\、]\s*(.+)", plan_text)
        if not steps:
            steps = [plan_text]  # 万一没解析出编号，就把整段当一步

        # 阶段2：逐步执行
        executed_results = []
        for i, step in enumerate(steps):
            exec_prompt = f"""原问题：{user_input}
完整计划：{plan_text}
之前步骤的执行结果：{executed_results}

现在请执行第{i+1}步：{step}
只需要给出这一步的具体结果，不用重复之前的内容。
"""
            step_result = self._invoke_with_delay([{"role": "user", "content": exec_prompt}])
            print(f"\n[执行第{i+1}步: {step}]\n{step_result}")
            executed_results.append(step_result)

        # 阶段3：汇总最终答案
        summary_prompt = f"""原问题：{user_input}
各步骤执行结果：{executed_results}

请综合以上信息，给出最终完整答案：
"""
        final_answer = self._invoke_with_delay([{"role": "user", "content": summary_prompt}])
        self.add_message(final_answer, "assistant")
        return final_answer

    def _invoke_with_delay(self, messages):
        time.sleep(3)
        return self.llm.invoke(messages)
