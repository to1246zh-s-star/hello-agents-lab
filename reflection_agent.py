import time
from agent import Agent

class ReflectionAgent(Agent):
    def __init__(self, name, llm, max_rounds: int = 3):
        super().__init__(name, llm)
        self.max_rounds = max_rounds

    def run(self, user_input: str) -> str:
        self.add_message(user_input, "user")

        # 第一轮：生成初稿
        draft = self._invoke_with_delay([
            {"role": "user", "content": f"请完成以下任务：{user_input}"}
        ])
        print(f"\n[初稿]\n{draft}")

        for round_num in range(self.max_rounds):
            # 批评者：挑毛病
            critique_prompt = f"""请仔细审查以下答案，指出其中的问题或可以改进的地方。
如果这个答案已经足够好，不需要再改进，请直接回复"满意"。

任务：{user_input}
当前答案：{draft}
"""
            critique = self._invoke_with_delay([{"role": "user", "content": critique_prompt}])
            print(f"\n[第{round_num+1}轮批评]\n{critique}")

            if "满意" in critique:
                break

            # 生成者：根据批评意见改进
            refine_prompt = f"""请根据以下批评意见，改进你的答案。

任务：{user_input}
上一版答案：{draft}
批评意见：{critique}

请给出改进后的完整答案：
"""
            draft = self._invoke_with_delay([{"role": "user", "content": refine_prompt}])
            print(f"\n[第{round_num+1}轮改进后]\n{draft}")

        self.add_message(draft, "assistant")
        return draft

    def _invoke_with_delay(self, messages):
        time.sleep(3)  # 避免连续请求触发限流
        return self.llm.invoke(messages)
