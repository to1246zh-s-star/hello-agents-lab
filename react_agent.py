import re
from agent import Agent

REACT_PROMPT_TEMPLATE = """你是一个可以使用工具的助手。你可以使用以下工具：
{tools}

请严格按照以下格式回答，每次只能执行一个步骤：
Thought: 分析当前该做什么
Action: 工具名[工具输入]  （如果不需要工具，直接写 Finish[最终答案]）

历史记录：
{history}

问题：{question}
"""

class ReActAgent(Agent):
    def __init__(self, name, llm, tools: dict, max_steps: int = 5):
        super().__init__(name, llm)
        self.tools = tools  # 形如 {"calculator": calculator函数}
        self.max_steps = max_steps

    def run(self, user_input: str) -> str:
        history_text = ""
        tools_desc = "\n".join([f"- {name}: 一个可调用的工具" for name in self.tools])

        for step in range(self.max_steps):
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                history=history_text,
                question=user_input
            )
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            print(f"\n--- 第{step+1}步 ---\n{response}")

            # 检查是否已经给出最终答案
            finish_match = re.search(r"Finish\[(.*?)\]", response)
            if finish_match:
                return finish_match.group(1)

            # 解析 Action，判断要不要调用工具
            action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", response)
            if action_match:
                tool_name = action_match.group(1)
                tool_input = action_match.group(2)

                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_input)
                else:
                    observation = f"错误：未知工具 {tool_name}"

                history_text += f"\n{response}\nObservation: {observation}"
            else:
                # 模型没按格式输出，把这次响应也记进历史，避免死循环
                history_text += f"\n{response}"

        return "达到最大步数限制，未能得到最终答案。"
