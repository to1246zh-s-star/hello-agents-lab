"""
对齐书里hello_agents.HelloAgentsLLM的最小封装:统一.invoke(prompt) -> str接口。
真实版走ModelScope(和第七、八章、sub_agent.py保持一致的调用方式)。
MockLLM仅用于本地没有.env/API Key时验证CodebaseMaintainer的流程是否走得通。
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class HelloAgentsLLM:
    def __init__(self, model: str = None):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        )
        self.model = model or os.getenv("LLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

    def invoke(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


class MockLLM:
    """仅用于本地无API Key环境下验证CodebaseMaintainer的流程结构,不产生真实回答。"""

    def invoke(self, prompt: str) -> str:
        if "问题" in prompt or "分析" in prompt:
            return "分析发现一个问题:模块耦合度较高,建议拆分。这是一个mock回答,用于验证流程结构。"
        if "计划" in prompt or "任务" in prompt:
            return "下一步任务计划:1. 补充测试 2. 重构核心模块。这是一个mock回答,用于验证流程结构。"
        return "这是mock LLM的示意回答,用于在没有真实API Key时验证CodebaseMaintainer的完整调用链路是否走通。"
