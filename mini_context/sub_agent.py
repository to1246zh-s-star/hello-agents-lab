"""
升级3:子代理架构(Sub-agent isolation)的真实代码实现。

说明:书本9.2.3节只在理论层面讨论了子代理架构("由主代理负责高层规划,
多个专长子代理在干净的上下文窗口中各自深挖,最后仅回传1000-2000 tokens的凝练摘要"),
没有给出配套代码。这里按这段理论 + 2026年业界"write/select/compress/isolate"
四件套里的isolate实践,补上代码实现。

和NoteTool的关系:子代理的"凝练摘要"应该写入NoteTool而不是只留在内存里,
这样主代理reset上下文之后依然能从笔记里找回子代理的结论(对应9.4节"结构化笔记"的价值)。
"""
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI


class SubAgent:
    """在一个全新的、干净的上下文里执行一个子任务,只返回1000-2000 tokens量级的凝练摘要。"""

    def __init__(self, model="Qwen/Qwen3-VL-8B-Instruct"):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        )
        self.model = model

    def execute(self, subtask_description: str, evidence: str, max_summary_tokens: int = 500) -> dict:
        """
        subtask_description: 独立的子任务指令,不携带主线程历史包袱
        evidence: 只传这个子任务需要的证据(比如TerminalTool探索出的文件内容)
        """
        messages = [
            {"role": "system", "content": (
                "你是一个专注的子任务执行者,在一个干净、独立的上下文里工作。"
                f"完成任务后,用不超过{max_summary_tokens}字的凝练摘要总结你的发现和结论,"
                "不要输出探索过程的细节,只保留结论、关键数据、和主代理需要知道的下一步建议。"
            )},
            {"role": "user", "content": f"任务:{subtask_description}\n\n证据:\n{evidence}"},
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.3)
        summary = resp.choices[0].message.content.strip()
        return {"subtask": subtask_description, "summary": summary}


class MainAgentOrchestrator:
    """
    主代理:分派子任务给多个SubAgent,把子代理返回的凝练摘要写入NoteTool持久化,
    主线程自身的上下文只积累这些摘要,不积累任何子代理的执行细节。
    """

    def __init__(self, note_tool, model="Qwen/Qwen3-VL-8B-Instruct"):
        self.sub_agent = SubAgent(model=model)
        self.note_tool = note_tool  # 复用9.4节实现的NoteTool

    def dispatch(self, subtasks: list) -> list:
        """subtasks: [{"description": ..., "evidence": ...}, ...]"""
        results = []
        for task in subtasks:
            result = self.sub_agent.execute(task["description"], task["evidence"])
            # 凝练摘要写入NoteTool,而不是只留在Python变量里
            self.note_tool.run({
                "action": "create",
                "title": f"子代理结论:{task['description'][:30]}",
                "content": result["summary"],
                "note_type": "conclusion",
                "tags": ["sub_agent_output"],
            })
            results.append(result)
        return results
