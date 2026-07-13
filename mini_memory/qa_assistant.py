"""实验8：整合案例 —— CLI版智能文档问答助手
组合 MemoryTool + AdvancedRAGTool
"""
from memory_tool import MemoryTool
from rag_advanced import AdvancedRAGTool


class QAAssistant:
    def __init__(self):
        self.memory = MemoryTool()
        self.rag = AdvancedRAGTool()
        self.stats = {"questions_asked": 0, "notes": 0}

    def load_knowledge(self, text: str):
        return self.rag.add_text(text)

    def ask(self, question: str, use_advanced: bool = True) -> str:
        # 1. 提问记入工作记忆
        self.memory.execute("add", content=f"提问: {question}", memory_type="working", importance=0.6)

        # 2. RAG检索生成答案
        if use_advanced:
            hits = self.rag.search_expanded(question, limit=3)
        else:
            hits = self.rag.search(question, limit=3)

        if not hits:
            answer = "⚠️ 知识库中没有找到相关内容"
        else:
            context = "\n\n".join([f"[参考{i+1}] {h['content']}" for i, h in enumerate(hits)])
            messages = [
                {"role": "system", "content": "你是一个基于给定参考资料回答问题的助手，只根据参考资料回答，不要编造。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
            ]
            resp = self.rag.client.chat.completions.create(model=self.rag.model, messages=messages)
            answer = resp.choices[0].message.content

        # 3. 问答事件记入情景记忆
        self.memory.execute("add", content=f"关于'{question}'的学习", memory_type="episodic", importance=0.7)
        self.stats["questions_asked"] += 1
        return answer

    def add_note(self, content: str):
        # 笔记是抽象知识，对应书里"add_note存入语义记忆"的设计（现在语义记忆已实现，不再用working占位）
        self.memory.execute("add", content=content, memory_type="semantic", importance=0.8)
        self.stats["notes"] += 1
        return "✅ 笔记已记录"

    def recall(self, query: str) -> str:
        return self.memory.execute("search", query=query)

    def report(self) -> str:
        return f"📊 学习报告 | 提问次数: {self.stats['questions_asked']} | 笔记数: {self.stats['notes']} | {self.memory.execute('stats')}"


if __name__ == "__main__":
    import os
    if os.path.exists("./mini_memory.db"):
        os.remove("./mini_memory.db")

    assistant = QAAssistant()
    assistant.load_knowledge("""# Python简介
Python是一种高级编程语言，由Guido van Rossum于1991年首次发布，设计哲学强调代码可读性。

# 机器学习基础
机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式，主要分为监督学习、无监督学习、强化学习三类。
""")

    print("=== 第一轮提问 ===")
    print(assistant.ask("Python是谁发明的？"))

    print("\n=== 第二轮提问 ===")
    print(assistant.ask("机器学习有哪几类？"))

    print("\n=== 添加学习笔记 ===")
    print(assistant.add_note("重点：Python强调可读性，机器学习分三大类"))

    print("\n=== 回顾学习历程 ===")
    print(assistant.recall("Python"))

    print("\n=== 生成报告 ===")
    print(assistant.report())

    print("\n✅ 实验8自测通过：完整链路（RAG问答 + 记忆记录 + 回顾 + 报告）跑通")
