"""
完整整合:书本三大件(ContextBuilder/NoteTool/TerminalTool) + 五项升级
(TF-IDF相关性/质量评估/子代理隔离/多智能体路由/Prompt Caching排布)。

真实使用时,把MockMemoryTool/MockRAGTool替换成第七、八章写好的真实实现即可,
接口是完全对齐的(都是.run({"action": ...})这套调用方式)。
"""
from relevance_scorer import TfidfContextBuilder
from context_builder import ContextConfig
from note_tool import NoteTool
from terminal_tool import TerminalTool
from context_quality import ContextQualityEvaluator
from prompt_cache_layout import reorder_for_prompt_caching


class ContextAwareAgentV2:
    def __init__(self, project_name: str, memory_tool=None, rag_tool=None, max_tokens: int = 4000):
        self.project_name = project_name
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool

        self.note_tool = NoteTool(workspace=f"./{project_name}_notes")
        self.terminal_tool = TerminalTool(workspace=f"./{project_name}_workspace")
        self.context_builder = TfidfContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=self.rag_tool,
            config=ContextConfig(max_tokens=max_tokens),
        )
        self.quality_evaluator = ContextQualityEvaluator()
        self.conversation_history = []

    def build_context(self, user_query: str, system_instructions: str = "") -> dict:
        # 1. 从NoteTool检索相关笔记,转成custom_packets(对齐书里9.4.4节的桥接方式)
        note_packets = self._notes_to_packets(user_query)

        # 2. 用ContextBuilder组装(内部会自动调用memory_tool/rag_tool)
        context_text = self.context_builder.build(
            user_query=user_query,
            conversation_history=self.conversation_history,
            system_instructions=system_instructions,
            custom_packets=note_packets,
        )

        # 3. 质量评估(习题2扩展功能)
        quality_report = self.quality_evaluator.evaluate(
            context_text, user_query, self.context_builder.config
        )

        # 4. Prompt caching重排(升级5)
        cache_layout = reorder_for_prompt_caching(context_text)

        return {
            "context_text": context_text,
            "quality_report": quality_report,
            "cache_layout": cache_layout,
        }

    def _notes_to_packets(self, query: str, limit: int = 3):
        from context_builder import ContextPacket
        from datetime import datetime

        try:
            notes = self.note_tool.run({"action": "search", "query": query, "limit": limit})
        except Exception as e:
            print(f"[WARNING] 笔记检索失败: {e}")
            return []

        packets = []
        for note in notes:
            content = f"[笔记:{note['title']}]\n{note['content']}"
            packets.append(ContextPacket(
                content=content,
                timestamp=datetime.fromisoformat(note["updated_at"]),
                token_count=self.context_builder._count_tokens(content),
                relevance_score=0.5,  # 交给TF-IDF重新计算,而不是硬编码0.75
                metadata={"type": "note", "note_type": note["type"], "note_id": note["note_id"]},
            ))
        return packets
