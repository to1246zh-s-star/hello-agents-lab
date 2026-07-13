"""实验5：MemoryTool —— 统一入口，命令模式分发，整合全部四种记忆类型
"""
from datetime import datetime

from memory_item import MemoryItem, MemoryConfig
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from perceptual_memory import PerceptualMemory


class MemoryTool:
    def __init__(self, config: MemoryConfig = None, user_id: str = "default_user"):
        self.user_id = user_id
        # 每个user_id用独立的SQLite文件，天然做到"不同用户的情景记忆互不干扰"
        self.config = config or MemoryConfig(database_path=f"./mini_memory_{user_id}.db")
        self.working = WorkingMemory(self.config)
        self.episodic = EpisodicMemory(self.config)
        self.semantic = SemanticMemory(self.config)
        self.perceptual = PerceptualMemory(self.config)
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def execute(self, action: str, **kwargs) -> str:
        """统一入口：命令模式分发"""
        handler = {
            "add": self._add,
            "search": self._search,
            "forget": self._forget,
            "consolidate": self._consolidate,
            "stats": self._stats,
            "summary": self._summary,
        }.get(action)
        if handler is None:
            return f"❌ 未知操作: {action}"
        return handler(**kwargs)

    def _add(self, content: str, memory_type: str = "working", importance: float = 0.5,
              modality: str = "text", **metadata) -> str:
        item = MemoryItem(content=content, memory_type=memory_type, importance=importance, metadata=metadata)
        if memory_type == "working":
            self.working.add(item)
        elif memory_type == "episodic":
            self.episodic.add(item, session_id=self.current_session_id)
        elif memory_type == "semantic":
            self.semantic.add(item)
        elif memory_type == "perceptual":
            self.perceptual.add(item, modality=modality)
        else:
            return f"❌ 未知记忆类型: {memory_type}"
        return f"✅ 记忆已添加 (ID: {item.id[:8]}..., type={memory_type})"

    def _search(self, query: str, memory_type: str = None, modality: str = "text", limit: int = 5) -> str:
        results = []
        if memory_type in (None, "working"):
            results += [("working", m.content, m.importance) for m in self.working.retrieve(query, limit)]
        if memory_type in (None, "episodic"):
            results += [("episodic", r["content"], r["importance"]) for r in self.episodic.retrieve(query, limit)]
        if memory_type in (None, "semantic"):
            results += [("semantic", item.content, item.importance) for item, v, g in self.semantic.retrieve(query, limit)]
        if memory_type in (None, "perceptual"):
            results += [("perceptual", m.content, m.importance) for m in self.perceptual.retrieve(query, modality=modality, limit=limit)]

        if not results:
            return f"🔍 未找到与 '{query}' 相关的记忆"

        label = {"working": "工作记忆", "episodic": "情景记忆", "semantic": "语义记忆", "perceptual": "感知记忆"}
        lines = [f"🔍 找到 {len(results)} 条相关记忆:"]
        for i, (mtype, content, importance) in enumerate(results[:limit], 1):
            preview = content[:60] + "..." if len(content) > 60 else content
            lines.append(f"{i}. [{label[mtype]}] {preview} (重要性: {importance:.2f})")
        return "\n".join(lines)

    def _forget(self, strategy: str = "importance_based", threshold: float = 0.2) -> str:
        """本实验版只对working memory实现遗忘（其他三种类型的遗忘留作练习，对应习题3）"""
        if strategy == "importance_based":
            before = len(self.working.memories)
            self.working.memories = [m for m in self.working.memories if m.importance >= threshold]
            count = before - len(self.working.memories)
            return f"🧹 已遗忘 {count} 条工作记忆（策略: importance_based, 阈值={threshold}）"
        return f"❌ 本实验版暂未实现策略: {strategy}"

    def _consolidate(self, importance_threshold: float = 0.7) -> str:
        """把重要的工作记忆固化为情景记忆（working -> episodic）"""
        to_move = [m for m in self.working.memories if m.importance >= importance_threshold]
        for m in to_move:
            self.episodic.add(m, session_id=self.current_session_id)
        self.working.memories = [m for m in self.working.memories if m.importance < importance_threshold]
        return f"🔄 已整合 {len(to_move)} 条记忆为长期记忆（working → episodic，阈值={importance_threshold}）"

    def _stats(self) -> str:
        return (f"📊 工作记忆: {len(self.working.memories)} 条 | "
                f"语义记忆节点: {self.semantic.graph.number_of_nodes()} 个 | "
                f"感知记忆(text/image/audio): "
                f"{len(self.perceptual.stores['text'].items)}/{len(self.perceptual.stores['image'].items)}/{len(self.perceptual.stores['audio'].items)} | "
                f"会话ID: {self.current_session_id}")

    def _summary(self, limit: int = 10) -> list:
        """汇总最近的记忆，供实验9的generate_report生成JSON报告使用（返回结构化列表而不是字符串）"""
        summary = []
        for m in self.working.memories[-limit:]:
            summary.append({"type": "working", "content": m.content, "importance": m.importance, "timestamp": m.timestamp})
        for item in self.semantic.items[-limit:]:
            summary.append({"type": "semantic", "content": item.content, "importance": item.importance, "timestamp": item.timestamp})
        summary.sort(key=lambda x: x["timestamp"], reverse=True)
        return summary[:limit]


if __name__ == "__main__":
    import os
    for f in os.listdir("."):
        if f.startswith("mini_memory_") and f.endswith(".db"):
            os.remove(f)

    tool = MemoryTool(user_id="kk")

    print("--- 添加四种类型的记忆 ---")
    print(tool.execute("add", content="用户问了关于Python函数的问题", memory_type="working", importance=0.6))
    print(tool.execute("add", content="用户完成了机器学习作业4", memory_type="episodic", importance=0.9))
    print(tool.execute("add", content="Python是一种编程语言", memory_type="semantic", importance=0.8))
    print(tool.execute("add", content="编程语言属于计算机科学的一个分支", memory_type="semantic", importance=0.7))
    print(tool.execute("add", content="截图内容：Python函数定义代码", memory_type="perceptual", importance=0.6, modality="image"))

    print("\n--- 全类型混合搜索 ---")
    print(tool.execute("search", query="Python相关内容"))

    print("\n--- 只搜语义记忆（验证图检索能力）---")
    print(tool.execute("search", query="计算机科学", memory_type="semantic"))

    print("\n--- 只搜感知记忆的image模态 ---")
    print(tool.execute("search", query="Python截图", memory_type="perceptual", modality="image"))

    print("\n--- 统计信息 ---")
    print(tool.execute("stats"))

    print("\n--- summary（给学习报告用）---")
    for item in tool.execute("summary", limit=5):
        print(f"  [{item['type']}] {item['content'][:30]} (importance={item['importance']})")

    print("\n--- 验证user_id隔离：换个user_id应该是全新的独立db文件 ---")
    tool_bob = MemoryTool(user_id="bob")
    assert tool.config.database_path != tool_bob.config.database_path
    print(f"  kk的db:  {tool.config.database_path}")
    print(f"  bob的db: {tool_bob.config.database_path}")

    print("\n✅ 实验5自测通过：四种记忆类型全部整合进统一入口，且支持summary汇总和user_id隔离")
