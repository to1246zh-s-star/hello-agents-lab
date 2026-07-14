"""
仅用于本地验证的轻量mock,接口形状对齐第八章MemoryTool/RAGTool的.run({"action":"search",...})调用方式。
真实项目里这里应该import第七、八章写好的真实MemoryTool/RAGTool。
"""
from datetime import datetime


class MockMemoryTool:
    def __init__(self):
        self._store = []

    def run(self, params: dict):
        if params.get("action") == "add":
            self._store.append({
                "content": params["content"],
                "timestamp": datetime.now(),
                "importance": params.get("importance", 0.5),
            })
            return {"status": "ok"}
        elif params.get("action") == "search":
            # 简化:直接返回全部(真实版本应做相关性排序,这里只为验证ContextBuilder的Select阶段)
            return {"results": self._store}


class MockRAGTool:
    def __init__(self, knowledge_base_path=None):
        self._docs = [
            "Pandas内存优化的核心策略包括: 1.使用合适的数据类型(如category代替object) 2.分块读取大文件 3.使用chunksize参数",
            "数据类型优化可以显著减少内存占用。例如,将int64降级为int32可以节省50%的内存。",
        ]

    def run(self, params: dict):
        if params.get("action") == "search":
            return {"results": [{"content": d, "score": 0.8, "timestamp": datetime.now()} for d in self._docs]}


class Message:
    def __init__(self, content, role, timestamp=None):
        self.content = content
        self.role = role
        self.timestamp = timestamp or datetime.now()
