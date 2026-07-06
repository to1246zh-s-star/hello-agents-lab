import os
import random
from tools import ToolRegistry, FunctionTool, ToolParameter

class MyAdvancedSearchTool:
    """
    模拟版多源搜索工具，用于在没有真实API Key时验证"智能后端选择+降级"的设计思路。
    真实场景下把 _search_with_tavily / _search_with_serpapi 换成书里7.5.3节的真实实现即可。
    """
    def __init__(self):
        self.name = "my_advanced_search"
        self.description = "智能搜索工具，模拟多源整合与降级机制"
        self.search_sources = []
        self._setup_search_sources()

    def _setup_search_sources(self):
        # 模拟检测：真实场景这里应该是 os.getenv("TAVILY_API_KEY") 等
        if os.getenv("MOCK_TAVILY_ENABLED", "true") == "true":
            self.search_sources.append("tavily")
            print("✅ [模拟] Tavily搜索源已启用")
        if os.getenv("MOCK_SERPAPI_ENABLED", "true") == "true":
            self.search_sources.append("serpapi")
            print("✅ [模拟] SerpApi搜索源已启用")
        print(f"🔧 可用搜索源: {', '.join(self.search_sources) or '无'}")

    def search(self, query: str) -> str:
        if not query.strip():
            return "❌ 错误: 搜索查询不能为空"
        if not self.search_sources:
            return "❌ 没有可用的搜索源，请配置TAVILY_API_KEY或SERPAPI_API_KEY"

        print(f"🔍 开始智能搜索: {query}")

        for source in self.search_sources:
            try:
                # 模拟第一个后端有10%概率"失败"，触发降级到下一个后端
                if source == "tavily" and random.random() < 0.3:
                    raise RuntimeError("模拟Tavily超时")

                if source == "tavily":
                    return f"📊 [模拟Tavily结果] 关于'{query}'的AI摘要答案：这是一个模拟的搜索结果示例。"
                elif source == "serpapi":
                    return f"🌐 [模拟SerpApi结果] 关于'{query}'的Google搜索结果：这是一个模拟的搜索结果示例。"
            except Exception as e:
                print(f"⚠️ {source} 搜索失败: {e}，尝试降级到下一个源")
                continue

        return "❌ 所有搜索源都失败了"

def create_advanced_search_registry():
    registry = ToolRegistry()
    search_tool_instance = MyAdvancedSearchTool()
    tool = FunctionTool(
        name="my_advanced_search",
        description="模拟的高级搜索工具，展示多源整合与降级设计",
        func=lambda query: search_tool_instance.search(query),
        parameters=[ToolParameter("query", "string", "搜索查询词", required=True)]
    )
    registry.register_tool(tool)
    return registry
