"""
升级4:多智能体上下文路由(Context Routing)。

行业术语(2026年多篇上下文工程综述提到的反模式):
- "Broadcaster"反模式:把同一份完整上下文原样发给流水线里的每一个子Agent,
  导致每个Agent都要处理大量与自己任务无关的信息,成本和噪音同时增加。
- "Hoarder"反模式:长会话中从不清理上下文,只增不减。

这里针对"Broadcaster"给出对策:每个子Agent通过自己的role_config,
只拿到"跟它这个角色相关"的那一份定制化上下文,而不是主Agent的完整上下文。
"""
from typing import Dict, List
from context_builder import ContextConfig


class RoleBasedContextRouter:
    """
    为流水线里的每个角色分别构建定制化上下文,而不是广播同一份完整上下文。
    每个角色可以有自己的:
    - max_tokens(角色需要的上下文量不同,不该一刀切)
    - system_instructions(角色定位不同)
    - 是否需要memory_tool/rag_tool检索(比如"总结员"角色可能完全不需要RAG证据)
    """

    def __init__(self, builder_cls, memory_tool=None, rag_tool=None):
        self.builder_cls = builder_cls  # 传入TfidfContextBuilder或原始ContextBuilder
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.role_configs: Dict[str, dict] = {}

    def register_role(self, role_name: str, max_tokens: int = 2000,
                       use_memory: bool = True, use_rag: bool = True,
                       system_instructions: str = ""):
        self.role_configs[role_name] = {
            "config": ContextConfig(max_tokens=max_tokens),
            "use_memory": use_memory,
            "use_rag": use_rag,
            "system_instructions": system_instructions,
        }

    def route(self, role_name: str, user_query: str, conversation_history: List = None,
               custom_packets: List = None) -> str:
        if role_name not in self.role_configs:
            raise ValueError(f"未注册的角色: {role_name}")

        rc = self.role_configs[role_name]
        builder = self.builder_cls(
            memory_tool=self.memory_tool if rc["use_memory"] else None,
            rag_tool=self.rag_tool if rc["use_rag"] else None,
            config=rc["config"],
        )
        return builder.build(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=rc["system_instructions"],
            custom_packets=custom_packets,
        )

    def route_report(self, role_name: str, user_query: str, **kwargs) -> dict:
        """附带token统计,方便对比"广播模式"vs"路由模式"的成本差异"""
        context = self.route(role_name, user_query, **kwargs)
        builder = self.builder_cls(config=self.role_configs[role_name]["config"])
        return {"role": role_name, "context": context, "tokens": builder._count_tokens(context)}
