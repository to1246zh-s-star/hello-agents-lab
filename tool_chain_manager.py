from typing import List, Dict, Any
from tools import ToolRegistry

class ToolChain:
    """工具链 - 支持多个工具的顺序执行"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        })

    def execute(self, registry: ToolRegistry, initial_input: str, context: Dict[str, Any] = None) -> str:
        context = context or {}
        context["input"] = initial_input
        print(f"🔗 开始执行工具链: {self.name}")

        for i, step in enumerate(self.steps, 1):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]

            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                return f"❌ 工具链执行失败: 模板变量 {e} 未找到"

            print(f"  步骤{i}: 使用 {tool_name} 处理 '{tool_input[:50]}...'")
            result = registry.execute_tool(tool_name, tool_input)
            context[output_key] = result
            print(f"  ✅ 步骤{i}完成，结果长度: {len(result)} 字符")

        final_result = context[self.steps[-1]["output_key"]]
        print(f"🎉 工具链 '{self.name}' 执行完成")
        return final_result

class ToolChainManager:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        self.chains[chain.name] = chain
        print(f"✅ 工具链 '{chain.name}' 已注册")

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        if chain_name not in self.chains:
            return f"❌ 工具链 '{chain_name}' 不存在"
        return self.chains[chain_name].execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        return list(self.chains.keys())

def create_research_chain() -> ToolChain:
    """搜索 -> 计算 的研究链"""
    chain = ToolChain(name="research_and_calculate", description="搜索信息并进行相关计算")
    chain.add_step(tool_name="my_advanced_search", input_template="{input}", output_key="search_result")
    chain.add_step(tool_name="my_calculator", input_template="2 + 2", output_key="calculation_result")
    return chain
