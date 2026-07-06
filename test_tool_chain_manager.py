from tools import ToolRegistry
from my_calculator_tool import create_calculator_registry
from my_advanced_search import create_advanced_search_registry
from tool_chain_manager import ToolChainManager, create_research_chain

# 把两个独立注册表里的工具，合并进同一个registry
calc_registry = create_calculator_registry()
search_registry = create_advanced_search_registry()

merged_registry = ToolRegistry()
for tool in calc_registry._tools.values():
    merged_registry.register_tool(tool)
for tool in search_registry._tools.values():
    merged_registry.register_tool(tool)

manager = ToolChainManager(merged_registry)
manager.register_chain(create_research_chain())

print("\n已注册工具链:", manager.list_chains())
print()

result = manager.execute_chain("research_and_calculate", "Python编程语言的历史")
print("\n最终结果:", result)
