from tools import ToolRegistry, CalculatorTool
from tool_chain import ToolChain, ToolChainStep

registry = ToolRegistry()
registry.register_tool(CalculatorTool())

chain = ToolChain(registry, steps=[
    ToolChainStep(tool_name="calculator", input_template="23+7"),
    ToolChainStep(tool_name="calculator", input_template="{previous}*3"),
    ToolChainStep(tool_name="calculator", input_template="{previous}/5"),
])

final_result = chain.run(original_input="不需要用到这个变量")
print("\n工具链最终结果:", final_result)
print("Python直接验算:", (23+7)*3/5)
