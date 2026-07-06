from tools import ToolRegistry, CalculatorTool

registry = ToolRegistry()
registry.register_tool(CalculatorTool())

print(registry.get_tools_description())
print()
print("execute_tool测试:", registry.execute_tool("calculator", "2+3*4"))
