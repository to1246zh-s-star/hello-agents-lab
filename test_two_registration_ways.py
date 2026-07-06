from tools import ToolRegistry, CalculatorTool, register_function, ToolParameter
import json

registry = ToolRegistry()

# 方式1: 继承Tool基类
registry.register_tool(CalculatorTool())

# 方式2: 函数直接注册（用同一个计算逻辑，方便对比）
def simple_calculator(expression):
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"计算错误: {e}"

register_function(
    registry, name="calculator_v2",
    description="计算数学表达式（函数注册版）",
    func=simple_calculator,
    parameters=[ToolParameter("expression", "string", "要计算的数学表达式", required=True)]
)

print("=== 两种方式生成的schema对比 ===")
for schema in registry.get_openai_schemas():
    print(json.dumps(schema, ensure_ascii=False, indent=2))
    print()
