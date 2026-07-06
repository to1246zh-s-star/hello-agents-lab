import ast
import operator
import math
from tools import ToolRegistry, FunctionTool, ToolParameter

def my_calculate(expression: str) -> str:
    """安全的数学计算函数：只解析AST中明确允许的节点，不用eval"""
    if not expression.strip():
        return "计算表达式不能为空"

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    functions = {
        'sqrt': math.sqrt,
        'pi': math.pi,
    }

    try:
        node = ast.parse(expression, mode='eval')
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except Exception:
        return "计算失败，请检查表达式格式"

def _eval_node(node, operators, functions):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op))
        return op(left, right)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]

def create_calculator_registry():
    registry = ToolRegistry()
    tool = FunctionTool(
        name="my_calculator",
        description="安全的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=lambda expression: my_calculate(expression),
        parameters=[ToolParameter("expression", "string", "数学表达式", required=True)]
    )
    registry.register_tool(tool)
    return registry
