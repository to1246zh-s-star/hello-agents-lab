from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable

class ToolParameter:
    def __init__(self, name, type_, description, required=True, default=None):
        self.name = name
        self.type = type_
        self.description = description
        self.required = required
        self.default = default

class Tool(ABC):
    name: str = "unnamed_tool"
    description: str = ""

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        ...

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        ...

    def to_openai_schema(self) -> dict:
        """把工具描述转换成OpenAI function calling要求的标准格式"""
        properties = {}
        required = []
        for p in self.get_parameters():
            prop = {"type": p.type, "description": p.description}
            if p.default is not None:
                prop["description"] = f"{p.description} (默认: {p.default})"
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

class CalculatorTool(Tool):
    name = "calculator"
    description = "计算数学表达式，输入如 2+3*4"

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("expression", "string", "要计算的数学表达式", required=True)]

    def run(self, parameters: Dict[str, Any]) -> str:
        expression = parameters.get("expression", "")
        try:
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

class FunctionTool(Tool):
    """把一个普通函数包装成Tool，用于函数直接注册的场景"""
    def __init__(self, name, description, func, parameters: List[ToolParameter]):
        self.name = name
        self.description = description
        self._func = func
        self._parameters = parameters

    def get_parameters(self) -> List[ToolParameter]:
        return self._parameters

    def run(self, parameters: Dict[str, Any]) -> str:
        return self._func(**parameters)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        if tool.name in self._tools:
            print(f"⚠️ 警告: 工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def get_tool(self, name: str) -> Tool:
        return self._tools.get(name)

    def has_tools(self) -> bool:
        return len(self._tools) > 0

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_openai_schemas(self) -> List[dict]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """获取所有可用工具的格式化描述，用于构建Agent提示词"""
        descriptions = [f"- {t.name}: {t.description}" for t in self._tools.values()]
        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def execute_tool(self, name: str, input_str: str) -> str:
        """简化调用接口：直接传字符串输入，自动适配到run()需要的字典参数"""
        tool = self.get_tool(name)
        if tool is None:
            return f"❌ 工具 '{name}' 不存在"
        params = tool.get_parameters()
        key = params[0].name if params else "input"
        return tool.run({key: input_str})


def register_function(registry: 'ToolRegistry', name, description, func, parameters: List[ToolParameter]):
    """函数直接注册的便捷方式"""
    tool = FunctionTool(name, description, func, parameters)
    registry.register_tool(tool)
