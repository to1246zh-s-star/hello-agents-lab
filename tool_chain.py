from typing import List, Dict, Any
from tools import ToolRegistry

class ToolChainStep:
    def __init__(self, tool_name: str, input_template: str):
        """
        input_template 支持用 {previous} 引用上一步的输出，
        或者用 {original_input} 引用最初的用户输入
        """
        self.tool_name = tool_name
        self.input_template = input_template

class ToolChain:
    def __init__(self, registry: ToolRegistry, steps: List[ToolChainStep]):
        self.registry = registry
        self.steps = steps

    def run(self, original_input: str) -> str:
        previous_output = ""
        for i, step in enumerate(self.steps):
            tool_input = step.input_template.format(
                previous=previous_output,
                original_input=original_input
            )
            tool = self.registry.get_tool(step.tool_name)
            if tool is None:
                raise ValueError(f"工具链第{i+1}步找不到工具: {step.tool_name}")

            result = tool.run({"expression": tool_input}) if step.tool_name == "calculator" else tool.run({"input": tool_input})
            print(f"[工具链第{i+1}步: {step.tool_name}] 输入={tool_input} -> 输出={result}")
            previous_output = result

        return previous_output
