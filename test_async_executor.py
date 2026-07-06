import time
import asyncio
from tools import ToolRegistry, FunctionTool, ToolParameter
from async_tool_executor import AsyncToolExecutor

def slow_tool(x: str) -> str:
    time.sleep(2)  # 模拟一个耗时2秒的工具（比如真实的网络请求）
    return f"处理完成: {x}"

registry = ToolRegistry()
registry.register_tool(FunctionTool(
    name="slow_tool", description="模拟耗时操作",
    func=slow_tool, parameters=[ToolParameter("x", "string", "输入", required=True)]
))

executor = AsyncToolExecutor(registry)

tasks = [
    {"tool_name": "slow_tool", "input_data": "任务A"},
    {"tool_name": "slow_tool", "input_data": "任务B"},
    {"tool_name": "slow_tool", "input_data": "任务C"},
]

start = time.time()
results = asyncio.run(executor.execute_tools_parallel(tasks))
elapsed = time.time() - start

print("\n结果:", results)
print(f"\n并行耗时: {elapsed:.1f}秒（如果串行执行，理论上需要约6秒）")
