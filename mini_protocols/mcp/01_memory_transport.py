from hello_agents.tools import MCPTool

# Memory Transport:不指定server_command,使用内置演示服务器
mcp_tool = MCPTool()

# 列出可用工具
result = mcp_tool.run({"action": "list_tools"})
print("可用工具:", result)

# 调用工具(内置演示服务器提供加法等基础工具)
result = mcp_tool.run({
    "action": "call_tool",
    "tool_name": "add",
    "arguments": {"a": 10, "b": 20}
})
print("MCP计算结果:", result)  # 书中预期输出: 30.0
