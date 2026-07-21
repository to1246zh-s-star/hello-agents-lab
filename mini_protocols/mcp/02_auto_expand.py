from dotenv import load_dotenv
load_dotenv()

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())

# 无需任何配置,自动使用内置演示服务器
mcp_tool = MCPTool(name="calculator")
agent.add_tool(mcp_tool)
# 预期打印: ✅ MCP工具 'calculator' 已展开为 6 个独立工具

response = agent.run("计算 25 乘以 16")
print(response)  # 书中预期: 25 乘以 16 的结果是 400
