from dotenv import load_dotenv
load_dotenv()

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="文件助手", llm=HelloAgentsLLM())

fs_tool = MCPTool(
    name="fs",
    description="访问本地文件系统",
    server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
)
agent.add_tool(fs_tool)

response = agent.run("请读取README.md文件，并总结其中的主要内容")
print(response)
