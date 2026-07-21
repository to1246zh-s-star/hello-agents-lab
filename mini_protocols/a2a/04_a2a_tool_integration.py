from dotenv import load_dotenv
load_dotenv()

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import A2ATool
from hello_agents.protocols import A2AServer
import threading, time, socket

def wait_for_port(host, port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False

researcher = A2AServer(name="researcher", description="负责搜索和分析资料的Agent")

@researcher.skill("research")
def handle_research(text: str) -> str:
    import re
    match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text
    return str({"topic": topic, "findings": f"关于{topic}的研究结果..."})

threading.Thread(target=lambda: researcher.run(host="localhost", port=5000), daemon=True).start()

if not wait_for_port("localhost", 5000):
    print("❌ 研究员Agent服务启动失败")
else:
    llm = HelloAgentsLLM()
    coordinator = SimpleAgent(name="协调者", llm=llm)

    researcher_tool = A2ATool(
        name="researcher",
        description="研究员Agent，可以搜索和分析资料",
        agent_url="http://localhost:5000"
    )
    coordinator.add_tool(researcher_tool)

    response = coordinator.run("请让研究员帮我研究AI在教育领域的应用")
    print(response)
