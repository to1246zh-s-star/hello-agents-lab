from hello_agents.protocols import A2AServer, A2AClient
import threading, time, socket

def wait_for_port(host, port, timeout=10):
    """轮询等待端口就绪,替代固定sleep(个人适配,非教材原文)"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False

researcher = A2AServer(name="researcher", description="负责搜索和分析资料的Agent", version="1.0.0")

@researcher.skill("research")
def handle_research(text: str) -> str:
    import re
    match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text
    result = {"topic": topic, "findings": f"关于{topic}的研究结果...", "sources": ["来源1", "来源2", "来源3"]}
    return str(result)

server_thread = threading.Thread(target=lambda: researcher.run(host="localhost", port=5000), daemon=True)
server_thread.start()

if not wait_for_port("localhost", 5000):
    print("❌ 研究员Agent服务启动失败或超时，请检查上面的报错")
else:
    print("✅ 研究员Agent服务已启动在 http://localhost:5000")
    client = A2AClient("http://localhost:5000")
    response = client.execute_skill("research", "research AI在医疗领域的应用")
    print(f"收到响应：{response.get('result')}")
