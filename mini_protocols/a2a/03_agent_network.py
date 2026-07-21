from hello_agents.protocols import A2AServer, A2AClient
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

# 1. 创建多个Agent服务
researcher = A2AServer(name="researcher", description="研究员")

@researcher.skill("research")
def do_research(text: str) -> str:
    import re
    match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text
    return str({"topic": topic, "findings": f"{topic}的研究结果"})

writer = A2AServer(name="writer", description="撰写员")

@writer.skill("write")
def write_article(text: str) -> str:
    import re
    match = re.search(r'write\s+(.+)', text, re.IGNORECASE)
    content = match.group(1).strip() if match else text
    try:
        data = eval(content)
        topic = data.get("topic", "未知主题")
        findings = data.get("findings", "无研究结果")
    except:
        topic, findings = "未知主题", content
    return f"# {topic}\n\n基于研究：{findings}\n\n文章内容..."

editor = A2AServer(name="editor", description="编辑")

@editor.skill("edit")
def edit_article(text: str) -> str:
    import re
    match = re.search(r'edit\s+(.+)', text, re.IGNORECASE)
    article = match.group(1).strip() if match else text
    result = {"article": article + "\n\n[已编辑优化]", "feedback": "文章质量良好", "approved": True}
    return str(result)

# 2. 启动所有服务
threading.Thread(target=lambda: researcher.run(port=5000), daemon=True).start()
threading.Thread(target=lambda: writer.run(port=5001), daemon=True).start()
threading.Thread(target=lambda: editor.run(port=5002), daemon=True).start()

ports_ready = all([
    wait_for_port("localhost", 5000),
    wait_for_port("localhost", 5001),
    wait_for_port("localhost", 5002),
])
if not ports_ready:
    print("❌ 有服务未能正常启动，请检查上面的报错")
else:
    # 3. 创建客户端连接到各个Agent
    researcher_client = A2AClient("http://localhost:5000")
    writer_client = A2AClient("http://localhost:5001")
    editor_client = A2AClient("http://localhost:5002")

    # 4. 协作流程
    def create_content(topic):
        research = researcher_client.execute_skill("research", f"research {topic}")
        research_data = research.get('result', '')
        article = writer_client.execute_skill("write", f"write {research_data}")
        article_content = article.get('result', '')
        final = editor_client.execute_skill("edit", f"edit {article_content}")
        return final.get('result', '')

    result = create_content("AI在医疗领域的应用")
    print(f"\n最终结果：\n{result}")
