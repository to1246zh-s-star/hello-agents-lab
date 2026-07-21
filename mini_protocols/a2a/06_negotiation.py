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

agent1 = A2AServer(name="agent1", description="Agent 1")

@agent1.skill("propose")
def handle_proposal(text: str) -> str:
    import re
    match = re.search(r'propose\s+(.+)', text, re.IGNORECASE)
    proposal_str = match.group(1).strip() if match else text
    try:
        proposal = eval(proposal_str)
        deadline = proposal.get("deadline")
        if deadline >= 7:
            result = {"accepted": True, "message": "接受提案"}
        else:
            result = {"accepted": False, "message": "时间太紧", "counter_proposal": {"deadline": 7}}
        return str(result)
    except:
        return str({"accepted": False, "message": "无效的提案格式"})

agent2 = A2AServer(name="agent2", description="Agent 2")

@agent2.skill("negotiate")
def negotiate_task(text: str) -> str:
    import re
    match = re.search(r'negotiate\s+task:(.+?)\s+deadline:(\d+)', text, re.IGNORECASE)
    if match:
        task, deadline = match.group(1).strip(), int(match.group(2))
        proposal = {"task": task, "deadline": deadline}
        return str({"status": "negotiating", "proposal": proposal})
    return str({"status": "error", "message": "无效的协商请求"})

threading.Thread(target=lambda: agent1.run(port=7000), daemon=True).start()
threading.Thread(target=lambda: agent2.run(port=7001), daemon=True).start()

if not all([wait_for_port("localhost", 7000), wait_for_port("localhost", 7001)]):
    print("❌ 有服务未能正常启动")
else:
    # 以下是补充的客户端调用演示(教材原文没给,自己补的最小验证)
    agent1_client = A2AClient("http://localhost:7000")

    # 场景1: deadline=10天,预期被接受
    print("场景1: 提案 deadline=10天")
    r1 = agent1_client.execute_skill("propose", str({"task": "写一份报告", "deadline": 10}))
    print(r1.get("result"))

    # 场景2: deadline=3天,预期被拒绝并给出反提案
    print("\n场景2: 提案 deadline=3天")
    r2 = agent1_client.execute_skill("propose", str({"task": "写一份报告", "deadline": 3}))
    print(r2.get("result"))
