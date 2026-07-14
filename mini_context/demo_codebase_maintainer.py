"""
复现书里9.6.4节的完整使用场景,用一个假的小代码库 + MockLLM + MockMemoryTool 在本地验证流程走通。
在AutoDL上跑真实版本时,把 llm=MockLLM() 换成 llm=HelloAgentsLLM(),
把 memory_tool=MockMemoryTool() 换成第八章写好的真实MemoryTool即可,接口完全兼容。
"""
import json
import shutil
import os

from codebase_maintainer import CodebaseMaintainer
from llm_client import MockLLM
from mock_tools import MockMemoryTool

# ---------- 准备一个假的小代码库,模拟书里"中型Flask应用"场景 ----------
FAKE_REPO = "./fake_flask_app"
if os.path.exists(FAKE_REPO):
    shutil.rmtree(FAKE_REPO)
os.makedirs(f"{FAKE_REPO}/app/models", exist_ok=True)
os.makedirs(f"{FAKE_REPO}/app/services", exist_ok=True)

with open(f"{FAKE_REPO}/app/models/user.py", "w") as f:
    f.write("class User:\n    def __init__(self, username, email):\n        # TODO: email应该加唯一约束\n        self.username = username\n        self.email = email\n")

with open(f"{FAKE_REPO}/app/services/order_service.py", "w") as f:
    f.write("def process_order(order_id):\n    # TODO: 这里嵌套太深,需要重构\n    order = get_order(order_id)\n    if order:\n        if order.status == 'pending':\n            pass\n")


# ========== 初始化助手 ==========
maintainer = CodebaseMaintainer(
    project_name="my_flask_app",
    codebase_path=FAKE_REPO,
    llm=MockLLM(),
    memory_tool=MockMemoryTool(),
)

# ========== 第一天:探索代码库 ==========
maintainer.explore()

# ========== 第二天:分析代码质量 ==========
maintainer.analyze()

# ========== 第三天:规划重构任务 ==========
maintainer.plan_next_steps()

maintainer.create_note(
    title="本周重构计划 - Week 1",
    content="## 目标\n完成数据模型层的优化\n\n## 任务清单\n- [ ] 为User.email添加唯一约束",
    note_type="task_state",
    tags=["refactoring", "week1"],
)

# ========== 查看笔记摘要和会话报告 ==========
summary = maintainer.note_tool.run({"action": "summary"})
print("📊 笔记摘要:")
print(json.dumps(summary, indent=2, ensure_ascii=False))

report = maintainer.generate_report()
print("\n📄 会话报告:")
print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
