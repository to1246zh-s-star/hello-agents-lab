"""
按《HelloAgents》第九章 9.6.3 节代码实现的 CodebaseMaintainer。
唯一的改动是import路径——书里从`hello_agents`这个pip包导入,
这里改成从本章前面几节实现的本地模块导入(context_builder/note_tool/terminal_tool/message/llm_client),
类的字段、方法、逻辑顺序全部保持和书里一致,没有做任何自己发挥的修改。
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from llm_client import HelloAgentsLLM
from context_builder import ContextBuilder, ContextConfig, ContextPacket
from note_tool import NoteTool
from terminal_tool import TerminalTool
from message import Message


class CodebaseMaintainer:
    """代码库维护助手 - 长程智能体示例

    整合 ContextBuilder + NoteTool + TerminalTool + MemoryTool
    实现跨会话的代码库维护任务管理
    """

    def __init__(
        self,
        project_name: str,
        codebase_path: str,
        llm: Optional[object] = None,
        memory_tool=None,
    ):
        self.project_name = project_name
        self.codebase_path = codebase_path
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化 LLM
        self.llm = llm or HelloAgentsLLM()

        # 初始化工具
        # 书里是 self.memory_tool = MemoryTool(user_id=project_name),
        # 这里改成依赖注入(传入第八章写好的真实MemoryTool,或测试用的MockMemoryTool),
        # 避免本文件反过来依赖mini_memory的具体实现细节
        self.memory_tool = memory_tool
        self.note_tool = NoteTool(workspace=f"./{project_name}_notes")
        self.terminal_tool = TerminalTool(workspace=codebase_path, timeout=60)

        # 初始化上下文构建器
        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=None,  # 本案例不使用 RAG
            config=ContextConfig(
                max_tokens=4000,
                reserve_ratio=0.15,
                min_relevance=0.2,
                enable_compression=True,
            ),
        )

        # 对话历史
        self.conversation_history: List[Message] = []

        # 统计信息
        self.stats = {
            "session_start": datetime.now(),
            "commands_executed": 0,
            "notes_created": 0,
            "issues_found": 0,
        }

        print(f"✅ 代码库维护助手已初始化: {project_name}")
        print(f"📁 工作目录: {codebase_path}")
        print(f"🆔 会话ID: {self.session_id}")

    def run(self, user_input: str, mode: str = "auto") -> str:
        """运行助手

        Args:
            user_input: 用户输入
            mode: 运行模式
                - "auto": 自动决策是否使用工具
                - "explore": 侧重代码探索
                - "analyze": 侧重问题分析
                - "plan": 侧重任务规划
        """
        print(f"\n{'='*80}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*80}\n")

        # 第一步:根据模式执行预处理
        pre_context = self._preprocess_by_mode(user_input, mode)

        # 第二步:检索相关笔记
        relevant_notes = self._retrieve_relevant_notes(user_input)
        note_packets = self._notes_to_packets(relevant_notes)

        # 第三步:构建优化的上下文
        context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self._build_system_instructions(mode),
            custom_packets=note_packets + pre_context,
        )

        # 第四步:调用 LLM
        print("🤖 正在思考...")
        response = self.llm.invoke(context)

        # 第五步:后处理
        self._postprocess_response(user_input, response)

        # 第六步:更新对话历史
        self._update_history(user_input, response)

        print(f"\n🤖 助手: {response}\n")
        print(f"{'='*80}\n")

        return response

    def _preprocess_by_mode(self, user_input: str, mode: str) -> List[ContextPacket]:
        """根据模式执行预处理,收集相关信息"""
        packets = []

        if mode == "explore" or mode == "auto":
            print("🔍 探索代码库结构...")
            structure = self.terminal_tool.run({"command": "find . -type f -name '*.py'"})
            self.stats["commands_executed"] += 1

            packets.append(ContextPacket(
                content=f"[代码库结构]\n{structure}",
                timestamp=datetime.now(),
                token_count=len(structure) // 4,
                relevance_score=0.6,
                metadata={"type": "code_structure", "source": "terminal"},
            ))

        if mode == "analyze":
            print("📊 分析代码质量...")
            loc = self.terminal_tool.run({"command": "find . -name '*.py' -exec wc -l {} +"})
            todos = self.terminal_tool.run({"command": "grep -rn TODO --include='*.py'"})
            self.stats["commands_executed"] += 2

            packets.append(ContextPacket(
                content=f"[代码统计]\n{loc}\n\n[待办事项]\n{todos}",
                timestamp=datetime.now(),
                token_count=(len(loc) + len(todos)) // 4,
                relevance_score=0.7,
                metadata={"type": "code_analysis", "source": "terminal"},
            ))

        if mode == "plan":
            print("📋 加载任务规划...")
            task_notes = self.note_tool.run({"action": "list", "note_type": "task_state", "limit": 3})

            if task_notes:
                content = "\n".join([f"- {note['title']}" for note in task_notes])
                packets.append(ContextPacket(
                    content=f"[当前任务]\n{content}",
                    timestamp=datetime.now(),
                    token_count=len(content) // 4,
                    relevance_score=0.8,
                    metadata={"type": "task_plan", "source": "notes"},
                ))

        return packets

    def _retrieve_relevant_notes(self, query: str, limit: int = 3) -> List[Dict]:
        """检索相关笔记"""
        try:
            blockers = self.note_tool.run({"action": "list", "note_type": "blocker", "limit": 2})
            search_results = self.note_tool.run({"action": "search", "query": query, "limit": limit})

            all_notes = {
                note.get("note_id") or note.get("id"): note
                for note in (blockers or []) + (search_results or [])
            }
            return list(all_notes.values())[:limit]

        except Exception as e:
            print(f"[WARNING] 笔记检索失败: {e}")
            return []

    def _notes_to_packets(self, notes: List[Dict]) -> List[ContextPacket]:
        """将笔记转换为上下文包"""
        packets = []

        relevance_map = {
            "blocker": 0.9,
            "action": 0.8,
            "task_state": 0.75,
            "conclusion": 0.7,
        }

        for note in notes:
            note_type = note.get("type", "general")
            relevance = relevance_map.get(note_type, 0.6)

            content = f"[笔记:{note.get('title', 'Untitled')}]\n类型: {note_type}\n\n{note.get('content', '')}"

            packets.append(ContextPacket(
                content=content,
                timestamp=datetime.fromisoformat(note.get("updated_at", datetime.now().isoformat())),
                token_count=len(content) // 4,
                relevance_score=relevance,
                metadata={
                    "type": "note",
                    "note_type": note_type,
                    "note_id": note.get("note_id") or note.get("id"),
                },
            ))

        return packets

    def _build_system_instructions(self, mode: str) -> str:
        """构建系统指令"""
        base_instructions = f"""你是 {self.project_name} 项目的代码库维护助手。

你的核心能力:
1. 使用 TerminalTool 探索代码库(ls, cat, grep, find等)
2. 使用 NoteTool 记录发现和任务
3. 基于历史笔记提供连贯的建议

当前会话ID: {self.session_id}
"""

        mode_specific = {
            "explore": """
当前模式: 探索代码库

你应该:
- 主动使用 terminal 命令了解代码结构
- 识别关键模块和文件
- 记录项目架构到笔记
""",
            "analyze": """
当前模式: 分析代码质量

你应该:
- 查找代码问题(重复、复杂度、TODO等)
- 评估代码质量
- 将发现的问题记录为 blocker 或 action 笔记
""",
            "plan": """
当前模式: 任务规划

你应该:
- 回顾历史笔记和任务
- 制定下一步行动计划
- 更新任务状态笔记
""",
            "auto": """
当前模式: 自动决策

你应该:
- 根据用户需求灵活选择策略
- 在需要时使用工具
- 保持回答的专业性和实用性
""",
        }

        return base_instructions + mode_specific.get(mode, mode_specific["auto"])

    def _postprocess_response(self, user_input: str, response: str):
        """后处理:分析回答,自动记录重要信息"""
        if any(keyword in response.lower() for keyword in ["问题", "bug", "错误", "阻塞"]):
            try:
                self.note_tool.run({
                    "action": "create",
                    "title": f"发现问题: {user_input[:30]}...",
                    "content": f"## 用户输入\n{user_input}\n\n## 问题分析\n{response[:500]}...",
                    "note_type": "blocker",
                    "tags": [self.project_name, "auto_detected", self.session_id],
                })
                self.stats["notes_created"] += 1
                self.stats["issues_found"] += 1
                print("📝 已自动创建问题笔记")
            except Exception as e:
                print(f"[WARNING] 创建笔记失败: {e}")

        elif any(keyword in user_input.lower() for keyword in ["计划", "下一步", "任务", "todo"]):
            try:
                self.note_tool.run({
                    "action": "create",
                    "title": f"任务规划: {user_input[:30]}...",
                    "content": f"## 讨论\n{user_input}\n\n## 行动计划\n{response[:500]}...",
                    "note_type": "action",
                    "tags": [self.project_name, "planning", self.session_id],
                })
                self.stats["notes_created"] += 1
                print("📝 已自动创建行动计划笔记")
            except Exception as e:
                print(f"[WARNING] 创建笔记失败: {e}")

    def _update_history(self, user_input: str, response: str):
        """更新对话历史"""
        self.conversation_history.append(Message(content=user_input, role="user", timestamp=datetime.now()))
        self.conversation_history.append(Message(content=response, role="assistant", timestamp=datetime.now()))

        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    # === 便捷方法 ===

    def explore(self, target: str = ".") -> str:
        return self.run(f"请探索 {target} 的代码结构", mode="explore")

    def analyze(self, focus: str = "") -> str:
        query = "请分析代码质量" + (f",重点关注{focus}" if focus else "")
        return self.run(query, mode="analyze")

    def plan_next_steps(self) -> str:
        return self.run("根据当前进度,规划下一步任务", mode="plan")

    def execute_command(self, command: str) -> str:
        result = self.terminal_tool.run({"command": command})
        self.stats["commands_executed"] += 1
        return result

    def create_note(self, title: str, content: str, note_type: str = "general", tags: List[str] = None) -> str:
        result = self.note_tool.run({
            "action": "create",
            "title": title,
            "content": content,
            "note_type": note_type,
            "tags": tags or [self.project_name],
        })
        self.stats["notes_created"] += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        try:
            note_summary = self.note_tool.run({"action": "summary"})
        except Exception:
            note_summary = {}

        return {
            "session_info": {
                "session_id": self.session_id,
                "project": self.project_name,
                "duration_seconds": duration,
            },
            "activity": {
                "commands_executed": self.stats["commands_executed"],
                "notes_created": self.stats["notes_created"],
                "issues_found": self.stats["issues_found"],
            },
            "notes": note_summary,
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        report = self.get_stats()

        if save_to_file:
            report_file = f"maintainer_report_{self.session_id}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["report_file"] = report_file
            print(f"📄 报告已保存: {report_file}")

        return report
