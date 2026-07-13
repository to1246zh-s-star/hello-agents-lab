"""实验8：PDFLearningAssistant —— 真实PDF版智能学习助手
组合 MemoryTool + AdvancedRAGTool，对齐书里8.4节的完整设计
"""
import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

from memory_tool import MemoryTool
from rag_advanced import AdvancedRAGTool


class PDFLearningAssistant:
    """智能文档问答助手（真实PDF版）"""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 每个user_id各自一份MemoryTool（独立.db文件）和RAGTool（独立命名空间+独立向量索引）
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = AdvancedRAGTool(namespace=f"pdf_{user_id}")

        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        }
        self.current_document = None
        self.qa_history = []  # 完整保存每一轮问答的问题+回答，避免只有终端打印、报告里读不到

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """加载真实PDF文档到知识库"""
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}

        start_time = time.time()
        result = self.rag_tool.execute("add_document", file_path=pdf_path, chunk_size=1000, chunk_overlap=200)
        process_time = time.time() - start_time

        if result.get("success", False):
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1

            self.memory_tool.execute(
                "add",
                content=f"加载了文档《{self.current_document}》",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id,
            )
            return {
                "success": True,
                "message": f"加载成功！共{result['chunks_added']}个分块 (耗时: {process_time:.1f}秒)",
                "document": self.current_document,
            }
        else:
            return {"success": False, "message": f"加载失败: {result.get('error', '未知错误')}"}

    def ask(self, question: str, use_advanced_search: bool = True) -> str:
        if not self.current_document:
            return "⚠️ 请先加载文档！"

        # 【MemoryTool】记录问题到工作记忆
        self.memory_tool.execute(
            "add", content=f"提问: {question}", memory_type="working", importance=0.6,
            session_id=self.session_id,
        )

        # 【RAGTool】使用高级检索获取答案
        answer = self.rag_tool.execute(
            "ask", question=question, limit=5,
            enable_advanced_search=use_advanced_search,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search,
        )

        # 【MemoryTool】记录到情景记忆（现在把回答内容也带上，而不只是记"问了这个问题"）
        answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
        self.memory_tool.execute(
            "add", content=f"Q: {question}\nA: {answer_preview}", memory_type="episodic", importance=0.7,
            event_type="qa_interaction", session_id=self.session_id,
        )

        # 完整保存这一轮问答，供generate_report读取（终端打印完就没了，这里存下来才能事后查）
        self.qa_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })

        self.stats["questions_asked"] += 1
        return answer

    def add_note(self, content: str, concept: Optional[str] = None) -> str:
        """添加学习笔记，带concept标签存入语义记忆"""
        self.memory_tool.execute(
            "add", content=content, memory_type="semantic", importance=0.8,
            concept=concept or "general", session_id=self.session_id,
        )
        self.stats["concepts_learned"] += 1
        return "✅ 笔记已记录"

    def recall(self, query: str, limit: int = 5) -> str:
        """回顾学习历程"""
        return self.memory_tool.execute("search", query=query, limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载",
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """生成学习报告，汇总统计+记忆摘要+RAG状态，导出JSON"""
        memory_summary = self.memory_tool.execute("summary", limit=10)
        rag_stats = self.rag_tool.execute("stats")

        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration,
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"],
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats,
            "qa_history": self.qa_history,  # 完整的问答记录，能直接读到每次的回答内容
        }

        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["report_file"] = report_file

        return report


if __name__ == "__main__":
    import os
    import time as _time
    for f in os.listdir("."):
        if (f.startswith("mini_memory_") and f.endswith(".db")) or f.startswith("learning_report_"):
            os.remove(f)

    assistant = PDFLearningAssistant(user_id="kk")

    print("./Happy-LLM-0727.pdf")
    # 换成你自己下载好的PDF路径，比如 "./Happy-LLM-0727.pdf"
    result = assistant.load_document("./Happy-LLM-0727.pdf")
    print(result)

    if not result["success"]:
        print("⚠️ 请把pdf_path换成你实际下载的PDF文件路径后重新运行")
    else:
        print("\n=== 第一轮提问 ===")
        print(assistant.ask("这份文档的核心内容是什么？"))

        _time.sleep(3)  # 每轮提问背后是好几次LLM调用(MQE+HyDE+生成)，主动歇3秒，减少触发免费池限流
        print("\n=== 第二轮提问 ===")
        print(assistant.ask("请举一个文档中提到的具体例子"))

        print("\n=== 添加学习笔记（带concept标签）===")
        print(assistant.add_note("这是我读完第一部分后的理解总结", concept="总体理解"))

        print("\n=== 回顾学习历程 ===")
        print(assistant.recall("文档"))

        print("\n=== 当前统计 ===")
        print(assistant.get_stats())

        print("\n=== 生成学习报告（导出JSON）===")
        report = assistant.generate_report(save_to_file=True)
        print(f"报告已保存: {report['report_file']}")

        print("\n✅ 实验8自测通过：真实PDF加载 + 高级检索问答 + 多层记忆 + JSON报告，完整链路跑通")
