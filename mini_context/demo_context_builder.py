"""复现书里9.3.4节的完整示例,验证ContextBuilder输出是否符合[Role&Policies]/[Task]/[Evidence]/[Context]/[Output]分区结构"""
from context_builder import ContextBuilder, ContextConfig
from mock_tools import MockMemoryTool, MockRAGTool, Message

memory_tool = MockMemoryTool()
rag_tool = MockRAGTool(knowledge_base_path="./knowledge_base")

config = ContextConfig(max_tokens=3000, reserve_ratio=0.2, min_relevance=0.2, enable_compression=True)
builder = ContextBuilder(memory_tool=memory_tool, rag_tool=rag_tool, config=config)

conversation_history = [
    Message(content="我正在开发一个数据分析工具", role="user"),
    Message(content="很好!数据分析工具通常需要处理大量数据。您计划使用什么技术栈?", role="assistant"),
    Message(content="我打算使用Python和Pandas,已经完成了CSV读取模块", role="user"),
    Message(content="不错的选择!Pandas在数据处理方面非常强大。接下来您可能需要考虑数据清洗和转换。", role="assistant"),
]

memory_tool.run({"action": "add", "content": "用户正在开发数据分析工具,使用Python和Pandas",
                  "memory_type": "semantic", "importance": 0.8})
memory_tool.run({"action": "add", "content": "已完成CSV读取模块的开发",
                  "memory_type": "episodic", "importance": 0.7})

context = builder.build(
    user_query="如何优化Pandas的内存占用?",
    conversation_history=conversation_history,
    system_instructions="你是一位资深的Python数据工程顾问。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例",
)

print("=" * 80)
print("构建的上下文:")
print("=" * 80)
print(context)
print("=" * 80)
