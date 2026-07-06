from my_advanced_search import create_advanced_search_registry

registry = create_advanced_search_registry()
print()

for query in ["Python编程语言的历史", "人工智能的最新发展"]:
    print(f"测试查询: {query}")
    result = registry.execute_tool("my_advanced_search", query)
    print(f"结果: {result}\n")
