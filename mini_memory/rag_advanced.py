"""实验7：MQE（多查询扩展）+ HyDE（假设文档嵌入）
"""
from rag_basic import RAGTool, call_llm_with_retry


class AdvancedRAGTool(RAGTool):
    def _prompt_mqe(self, query: str, n: int = 2) -> list:
        messages = [
            {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
        ]
        resp = call_llm_with_retry(self.client, self.model, messages)
        text = resp.choices[0].message.content or ""
        lines = [ln.strip("- \t") for ln in text.splitlines() if ln.strip()]
        return lines[:n] or [query]

    def _prompt_hyde(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
        ]
        resp = call_llm_with_retry(self.client, self.model, messages)
        return resp.choices[0].message.content or ""

    def search_expanded(self, query: str, limit: int = 3, enable_mqe: bool = True, enable_hyde: bool = True) -> list:
        expansions = [query]
        if enable_mqe:
            expansions += self._prompt_mqe(query)
        if enable_hyde:
            hyde_text = self._prompt_hyde(query)
            if hyde_text:
                expansions.append(hyde_text)

        print(f"  🔎 扩展出 {len(expansions)} 个查询用于检索")

        agg = {}
        for q in expansions:
            for hit in self.search(q, limit=limit * 2):
                key = hit["content"]
                if key not in agg or hit["score"] > agg[key]["score"]:
                    agg[key] = hit

        merged = sorted(agg.values(), key=lambda x: x["score"], reverse=True)
        return merged[:limit]

    def ask(self, question: str, limit: int = 5, enable_advanced_search: bool = True,
            enable_mqe: bool = None, enable_hyde: bool = None, **kwargs) -> str:
        """对齐书里PDFLearningAssistant调用RAGTool.execute("ask", enable_advanced_search=...)的接口。
        enable_mqe/enable_hyde不单独传时，默认跟随enable_advanced_search这个总开关。
        """
        use_mqe = enable_advanced_search if enable_mqe is None else enable_mqe
        use_hyde = enable_advanced_search if enable_hyde is None else enable_hyde

        if enable_advanced_search:
            hits = self.search_expanded(question, limit=limit, enable_mqe=use_mqe, enable_hyde=use_hyde)
        else:
            hits = self.search(question, limit=limit)

        if not hits:
            return "⚠️ 知识库中没有找到相关内容"

        context = "\n\n".join([f"[参考{i+1}，来源:{h['source']}] {h['content']}" for i, h in enumerate(hits)])
        messages = [
            {"role": "system", "content": "你是一个基于给定参考资料回答问题的助手，只根据参考资料回答，不要编造。"},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
        ]
        resp = call_llm_with_retry(self.client, self.model, messages)
        return resp.choices[0].message.content


if __name__ == "__main__":
    rag = AdvancedRAGTool(namespace="test")
    result = rag.execute("add_document", file_path="./Happy-LLM-0727.pdf", chunk_size=1000, chunk_overlap=200)
    print("加载结果:", result)

    if result["success"]:
        print("--- 基础检索 vs 扩展检索对比 ---")
        q = "AI是怎么理解语言的"  # 故意用口语化提问，测试是否比字面匹配更强

        basic_hits = rag.search(q, limit=2)
        print(f"\n[基础检索] 命中 {len(basic_hits)} 条:")
        for h in basic_hits:
            print(f"  score={h['score']:.3f} | 来源:{h['source']} | {h['content'][:40]}...")

        expanded_hits = rag.search_expanded(q, limit=2, enable_mqe=True, enable_hyde=True)
        print(f"\n[MQE+HyDE扩展检索] 命中 {len(expanded_hits)} 条:")
        for h in expanded_hits:
            print(f"  score={h['score']:.3f} | 来源:{h['source']} | {h['content'][:40]}...")

        print("\n✅ 实验7完成：对比两组结果，观察扩展检索是否找回了基础检索漏掉的相关分块")
    else:
        print("⚠️ 请把file_path换成你实际下载的PDF文件路径后重新运行")
