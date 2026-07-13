"""实验6：RAG基础链路 —— 真实PDF加载 + 段落感知分块 + TF-IDF检索 + LLM生成
"""
import os
import re
import time
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pypdf import PdfReader

load_dotenv()


def chunk_by_markdown_headers(text: str) -> List[str]:
    """按 #/##/### 标题切分，用于手写Markdown文本（PDF提取的纯文本走chunk_by_size）"""
    parts = re.split(r'\n(?=#{1,3}\s)', text)
    return [p.strip() for p in parts if p.strip()]


def extract_text_from_pdf(file_path: str) -> str:
    """用pypdf提取PDF全文（真实版用MarkItDown做PDF→Markdown转换，
    本地实验版用pypdf直接拿纯文本，牺牲了表格/图片结构，但文字类内容提取效果足够用）
    """
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages_text.append(t)
    return "\n\n".join(pages_text)


def chunk_by_size(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """段落感知的滑动窗口分块：按段落切，累积到接近chunk_size就切断，
    下一块开头保留上一块末尾chunk_overlap个字符，避免跨块的上下文断裂
    """
    paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = current + "\n" + para if current else para
        else:
            if current:
                chunks.append(current)
                current = current[-chunk_overlap:] + "\n" + para if chunk_overlap > 0 else para
            else:
                # 单个段落就超过chunk_size，直接按字符数硬切（罕见情况兜底）
                current = para[:chunk_size]
                chunks.append(current)
                current = para[chunk_size - chunk_overlap:]
    if current:
        chunks.append(current)
    return chunks


def call_llm_with_retry(client, model, messages, max_retries=5, base_delay=3):
    """带指数退避的LLM调用：免费API短时间内连续请求容易429限流，
    触发限流就等一下再重试，而不是直接崩溃（第七章踩坑表提到过这个问题）
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=model, messages=messages)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)  # 3s, 6s, 12s, 24s...
            print(f"  ⏳ 触发限流(429)，{wait}秒后自动重试（第{attempt + 1}/{max_retries}次）...")
            time.sleep(wait)


class RAGTool:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace  # 对应书里rag_namespace的隔离概念
        self.chunks: List[Dict] = []  # 每条存 {"content":..., "source":...}，多一个来源字段
        self.vectorizer = None
        self.vectors = None
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.model = os.getenv("LLM_MODEL_ID")

    def execute(self, action: str, **kwargs):
        """统一入口，和MemoryTool一样的命令模式设计"""
        handler = {
            "add_text": self._add_text,
            "add_document": self._add_document,
            "ask": self.ask,
            "search": self._search_action,
            "stats": self._stats,
        }.get(action)
        if handler is None:
            return {"success": False, "error": f"未知操作: {action}"}
        return handler(**kwargs)

    def _add_text(self, text: str, source: str = "manual_input") -> Dict:
        new_chunks = chunk_by_markdown_headers(text)
        self.chunks.extend([{"content": c, "source": source} for c in new_chunks])
        self._rebuild_index()
        return {"success": True, "chunks_added": len(new_chunks), "total_chunks": len(self.chunks)}

    def _add_document(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict:
        """加载真实PDF文档：提取文本 → 分块 → 向量化索引"""
        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}"}
        try:
            text = extract_text_from_pdf(file_path)
        except Exception as e:
            return {"success": False, "error": f"PDF解析失败: {e}"}

        if not text.strip():
            return {"success": False, "error": "PDF提取出的文本为空（可能是扫描版PDF，没有可提取的文字层）"}

        source = os.path.basename(file_path)
        new_chunks = chunk_by_size(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks.extend([{"content": c, "source": source} for c in new_chunks])
        self._rebuild_index()
        return {
            "success": True,
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
            "extracted_chars": len(text),
        }

    def _rebuild_index(self):
        if len(self.chunks) < 2:
            return
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        self.vectors = self.vectorizer.fit_transform([c["content"] for c in self.chunks])

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.chunks or self.vectors is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.vectors)[0]
        ranked = sorted(zip(self.chunks, sims), key=lambda x: x[1], reverse=True)
        return [{"content": c["content"], "source": c["source"], "score": float(s)}
                for c, s in ranked[:limit] if s > 0]

    def _search_action(self, query: str, limit: int = 3) -> Dict:
        results = self.search(query, limit)
        return {"success": True, "results": results}

    def ask(self, question: str, limit: int = 3, **kwargs) -> str:
        hits = self.search(question, limit)
        if not hits:
            return "⚠️ 知识库中没有找到相关内容"

        context = "\n\n".join([f"[参考{i+1}，来源:{h['source']}] {h['content']}" for i, h in enumerate(hits)])
        messages = [
            {"role": "system", "content": "你是一个基于给定参考资料回答问题的助手，只根据参考资料回答，不要编造。"},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
        ]
        resp = call_llm_with_retry(self.client, self.model, messages)
        return resp.choices[0].message.content

    def _stats(self) -> Dict:
        sources = {}
        for c in self.chunks:
            sources[c["source"]] = sources.get(c["source"], 0) + 1
        return {
            "namespace": self.namespace,
            "total_chunks": len(self.chunks),
            "sources": sources,
        }


if __name__ == "__main__":
    rag = RAGTool(namespace="test")

    # 把这里换成你自己下载好的PDF路径，比如 "./Happy-LLM-0727.pdf"
    result = rag.execute("add_document", file_path="./Happy-LLM-0727.pdf", chunk_size=1000, chunk_overlap=200)
    print("加载结果:", result)

    if result["success"]:
        print("\n--- 纯检索测试（不调用LLM）---")
        for h in rag.search("Transformer的自注意力机制是什么", limit=3):
            print(f"  score={h['score']:.3f} | 来源:{h['source']} | {h['content'][:50]}...")

        print("\n--- stats ---")
        print(rag.execute("stats"))

        print("\n--- 完整问答测试（调用LLM）---")
        answer = rag.ask("这份文档主要讲了什么内容？")
        print(f"回答: {answer}")

        print("\n✅ 实验6自测通过：真实PDF加载→分块→检索→问答全链路跑通")
    else:
        print("⚠️ 请把file_path换成你实际下载的PDF文件路径后重新运行")
