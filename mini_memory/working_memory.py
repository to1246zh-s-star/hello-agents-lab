"""实验2：工作记忆 —— 纯内存存储 + TTL自动清理 + 混合检索
"""
import math
from datetime import datetime, timedelta
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from memory_item import MemoryItem, MemoryConfig


class WorkingMemory:
    def __init__(self, config: MemoryConfig):
        self.max_capacity = config.working_memory_capacity
        self.max_age_minutes = config.working_memory_ttl
        self.memories: List[MemoryItem] = []

    def _expire_old_memories(self):
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.max_age_minutes)
        before = len(self.memories)
        self.memories = [m for m in self.memories if datetime.fromisoformat(m.timestamp) > cutoff]
        expired = before - len(self.memories)
        if expired:
            print(f"🧹 TTL清理：过期 {expired} 条")

    def _remove_lowest_priority_memory(self):
        if not self.memories:
            return
        lowest = min(self.memories, key=lambda m: m.importance)
        self.memories.remove(lowest)
        print(f"🧹 容量淘汰：移除重要性最低的记忆 (importance={lowest.importance})")

    def add(self, item: MemoryItem) -> str:
        self._expire_old_memories()
        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority_memory()
        self.memories.append(item)
        return item.id

    def _calculate_time_decay(self, timestamp: str) -> float:
        """指数衰减，模拟遗忘曲线"""
        age_hours = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() / 3600
        return max(0.1, math.exp(-0.1 * age_hours / 24))

    def _tfidf_scores(self, query: str) -> dict:
        """TF-IDF向量检索，语料太少时(<2条)自动降级为空"""
        if len(self.memories) < 2:
            return {}
        corpus = [m.content for m in self.memories] + [query]
        try:
            # 中文没有天然空格分词，用字符级n-gram代替词级分词，避免整句被当成一个token
            vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3)).fit_transform(corpus)
            sims = cosine_similarity(vec[-1], vec[:-1])[0]
            return {m.id: float(s) for m, s in zip(self.memories, sims)}
        except ValueError:
            return {}  # 全是停用词等边界情况

    def _keyword_score(self, query: str, content: str) -> float:
        q_words = set(query.lower())
        c_words = set(content.lower())
        if not q_words:
            return 0.0
        return len(q_words & c_words) / len(q_words)

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryItem]:
        self._expire_old_memories()
        vector_scores = self._tfidf_scores(query)

        scored = []
        for m in self.memories:
            v_score = vector_scores.get(m.id, 0.0)
            k_score = self._keyword_score(query, m.content)
            base_relevance = v_score * 0.7 + k_score * 0.3 if v_score > 0 else k_score
            time_decay = self._calculate_time_decay(m.timestamp)
            importance_weight = 0.8 + (m.importance * 0.4)
            final_score = base_relevance * time_decay * importance_weight
            if final_score > 0:
                scored.append((final_score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[:limit]]
        for m in results:
            m.touch()
        return results


if __name__ == "__main__":
    config = MemoryConfig(working_memory_capacity=3, working_memory_ttl=60)
    wm = WorkingMemory(config)

    wm.add(MemoryItem(content="用户问了Python函数的问题", importance=0.6))
    wm.add(MemoryItem(content="用户问了Python装饰器的问题", importance=0.7))
    wm.add(MemoryItem(content="今天天气不错", importance=0.2))

    results = wm.retrieve("Python函数怎么用", limit=2)
    print(f"检索到 {len(results)} 条：")
    for r in results:
        print(f"  - {r.content} (importance={r.importance})")

    # 验证容量淘汰：加第4条应该挤掉重要性最低的"今天天气不错"
    wm.add(MemoryItem(content="用户问了列表推导式", importance=0.9))
    contents = [m.content for m in wm.memories]
    assert "今天天气不错" not in contents, "容量淘汰逻辑有误"
    print("✅ 实验2自测通过：容量淘汰逻辑正确")
