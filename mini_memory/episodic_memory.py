"""实验4：情景记忆 —— SQLite结构化存储 + TF-IDF语义检索
"""
import json
import sqlite3
from datetime import datetime
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from memory_item import MemoryItem, MemoryConfig
from test_scoring import episodic_score, recency_exp_decay


class EpisodicMemory:
    def __init__(self, config: MemoryConfig):
        self.db_path = config.database_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                content TEXT,
                session_id TEXT,
                importance REAL,
                timestamp TEXT,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def add(self, item: MemoryItem, session_id: str = "default") -> str:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
            (item.id, item.content, session_id, item.importance, item.timestamp, json.dumps(item.metadata))
        )
        conn.commit()
        conn.close()
        return item.id

    def retrieve(self, query: str, limit: int = 5, session_id: Optional[str] = None) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if session_id:
            rows = conn.execute("SELECT * FROM episodes WHERE session_id = ?", (session_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM episodes").fetchall()
        conn.close()

        if not rows:
            return []

        # 结构化过滤后再做语义检索
        corpus = [r["content"] for r in rows] + [query]
        try:
            vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3)).fit_transform(corpus)
            sims = cosine_similarity(vec[-1], vec[:-1])[0]
        except ValueError:
            sims = [0.0] * len(rows)

        scored = []
        for row, sim in zip(rows, sims):
            age_hours = (datetime.now() - datetime.fromisoformat(row["timestamp"])).total_seconds() / 3600
            recency = recency_exp_decay(age_hours)
            score = episodic_score(float(sim), recency, row["importance"])
            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]


if __name__ == "__main__":
    import os
    if os.path.exists("./mini_memory.db"):
        os.remove("./mini_memory.db")  # 每次自测用干净数据库

    config = MemoryConfig(database_path="./mini_memory.db")
    em = EpisodicMemory(config)

    em.add(MemoryItem(content="2026年3月，用户张三完成了第一个Python项目", importance=0.8), session_id="s1")
    em.add(MemoryItem(content="用户在学习FMAN45机器学习课程，做了Assignment 4", importance=0.9), session_id="s1")
    em.add(MemoryItem(content="用户今天心情不错", importance=0.2), session_id="s1")

    results = em.retrieve("张三的Python项目进展", limit=2)
    print(f"检索到 {len(results)} 条：")
    for r in results:
        print(f"  - {r['content']} (importance={r['importance']})")

    assert "张三" in results[0]["content"], "最相关结果应该排第一"
    print("✅ 实验4自测通过")
