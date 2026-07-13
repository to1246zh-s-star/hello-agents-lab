"""实验4b：语义记忆 —— networkx模拟图数据库 + TF-IDF模拟向量检索
真实版用Neo4j+Qdrant，本地实验版用networkx+sklearn替代，检索逻辑完全一致
"""
import re
import math
from typing import List, Tuple
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from memory_item import MemoryItem, MemoryConfig
from test_scoring import semantic_score

# 简化版实体/关系抽取规则：只覆盖"X是Y"这类最常见的定义句
# 真实系统用NLP模型（spacy/HanLP）做命名实体识别，这里用规则演示"抽取"这一步在做什么
RELATION_PATTERNS = [
    (re.compile(r"^(.{1,10}?)是一种(.{1,20})"), "is_a"),
    (re.compile(r"^(.{1,10}?)是(.{1,20})"), "is"),
    (re.compile(r"^(.{1,10}?)属于(.{1,20})"), "belongs_to"),
    (re.compile(r"^(.{1,10}?)包括(.{1,20})"), "includes"),
]


def extract_entities_relations(content: str) -> List[Tuple[str, str, str]]:
    """极简规则抽取：返回 (实体1, 关系, 实体2) 三元组列表"""
    triples = []
    for pattern, relation in RELATION_PATTERNS:
        m = pattern.match(content.strip())
        if m:
            e1 = m.group(1).strip("，,。")
            e2 = m.group(2).strip("，,。、").split("，")[0].split(",")[0][:15]
            if e1 and e2:
                triples.append((e1, relation, e2))
            break  # 一句话只匹配第一个命中的规则，避免重复抽取
    return triples


class SemanticMemory:
    def __init__(self, config: MemoryConfig):
        self.items: List[MemoryItem] = []
        self.graph = nx.DiGraph()  # 用有向图模拟知识图谱，节点=实体，边=关系
        self.vectorizer = None
        self.vectors = None

    def add(self, item: MemoryItem) -> str:
        # 1. 存储原始记忆内容（对应向量库要存的东西）
        self.items.append(item)
        self._rebuild_vector_index()

        # 2. 抽取实体和关系，写入图
        triples = extract_entities_relations(item.content)
        for e1, relation, e2 in triples:
            self.graph.add_node(e1)
            self.graph.add_node(e2)
            self.graph.add_edge(e1, e2, relation=relation, memory_id=item.id)
        item.metadata["entities"] = [e1 for e1, _, _ in triples] + [e2 for _, _, e2 in triples]
        return item.id

    def _rebuild_vector_index(self):
        if len(self.items) < 2:
            return
        corpus = [it.content for it in self.items]
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        self.vectors = self.vectorizer.fit_transform(corpus)

    def _vector_search(self, query: str) -> dict:
        """返回 {memory_id: 相似度}"""
        if not self.items or self.vectors is None:
            return {}
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.vectors)[0]
        return {it.id: float(s) for it, s in zip(self.items, sims)}

    def _graph_search(self, query: str) -> dict:
        """图检索：如果query里提到了图中已有的实体，顺着关系找相关记忆
        直接相连(1跳)得分1.0，间接相连(2跳)得分0.5，模拟"关系越远，相关性越弱"
        """
        graph_scores = {}
        # 双向判断：实体名在query里，或者query在实体名里
        mentioned_entities = [n for n in self.graph.nodes if n in query or query in n]
        if not mentioned_entities:
            return {}

        for entity in mentioned_entities:
            for neighbor in list(self.graph.successors(entity)) + list(self.graph.predecessors(entity)):
                for _, _, data in self._edges_between(entity, neighbor):
                    mid = data.get("memory_id")
                    if mid:
                        graph_scores[mid] = max(graph_scores.get(mid, 0), 1.0)
                # 2跳邻居（间接关系，比如"Python是编程语言，编程语言属于计算机科学"）
                for neighbor2 in list(self.graph.successors(neighbor)) + list(self.graph.predecessors(neighbor)):
                    if neighbor2 == entity:
                        continue
                    for _, _, data in self._edges_between(neighbor, neighbor2):
                        mid = data.get("memory_id")
                        if mid:
                            graph_scores[mid] = max(graph_scores.get(mid, 0), 0.5)
        return graph_scores

    def _edges_between(self, a, b):
        edges = []
        if self.graph.has_edge(a, b):
            edges.append((a, b, self.graph.get_edge_data(a, b)))
        if self.graph.has_edge(b, a):
            edges.append((b, a, self.graph.get_edge_data(b, a)))
        return edges

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryItem]:
        vector_scores = self._vector_search(query)
        graph_scores = self._graph_search(query)

        scored = []
        for item in self.items:
            v_score = vector_scores.get(item.id, 0.0)
            g_score = graph_scores.get(item.id, 0.0)
            score = semantic_score(v_score, g_score, item.importance)
            if score > 0:
                scored.append((score, item, v_score, g_score))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(item, v, g) for _, item, v, g in scored[:limit]]


if __name__ == "__main__":
    config = MemoryConfig()
    sm = SemanticMemory(config)

    sm.add(MemoryItem(content="Python是一种编程语言", importance=0.9))
    sm.add(MemoryItem(content="编程语言属于计算机科学的一个分支", importance=0.7))
    sm.add(MemoryItem(content="今天天气不错", importance=0.2))

    print("--- 抽取到的关系三元组 ---")
    for e1, rel, e2, mid in [(u, d["relation"], v, d["memory_id"]) for u, v, d in sm.graph.edges(data=True)]:
        print(f"  {e1} --[{rel}]--> {e2} (来自记忆 {mid[:8]}...)")

    print("\n--- 检索测试：向量检索能找到的 ---")
    results = sm.retrieve("Python编程", limit=3)
    for item, v, g in results:
        print(f"  score计算: vector={v:.2f}, graph={g:.2f} | {item.content}")

    print("\n--- 检索测试：验证图检索能发现'隐含关联' ---")
    # 关键验证点：问"计算机科学"，向量检索大概率找不到"Python"那条(字面完全不重合)
    # 但图检索能通过 Python->编程语言->计算机科学 这条2跳路径找到它
    results2 = sm.retrieve("计算机科学", limit=3)
    contents = [item.content for item, v, g in results2]
    print(f"命中: {contents}")
    assert any("Python" in c for c in contents), "图检索应该能通过2跳关系找到Python这条记忆"
    print("✅ 图检索验证通过：纯向量检索找不到的关联，图检索找到了")

    print("\n✅ 实验4b自测通过")
