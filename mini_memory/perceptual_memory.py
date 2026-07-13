"""实验4c：感知记忆 —— 按模态分离的向量空间
真实版用CLIP编码图像、CLAP编码音频；本地实验版没有这些模型，
用"文字描述模拟多模态"（假装已经有个字幕模型把图片/音频转成了文字），
重点演示"为什么要按模态分开存"，而不是真的做图像/音频识别
"""
from datetime import datetime
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from memory_item import MemoryItem, MemoryConfig
from test_scoring import perceptual_score, recency_exp_decay


class ModalityStore:
    """单个模态的独立向量空间，模拟"text/image/audio各开一个collection" """
    def __init__(self):
        self.items: List[MemoryItem] = []
        self.vectorizer = None
        self.vectors = None

    def add(self, item: MemoryItem):
        self.items.append(item)
        if len(self.items) >= 2:
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            self.vectors = self.vectorizer.fit_transform([it.content for it in self.items])

    def search(self, query: str) -> dict:
        if len(self.items) < 2:
            return {}
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.vectors)[0]
        return {it.id: float(s) for it, s in zip(self.items, sims)}


class PerceptualMemory:
    def __init__(self, config: MemoryConfig):
        # 按模态分离的独立存储，对应书里"避免维度不匹配"的设计
        self.stores = {
            "text": ModalityStore(),
            "image": ModalityStore(),
            "audio": ModalityStore(),
        }

    def add(self, item: MemoryItem, modality: str = "text") -> str:
        if modality not in self.stores:
            raise ValueError(f"不支持的模态: {modality}")
        item.metadata["modality"] = modality
        self.stores[modality].add(item)
        return item.id

    def retrieve(self, query: str, modality: str = "text", limit: int = 5) -> List[MemoryItem]:
        """同模态检索：只在指定模态的向量空间里找"""
        store = self.stores.get(modality)
        if store is None:
            return []
        vector_scores = store.search(query)

        scored = []
        for item in store.items:
            v_score = vector_scores.get(item.id, 0.0)
            age_hours = (datetime.now() - datetime.fromisoformat(item.timestamp)).total_seconds() / 3600
            recency = recency_exp_decay(age_hours)
            score = perceptual_score(v_score, recency, item.importance)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]


if __name__ == "__main__":
    config = MemoryConfig()
    pm = PerceptualMemory(config)

    # text模态：正常文字记忆
    pm.add(MemoryItem(content="用户问了关于Python函数的问题", importance=0.6), modality="text")
    pm.add(MemoryItem(content="用户讨论了机器学习模型训练", importance=0.7), modality="text")

    # image模态：假装已经有字幕模型，把图片转成了这段文字描述
    pm.add(MemoryItem(content="截图内容：一段Python函数定义代码，包含for循环",
                       importance=0.7, metadata={"raw_data": "./uploads/code_screenshot.png"}),
           modality="image")
    pm.add(MemoryItem(content="截图内容：机器学习训练过程的loss曲线图表",
                       importance=0.6, metadata={"raw_data": "./uploads/loss_chart.png"}),
           modality="image")

    print("--- 验证1：text模态检索只在text空间里找，不会混入image模态的内容 ---")
    text_results = pm.retrieve("Python相关内容", modality="text", limit=5)
    print(f"命中 {len(text_results)} 条 text 记忆:")
    for r in text_results:
        print(f"  - [{r.metadata['modality']}] {r.content}")
    assert all(r.metadata["modality"] == "text" for r in text_results), "text检索不应该混入其他模态"

    print("\n--- 验证2：image模态检索只在image空间里找 ---")
    image_results = pm.retrieve("Python代码截图", modality="image", limit=5)
    print(f"命中 {len(image_results)} 条 image 记忆:")
    for r in image_results:
        print(f"  - [{r.metadata['modality']}] {r.content}")
    assert all(r.metadata["modality"] == "image" for r in image_results), "image检索不应该混入其他模态"
    assert len(image_results) > 0 and "code_screenshot" in image_results[0].metadata["raw_data"]

    print("\n✅ 实验4c自测通过：模态隔离生效，text查询不会检索到image内容，反之亦然")
