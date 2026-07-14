"""
升级1:relevance打分从Jaccard换成TF-IDF字符ngram余弦相似度。

为什么不直接改context_builder.py里的_calculate_relevance:
保留书本原始实现完整可对照,升级版做成"可插拔子类",继承ContextBuilder只覆写这一个方法。
这样"书本忠实版"和"生产可用版"两份代码都在,方便对比和答辩时讲清楚差异。
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from context_builder import ContextBuilder


class TfidfContextBuilder(ContextBuilder):
    """
    用TF-IDF(char analyzer, 2-3gram)替换书本的Jaccard相关性计算。
    延续第八章mini_memory里验证过的思路:char ngram能绕开中文分词依赖,
    "用户"和"用户们"、"优化Pandas"和"Pandas优化"这类局部重合能被抓住,
    而Jaccard按空格分词在中文里基本不产生任何交集。
    """

    def _calculate_relevance(self, content: str, query: str) -> float:
        if not content.strip() or not query.strip():
            return 0.0
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        try:
            vectors = vectorizer.fit_transform([content, query])
        except ValueError:
            # 内容过短,char ngram词表为空
            return 0.0
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(max(0.0, min(1.0, sim)))
