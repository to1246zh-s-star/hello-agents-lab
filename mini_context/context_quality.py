"""
对应书本习题2第二问:"为ContextBuilder添加一个上下文质量评估功能"。
评估三个维度:信息密度、相关性、完整性,并给出优化建议。
不依赖LLM调用,纯规则+统计,免费且可在任何机器上跑。
"""
import re
from typing import Dict, Any


class ContextQualityEvaluator:
    def evaluate(self, built_context: str, user_query: str, config) -> Dict[str, Any]:
        sections = self._parse_sections(built_context)

        density = self._evaluate_density(sections)
        relevance = self._evaluate_relevance(sections, user_query)
        completeness = self._evaluate_completeness(sections)

        overall = round((density["score"] + relevance["score"] + completeness["score"]) / 3, 3)

        suggestions = []
        suggestions.extend(density["suggestions"])
        suggestions.extend(relevance["suggestions"])
        suggestions.extend(completeness["suggestions"])

        return {
            "overall_score": overall,
            "density": density,
            "relevance": relevance,
            "completeness": completeness,
            "suggestions": suggestions or ["未发现明显问题"],
        }

    def _parse_sections(self, text: str) -> Dict[str, str]:
        pattern = r"\[(Role & Policies|Task|Evidence|Context|Output)\]\n(.*?)(?=\n\[|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)
        return {name: content.strip() for name, content in matches}

    def _evaluate_density(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """信息密度:警惕[Context]区堆了大量重复/水分内容(比如同一句话反复出现)"""
        context_text = sections.get("Context", "")
        lines = [l.strip() for l in context_text.split("\n") if l.strip()]
        unique_lines = set(lines)

        duplication_ratio = 1 - (len(unique_lines) / len(lines)) if lines else 0.0
        score = max(0.0, 1.0 - duplication_ratio)

        suggestions = []
        if duplication_ratio > 0.2:
            suggestions.append(f"[Context]区重复内容占比{duplication_ratio*100:.0f}%,建议在Select阶段做去重")

        return {"score": round(score, 3), "duplication_ratio": round(duplication_ratio, 3), "suggestions": suggestions}

    def _evaluate_relevance(self, sections: Dict[str, str], user_query: str) -> Dict[str, Any]:
        """相关性:粗略估计Evidence/Context内容和query的字符级重合度,识别"文不对题"的证据"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        combined = (sections.get("Evidence", "") + "\n" + sections.get("Context", "")).strip()
        if not combined or not user_query.strip():
            return {"score": 0.0, "suggestions": ["[Evidence]和[Context]均为空,上下文可能信息不足"]}

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        try:
            vecs = vectorizer.fit_transform([combined, user_query])
            sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
        except ValueError:
            sim = 0.0

        suggestions = []
        if sim < 0.05:
            suggestions.append("检索到的证据/上下文与用户问题字面相关性很低,建议检查检索query或提高min_relevance阈值")

        return {"score": round(sim, 3), "suggestions": suggestions}

    def _evaluate_completeness(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """完整性:必要分区是否缺失(尤其是被Compress阶段整段丢弃后)"""
        required = ["Task", "Output"]
        recommended = ["Role & Policies", "Evidence", "Context"]

        missing_required = [s for s in required if s not in sections]
        missing_recommended = [s for s in recommended if s not in sections]

        score = 1.0 - 0.5 * len(missing_required) - 0.15 * len(missing_recommended)
        score = max(0.0, score)

        suggestions = []
        if missing_required:
            suggestions.append(f"缺少关键分区: {missing_required},上下文可能因Compress阶段被过度截断")
        if missing_recommended:
            suggestions.append(f"缺少建议分区: {missing_recommended},可能是检索没召回任何内容")

        return {"score": round(score, 3), "missing_required": missing_required,
                "missing_recommended": missing_recommended, "suggestions": suggestions}
