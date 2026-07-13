"""实验3：独立验证四种记忆类型的评分公式
不依赖真实存储，只验证数学逻辑
"""
import math


def working_score(vector_sim, keyword_sim, time_decay, importance):
    base = vector_sim * 0.7 + keyword_sim * 0.3 if vector_sim > 0 else keyword_sim
    return base * time_decay * (0.8 + importance * 0.4)


def episodic_score(vector_sim, recency, importance):
    base = vector_sim * 0.8 + recency * 0.2
    return base * (0.8 + importance * 0.4)


def semantic_score(vector_sim, graph_sim, importance):
    base = vector_sim * 0.7 + graph_sim * 0.3
    return base * (0.8 + importance * 0.4)


def perceptual_score(vector_sim, recency, importance):
    base = vector_sim * 0.8 + recency * 0.2
    return base * (0.8 + importance * 0.4)


def recency_exp_decay(age_hours, decay_factor=0.1):
    return max(0.1, math.exp(-decay_factor * age_hours / 24))


if __name__ == "__main__":
    # 场景：同样是0.9的向量相似度，重要性权重范围应该是[0.8, 1.2]
    low_importance = episodic_score(0.9, 0.5, importance=0.0)
    high_importance = episodic_score(0.9, 0.5, importance=1.0)
    print(f"重要性=0 -> 得分 {low_importance:.3f}")
    print(f"重要性=1 -> 得分 {high_importance:.3f}")
    assert high_importance / low_importance == 1.5, "重要性权重范围应为[0.8,1.2]，比值应为1.5"
    print("✅ 重要性权重范围验证通过")

    # 验证：情景记忆里时间近因性权重(0.2) < 语义记忆里图检索权重(0.3)
    # 这是本章的核心面试点——用同样的vector_sim对比两种类型对"次要因素"的依赖程度
    ep = episodic_score(vector_sim=0.6, recency=1.0, importance=0.5)  # recency拉满
    se = semantic_score(vector_sim=0.6, graph_sim=1.0, importance=0.5)  # graph_sim拉满
    print(f"情景记忆(recency=1.0拉满) -> {ep:.3f}")
    print(f"语义记忆(graph_sim=1.0拉满) -> {se:.3f}")
    assert se > ep, "语义记忆对图检索的权重(0.3)应该比情景记忆对时间近因性的权重(0.2)更高"
    print("✅ 权重差异验证通过：语义记忆更依赖图检索，情景记忆更依赖时间近因性")

    # 验证指数衰减：24小时内保持高分，之后逐渐衰减，且不会衰减到0以下
    print(f"1小时后衰减分: {recency_exp_decay(1):.3f}")
    print(f"24小时后衰减分: {recency_exp_decay(24):.3f}")
    print(f"240小时(10天)后衰减分: {recency_exp_decay(240):.3f}")
    assert recency_exp_decay(240) >= 0.1, "衰减应该有0.1的最低兜底"
    print("✅ 实验3全部自测通过")
