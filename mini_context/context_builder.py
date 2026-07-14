"""
按《HelloAgents》第九章 9.3 节代码逐字实现的 ContextBuilder。
不做任何"自己发挥"的改动——评分方式(Jaccard相关性+指数衰减新近性)、
GSSC四阶段的具体逻辑、_count_tokens的估算公式,全部与书里一致。
"""
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class ContextPacket:
    """候选信息包

    Attributes:
        content: 信息内容
        timestamp: 时间戳
        token_count: Token 数量
        relevance_score: 相关性分数(0.0-1.0)
        metadata: 可选的元数据
    """
    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class ContextConfig:
    """上下文构建配置

    Attributes:
        max_tokens: 最大 token 数量
        reserve_ratio: 为系统指令预留的比例(0.0-1.0)
        min_relevance: 最低相关性阈值
        enable_compression: 是否启用压缩
        recency_weight: 新近性权重(0.0-1.0)
        relevance_weight: 相关性权重(0.0-1.0)
    """
    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7

    def __post_init__(self):
        assert 0.0 <= self.reserve_ratio <= 1.0, "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, "min_relevance 必须在 [0, 1] 范围内"
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6, \
            "recency_weight + relevance_weight 必须等于 1.0"


class ContextBuilder:
    """GSSC(Gather-Select-Structure-Compress)上下文构建器。"""

    def __init__(self, memory_tool=None, rag_tool=None, config: Optional[ContextConfig] = None):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()

    # ---------- 对外唯一入口 ----------
    def build(
        self,
        user_query: str,
        conversation_history: Optional[List] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
    ) -> str:
        packets = self._gather(user_query, conversation_history, system_instructions, custom_packets)

        available_tokens = int(self.config.max_tokens * (1 - self.config.reserve_ratio)) \
            if False else self.config.max_tokens
        # 说明:书里 reserve_ratio 的字面用途是"为系统指令预留比例"这一设计意图,
        # 但9.3.3给出的_select实现中系统指令是按"实际占用"扣除、而不是按reserve_ratio硬预留一个池子。
        # 这里保持和书中_select函数完全一致的行为(见下方_select),reserve_ratio暂未在计算里二次使用,
        # 这是书中原始实现的一个已知含糊点,会在下方"忠实实现的思考"里详细说明,不做擅自"修正"。

        selected = self._select(packets, user_query, self.config.max_tokens)
        structured = self._structure(selected, user_query)

        if self.config.enable_compression:
            structured = self._compress(structured, self.config.max_tokens)

        return structured

    # ---------- Gather ----------
    def _gather(
        self,
        user_query: str,
        conversation_history: Optional[List] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
    ) -> List[ContextPacket]:
        packets = []

        # 1. 添加系统指令(最高优先级,不参与评分)
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                timestamp=datetime.now(),
                token_count=self._count_tokens(system_instructions),
                relevance_score=1.0,
                metadata={"type": "system_instruction", "priority": "high"}
            ))

        # 2. 从记忆系统检索相关记忆
        if self.memory_tool:
            try:
                memory_results = self.memory_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 10,
                    "min_importance": 0.3
                })
                memory_packets = self._parse_memory_results(memory_results, user_query)
                packets.extend(memory_packets)
            except Exception as e:
                print(f"[WARNING] 记忆检索失败: {e}")

        # 3. 从 RAG 系统检索相关知识
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5,
                    "min_score": 0.3
                })
                rag_packets = self._parse_rag_results(rag_results, user_query)
                packets.extend(rag_packets)
            except Exception as e:
                print(f"[WARNING] RAG检索失败: {e}")

        # 4. 添加对话历史(仅保留最近的5条)
        if conversation_history:
            recent_history = conversation_history[-5:]
            for msg in recent_history:
                packets.append(ContextPacket(
                    content=f"{msg.role}: {msg.content}",
                    timestamp=msg.timestamp if hasattr(msg, "timestamp") else datetime.now(),
                    token_count=self._count_tokens(msg.content),
                    relevance_score=0.6,
                    metadata={"type": "conversation_history", "role": msg.role}
                ))

        # 5. 添加自定义信息包(比如NoteTool检索出的笔记)
        if custom_packets:
            packets.extend(custom_packets)

        print(f"[ContextBuilder] 汇集了 {len(packets)} 个候选信息包")
        return packets

    def _parse_memory_results(self, memory_results, query) -> List[ContextPacket]:
        """书中未给出具体实现,这里按最自然的方式桥接:
        假设memory_tool.run(search)返回 {"results": [{"content", "timestamp", "importance"}, ...]}"""
        packets = []
        for item in memory_results.get("results", []):
            packets.append(ContextPacket(
                content=f"[记忆] {item['content']}",
                timestamp=item.get("timestamp", datetime.now()),
                token_count=self._count_tokens(item["content"]),
                relevance_score=0.5,  # 交给_select里的_calculate_relevance重新计算
                metadata={"type": "episodic_or_semantic_memory"}
            ))
        return packets

    def _parse_rag_results(self, rag_results, query) -> List[ContextPacket]:
        packets = []
        for item in rag_results.get("results", []):
            packets.append(ContextPacket(
                content=item["content"],
                timestamp=item.get("timestamp", datetime.now()),
                token_count=self._count_tokens(item["content"]),
                relevance_score=item.get("score", 0.5),
                metadata={"type": "rag_result"}
            ))
        return packets

    # ---------- Select ----------
    def _select(self, packets: List[ContextPacket], user_query: str, available_tokens: int) -> List[ContextPacket]:
        system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        other_packets = [p for p in packets if p.metadata.get("type") != "system_instruction"]

        system_tokens = sum(p.token_count for p in system_packets)
        remaining_tokens = available_tokens - system_tokens

        if remaining_tokens <= 0:
            print("[WARNING] 系统指令已占满所有token预算")
            return system_packets

        scored_packets = []
        for packet in other_packets:
            if packet.relevance_score == 0.5:
                relevance = self._calculate_relevance(packet.content, user_query)
                packet.relevance_score = relevance

            recency = self._calculate_recency(packet.timestamp)

            combined_score = (
                self.config.relevance_weight * packet.relevance_score +
                self.config.recency_weight * recency
            )

            if packet.relevance_score >= self.config.min_relevance:
                scored_packets.append((combined_score, packet))

        scored_packets.sort(key=lambda x: x[0], reverse=True)

        selected = system_packets.copy()
        current_tokens = system_tokens

        for score, packet in scored_packets:
            if current_tokens + packet.token_count <= available_tokens:
                selected.append(packet)
                current_tokens += packet.token_count
            else:
                break

        print(f"[ContextBuilder] 选择了 {len(selected)} 个信息包,共 {current_tokens} tokens")
        return selected

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Jaccard相似度(书中原始实现:简单关键词重叠,非向量相似度)"""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        intersection = content_words & query_words
        union = content_words | query_words
        return len(intersection) / len(union) if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        """指数衰减模型:24小时内保持高分,之后逐渐衰减"""
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        decay_factor = 0.1
        recency_score = math.exp(-decay_factor * age_hours / 24)
        return max(0.1, min(1.0, recency_score))

    # ---------- Structure ----------
    def _structure(self, selected_packets: List[ContextPacket], user_query: str) -> str:
        system_instructions = []
        evidence = []
        context = []

        for packet in selected_packets:
            packet_type = packet.metadata.get("type", "general")
            if packet_type == "system_instruction":
                system_instructions.append(packet.content)
            elif packet_type in ["rag_result", "knowledge"]:
                evidence.append(packet.content)
            else:
                context.append(packet.content)

        sections = []
        if system_instructions:
            sections.append("[Role & Policies]\n" + "\n".join(system_instructions))
        sections.append(f"[Task]\n{user_query}")
        if evidence:
            sections.append("[Evidence]\n" + "\n---\n".join(evidence))
        if context:
            sections.append("[Context]\n" + "\n".join(context))
        sections.append("[Output]\n请基于以上信息,提供准确、有据的回答。")

        return "\n\n".join(sections)

    # ---------- Compress ----------
    def _compress(self, context: str, max_tokens: int) -> str:
        current_tokens = self._count_tokens(context)
        if current_tokens <= max_tokens:
            return context

        print(f"[ContextBuilder] 上下文超限({current_tokens} > {max_tokens}),执行压缩")

        sections = context.split("\n\n")
        compressed_sections = []
        current_total = 0

        for section in sections:
            section_tokens = self._count_tokens(section)
            if current_total + section_tokens <= max_tokens:
                compressed_sections.append(section)
                current_total += section_tokens
            else:
                remaining_tokens = max_tokens - current_total
                if remaining_tokens > 50:
                    truncated = self._truncate_text(section, remaining_tokens)
                    compressed_sections.append(truncated + "\n[... 内容已压缩 ...]")
                break

        compressed_context = "\n\n".join(compressed_sections)
        final_tokens = self._count_tokens(compressed_context)
        print(f"[ContextBuilder] 压缩完成: {current_tokens} -> {final_tokens} tokens")
        return compressed_context

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        token_count = self._count_tokens(text)
        char_per_token = len(text) / token_count if token_count > 0 else 4
        max_chars = int(max_tokens * char_per_token)
        return text[:max_chars]

    def _count_tokens(self, text: str) -> int:
        """书中原始的粗估算:中文1字符≈1token,英文1单词≈1.3token"""
        chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        english_words = len([w for w in text.split() if w])
        return int(chinese_chars + english_words * 1.3)
