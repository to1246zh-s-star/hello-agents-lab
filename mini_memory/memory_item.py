"""实验1：记忆系统的基础数据结构
对应书中 MemoryItem / MemoryConfig
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class MemoryItem:
    """标准化的记忆项，四种记忆类型最终都落到这个结构上"""
    content: str
    memory_type: str = "working"          # working / episodic / semantic / perceptual
    importance: float = 0.5               # 0.0-1.0，模拟人脑对信息重要性的评估
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0                 # 后面做"智能遗忘"要用到的访问频率

    def touch(self):
        """每次被检索命中时调用，用于访问频率统计"""
        self.access_count += 1


@dataclass
class MemoryConfig:
    """记忆系统全局配置"""
    working_memory_capacity: int = 50
    working_memory_ttl: int = 60          # 分钟
    database_path: str = "./mini_memory.db"
    vector_dim: int = 0                   # TF-IDF模式下不固定维度，训练后确定


if __name__ == "__main__":
    # 自测：验证基础结构可正常创建
    item = MemoryItem(content="用户叫张三，正在学Python", memory_type="episodic", importance=0.8)
    print(f"✅ 创建成功: id={item.id[:8]}..., type={item.memory_type}, importance={item.importance}")
    assert 0.0 <= item.importance <= 1.0
    print("✅ 实验1自测通过")
