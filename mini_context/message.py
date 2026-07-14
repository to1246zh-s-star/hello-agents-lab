"""对齐书里hello_agents.core.message.Message的最小实现,只包含本章用到的字段。"""
from datetime import datetime


class Message:
    def __init__(self, content: str, role: str, timestamp: datetime = None):
        self.content = content
        self.role = role
        self.timestamp = timestamp or datetime.now()
