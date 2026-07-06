from typing import Literal
from dataclasses import dataclass

Role = Literal["user", "assistant", "system", "tool"]

@dataclass
class Message:
    content: str
    role: Role

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}
