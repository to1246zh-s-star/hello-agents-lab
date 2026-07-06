import os
from pydantic import BaseModel

class Config(BaseModel):
    default_model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1024

    @classmethod
    def from_env(cls):
        return cls(
            default_model=os.getenv("LLM_MODEL_ID", "default"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
        )
