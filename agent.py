from abc import ABC, abstractmethod
from message import Message

class Agent(ABC):
    def __init__(self, name: str, llm):
        self.name = name
        self.llm = llm
        self._history = []

    @abstractmethod
    def run(self, user_input: str) -> str:
        ...

    def add_message(self, content: str, role: str):
        self._history.append(Message(content, role))

    def get_history(self):
        return self._history

    def clear_history(self):
        self._history = []
