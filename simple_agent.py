import os
import json
from openai import OpenAI
from agent import Agent

class MySimpleAgent(Agent):
    def __init__(self, name, llm, system_prompt="你是一个有用的助手",
                 tool_registry=None, enable_tool_calling=False):
        super().__init__(name, llm)
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling

        # 直接建一个openai client，用于支持function calling和流式（这两个能力
        # 需要访问更底层的API参数，我们自己实现的llm.invoke()不一定封装了这些）
        self._client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )
        self._model = os.getenv("LLM_MODEL_ID")

    # ---------- 基础对话 ----------
    def run(self, user_input: str) -> str:
        self.add_message(user_input, "user")
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [m.to_dict() for m in self.get_history()]

        if self.enable_tool_calling and self.tool_registry and self.tool_registry.has_tools():
            return self._run_with_tools(messages)

        response = self._client.chat.completions.create(
            model=self._model, messages=messages
        )
        response_text = response.choices[0].message.content
        self.add_message(response_text, "assistant")
        return response_text

    # ---------- 带工具调用的对话 ----------
    def _run_with_tools(self, messages):
        tools_schema = self.tool_registry.get_openai_schemas()

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools_schema
        )
        message = response.choices[0].message

        # 模型判断需要调用工具
        if message.tool_calls:
            messages.append(message.model_dump())
            for call in message.tool_calls:
                tool_name = call.function.name
                tool_args = json.loads(call.function.arguments)
                tool = self.tool_registry.get_tool(tool_name)
                result = tool.run(tool_args) if tool else f"未知工具: {tool_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })

            # 把工具结果喂回去，让模型生成最终回复
            final_response = self._client.chat.completions.create(
                model=self._model, messages=messages
            )
            final_text = final_response.choices[0].message.content
            self.add_message(final_text, "assistant")
            return final_text
        else:
            # 模型判断不需要工具，直接回答
            response_text = message.content
            self.add_message(response_text, "assistant")
            return response_text

    # ---------- 流式响应 ----------
    def stream_run(self, user_input: str):
        self.add_message(user_input, "user")
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [m.to_dict() for m in self.get_history()]

        stream = self._client.chat.completions.create(
            model=self._model, messages=messages, stream=True
        )

        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full_text += delta
                yield delta
        print()  # 换行
        self.add_message(full_text, "assistant")

    # ---------- 动态工具管理 ----------
    def add_tool(self, tool):
        if self.tool_registry is None:
            from tools import ToolRegistry
            self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(tool)
        self.enable_tool_calling = True

    def has_tools(self) -> bool:
        return self.tool_registry is not None and self.tool_registry.has_tools()

    def list_tools(self):
        return self.tool_registry.list_tools() if self.tool_registry else []
