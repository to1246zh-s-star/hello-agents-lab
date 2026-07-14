"""
升级5:Prompt Caching感知的上下文排布。

背景(2026年主流LLM API的通行做法,含Anthropic的prompt caching):
缓存命中要求"前缀完全一致"——只要prefix里有一个token变了,这段缓存就失效,
后面全部要重新计算。所以上下文里"稳定不变的部分"应该放在最前面,
"每次都不同的动态部分"应该放在最后面,而不是随便拼接。

ContextBuilder._structure()按[Role&Policies]→[Task]→[Evidence]→[Context]→[Output]
的固定顺序输出,这个顺序本身对caching并不友好:
[Task]是最动态的内容(每次query都变),却被排在了第二位,
会导致[Evidence]和[Context]即使内容不变,也因为前面的[Task]变了而无法复用缓存。

这里提供一个重排函数,把稳定部分(Role&Policies、Evidence、Context里的"旧"部分)
挪到前面,把动态部分(当前Task、Output指令)挪到后面。
"""
import re
from typing import Dict


def reorder_for_prompt_caching(built_context: str) -> Dict[str, str]:
    """
    将ContextBuilder产出的文本拆成:
    - cacheable_prefix: 稳定、可复用的部分(Role&Policies + Evidence + Context)
    - dynamic_suffix: 每次必变的部分(Task + Output)

    实际调用LLM时:
    messages = [
        {"role": "system", "content": cacheable_prefix, "cache_control": {"type": "ephemeral"}},  # 示意,具体字段以实际API为准
        {"role": "user", "content": dynamic_suffix},
    ]
    """
    pattern = r"\[(Role & Policies|Task|Evidence|Context|Output)\]\n(.*?)(?=\n\[|\Z)"
    matches = re.findall(pattern, built_context, re.DOTALL)
    sections = {name: content.strip() for name, content in matches}

    stable_order = ["Role & Policies", "Evidence", "Context"]
    dynamic_order = ["Task", "Output"]

    cacheable_prefix = "\n\n".join(
        f"[{name}]\n{sections[name]}" for name in stable_order if name in sections
    )
    dynamic_suffix = "\n\n".join(
        f"[{name}]\n{sections[name]}" for name in dynamic_order if name in sections
    )

    return {"cacheable_prefix": cacheable_prefix, "dynamic_suffix": dynamic_suffix}
