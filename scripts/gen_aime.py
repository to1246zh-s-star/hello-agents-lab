"""生成 AIME 风格数学题（自定义实现，库不提供生成器，只提供读取器）"""
from dotenv import load_dotenv; load_dotenv(override=True)
from hello_agents import HelloAgentsLLM
import json, re, time
from datetime import datetime

llm = HelloAgentsLLM()

GEN_PROMPT = """请创作一道AIME风格的数学题目（美国数学邀请赛，答案必须是0-999的整数）。
主题：{topic}
要求：
1. 题目表述清晰、无歧义
2. 难度中等偏难，需要多步推理，不能用简单公式一步得出
3. 答案必须是 0-999 之间的整数
4. 提供完整解答过程

请严格按以下JSON格式输出，不要有多余文字：
{{"problem": "题目内容", "answer": "答案数字", "solution": "完整解答过程", "topic": "{topic}", "difficulty": "medium"}}"""

TOPICS = ["代数", "几何", "数论", "组合", "概率"]

def extract_json(text):
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None

def generate_batch(n=10):
    results = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        prompt = GEN_PROMPT.format(topic=topic)
        raw = llm.invoke([{"role": "user", "content": prompt}])
        item = extract_json(raw)
        if item and item.get("answer", "").strip().lstrip("-").isdigit():
            item["id"] = f"gen_{i:03d}"
            results.append(item)
            print(f"[{i+1}/{n}] ✅ {topic} - 答案={item['answer']}")
        else:
            print(f"[{i+1}/{n}] ❌ 解析失败或答案非法整数，跳过")
        time.sleep(2.5)   # 避免 ModelScope 限流
    return results

if __name__ == "__main__":
    problems = generate_batch(n=10)   # 先10道验证流程，跑通后可加量到30
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data_generation/generated_data/aime_generated_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 生成 {len(problems)}/10 道合格题目，已保存: {out_path}")
