"""绕开 0.2.7 的 level 类型 bug：先全量加载，再手动按 int(level) 过滤"""
from dotenv import load_dotenv; load_dotenv(override=True)
from hello_agents.evaluation import GAIADataset

def load_gaia_by_level(level: int, split: str = "validation"):
    ds = GAIADataset(level=None, split=split)   # 不传 level，避免内部 bug 过滤
    all_items = ds.load()                        # 拿到全部 165 条
    # 手动转 int 再筛（兼容 level 是 str 或 int 两种情况）
    filtered = [it for it in all_items if int(it.get("level", -1)) == level]
    return filtered

if __name__ == "__main__":
    for lv in [1, 2, 3]:
        items = load_gaia_by_level(lv)
        print(f"Level {lv}: {len(items)} 条")
