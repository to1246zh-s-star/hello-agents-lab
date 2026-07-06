import time
from openai import RateLimitError

def call_with_retry(func, max_retries=3, base_delay=3):
    """
    对可能触发限流的函数调用做自动重试。
    func: 一个无参数的函数（用lambda包装你的实际调用）
    """
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = base_delay * (2 ** attempt)  # 3s -> 6s -> 12s
            print(f"\n[限流，第{attempt+1}次重试，等待{wait_time}秒...]")
            time.sleep(wait_time)
    raise Exception("多次重试后仍然被限流，请稍后再试")
