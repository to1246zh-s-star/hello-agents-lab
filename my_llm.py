from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """
    继承HelloAgentsLLM，演示如何在不修改源码的前提下扩展新的provider。
    """
    def __init__(self, provider=None, **kwargs):
        if provider == "my_custom_provider":
            # 这里写你自己的初始化逻辑
            # 比如：固定某个base_url、传入特殊的请求头等
            print("走的是自定义分支：my_custom_provider")
            custom_base_url = "https://api-inference.modelscope.cn/v1/"
            super().__init__(provider="modelscope", base_url=custom_base_url, **kwargs)
        else:
            # 其他情况，完全交还给父类处理（比如默认的modelscope自动检测）
            print("走的是默认父类逻辑")
            super().__init__(provider=provider, **kwargs)
