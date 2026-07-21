from hello_agents.protocols.a2a.implementation import A2AServer, A2A_AVAILABLE

def create_calculator_agent():
    if not A2A_AVAILABLE:
        print("❌ A2A SDK 未安装，请运行: pip install a2a-sdk")
        return None

    calculator = A2AServer(
        name="calculator-agent",
        description="专业的数学计算智能体",
        version="1.0.0",
        capabilities={
            "math": ["addition", "subtraction", "multiplication", "division"],
            "advanced": ["power", "sqrt", "factorial"]
        }
    )

    @calculator.skill("add")
    def add_numbers(query: str) -> str:
        parts = query.replace("计算", "").replace("加", "+").replace("加上", "+")
        if "+" in parts:
            numbers = [float(x.strip()) for x in parts.split("+")]
            result = sum(numbers)
            return f"计算结果: {' + '.join(map(str, numbers))} = {result}"
        return "请使用格式: 计算 5 + 3"

    @calculator.skill("multiply")
    def multiply_numbers(query: str) -> str:
        parts = query.replace("计算", "").replace("乘以", "*").replace("×", "*")
        if "*" in parts:
            numbers = [float(x.strip()) for x in parts.split("*")]
            result = 1
            for num in numbers:
                result *= num
            return f"计算结果: {' × '.join(map(str, numbers))} = {result}"
        return "请使用格式: 计算 5 * 3"

    @calculator.skill("info")
    def get_info(query: str) -> str:
        return f"我是 {calculator.name}，支持的技能: {list(calculator.skills.keys())}"

    print(f"✅ 计算器智能体创建成功，支持技能: {list(calculator.skills.keys())}")
    return calculator

calc_agent = create_calculator_agent()
if calc_agent:
    for query in ["获取信息", "计算 10 + 5", "计算 6 * 7"]:
        if "信息" in query:
            result = calc_agent.skills["info"](query)
        elif "+" in query:
            result = calc_agent.skills["add"](query)
        elif "*" in query or "×" in query:
            result = calc_agent.skills["multiply"](query)
        else:
            result = "未知查询类型"
        print(f"  📝 查询: {query}  🤖 回复: {result}")
