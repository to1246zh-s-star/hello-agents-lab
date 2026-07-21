from dotenv import load_dotenv
load_dotenv()

from hello_agents.protocols import ANPDiscovery, register_service
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools.builtin import ANPTool
import random

llm = HelloAgentsLLM()
discovery = ANPDiscovery()

for i in range(10):
    register_service(
        discovery=discovery, service_id=f"compute_node_{i}", service_name=f"计算节点{i}",
        service_type="compute", capabilities=["data_processing", "ml_training"],
        endpoint=f"http://node{i}:8000",
        metadata={
            "load": random.uniform(0.1, 0.9),
            "cpu_cores": random.choice([4, 8, 16]),
            "memory_gb": random.choice([16, 32, 64]),
            "gpu": random.choice([True, False])
        }
    )
print(f"✅ 注册了 {len(discovery.list_all_services())} 个计算节点")

scheduler = SimpleAgent(
    name="任务调度器", llm=llm,
    system_prompt="""你是一个智能任务调度器，负责：
1. 分析任务需求
2. 选择最合适的计算节点
3. 分配任务
选择节点时考虑：负载、CPU核心数、内存、GPU等因素。"""
)
anp_tool = ANPTool(name="service_discovery", description="服务发现工具，可以查找和选择计算节点", discovery=discovery)
scheduler.add_tool(anp_tool)

def assign_task(task_description):
    print(f"\n任务：{task_description}\n{'='*50}")
    response = scheduler.run(f"""
    请为以下任务选择最合适的计算节点：
    {task_description}
    要求：1.列出所有可用节点 2.分析每个节点的特点 3.选择最合适的节点 4.说明选择理由
    """)
    print(response)

assign_task("训练一个大型深度学习模型，需要GPU支持")
assign_task("处理大量文本数据，需要高内存")
assign_task("运行轻量级数据分析任务")
