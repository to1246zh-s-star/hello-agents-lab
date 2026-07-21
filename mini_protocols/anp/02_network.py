from hello_agents.protocols import ANPDiscovery, register_service, ANPNetwork

discovery = ANPDiscovery()
register_service(discovery=discovery, service_id="nlp_agent_1", service_name="NLP处理专家A",
                  service_type="nlp", capabilities=["text_analysis"], endpoint="http://localhost:8001",
                  metadata={"load": 0.3})
register_service(discovery=discovery, service_id="nlp_agent_2", service_name="NLP处理专家B",
                  service_type="nlp", capabilities=["text_analysis"], endpoint="http://localhost:8002",
                  metadata={"load": 0.7})

network = ANPNetwork(network_id="ai_cluster")
for service in discovery.list_all_services():
    network.add_node(service.service_id, service.endpoint)
network.connect_nodes("nlp_agent_1", "nlp_agent_2")

stats = network.get_network_stats()
print(f"✅ 网络构建完成，共 {stats['total_nodes']} 个节点")
