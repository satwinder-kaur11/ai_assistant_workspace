from app.agent.graph import build_graph

def test_agent_graph_init():
    graph = build_graph()
    assert graph is not None
