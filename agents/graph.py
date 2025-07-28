from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.shared.state import LitState
from agents.planning_agent.graph import planning_graph

workflow = StateGraph(LitState)

workflow.add_node("planning", planning_graph)

workflow.add_edge(START, "planning")
workflow.add_edge("planning", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)