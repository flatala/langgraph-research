from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.shared.state.main_state import AgentState
from agents.planning_agent.graph import planning_graph
from agents.refinement_agent.graph import refinement_graph
from agents.latex_builder import build_latex

workflow = StateGraph(AgentState)

workflow.add_node("planning", planning_graph)
workflow.add_node("refinement", refinement_graph)
workflow.add_node("build_latex", build_latex)

workflow.add_edge(START, "planning")
workflow.add_edge("planning", "refinement")
workflow.add_edge("refinement", "build_latex")
workflow.add_edge("build_latex", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)