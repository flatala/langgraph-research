from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.shared.state.main_state import AgentState
from agents.refinement_agent.nodes.refinement_nodes import (
    initialise_refinement_progress
)

workflow = StateGraph(AgentState)
workflow.add_node("initialise_refinement_progress", initialise_refinement_progress)

workflow.add_edge(START, "initialise_refinement_progress")
workflow.add_edge("initialise_refinement_progress", END)

refinement_graph = workflow.compile()