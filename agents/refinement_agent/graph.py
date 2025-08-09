from langgraph.graph import StateGraph, START, END
from agents.shared.state.main_state import AgentState
from agents.refinement_agent.nodes.state_management_nodes import (
    initialise_refinement_progress,
    decide_refinement_stage,
    complete_refinement
)
from agents.refinement_agent.nodes.refinement_nodes import (
    prepare_subsection_context
)

workflow = StateGraph(AgentState)

workflow.add_node("initialise_refinement_progress", initialise_refinement_progress)
workflow.add_node("router", decide_refinement_stage)  # Central routing hub
workflow.add_node("prepare_subsection_context", prepare_subsection_context)
workflow.add_node("complete_refinement", complete_refinement)

ACTION_NODES = ["prepare_subsection_context"]
ROUTE_MAP = {
    "prepare_subsection_context": "prepare_subsection_context",
    "complete_refinement": "complete_refinement",
    # "write_subsection": "write_subsection",
    # "review_content": "review_content", 
    # "review_grounding": "review_grounding",
    # "process_feedback": "process_feedback",
    # "start_revision": "start_revision",
    # "advance_to_next": "advance_to_next",
}

workflow.add_edge(START, "initialise_refinement_progress")
workflow.add_edge("initialise_refinement_progress", "router")
workflow.add_conditional_edges("router", decide_refinement_stage, ROUTE_MAP)

# set up routing for all the refinement nodes
for node in ACTION_NODES:
    workflow.add_edge(node, "router")

workflow.add_edge("complete_refinement", END)

refinement_graph = workflow.compile()