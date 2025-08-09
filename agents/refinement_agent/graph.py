from langgraph.graph import StateGraph, START, END
from agents.shared.state.main_state import AgentState
from agents.refinement_agent.nodes.state_management_nodes import (
    initialise_refinement_progress,
    decide_refinement_stage,
    complete_refinement,
    advance_to_next
)
from agents.refinement_agent.nodes.refinement_nodes import (
    prepare_subsection_context,
    write_subsection,
    review_content,
    review_grounding,
    process_feedback,
    start_revision
)

workflow = StateGraph(AgentState)

workflow.add_node("initialise_refinement_progress", initialise_refinement_progress)
workflow.add_node("prepare_subsection_context", prepare_subsection_context)
workflow.add_node("write_subsection", write_subsection)
workflow.add_node("review_content", review_content)
workflow.add_node("review_grounding", review_grounding)
workflow.add_node("process_feedback", process_feedback)
workflow.add_node("start_revision", start_revision)
workflow.add_node("advance_to_next", advance_to_next)
workflow.add_node("complete_refinement", complete_refinement)

ACTION_NODES = [
    "prepare_subsection_context",
    "write_subsection", 
    "review_content",
    "review_grounding",
    "process_feedback",
    "start_revision",
    "advance_to_next"
]

ROUTE_MAP = {
    "prepare_subsection_context": "prepare_subsection_context",
    "write_subsection": "write_subsection",
    "review_content": "review_content", 
    "review_grounding": "review_grounding",
    "process_feedback": "process_feedback",
    "start_revision": "start_revision",
    "advance_to_next": "advance_to_next",
    "complete_refinement": "complete_refinement"
}

workflow.add_edge(START, "initialise_refinement_progress")
workflow.add_conditional_edges("initialise_refinement_progress", decide_refinement_stage, ROUTE_MAP)
for node in ACTION_NODES:
    workflow.add_conditional_edges(node, decide_refinement_stage, ROUTE_MAP)
workflow.add_edge("complete_refinement", END)

refinement_graph = workflow.compile()