from langgraph.graph import StateGraph, START, END
from agents.shared.state.main_state import AgentState
from agents.refinement_agent.nodes.state_management import (
    initialise_refinement_progress,
    decide_refinement_stage,
    complete_refinement,
    cleanup_temp_cache,
    advance_to_next
)
from agents.refinement_agent.nodes.writing import prepare_subsection_context, write_subsection
from agents.refinement_agent.nodes.review_content import review_content
from agents.refinement_agent.nodes.review_grounding import review_grounding
from agents.refinement_agent.nodes.feedback_processing import (
    process_content_feedback,
    process_grounding_feedback
)

workflow = StateGraph(AgentState)

# Register all nodes
workflow.add_node("initialise_refinement_progress", initialise_refinement_progress)
workflow.add_node("prepare_subsection_context", prepare_subsection_context)
workflow.add_node("write_subsection", write_subsection)
workflow.add_node("review_content", review_content)
workflow.add_node("process_content_feedback", process_content_feedback)
workflow.add_node("review_grounding", review_grounding)
workflow.add_node("process_grounding_feedback", process_grounding_feedback)
workflow.add_node("advance_to_next", advance_to_next)
workflow.add_node("complete_refinement", complete_refinement)
workflow.add_node("cleanup_temp_cache", cleanup_temp_cache)

# List of action nodes that route through decide_refinement_stage
ACTION_NODES = [
    "prepare_subsection_context",
    "write_subsection",
    "review_content",
    "process_content_feedback",
    "review_grounding",
    "process_grounding_feedback",
    "advance_to_next"
]

# Route map for conditional edges
ROUTE_MAP = {
    "prepare_subsection_context": "prepare_subsection_context",
    "write_subsection": "write_subsection",
    "review_content": "review_content",
    "process_content_feedback": "process_content_feedback",
    "review_grounding": "review_grounding",
    "process_grounding_feedback": "process_grounding_feedback",
    "advance_to_next": "advance_to_next",
    "complete_refinement": "complete_refinement"
}

# Define edges
workflow.add_edge(START, "initialise_refinement_progress")
workflow.add_conditional_edges("initialise_refinement_progress", decide_refinement_stage, ROUTE_MAP)

# All action nodes route back through decide_refinement_stage
for node in ACTION_NODES:
    workflow.add_conditional_edges(node, decide_refinement_stage, ROUTE_MAP)

workflow.add_edge("complete_refinement", "cleanup_temp_cache")
workflow.add_edge("cleanup_temp_cache", END)

refinement_graph = workflow.compile()
