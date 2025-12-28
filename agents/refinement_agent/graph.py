from langgraph.graph import StateGraph, START, END
from agents.refinement_agent.nodes.writing import prepare_subsection_context, write_subsection
from agents.refinement_agent.nodes.review_content import review_content
from agents.refinement_agent.nodes.review_grounding import review_grounding
from agents.refinement_agent.nodes.feedback_processing import process_content_feedback, process_grounding_feedback
from agents.shared.state.main_state import AgentState
from agents.refinement_agent.nodes.state_management import (
    initialise_refinement_progress,
    complete_refinement,
    cleanup_temp_cache,
    advance_to_next,
    content_review_passed,
    grounding_review_passed,
    has_more_subsections,
)

workflow = StateGraph(AgentState)

# register nodes
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

# linear edges (single destination)
workflow.add_edge(START, "initialise_refinement_progress")
workflow.add_edge("initialise_refinement_progress", "prepare_subsection_context")
workflow.add_edge("prepare_subsection_context", "write_subsection")
workflow.add_edge("write_subsection", "review_content")
workflow.add_edge("review_content", "process_content_feedback")
workflow.add_edge("review_grounding", "process_grounding_feedback")
workflow.add_edge("complete_refinement", "cleanup_temp_cache")
workflow.add_edge("cleanup_temp_cache", END)

# conditional edges (2 destinations each)

workflow.add_conditional_edges(
    "process_content_feedback",
    content_review_passed,
    {"passed": "review_grounding", "retry": "review_content"}
)

workflow.add_conditional_edges(
    "process_grounding_feedback",
    grounding_review_passed,
    {"passed": "advance_to_next", "retry": "review_grounding"}
)

workflow.add_conditional_edges(
    "advance_to_next",
    has_more_subsections,
    {"continue": "prepare_subsection_context", "complete": "complete_refinement"}
)

refinement_graph = workflow.compile()
