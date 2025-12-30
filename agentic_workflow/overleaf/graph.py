"""Overleaf/LaTeX export graph definition."""

from langgraph.graph import StateGraph, START, END

from agentic_workflow.shared.state.main_state import AgentState
from agentic_workflow.overleaf.nodes.metadata import (
    collect_papers,
    generate_title,
    generate_section_intros,
    generate_subsection_titles,
)
from agentic_workflow.overleaf.nodes.latex_generation import (
    generate_latex_content,
    generate_bibtex_content,
)
from agentic_workflow.overleaf.nodes.export import create_zip_export


workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("collect_papers", collect_papers)
workflow.add_node("generate_title", generate_title)
workflow.add_node("generate_section_intros", generate_section_intros)
workflow.add_node("generate_subsection_titles", generate_subsection_titles)
workflow.add_node("generate_latex_content", generate_latex_content)
workflow.add_node("generate_bibtex_content", generate_bibtex_content)
workflow.add_node("create_zip_export", create_zip_export)

# Define edges
workflow.add_edge(START, "collect_papers")
workflow.add_edge("collect_papers", "generate_title")
workflow.add_edge("generate_title", "generate_section_intros")
workflow.add_edge("generate_section_intros", "generate_subsection_titles")
workflow.add_edge("generate_subsection_titles", "generate_latex_content")
workflow.add_edge("generate_latex_content", "generate_bibtex_content")
workflow.add_edge("generate_bibtex_content", "create_zip_export")
workflow.add_edge("create_zip_export", END)

overleaf_graph = workflow.compile()
