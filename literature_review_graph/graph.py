from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from typing_extensions import TypedDict, Annotated
from dataclasses import field
from shared.state import reduce_docs
from .state import LitState
from shared.utils import get_llm      


def plan_review(state: LitState) -> dict:
    """Ask the LLM for a JSON array of section titles."""
    prompt = (
        "You are an expert research assistant.\n"
        f"Generate an outline for a literature review on '{state['topic']}' "
        f"using papers {state['paper_recency']}. "
        "Return a JSON array with section titles only."
    )
    llm = get_llm()    
    outline_json = llm.invoke(prompt).content
    return {"plan": outline_json}


def refine_section(state: LitState) -> dict:
    """Draft the first section (placeholder logic)."""
    first_title = state["plan"].splitlines()[0]   # crude; adjust when outline is JSON
    prompt = (
        f"Write a clear, 2-3 sentence draft for the section: '{first_title}'. "
        "Assume the target reader is a grad student."
    )
    llm = get_llm()    
    draft = llm.invoke(prompt).content
    return {"draft_sections": [draft]}


def verify_section(state: LitState) -> dict:
    """Light-weight factuality pass—returns the text unchanged if it looks fine."""
    draft = state["draft_sections"][0]
    prompt = (
        "Check the following paragraph for factual consistency. "
        "If it is correct, return it unchanged; otherwise, return a corrected version.\n\n"
        f"{draft}"
    )
    llm = get_llm()    
    verified = llm.invoke(prompt).content
    return {"verified_sections": [verified]}


builder = StateGraph(LitState)
builder.add_node("plan_literature_review", plan_review)
builder.add_node("refine_section", refine_section)
builder.add_node("verify_section", verify_section)

builder.add_edge(START, "plan_literature_review")
builder.add_edge("plan_literature_review", "refine_section")
builder.add_edge("refine_section", "verify_section")
builder.add_edge("verify_section", END)

graph = builder.compile()      