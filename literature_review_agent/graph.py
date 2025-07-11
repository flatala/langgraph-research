from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from typing_extensions import TypedDict, Annotated
from dataclasses import field
from literature_review_agent.state import LitState, Section
from literature_review_agent.utils import get_llm      
from typing import List
import json, re

def plan_review(state: LitState) -> dict:
    """Return a structured literature-review plan."""
    prompt = (
        "You are an expert research assistant.\n"
        f"Generate a literature-review plan on '{state.topic}' "
        f"using papers {state.paper_recency}.\n"
        "Return JSON exactly in this format:\n"
        "[\n"
        "  {\"title\": <section title>,\n"
        "   \"key_points\": [\n"
        "       {\"text\": <point>, \"papers\": [<url1>, <url2>, ...]},\n"
        "       ...\n"
        "   ]},\n"
        "  ...\n"
        "]"
    )

    def parse_llm_plan(raw: str) -> List[dict]:
        """
        Strip markdown code fences and return a Python list.
        Raises JSONDecodeError if the cleaned string is still invalid.
        """
        cleaned = re.sub(r"^```[\w-]*\n|\n```$", "", raw.strip(), flags=re.S)
        return json.loads(cleaned)

    
    response_text = get_llm().invoke(prompt).content   
    print(f"LLM response: {response_text}")

    try:
        plan: List[Section] = parse_llm_plan(response_text)
    except json.JSONDecodeError as err:
        raise ValueError("LLM did not return valid JSON") from err

    return {"plan": plan}


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
builder.add_edge("plan_literature_review", END)

# builder.add_edge("plan_literature_review", "refine_section")
# builder.add_edge("refine_section", "verify_section")
# builder.add_edge("verify_section", END)

graph = builder.compile()      