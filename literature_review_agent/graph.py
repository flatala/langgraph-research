from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import chat_agent_executor, ToolNode, tools_condition
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from literature_review_agent.state import LitState, Section
from literature_review_agent.utils import get_text_llm, get_orchestrator_llm
from literature_review_agent.configuration import Configuration  
from literature_review_agent.tools import arxiv_search, summarise_text

from typing import List, Optional
from typing_extensions import TypedDict, Annotated
from dataclasses import field
from pprint import pprint
import json, re


async def prepare_search_queries(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    cfg = Configuration.from_runnable_config(config)

    prompt = cfg.query_refinement_prompt.format(
        query_count=cfg.refined_query_count,
        topic=state.topic,
    )

    llm = (
        get_text_llm(cfg=cfg)
        .with_config({"response_format": {"type": "json_object"}})
    )

    messages = state.messages.copy()
    messages.append(HumanMessage(content=prompt))

    ai_msg: AIMessage = await llm.ainvoke(messages)
    messages.append(ai_msg)

    raw = re.sub(r"^```[\w-]*\n|\n```$", "", ai_msg.content.strip(), flags=re.S)
    data = json.loads(raw)
    if isinstance(data, dict) and "queries" in data:
        queries: List[str] = data["queries"]
    elif isinstance(data, list):
        queries = data
    else:
        raise ValueError("Unexpected JSON structure returned by LLM")

    return {
        "search_queries": queries,
        "messages": messages,
    }


async def plan_review(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    cfg = Configuration.from_runnable_config(config)

    queries_str = "; ".join(state.search_queries)
    prompt = cfg.research_prompt.format(
        topic=state.topic,
        paper_recency=state.paper_recency,
        search_queries=queries_str
    )

    tools = [arxiv_search]
    llm = (
        get_orchestrator_llm(cfg=cfg)
        .bind_tools(tools, tool_choice="auto")
        .with_config({"response_format": {"type": "json_object"}})
    )

    tool_map = {t.name: t for t in tools}

    messages = state.messages.copy()  
    messages.append(HumanMessage(content=prompt))

    while True:
        ai_msg: AIMessage = await llm.ainvoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:         
            break

        for call in ai_msg.tool_calls:
            tool = tool_map[call["name"]]

            if call["name"] == "arxiv_search":
                papers = await tool.ainvoke(call["args"])

                slim_papers = [
                    {
                        "title": p["title"],
                        "url":   p["url"],
                        "comment": p["summary"],
                    }
                    for p in papers
                ]

                result_for_history = slim_papers

            else:
                result_for_history = await tool.ainvoke(call["args"])

            messages.append(
                ToolMessage(
                    name=call["name"],
                    tool_call_id=call["id"],
                    content=json.dumps(result_for_history)
                )
            )

    text = re.sub(r"^```[\w-]*\n|\n```$", "", messages[-1].content.strip(), flags=re.S)
    plan: List[Section] = json.loads(text)

    return {
        "plan": plan,
        "messages": messages,
    }

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
tools_node = ToolNode([arxiv_search])

builder.add_node("prepare_search_queries", prepare_search_queries) 
builder.add_node("plan_literature_review", plan_review)
builder.add_node("tools", tools_node)

builder.add_edge(START, "prepare_search_queries")
builder.add_edge("prepare_search_queries", "plan_literature_review")
builder.add_conditional_edges(
    "plan_literature_review",
    tools_condition,              
    {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "plan_literature_review")

graph = builder.compile()