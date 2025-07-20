from langgraph.graph import StateGraph, START, END
from literature_review_agent.state import LitState
from literature_review_agent.tools import arxiv_search
from literature_review_agent.nodes.rag_nodes import prepare_rag_knowledge_base
from literature_review_agent.nodes.general_nodes import AsyncToolNode, route_tools
from literature_review_agent.nodes.reserch_plan_nodes import (
    decide_on_start_stage, 
    prepare_search_queries, 
    send_plan_prompt, 
    plan_literature_review, 
    parse_plan, 
    load_cached_plan
)

workflow = StateGraph(LitState)

# Define all nodes
workflow.add_node("load_cached_plan", load_cached_plan)
workflow.add_node("prepare_search_queries", prepare_search_queries)
workflow.add_node("send_plan_prompt", send_plan_prompt)
workflow.add_node("plan_literature_review", plan_literature_review)
workflow.add_node("tools", AsyncToolNode([arxiv_search]))
workflow.add_node("parse_plan", parse_plan)
workflow.add_node("prepare_rag_knowledge_base", prepare_rag_knowledge_base)

# First decide on the initial starting stage
workflow.add_conditional_edges(
    START,
    decide_on_start_stage,
    {"load_cached_plan": "load_cached_plan", "prepare_search_queries": "prepare_search_queries"},
)

# If a cached plan is loaded, go directly to rag setup
workflow.add_edge("load_cached_plan", "prepare_rag_knowledge_base")

# Refine search queries and send the system prompt
workflow.add_edge("prepare_search_queries", "send_plan_prompt")
workflow.add_edge("send_plan_prompt", "plan_literature_review")

# Loop between tool calls and ai model refinement
workflow.add_conditional_edges(
    "plan_literature_review",
    route_tools,
    {"tools": "tools", "__end__": "parse_plan"},
)
workflow.add_edge("tools", "plan_literature_review")

# Parse the plan, save json, setup rag
workflow.add_edge("parse_plan", "prepare_rag_knowledge_base")
workflow.add_edge("prepare_rag_knowledge_base", END)

graph = workflow.compile()
