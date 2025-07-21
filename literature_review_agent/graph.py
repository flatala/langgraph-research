from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from literature_review_agent.state import LitState
from literature_review_agent.tools import arxiv_search, human_assistance
from literature_review_agent.nodes.rag_nodes import prepare_rag_knowledge_base
from literature_review_agent.nodes.general_nodes import AsyncToolNode, route_tools, set_workflow_completed_flag
from literature_review_agent.nodes.reserch_plan_nodes import (
    decide_on_start_stage, 
    plan_literature_review, 
    parse_plan, 
    load_cached_plan,
    append_system_prompt,
    refine_problem_statement,
    parse_queries_add_plan_prompt
)

workflow = StateGraph(LitState)

# Define all nodes
workflow.add_node("load_cached_plan", load_cached_plan)
workflow.add_node("append_system_prompt", append_system_prompt)
workflow.add_node("refine_problem_statement", refine_problem_statement)
workflow.add_node("tools_1", AsyncToolNode([human_assistance]))
workflow.add_node("parse_queries_add_plan_prompt", parse_queries_add_plan_prompt)
workflow.add_node("plan_literature_review", plan_literature_review)
workflow.add_node("tools_2", AsyncToolNode([arxiv_search]))
workflow.add_node("parse_plan", parse_plan)
workflow.add_node("prepare_rag_knowledge_base", prepare_rag_knowledge_base)
workflow.add_node("set_workflow_completed_flag", set_workflow_completed_flag)

# First decide on the initial starting stage
workflow.add_conditional_edges(
    START,
    decide_on_start_stage,
    {"load_cached_plan": "load_cached_plan", "start_from_scratch": "append_system_prompt"},
)

# If a cached plan is loaded, go directly to rag setup
workflow.add_edge("load_cached_plan", "prepare_rag_knowledge_base")

# if we do refinement, append system + user prompts
workflow.add_edge("append_system_prompt", "refine_problem_statement")

# Loop between llm and human inputs
workflow.add_conditional_edges(
    "refine_problem_statement",
    route_tools,
    {"tools": "tools_1", "__end__": "parse_queries_add_plan_prompt"})
workflow.add_edge("tools_1", "refine_problem_statement")

# Go to the planning phase
workflow.add_edge("parse_queries_add_plan_prompt", "plan_literature_review")

# Loop between tool calls and ai model refinement
workflow.add_conditional_edges(
    "plan_literature_review",
    route_tools,
    {"tools": "tools_2", "__end__": "parse_plan"},
)
workflow.add_edge("tools_2", "plan_literature_review")

# Parse the plan, save json, setup rag
workflow.add_edge("parse_plan", "prepare_rag_knowledge_base")
workflow.add_edge("prepare_rag_knowledge_base", "set_workflow_completed_flag")
workflow.add_edge("set_workflow_completed_flag", END)

# compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
