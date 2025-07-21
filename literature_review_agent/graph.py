from langgraph.graph import StateGraph, START, END
from literature_review_agent.state import LitState
from literature_review_agent.tools import arxiv_search, human_assistance
from literature_review_agent.nodes.rag_nodes import prepare_rag_knowledge_base
from literature_review_agent.nodes.general_nodes import AsyncToolNode, route_tools, set_workflow_completed_flag
from literature_review_agent.nodes.reserch_plan_nodes import (
    decide_on_start_stage, 
    # prepare_search_queries, 
    # send_plan_prompt, 
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
workflow.add_node("human_assistance_tool", AsyncToolNode([human_assistance]))
workflow.add_node("parse_queries_add_plan_prompt", parse_queries_add_plan_prompt)

# old shit
# workflow.add_node("prepare_search_queries", prepare_search_queries)

# workflow.add_node("send_plan_prompt", send_plan_prompt)
workflow.add_node("plan_literature_review", plan_literature_review)
workflow.add_node("tools", AsyncToolNode([arxiv_search]))
workflow.add_node("parse_plan", parse_plan)
workflow.add_node("prepare_rag_knowledge_base", prepare_rag_knowledge_base)

workflow.add_node("set_workflow_completed_flag", set_workflow_completed_flag)


# workflow.add_edge(prepend_system_prompt, ""human_assistance)

# two tool nodes

# workflow.add_conditional_edges(
#     hooman assistance,
#     route_tools,
#     {"tools": "tools", "prepare_search_queries": "prepare_search_queries"},
# )

# First decide on the initial starting stage
workflow.add_conditional_edges(
    START,
    decide_on_start_stage,
    {"load_cached_plan": "load_cached_plan", "refine_problem_statement": "append_system_prompt"},
)

# If a cached plan is loaded, go directly to rag setup
workflow.add_edge("load_cached_plan", "prepare_rag_knowledge_base")

# if we do refinement
workflow.add_edge("append_system_prompt", "refine_problem_statement")

workflow.add_conditional_edges(
    "refine_problem_statement",
    route_tools,
    {"tools": "human_assistance_tool", "__end__": "parse_queries_add_plan_prompt"},
)
workflow.add_edge("human_assistance_tool", "refine_problem_statement")


workflow.add_edge("parse_queries_add_plan_prompt", "plan_literature_review")

# Refine search queries and send the system prompt
# workflow.add_edge("prepare_search_queries", "send_plan_prompt")
# workflow.add_edge("send_plan_prompt", "plan_literature_review")

# Loop between tool calls and ai model refinement
workflow.add_conditional_edges(
    "plan_literature_review",
    route_tools,
    {"tools": "tools", "__end__": "parse_plan"},
)
workflow.add_edge("tools", "plan_literature_review")

# Parse the plan, save json, setup rag
workflow.add_edge("parse_plan", "prepare_rag_knowledge_base")
workflow.add_edge("prepare_rag_knowledge_base", "set_workflow_completed_flag")
workflow.add_edge("set_workflow_completed_flag", END)

graph = workflow.compile()
