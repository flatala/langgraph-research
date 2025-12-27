from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from agents.shared.state.main_state import AgentState
from agents.planning_agent.tools import arxiv_search, human_assistance, web_search
from agents.planning_agent.nodes.reserch_plan_nodes import (
    plan_literature_review,
    parse_plan,
    append_system_prompt,
    refine_problem_statement,
    parse_queries_add_plan_prompt,
    reflect_on_papers
)

workflow = StateGraph(AgentState)

# Define all nodes
workflow.add_node("append_system_prompt", append_system_prompt)
workflow.add_node("refine_problem_statement", refine_problem_statement)
workflow.add_node("tools_1", ToolNode([human_assistance, web_search]))
workflow.add_node("parse_queries_add_plan_prompt", parse_queries_add_plan_prompt)
workflow.add_node("plan_literature_review", plan_literature_review)
workflow.add_node("tools_2", ToolNode([arxiv_search, web_search]))
workflow.add_node("reflect_on_papers", reflect_on_papers)
workflow.add_node("parse_plan", parse_plan)

# Start with system prompt
workflow.add_edge(START, "append_system_prompt")

# if we do refinement, append system + user prompts
workflow.add_edge("append_system_prompt", "refine_problem_statement")

# Loop between llm and human inputs
workflow.add_conditional_edges(
    "refine_problem_statement",
    tools_condition,
    {"tools": "tools_1", "__end__": "parse_queries_add_plan_prompt"})
workflow.add_edge("tools_1", "refine_problem_statement")

# Go to the planning phase
workflow.add_edge("parse_queries_add_plan_prompt", "plan_literature_review")

# Loop between tool calls and ai model refinement with reflection
workflow.add_conditional_edges(
    "plan_literature_review",
    tools_condition,
    {"tools": "tools_2", "__end__": "parse_plan"},
)

workflow.add_edge("tools_2", "reflect_on_papers")
workflow.add_edge("reflect_on_papers", "plan_literature_review")

# Parse the plan, save json, setup rag
workflow.add_edge("parse_plan", END)

# compile with checkpointer=True to maintain internal state while coordinating with parent
planning_graph = workflow.compile(checkpointer=True)
