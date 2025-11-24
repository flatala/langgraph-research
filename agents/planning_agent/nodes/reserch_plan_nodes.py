from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.shared.state.main_state import AgentState
from agents.shared.state.planning_components import Plan
from agents.shared.utils.llm_utils import get_text_llm, get_orchestrator_llm
from agents.planning_agent.agent_config import PlanningAgentConfiguration as Configuration
from agents.planning_agent.tools import arxiv_search, human_assistance, web_search

from typing import List, Optional
import json


def append_system_prompt(state: AgentState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    # Add system prompt
    system_msg = SystemMessage(content=cfg.system_prompt)
    
    # Add user prompt for query refinement
    user_prompt = cfg.query_refinement_prompt.format(
        query_count=cfg.refined_query_count,
        topic=state.topic,
    )
    user_msg = HumanMessage(content=user_prompt)

    print("Appended system prompt and problem statement refinement prompt.\n")

    return {"messages": [system_msg, user_msg]}


async def refine_problem_statement(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = (
        get_text_llm(cfg=cfg)
        .bind_tools([human_assistance, web_search])
        .with_config({"response_format": {"type": "json_object"}})
    )

    messages = state.messages
    ai_msg: AIMessage = await llm.ainvoke(messages) # type: ignore

    return {
        "messages": [ai_msg],
    }


async def parse_queries_add_plan_prompt(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    last_message: AIMessage = state.messages[-1]
    content = last_message.content.strip()

    # Debug: print what we're trying to parse
    print(f"Attempting to parse queries from content (first 200 chars): {content[:200]}")

    if not content:
        raise ValueError("LLM returned empty content. Expected JSON array of search queries.")

    try:
        queries: List[str] = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed. Full content:\n{content}\n")
        raise ValueError(f"Failed to parse search queries as JSON. Error: {e}") from e

    cfg = Configuration.from_runnable_config(config)

    queries_str = "; ".join(state.search_queries or [])
    prompt = cfg.research_prompt.format(
        topic=state.topic,
        paper_recency=state.paper_recency,
        search_queries=queries_str
    )

    new_msg = HumanMessage(content=prompt)

    return {
        "messages": [new_msg],
        "search_queries": queries,
    }


async def plan_literature_review(state: AgentState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    llm = (
        get_orchestrator_llm(cfg=cfg)
        .bind_tools([arxiv_search, web_search])
        .with_config({"response_format": {"type": "json_object"}})
    )

    messages = state.messages
    ai_msg = await llm.ainvoke(messages)

    print("Executed a planning step.\n")

    return {"messages": [ai_msg]}

# TODO: use a helper agent / tool that verifies if the collected papers meet the conditions
# and returns a response stating if one hsould continue or not


async def reflect_on_papers(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    """AI-initiated reflection step to analyze found papers and determine next actions."""
    cfg = Configuration.from_runnable_config(config)
    
    # Create reflection prompt
    reflection_content = cfg.reflection_prompt.format(
        topic=state.topic,
        paper_count=cfg.paper_count
    )
    reflection_msg = HumanMessage(content=reflection_content)
    
    # Use text LLM for reflection (no tools needed)
    llm = get_orchestrator_llm(cfg=cfg)
    
    messages = state.messages + [reflection_msg]
    ai_msg = await llm.ainvoke(messages)

    # This prompt helps the lm decide on the next step.
    next_step_prompt = cfg.reflection_next_step_prompt.format()
    next_step_msg = HumanMessage(content=next_step_prompt)

    print("Executed reflection on found papers.\n")

    return {"messages": [reflection_msg, ai_msg, next_step_msg]}


async def parse_plan(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    """Process the plan to extract key points and papers."""
    from data.database.crud import ReviewDB

    last_message: AIMessage = state.messages[-1]
    # Parse JSON into Pydantic Plan model
    plan: Plan = Plan.model_validate_json(last_message.content.strip())
    print("Literature survey plan extracted from LLM response.\n")

    # Save plan to database
    db = ReviewDB()
    db.save_plan(state.review_id, plan)

    print("✓ Planning stage complete. Returning control to parent graph.\n")

    return {
        "plan": plan,
    }