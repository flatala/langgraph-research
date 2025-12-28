from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import ValidationError
from typing import List, Optional
from data.database.crud import ReviewDB

from agents.shared.state.main_state import AgentState
from agents.shared.state.planning_components import Plan
from agents.shared.utils.llm_utils import get_text_llm, get_orchestrator_llm, invoke_llm_with_json_retry
from agents.shared.utils.json_utils import clean_and_parse_json
from agents.planning_agent.agent_config import PlanningAgentConfiguration as Configuration
from agents.planning_agent.tools import arxiv_search, human_assistance, web_search

import json
import re
import logging

logger = logging.getLogger(__name__)

def append_system_prompt(state: AgentState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    # append prompts
    system_msg = SystemMessage(content=cfg.system_prompt)
    user_prompt = cfg.query_refinement_prompt.format(query_count=cfg.refined_query_count, topic=state.topic,)
    user_msg = HumanMessage(content=user_prompt)

    logger.info("Appended system prompt and problem statement refinement prompt.")
    return {"messages": [system_msg, user_msg]}


async def refine_problem_statement(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = get_text_llm(cfg=cfg).bind_tools([human_assistance, web_search]).with_config({"response_format": {"type": "json_object"}})
    messages = state.messages
    ai_msg: AIMessage = await llm.ainvoke(messages) # type: ignore

    return {
        "messages": [ai_msg],
    }


async def parse_queries_add_plan_prompt(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    last_message: AIMessage = state.messages[-1]
    content = last_message.content.strip()

    # Debug: print what we're trying to parse
    logger.debug(f"Attempting to parse queries from content (first 200 chars): {content[:200]}")
    if not content:
        raise ValueError("LLM returned empty content. Expected JSON array of search queries.")

    # parse queries
    queries: List[str] = clean_and_parse_json(content)

    # append research plan prompt
    cfg = Configuration.from_runnable_config(config)
    queries_str = "; ".join(state.search_queries or [])
    prompt = cfg.research_prompt.format(topic=state.topic, paper_recency=state.paper_recency, search_queries=queries_str)
    new_msg = HumanMessage(content=prompt)

    return {
        "messages": [new_msg],
        "search_queries": queries,
    }


async def plan_literature_review(state: AgentState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    llm = get_orchestrator_llm(cfg=cfg).bind_tools([arxiv_search, web_search]).with_config({"response_format": {"type": "json_object"}})
    messages = state.messages
    ai_msg = await llm.ainvoke(messages)

    logger.info("Executed a planning step.")
    return {"messages": [ai_msg]}


async def reflect_on_papers(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    """AI-initiated reflection step to analyze found papers and determine next actions."""
    cfg = Configuration.from_runnable_config(config)
    
    # prepare prompt
    reflection_content = cfg.reflection_prompt.format(topic=state.topic, paper_count=cfg.paper_count)
    reflection_msg = HumanMessage(content=reflection_content)
    
    # invoke LLM
    llm = get_orchestrator_llm(cfg=cfg)
    messages = state.messages + [reflection_msg]
    ai_msg = await llm.ainvoke(messages)

    # this prompt helps the lm decide on the next step.
    next_step_prompt = cfg.reflection_next_step_prompt.format()
    next_step_msg = HumanMessage(content=next_step_prompt)
    logger.info("Executed reflection on found papers.")

    return {"messages": [reflection_msg, ai_msg, next_step_msg]}


async def parse_plan(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    """Process the plan to extract key points and papers."""
    last_message: AIMessage = state.messages[-1]
    content = last_message.content.strip()

    # parse JSON into Pydantic Plan model
    plan_data = clean_and_parse_json(content)
    plan: Plan = Plan.model_validate(plan_data)
    logger.info("Literature survey plan extracted from LLM response.")

    # save plan to database
    db = ReviewDB()
    db.save_plan(state.review_id, plan)

    logger.info("Planning stage complete. Returning control to parent graph.")

    return {
        "plan": plan,
    }