from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.shared.state.main_state import AgentState, CachingOptions
from agents.shared.state.planning_components import Plan
from agents.shared.utils.llm_utils import get_text_llm, get_orchestrator_llm
from agents.planning_agent.agent_config import PlanningAgentConfiguration as Configuration  
from agents.planning_agent.tools import arxiv_search, human_assistance

from typing import List, Optional
from datetime import datetime
from pathlib import Path

import hashlib
import json


PLAN_CACHE_PATH = 'cache/plans/'


async def decide_on_start_stage(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    if state["caching_options"] is not None and state["caching_options"]["cached_plan_id"] is not None:
        print("Using cached plan...\n")
        return "load_cached_plan"
    else:
        print("Starting from scratch...\n")
        return "start_from_scratch"


def append_system_prompt(state: AgentState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    # Add system prompt
    system_msg = SystemMessage(content=cfg.system_prompt)
    
    # Add user prompt for query refinement
    user_prompt = cfg.query_refinement_prompt.format(
        query_count=cfg.refined_query_count,
        topic=state.get("topic"),
    )
    user_msg = HumanMessage(content=user_prompt)

    messages = state["messages"]
    new_messages = messages + [system_msg, user_msg]

    print("Appended system prompt and problem statement refinement prompt.\n")

    return {"messages": new_messages}


async def refine_problem_statement(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = (
        get_text_llm(cfg=cfg)
        .bind_tools([human_assistance])
        .with_config({"response_format": {"type": "json_object"}})
    )

    messages = state.get("messages")
    ai_msg: AIMessage = await llm.ainvoke(messages)

    return {
        "messages": messages + [ai_msg],
    }


async def parse_queries_add_plan_prompt(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    last_message: AIMessage = state.get("messages")[-1]
    queries: List[str] = json.loads(last_message.content.strip())

    cfg = Configuration.from_runnable_config(config)

    queries_str = "; ".join(state["search_queries"])
    prompt = cfg.research_prompt.format(
        topic=state["topic"],
        paper_recency=state["paper_recency"],
        search_queries=queries_str
    )

    messages = state["messages"]
    new_msg = HumanMessage(content=prompt)

    return {
        "messages": messages + [new_msg],
        "search_queries": queries,
    }


async def plan_literature_review(state: AgentState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    llm = (
        get_orchestrator_llm(cfg=cfg)
        .bind_tools([arxiv_search])
        .with_config({"response_format": {"type": "json_object"}})
    )

    messages = state["messages"]
    ai_msg = await llm.ainvoke(messages)

    print("Executed a planning step.\n")

    return {"messages": messages + [ai_msg]}

# TODO: use a helper agent / tool that verifies if the collected papers meet the conditions
# and returns a response stating if one hsould continue or not


async def reflect_on_papers(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    """AI-initiated reflection step to analyze found papers and determine next actions."""
    cfg = Configuration.from_runnable_config(config)
    
    # Create reflection prompt
    reflection_content = cfg.reflection_prompt.format(
        topic=state.get("topic"),
        paper_count=cfg.paper_count
    )
    reflection_msg = HumanMessage(content=reflection_content)
    
    # Use text LLM for reflection (no tools needed)
    llm = get_orchestrator_llm(cfg=cfg)
    
    messages = state["messages"] + [reflection_msg]
    ai_msg = await llm.ainvoke(messages)

    # This prompt helps the lm decide on the next step.
    next_step_prompt = cfg.reflection_next_step_prompt.format()
    next_step_msg = HumanMessage(content=next_step_prompt)
    
    print("Executed reflection on found papers.\n")
    
    return {"messages": messages + [ai_msg, next_step_msg]}


async def parse_plan(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    """Process the plan to extract key points and papers."""

    last_message: AIMessage = state.get("messages")[-1]
    plan: Plan = json.loads(last_message.content.strip())
    print("Litarture survey plan extracted from LLM response.\n")

    # prepare cache key
    topic = state.get("topic")
    now = datetime.now()
    to_hash = f"{topic}_{now.strftime('%Y%m%d_%H%M%S')}"
    md5_hex = hashlib.md5(to_hash.encode()).hexdigest()
    parts = [md5_hex[i:i+8] for i in range(0, 32, 8)]
    final_id = '-'.join(parts)

    # ensure that the directory exists
    plan_path = Path(PLAN_CACHE_PATH) / f"{final_id}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    # write json
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=4)

    print(f"Saved plan at {plan_path}\n")

    return {
        "plan": plan,
    }


async def load_cached_plan(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    caching_options: CachingOptions = state.get("caching_options")
    if caching_options and caching_options.get("cached_plan_id"):
        plan_id = caching_options["cached_plan_id"]
        plan_path = Path(PLAN_CACHE_PATH) / f"{plan_id}.json"
        with open(plan_path, "r") as f:
            plan: Plan  = json.load(f)
            print(f"Loaded cached plan with id {plan_id}.\n")
            return { "plan": plan }
    else:
        raise ValueError("No cached plan ID provided in state.")