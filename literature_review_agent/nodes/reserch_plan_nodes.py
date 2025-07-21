from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from literature_review_agent.state import LitState, Plan, CachingOptions
from literature_review_agent.utils import get_text_llm, get_orchestrator_llm
from literature_review_agent.configuration import Configuration  
from literature_review_agent.tools import arxiv_search, human_assistance

from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json, re
import hashlib


PLAN_CACHE_PATH = 'cache/plans/'


async def decide_on_start_stage(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    if state["caching_options"] is not None and state["caching_options"]["cached_plan_id"] is not None:
        print("Using cached plan...\n")
        return "load_cached_plan"
    else:
        print("Starting from scratch...\n")
        return "refine_problem_statement"

# include system prompt + human prompt and use both
def append_system_prompt(state: LitState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    prompt = cfg.query_refinement_prompt.format(
        query_count=cfg.refined_query_count,
        topic=state.get("topic"),
    )

    messages = state["messages"]
    new_msg = HumanMessage(content=prompt)

    print("Appended problem statement refinement prompt.\n")

    return {"messages": messages + [new_msg]}


async def refine_problem_statement(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = (
        get_text_llm(cfg=cfg)
        .bind_tools([human_assistance])
        .with_config({"response_format": {"type": "json_object"}})
    )

    messages = state.get("messages").copy()
    ai_msg: AIMessage = await llm.ainvoke(messages)

    return {
        # "search_queries": queries,
        "messages": messages + [ai_msg],
    }


async def parse_queries_add_plan_prompt(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    last_message: AIMessage = state.get("messages")[-1]
    queries: List[str] = json.loads(last_message.content.strip())

    cfg = Configuration.from_runnable_config(config)

    queries_str = "; ".join(state["search_queries"])
    prompt = cfg.research_prompt.format(
        topic=state["topic"],
        paper_recency=state["paper_recency"],
        search_queries=queries_str,
    )

    messages = state["messages"]
    new_msg = HumanMessage(content=prompt)

    return {
        "messages": messages + [new_msg],
        "search_queries": queries,
    }



# # TODO: maybe combien this with query refinement and 
# # add somr human-in-the -loop call for extra questions
# def send_plan_prompt(state: LitState, *, config=None):
#     cfg = Configuration.from_runnable_config(config)

#     queries_str = "; ".join(state["search_queries"])
#     prompt = cfg.research_prompt.format(
#         topic=state["topic"],
#         paper_recency=state["paper_recency"],
#         search_queries=queries_str,
#     )

#     messages = state["messages"]
#     new_msg = HumanMessage(content=prompt)

#     print("Appended instruction prompt.\n")

#     return {"messages": messages + [new_msg]}







# async def prepare_search_queries(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
#     cfg = Configuration.from_runnable_config(config)

#     prompt = cfg.query_refinement_prompt.format(
#         query_count=cfg.refined_query_count,
#         topic=state.get("topic"),
#     )

#     llm = (
#         get_text_llm(cfg=cfg)
#         .with_config({"response_format": {"type": "json_object"}})
#     )

#     messages = state.get("messages").copy()
#     messages.append(HumanMessage(content=prompt))

#     ai_msg: AIMessage = await llm.ainvoke(messages)
#     messages.append(ai_msg)

#     raw = re.sub(r"^```[\w-]*\n|\n```$", "", ai_msg.content.strip(), flags=re.S)
#     data = json.loads(raw)
#     if isinstance(data, dict) and "queries" in data:
#         queries: List[str] = data["queries"]
#     elif isinstance(data, list):
#         queries = data
#     else:
#         raise ValueError("Unexpected JSON structure returned by LLM")

#     print("Refined search queries.\n")

#     return {
#         "search_queries": queries,
#         "messages": messages,
#     }

# # TODO: maybe combien this with query refinement and 
# # add somr human-in-the -loop call for extra questions
# def send_plan_prompt(state: LitState, *, config=None):
#     cfg = Configuration.from_runnable_config(config)

#     queries_str = "; ".join(state["search_queries"])
#     prompt = cfg.research_prompt.format(
#         topic=state["topic"],
#         paper_recency=state["paper_recency"],
#         search_queries=queries_str,
#     )

#     messages = state["messages"].copy()
#     messages.append(HumanMessage(content=prompt))

#     print("Appended instruction prompt.\n")

#     return {"messages": messages}



async def plan_literature_review(state: LitState, *, config=None):
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


async def parse_plan(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
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

    print(f"Saved plan with id {hash} at {plan_path}.\n")

    return {
        "plan": plan,
    }


async def load_cached_plan(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
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