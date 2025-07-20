from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS

from literature_review_agent.state import LitState, Plan, CachingOptions
from literature_review_agent.utils import get_text_llm, get_orchestrator_llm, reduce_docs
from literature_review_agent.custom_nodes import AsyncToolNode
from literature_review_agent.configuration import Configuration  
from literature_review_agent.tools import arxiv_search, summarise_text

from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json, re
import hashlib


PLAN_CACHE_PATH = 'cache/plans/'

# handling of caching the stages

async def decide_on_start_stage(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    if state["caching_options"] is not None and state["caching_options"]["cached_plan_id"] is not None:
        print("Using cached plan...\n")
        return "load_cached_plan"
    else:
        print("Starting from scratch...\n")
        return "prepare_search_queries"

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


# Actualworkflow starts here


async def prepare_search_queries(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    cfg = Configuration.from_runnable_config(config)

    prompt = cfg.query_refinement_prompt.format(
        query_count=cfg.refined_query_count,
        topic=state.get("topic"),
    )

    llm = (
        get_text_llm(cfg=cfg)
        .with_config({"response_format": {"type": "json_object"}})
    )

    messages = state.get("messages").copy()
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

    print("Refined search queries.\n")

    return {
        "search_queries": queries,
        "messages": messages,
    }


def send_plan_prompt(state: LitState, *, config=None):
    cfg = Configuration.from_runnable_config(config)

    queries_str = "; ".join(state["search_queries"])
    prompt = cfg.research_prompt.format(
        topic=state["topic"],
        paper_recency=state["paper_recency"],
        search_queries=queries_str,
    )

    messages = state["messages"].copy()
    messages.append(HumanMessage(content=prompt))

    print("Appended instruction prompt.\n")

    return {"messages": messages}


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

    
def route_tools(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """

    print("Checking for tool calls...\n")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


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


async def prepare_rag_knowledge_base(state: LitState, *, config=None) -> dict:
    print("Preparing the RAG setup...")

    # 1. Collect all unique arXiv URLs from the plan
    arxiv_urls = set()
    plan_dict: Plan = state.get("plan", {})
    for section in plan_dict.get("plan", []): 
        for kp in section["key_points"]:
            for paper in kp["papers"]:
                url = paper.get("url")
                if url and "arxiv.org" in url:
                    # Removing version suffix (e.g., v1, v2) for clean querying
                    url = re.sub(r'v\d+$', '', url)
                    arxiv_urls.add(url)

    # 2. Download all full-text PDFs
    docs = []
    for url in arxiv_urls:
        arxiv_id = url.split("/")[-1]
        try:
            loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_full_text=True)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    # 3. Chunk the docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # 4. Embed and index (batching because of the total size)
    embeddings = OpenAIEmbeddings()
    max_chunks_per_batch = 250
    db = None
    for i in range(0, len(split_docs), max_chunks_per_batch):
        batch = split_docs[i:i + max_chunks_per_batch]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    print("Rag setup ready...\n")

    return {
        "retriever": db.as_retriever(),
        "documents": reduce_docs(state.get("documents"), split_docs)
    }

workflow = StateGraph(LitState)

workflow.add_node("load_cached_plan", load_cached_plan)
workflow.add_node("prepare_search_queries", prepare_search_queries)
workflow.add_node("send_plan_prompt", send_plan_prompt)
workflow.add_node("plan_literature_review", plan_literature_review)
workflow.add_node("tools", AsyncToolNode([arxiv_search]))
workflow.add_node("parse_plan", parse_plan)
workflow.add_node("prepare_rag_knowledge_base", prepare_rag_knowledge_base)

# first decide on the initial starting stage
workflow.add_conditional_edges(
    START,
    decide_on_start_stage,
    {"load_cached_plan": "load_cached_plan", "prepare_search_queries": "prepare_search_queries"},
)

# if a cached plan is loaded, go directly to rag setup
workflow.add_edge("load_cached_plan", "prepare_rag_knowledge_base")

# refien search queries and send the system prompt
workflow.add_edge("prepare_search_queries", "send_plan_prompt")
workflow.add_edge("send_plan_prompt", "plan_literature_review")

# loop between tool calls and ai model refinement
workflow.add_conditional_edges(
    "plan_literature_review",
    route_tools,
    {"tools": "tools", "__end__": "parse_plan"},
)
workflow.add_edge("tools", "plan_literature_review")

# parse the plan, save json, setup rag
workflow.add_edge("parse_plan", "prepare_rag_knowledge_base")
workflow.add_edge("prepare_rag_knowledge_base", END)

graph = workflow.compile()

# def refine_section(state: LitState) -> dict:
#     """Draft the first section (placeholder logic)."""
#     first_title = state["plan"].splitlines()[0]
#     prompt = (
#         f"Write a clear, 2-3 sentence draft for the section: '{first_title}'. "
#         "Assume the target reader is a grad student."
#     )
#     llm = get_text_llm()    
#     draft = llm.invoke(prompt).content
#     return {"draft_sections": [draft]}


# def verify_section(state: LitState) -> dict:
#     """Light-weight factuality pass—returns the text unchanged if it looks fine."""
#     draft = state["draft_sections"][0]
#     prompt = (
#         "Check the following paragraph for factual consistency. "
#         "If it is correct, return it unchanged; otherwise, return a corrected version.\n\n"
#         f"{draft}"
#     )
#     llm = get_text_llm()    
#     verified = llm.invoke(prompt).content
#     return {"verified_sections": [verified]}


# def get_context_for_section(section, retriever, k=8):
#     query = section['outline'] + "\n" + " ".join([kp["text"] for kp in section["key_points"]])
#     docs = retriever.get_relevant_documents(query)  # or .similarity_search(query, k=k)
#     return "\n".join([doc.page_content for doc in docs])