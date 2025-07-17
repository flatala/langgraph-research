from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import chat_agent_executor, ToolNode, tools_condition
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS

from literature_review_agent.state import LitState, Plan
from literature_review_agent.utils import get_text_llm, get_orchestrator_llm, reduce_docs
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
                result_for_history = papers

            else:
                result_for_history = await tool.ainvoke(call["args"])

            messages.append(
                ToolMessage(
                    name=call["name"],
                    tool_call_id=call["id"],
                    content=json.dumps(result_for_history)
                )
            )

    # text = re.sub(r"^```[\w-]*\n|\n```$", "", messages[-1].content.strip(), flags=re.S)
    plan: Plan = json.loads(messages[-1].content.strip())

    return {
        "plan": plan,
        "messages": messages,
    }


async def prepare_rag_knowledge_base(state: LitState, *, config=None) -> dict:
    import re

    # 1. Collect all unique arXiv URLs from the plan
    arxiv_urls = set()
    for section in state.plan["plan"]:
        for kp in section["key_points"]:
            for paper in kp["papers"]:
                url = paper.get("url", "")
                if url and "arxiv.org" in url:
                    # Remove version suffix (e.g., v1, v2) for clean querying
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

    return {
        "retriever": db.as_retriever(),
        "documents": reduce_docs(state.documents, split_docs),
        "messages": state.messages,
    }


builder = StateGraph(LitState)
tools_node = ToolNode([arxiv_search])

builder.add_node("prepare_search_queries", prepare_search_queries) 
builder.add_node("plan_literature_review", plan_review)
builder.add_node("tools", tools_node)
builder.add_node("prepare_rag_knowledge_base", prepare_rag_knowledge_base)

builder.add_edge(START, "prepare_search_queries")
builder.add_edge("prepare_search_queries", "plan_literature_review")
builder.add_conditional_edges(
    "plan_literature_review",
    tools_condition,              
    {"tools": "tools", "__end__": "prepare_rag_knowledge_base"}
)
builder.add_edge("tools", "plan_literature_review")
builder.add_edge("prepare_rag_knowledge_base", END)

graph = builder.compile()



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