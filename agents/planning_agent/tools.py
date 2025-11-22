from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import ArxivLoader
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from langgraph.types import interrupt

from agents.planning_agent.agent_config import PlanningAgentConfiguration as Configuration
from agents.shared.utils.llm_utils import get_text_llm

from typing import List, Dict
from typing_extensions import Annotated


@tool("arxiv_search")
async def arxiv_search(query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]) -> List[Dict]:
    """
    Return recent arXiv papers that match *query*.
    Each item has title, url, summary (one paragraph) and year.
    """

    print(f"Starting ArXiv search...\n")
    cfg = Configuration.from_runnable_config(config)
    docs = ArxivLoader(query=query, max_results=cfg.max_search_results).get_summaries_as_docs()
    results = []
    for d in docs:
        m = d.metadata
        results.append(
            {
                "title":   m["Title"],
                "url":     m["Entry ID"],
                "summary": d.page_content,
                "year":    str(m["Published"].year) if "Published" in m else "",
            }
        )
    return results


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    print(f"\nHuman input received.\n")
    return human_response["data"]


@tool("summarise_text")
async def summarise_text(text: str, *, config: Annotated[RunnableConfig, InjectedToolArg]) -> str:
    """
    Return a 1-sentence summary of *text*.
    """
    llm = get_text_llm(Configuration.from_runnable_config(config))
    prompt = f"Summarise in one sentence:\n\n{text}"
    return (await llm.ainvoke(prompt)).content.strip()
