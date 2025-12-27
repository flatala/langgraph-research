from langchain_community.document_loaders import ArxivLoader
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from langgraph.types import interrupt
from tavily import AsyncTavilyClient

from agents.planning_agent.agent_config import PlanningAgentConfiguration as Configuration

from typing import List, Dict
from typing_extensions import Annotated
import asyncio
import logging

logger = logging.getLogger(__name__)


@tool("arxiv_search")
async def arxiv_search(query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]) -> List[Dict]:
    """
    Return recent arXiv papers that match *query*.
    Each item has title, url, summary (one paragraph) and year.
    """
    logger.info(f"Starting ArXiv search...")
    cfg = Configuration.from_runnable_config(config)

    # retry with exponential backoff on rate limits
    max_retries = 5
    for attempt in range(max_retries):
        try:
            docs = ArxivLoader(query=query, load_max_docs=cfg.max_search_results).get_summaries_as_docs()
            break
        except Exception as e:
            if ("429" in str(e) or "503" in str(e)) and attempt < max_retries - 1:
                wait = 3 * (attempt + 1)  # 3s, 6s, 9s, 12s
                logger.warning(f"ArXiv rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
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


@tool("web_search")
async def web_search(query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]) -> List[Dict]:
    """
    Search the web for current information, news, and recent developments.
    Use this for topics requiring up-to-date information beyond academic papers,
    such as recent industry trends, current events, or practical applications.
    Each result includes title, URL, content snippet, and relevance score.
    """
    logger.info(f"Starting Tavily web search for: '{query}'...")
    cfg = Configuration.from_runnable_config(config)

    if not cfg.tavily_api_key:
        raise ValueError("TAVILY_API_KEY not configured. Please set it in your .env file.")

    # Initialize Tavily client
    tavily = AsyncTavilyClient(api_key=cfg.tavily_api_key)

    # Execute search
    response = await tavily.search(
        query=query,
        max_results=cfg.tavily_max_results,
        search_depth=cfg.tavily_search_depth,
        include_answer=True,
        include_raw_content=False
    )

    # Format results
    results = []
    for result in response.get("results", []):
        results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "score": result.get("score", 0.0)
        })

    logger.info(f"Found {len(results)} web results.")
    return results


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    logger.info(f"Human input received.")
    return human_response["data"]
