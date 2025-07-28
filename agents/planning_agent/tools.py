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
    cfg   = Configuration.from_runnable_config(config)
    docs  = ArxivLoader(query=query, max_results=cfg.max_search_results).get_summaries_as_docs()
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


# # https://github.com/langchain-ai/data-enrichment/blob/main/src/enrichment_agent/utils.py
# @tool("web_search")
# async def search(
#     query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
# ) -> Optional[list[dict[str, Any]]]:
#     """Query a search engine.

#     This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
#     for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
#     """
#     configuration = Configuration.from_runnable_config(config)
#     wrapped = TavilySearchResults(max_results=configuration.max_search_results)
#     result = await wrapped.ainvoke({"query": query})
#     return cast(list[dict[str, Any]], result)


# _INFO_PROMPT = """You are doing web research on behalf of a user. You are trying to find out this information:

# <info>
# {info}
# </info>

# You just scraped the following website: {url}

# Based on the website content below, jot down some notes about the website.

# <Website content>
# {content}
# </Website content>"""

# @tool("scrape_website")
# async def scrape_website(
#     url: str,
#     *,
#     state: Annotated[LitState, InjectedState],
#     config: Annotated[RunnableConfig, InjectedToolArg],
# ) -> str:
#     """Scrape and summarize content from a given URL.

#     Returns:
#         str: A summary of the scraped content, tailored to the extraction schema.
#     """
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             content = await response.text()

#     p = _INFO_PROMPT.format(
#         info=json.dumps(state.extraction_schema, indent=2),
#         url=url,
#         content=content[:40_000],
#     )
#     raw_model = get_llm(config)
#     result = await raw_model.ainvoke(p)
#     return str(result.content)