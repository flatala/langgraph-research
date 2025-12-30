from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from dataclasses import fields
from agentic_workflow.planning import prompts
from agentic_workflow.shared.main_config import MainConfiguration
import os

@dataclass(kw_only=True)
class PlanningAgentConfiguration(MainConfiguration):
    """Full configuration for the planning agent, extending MainConfiguration."""

    # PROMPTS

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt that establishes the assistant's role as a literature review specialist."
        },
    )

    query_refinement_prompt: str = field(
        default=prompts.PREPARE_SEARCH_QUERIES_PROMPT,
        metadata={
            "description": "The prompt template to use for the agent's search query refinement phase. "
            "Expects two f-string arguments: {query_count} and {topic}."
        },
    )

    research_prompt: str = field(
        default=prompts.PLAN_PROMPT,
        metadata={
            "description": "The prompt template to use for the agent's literature review planning phase. "
            "Expects three f-string arguments: {topic}, {paper_recency}, and {search_queries}."
        },
    )

    reflection_prompt: str = field(
        default=prompts.REFLECTION_PROMPT,
        metadata={
            "description": "The prompt template for AI-initiated reflection on found papers. "
            "Expects one f-string argument: {topic}."
        },
    )

    reflection_next_step_prompt: str = field(
        default=prompts.REFLECTION_NEXT_STEP_PROMPT,
        metadata={
            "description": "The prompt template for determining the next steps after reflection. "
        }
    )

    # NUMERIC PARAMETERS

    refined_query_count: int = field(
        default=15,
        metadata={
            "description": "The number of queries to prepare in the query refinement stage."
        },
    )

    paper_count: int = field(
        default=15,
        metadata={
            "description": "The number of research papers to find and review."
        },
    )

    max_search_results: int = field(
        default=35,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    # TAVILY SEARCH CONFIGURATION

    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        metadata={
            "description": "Tavily API key for web search functionality."
        },
    )

    tavily_max_results: int = field(
        default=5,
        metadata={
            "description": "Maximum number of search results to return from Tavily web search."
        },
    )

    tavily_search_depth: str = field(
        default="advanced",
        metadata={
            "description": "Tavily search depth: 'basic' for faster results or 'advanced' for more comprehensive search."
        },
    )

    # HELPERS

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "PlanningAgentConfiguration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})