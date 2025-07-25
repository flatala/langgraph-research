from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from literature_review_agent import prompts
import os

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    # MODELS

    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    orchestrator_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="o3",
        metadata={"description": "The model planning the literature review process."},
    )

    text_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4.1-2025-04-14",
        metadata={"description": "OpenAI model name, e.g. 'gpt-4o-mini'."},
    )

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


    # NUMERIC

    refined_query_count: int = field(
        default=15,
        metadata={
            "description": "The number of queries to prepare in the query refinement stage."
        },
    )

    max_search_results: int = field(
        default=35,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )


    # HELPERS

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})