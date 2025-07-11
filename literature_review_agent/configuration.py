from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from literature_review_agent import prompts
import os

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    orchestrator_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="o3-mini-2025-01-31",
        metadata={"description": "The model planning the literature review process."},
    )

    text_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o-mini",
        metadata={"description": "OpenAI model name, e.g. 'gpt-4o-mini'."},
    )

    research_prompt: str = field(
        default=prompts.PLAN_PROMPT,
        metadata={
            "description": "The prompt template to use for the agent's literature review planning phase. "
            "Expects two f-string arguments: {topic} and {paper_recency}."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})