"""Configuration for Overleaf/LaTeX export agent."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional
from langchain_core.runnables import RunnableConfig, ensure_config

from agentic_workflow.overleaf import prompts
from agentic_workflow.shared.main_config import MainConfiguration


@dataclass(kw_only=True)
class OverleafAgentConfiguration(MainConfiguration):
    """Full configuration for the overleaf agent, extending MainConfiguration."""

    # PROMPTS

    generate_title_prompt: str = field(
        default=prompts.GENERATE_TITLE_PROMPT,
        metadata={
            "description": "Prompt template for generating academic survey title. "
            "Expects f-string arguments: topic (str), section_titles (str)"
        },
    )

    generate_section_intro_prompt: str = field(
        default=prompts.GENERATE_SECTION_INTRO_PROMPT,
        metadata={
            "description": "Prompt template for generating section introductions. "
            "Expects f-string arguments: section_title (str), section_outline (str), subsection_summaries (str)"
        },
    )

    generate_subsection_title_prompt: str = field(
        default=prompts.GENERATE_SUBSECTION_TITLE_PROMPT,
        metadata={
            "description": "Prompt template for generating subsection titles. "
            "Expects f-string arguments: key_point_text (str)"
        },
    )

    format_math_prompt: str = field(
        default=prompts.FORMAT_MATH_PROMPT,
        metadata={
            "description": "System prompt for LLM to format mathematical expressions for LaTeX."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "OverleafAgentConfiguration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
