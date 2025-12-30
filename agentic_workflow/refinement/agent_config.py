from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from dataclasses import fields
from agentic_workflow.refinement import prompts
from agentic_workflow.shared.main_config import MainConfiguration

@dataclass(kw_only=True)
class RefinementAgentConfiguration(MainConfiguration):
    """Full configuration for the refinement agent, extending MainConfiguration."""

    # PROMPTS

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt that establishes the assistant's role as a literature review writer."
        },
    )

    write_subsection_prompt: str = field(
        default=prompts.WRITE_SUBSECTION_PROMPT,
        metadata={
            "description": "The prompt template to use for writing subsections based on key points and paper segments. "
            "Expects f-string arguments for preceeding_sections, key_point_text, section_title, section_outline, subsection_index, "
            "total_subsections, and paper_segments."
        },
    )

    content_review_prompt: str = field(
        default=prompts.CONTENT_REVIEW_PROMPT,
        metadata={
            "description": "The prompt template to use for reviewing the quality of generated subsections. "
            "Expects the following arguments: minimum_score (int), key_point (str), subsection (str)"
        },
    )

    citation_identification_prompt: str = field(
        default=prompts.CITATION_IDENTIFICATION_PROMPT,
        metadata={
            "description": "The prompt template to use for identifying citations and extracting supported claims from paper segments. "
            "Expects f-string argument: paper_segment (str)"
        },
    )

    verify_claim_prompt: str = field(
        default=prompts.VERIFY_CLAIM_PROMPT,
        metadata={
            "description": "The prompt template to use for verifying a single citation claim against the paper content. "
            "Expects f-string arguments: citation (str), claim (str), supporting_content (str)"
        },
    )

    grounding_refinement_prompt: str = field(
        default=prompts.GROUNDING_REFINEMENT_PROMPT,
        metadata={
            "description": "The prompt template to use for refining subsections to fix specific grounding issues. "
            "Expects f-string arguments: citation (str), supported_claim (str), error_type (str), explanation (str), "
            "correction_suggestion (str), current_subsection (str), full_paper_content (str)"
        },
    )

    content_feedback_prompt: str = field(
        default=prompts.CONTENT_FEEDBACK_PROMPT,
        metadata={
            "description": "The prompt template for content review feedback. "
            "Expects f-string arguments: issues_list (str), score (int), reasoning (str)"
        },
    )

    grounding_feedback_prompt: str = field(
        default=prompts.GROUNDING_FEEDBACK_PROMPT,
        metadata={
            "description": "The prompt template for grounding review feedback with tool availability. "
            "Expects f-string arguments: issues_list (str), available_papers (str)"
        },
    )

    minimum_score: int = field(
        default=7,
        metadata={
            "description": "Minimum quality score required for content review to pass (1-10 scale)"
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "RefinementAgentConfiguration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
