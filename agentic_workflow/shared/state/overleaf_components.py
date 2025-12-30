"""Components for the Overleaf/LaTeX export process."""

from pydantic import BaseModel, Field
from typing import Dict, Optional

from agentic_workflow.shared.state.refinement_components import PaperWithSegements


class OverleafProgress(BaseModel):
    """Progress tracking for LaTeX generation."""

    # Collected papers
    papers_map: Dict[str, PaperWithSegements] = Field(default_factory=dict)

    # Generated metadata
    generated_title: Optional[str] = None
    section_intros: Dict[int, str] = Field(default_factory=dict)
    subsection_titles: Dict[str, str] = Field(default_factory=dict)  # "section,subsection" -> title

    # Generated content
    latex_content: Optional[str] = None
    bibtex_content: Optional[str] = None
