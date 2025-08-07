from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from langchain_core.documents import Document

class SubsectionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    NEEDS_REVISION = "needs_revision"
    APPROVED = "approved"

class SectionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class ReviewType(str, Enum):
    CONTENT = "content"
    GROUNDING = "grounding"

class ReviewFeedback(BaseModel):
    review_type: ReviewType
    passed: bool
    feedback: str
    suggestions: Optional[List[str]] = None

class PaperWithSegements(BaseModel):
    title: str
    authors: List[str]
    arxiv_id: str
    arxiv_url: str
    citation: str
    content: Document
    relevant_segments: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

class Subsection(BaseModel):
    subsection_index: int
    papers: List[PaperWithSegements]
    key_point_text: str
    content: str
    revision_count: int = 0
    feedback_history: List[ReviewFeedback] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list) 

class Section(BaseModel):
    section_index: int
    section_title: str
    section_outline: str
    section_introduction: str
    subsections: List[Subsection] = Field(default_factory=list)
    section_markdown: str = ""

class RefinementProgress(BaseModel):
    total_sections: int
    subsections_per_section: Dict[int, int] = Field(default_factory=dict)

    current_section_index: int = 0
    current_section_status: SectionStatus = SectionStatus.NOT_STARTED

    current_subsection_index: int = 0
    current_subsection_status: SubsectionStatus = SubsectionStatus.NOT_STARTED
    
    current_review_status: Optional[ReviewType] = None

    completed_sections: List[int] = Field(default_factory=list)
    completed_subsections: Dict[int, List[int]] = Field(default_factory=dict)