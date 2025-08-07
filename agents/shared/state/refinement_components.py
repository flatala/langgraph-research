from typing_extensions import TypedDict, List, Optional, Dict, Any
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

class ReviewFeedback(TypedDict):
    review_type: ReviewType
    passed: bool
    feedback: str
    suggestions: Optional[List[str]]

class PaperWithSegements(TypedDict):
    title: str
    authors: List[str]
    arxiv_id: str
    arxiv_url: str
    citation: str
    content: Document
    relevant_segments: List[str]

class Subsection(TypedDict):
    subsection_index: int
    papers: List[PaperWithSegements]
    key_point_text: str
    content: str
    status: SubsectionStatus
    revision_count: int
    feedback_history: List[ReviewFeedback]
    citations: List[Dict[str, Any]] 

class Section(TypedDict):
    section_index: int
    section_title: str
    section_outline: str
    section_introduction: str
    subsections: List[Subsection]
    section_markdown: str
    status: SectionStatus    

class RefinementProgress(TypedDict):
    total_sections: int
    subsections_per_section: Dict[int, int]

    current_section_index: int
    current_subsection_index: int

    completed_sections: List[int]
    completed_subsections: Dict[int, List[int]]