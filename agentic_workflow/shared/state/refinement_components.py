from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class SubsectionStatus(str, Enum):
    READY_FOR_CONTEXT_PREP = "ready_for_context_prep"
    READY_FOR_WRITING = "ready_for_writing"
    READY_FOR_CONTENT_REVIEW = "ready_for_content_review"
    READY_FOR_CONTENT_REVISION = "ready_for_content_revision"  # Content issues found
    READY_FOR_GROUNDING_REVIEW = "ready_for_grounding_review"
    READY_FOR_GROUNDING_REVISION = "ready_for_grounding_revision"  # Grounding issues found
    COMPLETED = "completed" 

class SectionStatus(str, Enum):
    NOT_STARTED = "not_started"       
    IN_PROGRESS = "in_progress"      
    COMPLETED = "completed"           

class ContentReviewFineGrainedResult(BaseModel):
    reviewed_text: str
    error_type: str  # e.g., "clarity", "conciseness", "flow", "grammar", "vague", "style", "accuracy"
    explanation: str
    correction_suggestion: str

class ContentReviewOverallAssessment(BaseModel):
    score: int
    meets_minimum: bool
    reasoning: str

class GroundingCheckResult(BaseModel):
    paper_ids: Optional[List[str]] = None
    citation: str
    supported_claim: str
    # status: "valid" or "invalid"
    status: str
    # Optional details if invalid
    error_type: Optional[str] = None # misrepresentation, hallucination, etc
    explanation: Optional[str] = None
    correction_suggestion: Optional[str] = None

class ReviewRound(BaseModel):
    # Content review results
    content_review_results: Optional[List[ContentReviewFineGrainedResult]] = None
    content_overall_assessment: Optional[ContentReviewOverallAssessment] = None
    content_review_passed: bool = False
    
    # Grounding review results  
    grounding_review_results: Optional[List[GroundingCheckResult]] = None
    grounding_review_passed: bool = False

class CitationClaim(BaseModel):
    citation: str
    cited_papers: List[str]
    supported_claim: str

class CitationExtraction(BaseModel):
    citation_claims: List[CitationClaim]
    total_citations: int
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'CitationExtraction':
        """Parse JSON data into CitationExtraction Pydantic model with validation."""
        citation_claims = []
        
        for claim_data in json_data.get("citation_claims", []):
            citation_claim = CitationClaim(
                citation=claim_data.get("citation", ""),
                cited_papers=claim_data.get("cited_papers", []),
                supported_claim=claim_data.get("supported_claim", ""),
            )
            citation_claims.append(citation_claim)
        
        return cls(
            citation_claims=citation_claims,
            total_citations=json_data.get("total_citations", 0)
        )

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
    subsection_title: str
    papers: List[PaperWithSegements]
    key_point_text: str
    content: str

    revision_count: int = 0
    review_history: List[ReviewRound] = Field(default_factory=list)
    refinement_messages: List[BaseMessage] = Field(default_factory=list)  # Continuous message thread

    citations: List[CitationClaim] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True 

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
    current_subsection_status: SubsectionStatus = SubsectionStatus.READY_FOR_CONTEXT_PREP
    
    current_review_status: Optional[str] = None

    completed_sections: List[int] = Field(default_factory=list)
    completed_subsections: Dict[int, List[int]] = Field(default_factory=dict)
