from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from langchain_core.documents import Document
from textwrap import indent

class SubsectionStatus(str, Enum):
    READY_FOR_CONTEXT_PREP = "ready_for_context_prep"        
    READY_FOR_WRITING = "ready_for_writing"                  
    READY_FOR_CONTENT_REVIEW = "ready_for_content_review"   
    READY_FOR_GROUNDING_REVIEW = "ready_for_grounding_review" 
    READY_FOR_FEEDBACK = "ready_for_feedback"                
    READY_FOR_REVISION = "ready_for_revision"              
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
    content_review_messages: list = Field(default_factory=list)
    grounding_review_messages: list = Field(default_factory=list)

    citations: List[CitationClaim] = Field(default_factory=list) 

class Section(BaseModel):
    section_index: int
    section_title: str
    section_outline: str
    section_introduction: str
    subsections: List[Subsection] = Field(default_factory=list)
    section_markdown: str = ""
    
    def print_section(self, include_segments: bool = False) -> str:
        """
        Nicely format the section and subsections for quick inspection.
        """
        lines: List[str] = []
        
        # Section header
        lines.append(f"\n{'='*60}")
        lines.append(f"Section {self.section_index + 1}: {self.section_title}")
        lines.append(f"{'='*60}")
        lines.append(f"Outline: {self.section_outline}")
        
        if self.section_introduction:
            lines.append(f"Introduction: {self.section_introduction}")
        
        lines.append(f"Subsections: {len([s for s in self.subsections if s is not None])}")
        lines.append("-" * 60)
        
        # Process each subsection
        for i, subsection in enumerate(self.subsections):
            if subsection is None:
                lines.append(f"  Subsection {i + 1}: [Not initialized]")
                continue
                
            lines.append(f"\n  Subsection {i + 1}: {subsection.key_point_text}")
            lines.append(f"    Status: Revision #{subsection.revision_count}")
            
            # Full content
            if subsection.content:
                lines.append("    Content:")
                # Indent each line of content
                content_lines = subsection.content.split('\n')
                for content_line in content_lines:
                    lines.append(f"      {content_line}")
            else:
                lines.append("    Content: [No content generated yet]")
            
            lines.append("")  # blank line between subsections
        
        formatted = "\n".join(lines)
        return formatted

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
