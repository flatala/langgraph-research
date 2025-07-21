from typing_extensions import TypedDict, Optional, Any, List
from dataclasses import dataclass, field
from langchain_core.documents import Document

class PaperRef(TypedDict):
    title: str     
    year: int
    url: str  
    summary: str
    citation_reason: str

class KeyPoint(TypedDict):
    text: str
    papers: List[PaperRef]  

class Section(TypedDict):
    number: int
    title: str
    outline: str
    key_points: List[KeyPoint]

class Plan(TypedDict):
    plan: List[Section]
    reasoning: str

class CachingOptions(TypedDict):
    cached_plan_id: Optional[str] = None
    cached_section_ids: Optional[List[str]] = None

# TODO: cleanup the usage of Optionals in the State
@dataclass(kw_only=True)
class LitState(TypedDict):
    caching_options: Optional[CachingOptions]

    messages: list
    
    retriever: Optional[Any] = field(default=None)
    documents: Optional[List[Document]] = field(default=None)

    topic: str
    paper_recency: str   

    search_queries: List[str]    
    plan: Plan    

    draft_sections: List[str]          
    verified_sections: List[str]

    completed: bool = field(default=False)