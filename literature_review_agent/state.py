from typing_extensions import TypedDict, Annotated, Optional, Any, List
from langchain_core.documents import Document
from dataclasses import dataclass, field
from literature_review_agent.utils import reduce_docs

class PaperRef(TypedDict):
    title: str     
    url: str  
    summary: str

class KeyPoint(TypedDict):
    text: str
    papers: List[PaperRef]  

class Section(TypedDict):
    title: str
    key_points: List[KeyPoint]

@dataclass(kw_only=True)
class LitState:
    messages: list
    topic: str
    paper_recency: str                
    plan: List[Section]                
    draft_sections: List[str]          
    verified_sections: List[str]
    info: Optional[dict[str, Any]] = field(default=None)

    # documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)


