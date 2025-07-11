from typing_extensions import TypedDict, Annotated
from langchain_core.documents import Document
from dataclasses import dataclass, field
import operator
from shared.state import reduce_docs

@dataclass(kw_only=True)
class LitState(TypedDict):
    topic: str
    paper_recency: str               
    plan: str
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    draft_sections: list[str] 
    verified_sections: list[str]
