from typing_extensions import TypedDict, Annotated
from langchain_core.documents import Document
import operator

class LitState(TypedDict):
    topic: str
    paper_recency: str               
    plan: str
    documents: Annotated[list]
    draft_section: str
    verified_section: str
