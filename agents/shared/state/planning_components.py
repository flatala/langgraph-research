from typing_extensions import TypedDict, List

class PaperRef(TypedDict):
    title: str     
    year: int
    url: str  
    summary: str
    citation_reason: str

class KeyPoint(TypedDict):
    text: str
    papers: List[PaperRef]  

class SectionPlan(TypedDict):
    number: int
    title: str
    outline: str
    key_points: List[KeyPoint]

class Plan(TypedDict):
    plan: List[SectionPlan]
    reasoning: str