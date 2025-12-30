from pydantic import BaseModel, Field
from typing import List

class PaperRef(BaseModel):
    title: str     
    year: int
    url: str  
    summary: str
    citation_reason: str

class KeyPoint(BaseModel):
    text: str
    papers: List[PaperRef] = Field(default_factory=list)

class SectionPlan(BaseModel):
    number: int
    title: str
    outline: str
    key_points: List[KeyPoint] = Field(default_factory=list)

class Plan(BaseModel):
    plan: List[SectionPlan] = Field(default_factory=list)
    reasoning: str = ""