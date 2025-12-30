from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from langgraph.graph.message import add_messages

from agentic_workflow.shared.state.planning_components import Plan
from agentic_workflow.shared.state.refinement_components import RefinementProgress, Section
from agentic_workflow.shared.state.overleaf_components import OverleafProgress

class AgentState(BaseModel):
    # initial params
    topic: str
    paper_recency: str
    completed: bool
    review_id: str  # Database review ID

    # history of messages
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # arxiv search queries and survey plan
    search_queries: Optional[List[str]] = None
    plan: Optional[Plan] = None

    # survey refinement
    refinement_progress: Optional[RefinementProgress] = None
    literature_survey: List[Section] = Field(default_factory=list)

    # overleaf/latex export
    overleaf_progress: Optional[OverleafProgress] = None
    latex_export_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

