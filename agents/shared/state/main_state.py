from typing_extensions import TypedDict, Optional, List
from dataclasses import dataclass, field
from agents.shared.state.planning_components import Plan

class CachingOptions(TypedDict):
    cached_plan_id: Optional[str] = None
    cached_section_ids: Optional[List[str]] = None

@dataclass(kw_only=True)
class AgentState(TypedDict):

    # intial params
    caching_options: Optional[CachingOptions] = field(default=None)
    topic: str
    paper_recency: str  

    # history of messages
    messages: list

    # arxiv search queries and survey plan
    search_queries: Optional[List[str]] = field(default=None)   
    plan: Optional[Plan] = field(default=None)    

    completed: bool

