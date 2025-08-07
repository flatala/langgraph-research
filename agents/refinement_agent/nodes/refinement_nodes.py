from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import RefinementProgress, SectionStatus, SubsectionStatus, ReviewType
from agents.shared.utils.llm_utils import get_text_llm, get_orchestrator_llm
from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration

from typing import List, Optional
from datetime import datetime
from pathlib import Path
from pprint import pprint

import hashlib
import json

def initialise_refinement_progress(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
    """
    Initialize the refinement progress in the agent state.
    """

    total_sections = len(state.plan.plan)
    subsections_per_section = {
        i: len(section.key_points) for i, 
        section in enumerate(state.plan.plan)
    }

    progress = RefinementProgress(
        total_sections=total_sections,
        subsections_per_section=subsections_per_section,
        current_section_index=0,
        current_section_status=SectionStatus.NOT_STARTED,
        current_subsection_index=0,
        current_subsection_status=SubsectionStatus.NOT_STARTED,
        current_review_status=None,
        completed_sections=[],
        completed_subsections={},
    )

    print("Refinement progress initialized.\n")
    pprint(progress)

    return { "refinement_progress": progress }
    

# def decide_on_refinement_step(state: AgentState, *, config: Optional[RunnableConfig] = None) -> dict:
#     """
#     Decide the next step in the refinement process based on the current state.
#     """
    
#     progress = state["refinement_progress"]
#     current_section_index = progress["current_section_index"]
    
#     if current_section_index < state["total_sections"]:
#         if(completed_sec)
        
    
#     if