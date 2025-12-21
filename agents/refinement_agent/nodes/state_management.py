from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import RefinementProgress, SectionStatus, SubsectionStatus
from agents.shared.utils.llm_utils import get_text_llm, get_orchestrator_llm
from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration

from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
from pprint import pprint

import hashlib
import json

def initialise_refinement_progress(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Initialize the refinement progress in the agent state.
    """

    print("\n" + "="*60)
    print("🎯 REFINEMENT STAGE STARTING")
    print("="*60 + "\n")

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
        current_subsection_status=SubsectionStatus.READY_FOR_CONTEXT_PREP,
        current_review_status=None,
        completed_sections=[],
        completed_subsections={},
    )

    print("Refinement progress initialized.\n")
    pprint(progress)

    return { "refinement_progress": progress }
    

def decide_refinement_stage(state: AgentState, *, config: Optional[RunnableConfig] = None) -> str:
    """
    Route to next stage based on current refinement progress.
    Clear 1:1 mapping between status and action.
    """
    progress = state.refinement_progress
    
    if not progress or progress.current_section_index >= progress.total_sections:
        return "complete_refinement"
    
    status = progress.current_subsection_status
    
    route_map = {
        SubsectionStatus.READY_FOR_CONTEXT_PREP: "prepare_subsection_context",
        SubsectionStatus.READY_FOR_WRITING: "write_subsection",
        SubsectionStatus.READY_FOR_CONTENT_REVIEW: "review_content",
        SubsectionStatus.READY_FOR_GROUNDING_REVIEW: "review_grounding", 
        SubsectionStatus.READY_FOR_FEEDBACK: "process_feedback",
        SubsectionStatus.READY_FOR_REVISION: "start_revision",
        SubsectionStatus.COMPLETED: "advance_to_next"
    }
    
    return route_map.get(status, "complete_refinement")


def advance_to_next(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Advance to next subsection or section.
    Status: COMPLETED → READY_FOR_CONTEXT_PREP (next subsection) or section advance
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    # Check if more subsections in current section
    total_subsections = progress.subsections_per_section[current_section_idx]
    next_subsection_idx = current_subsection_idx + 1
    
    if next_subsection_idx >= total_subsections:
        # Section complete - advance to next section
        next_section_idx = current_section_idx + 1
        completed_sections = list(progress.completed_sections)
        completed_sections.append(current_section_idx)
        
        print(f"🎉 Section {current_section_idx+1} completed! Moving to section {next_section_idx+1}")
        
        return {
            "refinement_progress": progress.model_copy(update={
                "current_section_index": next_section_idx,
                "current_section_status": SectionStatus.NOT_STARTED,
                "current_subsection_index": 0,
                "current_subsection_status": SubsectionStatus.READY_FOR_CONTEXT_PREP,
                "completed_sections": completed_sections
            })
        }
    else:
        # Move to next subsection
        print(f"➡️  Moving to subsection {next_subsection_idx+1} of section {current_section_idx+1}")
        
        return {
            "refinement_progress": progress.model_copy(update={
                "current_subsection_index": next_subsection_idx,
                "current_subsection_status": SubsectionStatus.READY_FOR_CONTEXT_PREP
            })
        }
    

def complete_refinement(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Complete the entire refinement process.
    """
    print("🎉 Literature survey refinement completed!")

    # Calculate stats
    total_sections = len(state.literature_survey)
    total_subsections = sum(len(section.subsections) for section in state.literature_survey if section.subsections)
    total_revisions = sum(
        subsection.revision_count
        for section in state.literature_survey
        for subsection in (section.subsections or [])
        if subsection
    )

    print(f"📊 Final stats:")
    print(f"   Sections: {total_sections}")
    print(f"   Subsections: {total_subsections}")
    print(f"   Total revisions: {total_revisions}")

    return {"completed": True}


def cleanup_temp_cache(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Clean up temporary paper cache after review is complete.
    """
    from data.temp_cache.paper_cache import PaperCache

    paper_cache = PaperCache(state.review_id)
    paper_cache.cleanup()

    return {}
