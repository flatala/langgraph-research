from langchain_core.runnables import RunnableConfig

from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import RefinementProgress, SubsectionStatus

from typing import Dict, Optional


def process_feedback(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Process all feedback and determine if subsection is approved or needs revision.
    Status: READY_FOR_FEEDBACK → COMPLETED (if approved) or READY_FOR_REVISION (if not)
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print("📊 Processing feedback...")
    
    # Get current subsection feedback
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    feedback_history = current_subsection.feedback_history
    
    # Check if both content and grounding passed in the latest review round
    if feedback_history:
        latest_round = feedback_history[-1]
        content_passed = latest_round.content_review_passed
        grounding_passed = latest_round.grounding_review_passed
    else:
        content_passed = False
        grounding_passed = False
    
    if content_passed and grounding_passed:
        print("✅ All reviews passed - subsection approved!")
        next_status = SubsectionStatus.COMPLETED
    else:
        print("❌ Reviews failed - subsection needs revision")
        next_status = SubsectionStatus.READY_FOR_REVISION
    
    return {
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": next_status
        })
    }

def start_revision(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Start revision process: increment revision count and go back to writing.
    Status: READY_FOR_REVISION → READY_FOR_WRITING
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    # Update revision count
    current_section = state.literature_survey[current_section_idx]
    current_subsection = current_section.subsections[current_subsection_idx]
    
    updated_subsection = current_subsection.model_copy(update={
        "revision_count": current_subsection.revision_count + 1
    })
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    print(f"🔄 Starting revision #{updated_subsection.revision_count} for subsection {current_subsection_idx+1}")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_WRITING
        })
    }