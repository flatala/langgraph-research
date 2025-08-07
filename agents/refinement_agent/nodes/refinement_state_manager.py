"""
Lean Refinement State Management

Essential nodes for subsection-by-subsection literature survey refinement using clear status enums.
"""

from typing import Dict, Optional
from langchain_core.runnables import RunnableConfig

from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    RefinementProgress, Section, Subsection, PaperWithSegements,
    SectionStatus, SubsectionStatus, ReviewType, ReviewFeedback
)


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


async def prepare_subsection_context(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Comprehensive preparation: RAG + context + subsection setup.
    Status: READY_FOR_CONTEXT_PREP → READY_FOR_WRITING
    """
    progress = state.refinement_progress
    plan = state.plan
    
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    # Get current key point from plan
    section_plan = plan.plan[current_section_idx]
    key_point = section_plan.key_points[current_subsection_idx]
    
    print(f"📝 Preparing context for Section {current_section_idx+1}, Subsection {current_subsection_idx+1}")
    print(f"Key point: {key_point.text}")
    
    # TODO: Implement actual RAG retrieval here
    papers_with_segments = []
    for paper_ref in key_point.papers:
        paper_with_segments = PaperWithSegements(
            title=paper_ref.title,
            authors=["Author"],  # TODO: Extract from actual paper
            arxiv_id="placeholder",
            arxiv_url=paper_ref.url,
            citation=f"({paper_ref.title}, {paper_ref.year})",
            content=None,  # TODO: Load actual document
            relevant_segments=["TODO: Retrieve relevant segments using RAG"]
        )
        papers_with_segments.append(paper_with_segments)
    
    # Create subsection with all context
    subsection = Subsection(
        subsection_index=current_subsection_idx,
        papers=papers_with_segments,
        key_point_text=key_point.text,
        content="",
        revision_count=0,
        feedback_history=[],
        citations=[]
    )
    
    # Ensure section exists and add subsection
    literature_survey = list(state.literature_survey)
    
    # Create section if it doesn't exist
    if current_section_idx >= len(literature_survey):
        new_section = Section(
            section_index=current_section_idx,
            section_title=section_plan.title,
            section_outline=section_plan.outline,
            section_introduction="",
            subsections=[],
            section_markdown=""
        )
        literature_survey.append(new_section)
    
    # Add/update subsection
    current_section = literature_survey[current_section_idx]
    updated_section = current_section.model_copy()
    
    # Extend subsections list if needed
    while len(updated_section.subsections) <= current_subsection_idx:
        updated_section.subsections.append(None)
    
    updated_section.subsections[current_subsection_idx] = subsection
    literature_survey[current_section_idx] = updated_section
    
    print(f"✅ Context prepared: {len(papers_with_segments)} papers with segments")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_section_status": SectionStatus.IN_PROGRESS,
            "current_subsection_status": SubsectionStatus.READY_FOR_WRITING
        })
    }


async def write_subsection(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Write content for current subsection.
    Status: READY_FOR_WRITING → READY_FOR_CONTENT_REVIEW
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print(f"✍️  Writing subsection {current_subsection_idx+1} of section {current_section_idx+1}")
    
    # Get current subsection
    current_section = state.literature_survey[current_section_idx]
    current_subsection = current_section.subsections[current_subsection_idx]
    
    # TODO: Implement actual writing logic with LLM
    placeholder_content = f"""
### {current_subsection.key_point_text}

[TODO: Generate actual content using LLM with:]
- Key point: {current_subsection.key_point_text}
- Papers: {len(current_subsection.papers)} papers with segments
- Previous context for flow

This subsection discusses {current_subsection.key_point_text} based on the retrieved paper segments.
""".strip()
    
    # Update subsection with content
    updated_subsection = current_subsection.model_copy(update={
        "content": placeholder_content
    })
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    print("✅ Content written, ready for content review")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_CONTENT_REVIEW
        })
    }


async def review_content(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Perform content quality review.
    Status: READY_FOR_CONTENT_REVIEW → READY_FOR_GROUNDING_REVIEW (if pass) or READY_FOR_FEEDBACK (if fail)
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print("🔍 Reviewing content quality...")
    
    # Get current subsection
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    
    # TODO: Implement actual content review with LLM
    # For now, simple logic for testing
    review_passed = (current_subsection.revision_count % 2 == 0)
    
    feedback = ReviewFeedback(
        review_type=ReviewType.CONTENT,
        passed=review_passed,
        feedback=f"TODO: Implement actual content review. Current: {'PASS' if review_passed else 'FAIL'}",
        suggestions=["Improve clarity", "Add more examples"] if not review_passed else None
    )
    
    # Add feedback to subsection
    updated_subsection = current_subsection.model_copy()
    updated_subsection.feedback_history.append(feedback)
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    # Determine next status
    if review_passed:
        next_status = SubsectionStatus.READY_FOR_GROUNDING_REVIEW
        print("✅ Content review passed, ready for grounding review")
    else:
        next_status = SubsectionStatus.READY_FOR_FEEDBACK
        print("❌ Content review failed, ready for feedback processing")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": next_status
        })
    }


async def review_grounding(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Perform grounding/citation review.
    Status: READY_FOR_GROUNDING_REVIEW → READY_FOR_FEEDBACK
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print("🔍 Reviewing grounding and citations...")
    
    # Get current subsection
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    
    # TODO: Implement actual grounding review with LLM
    # For now, simple logic for testing
    review_passed = (current_subsection.revision_count % 3 != 1)  # Different pattern than content
    
    feedback = ReviewFeedback(
        review_type=ReviewType.GROUNDING,
        passed=review_passed,
        feedback=f"TODO: Implement actual grounding review. Current: {'PASS' if review_passed else 'FAIL'}",
        suggestions=["Fix citations", "Verify claims"] if not review_passed else None
    )
    
    # Add feedback to subsection
    updated_subsection = current_subsection.model_copy()
    updated_subsection.feedback_history.append(feedback)
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    if review_passed:
        print("✅ Grounding review passed, ready for feedback processing")
    else:
        print("❌ Grounding review failed, ready for feedback processing")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_FEEDBACK
        })
    }


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
    
    # Check if both content and grounding passed
    content_passed = any(f.review_type == ReviewType.CONTENT and f.passed for f in feedback_history)
    grounding_passed = any(f.review_type == ReviewType.GROUNDING and f.passed for f in feedback_history)
    
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


# Utility functions
def print_current_status(state: AgentState) -> str:
    """Print current refinement status for debugging."""
    progress = state.refinement_progress
    if not progress:
        return "Refinement not started"
    
    status_text = f"""
📝 Refinement Status:
   Section: {progress.current_section_index + 1}/{progress.total_sections}
   Subsection: {progress.current_subsection_index + 1}/{progress.subsections_per_section.get(progress.current_section_index, 0)}
   Status: {progress.current_subsection_status.value}
   Completed: {len(progress.completed_sections)} sections, {sum(len(v) for v in progress.completed_subsections.values())} subsections
"""
    print(status_text)
    return status_text