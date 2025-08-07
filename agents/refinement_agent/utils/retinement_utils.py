from agents.shared.state.main_state import AgentState

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