from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import RefinementProgress, SubsectionStatus, GroundingCheckResult
from agents.shared.utils.llm_utils import get_orchestrator_llm

from typing import Dict, Optional, List
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv(                
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,         
)


def process_feedback(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Process all feedback and determine if subsection is approved or needs revision.
    Status: READY_FOR_FEEDBACK → COMPLETED (if approved) or READY_FOR_REVISION (if not)
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index

    # get current subsection feedback
    logger.info("Processing feedback...")
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    feedback_history = current_subsection.review_history
    
    # check if both content and grounding passed in the latest review round
    if feedback_history:
        latest_round = feedback_history[-1]
        content_passed = latest_round.content_review_passed
        grounding_passed = latest_round.grounding_review_passed
    else:
        content_passed = False
        grounding_passed = False
    
    if content_passed and grounding_passed:
        logger.info("All reviews passed - subsection approved!")
        next_status = SubsectionStatus.COMPLETED
    else:
        logger.info("Reviews failed - subsection needs revision")
        next_status = SubsectionStatus.READY_FOR_REVISION
    
    return {
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": next_status
        })
    }


async def start_revision(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Start revision process: First refine grounding issues, then increment revision count and go back to writing.
    Status: READY_FOR_REVISION → READY_FOR_WRITING
    """
    cfg = Configuration.from_runnable_config(config)
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    # get current subsection
    current_section = state.literature_survey[current_section_idx]
    current_subsection = current_section.subsections[current_subsection_idx]
    
    # first, refine grounding issues if any exist
    refined_subsection = await _refine_grounding_issues(cfg, current_subsection)
    
    # update revision count
    updated_subsection = refined_subsection.model_copy(update={
        "revision_count": refined_subsection.revision_count + 1
    })
    
    # update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    logger.info(f"Starting revision #{updated_subsection.revision_count} for subsection {current_subsection_idx+1}")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_WRITING
        })
    }


async def _refine_grounding_issues(cfg: Configuration, subsection) -> object:
    """
    Refine grounding issues one by one.
    Returns the subsection with refined content.
    """
    if not subsection.review_history:
        return subsection
    
    # get latest review round
    latest_review = subsection.review_history[-1]
    grounding_results = latest_review.grounding_review_results or []
    
    # filter only invalid issues
    invalid_issues = [issue for issue in grounding_results if issue.status == "invalid"]
    if not invalid_issues:
        logger.info("No grounding issues to refine")
        return subsection
    
    # process each issue one by one
    logger.info(f"Refining {len(invalid_issues)} grounding issues...")
    current_content = subsection.content
    updated_subsection = subsection.model_copy()
    for i, issue in enumerate(invalid_issues, 1):
        logger.info(f"Fixing issue {i}/{len(invalid_issues)} - {issue.error_type}: {issue.citation}")
        
        # get the paper content for this issue
        papers = []
        for paper_id in issue.paper_ids:
            paper = next((p for p in subsection.papers if p.arxiv_id == paper_id), None)
            if not paper:
                raise ValueError(f"Paper with ID {paper_id} not found in subsection papers")
            papers.append(paper)

        # format the content of papers
        papers_content = "\n\n".join(
            f"# Paper {i}\n"
            f"- **Title**: {p.title}\n"
            f"- **arXiv ID**: {p.arxiv_id}\n"
            f"- **Content**:\n{(p.content.page_content if p.content else '').strip()}"
            for i, p in enumerate(papers, 1)
        )
        
        # create refinement prompt
        refinement_prompt = cfg.grounding_refinement_prompt.format(
            citation=issue.citation,
            supported_claim=issue.supported_claim,
            error_type=issue.error_type,
            explanation=issue.explanation,
            correction_suggestion=issue.correction_suggestion,
            current_subsection=current_content,
            full_paper_content=papers_content
        )
        
        # prepare messages    
        system_msg = SystemMessage(content="You are an expert academic writing assistant specializing in literature review refinement.")
        user_msg = HumanMessage(content=refinement_prompt)
        messages = [system_msg, user_msg]
        
        # get LLM and refine content
        logger.info(f"Refining content with LLM...")
        llm = get_orchestrator_llm(cfg=cfg)
        ai_response = await llm.ainvoke(messages)
        refined_content = ai_response.content.strip()
        
        # update current content for next iteration
        current_content = refined_content
        logger.info(f"Issue {i} refined successfully")
    
    # update subsection with final refined content
    updated_subsection.content = current_content
    logger.info(f"All {len(invalid_issues)} grounding issues have been refined")
    
    return updated_subsection
