from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from typing import Dict, Optional, List
from pathlib import Path
from dotenv import load_dotenv

from agents.shared.utils.llm_utils import get_text_llm, get_embedding_model
from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.refinement_agent.tools import create_search_paper_fragments_tool
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    RefinementProgress, SubsectionStatus, Section, Subsection
)

import logging

logger = logging.getLogger(__name__)
load_dotenv(
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,
)


async def process_content_feedback(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Process content review feedback and refine the subsection if needed.
    Uses the continuous message thread for context.

    Status transitions:
    - If content review passed → READY_FOR_GROUNDING_REVIEW
    - If content issues found → refine and go back to READY_FOR_CONTENT_REVIEW
    """
    cfg = Configuration.from_runnable_config(config)
    progress: RefinementProgress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index

    # get latest review round
    current_section: Section = state.literature_survey[current_section_idx]
    current_subsection: Subsection = current_section.subsections[current_subsection_idx]
    if not current_subsection.review_history:
        logger.warning("No review history found, passing to grounding review")
        return {
            "refinement_progress": progress.model_copy(update={
                "current_subsection_status": SubsectionStatus.READY_FOR_GROUNDING_REVIEW
            })
        }

    # check if content review passed
    latest_review = current_subsection.review_history[-1]
    if latest_review.content_review_passed:
        logger.info("Content review passed, subsection completed!")
        return {
            "refinement_progress": progress.model_copy(update={
                "current_subsection_status": SubsectionStatus.COMPLETED
            })
        }

    # content review failed - need to fix issues
    logger.info("Content review failed, fixing issues...")
    
    # prepare prompt
    issues_list = _format_content_issues(latest_review.content_review_results)
    overall = latest_review.content_overall_assessment
    feedback_prompt = cfg.content_feedback_prompt.format(
        issues_list=issues_list,
        score=overall.score if overall else "N/A",
        reasoning=overall.reasoning if overall else "No reasoning provided"
    )

    # get the message thread and add feedback
    messages = list(current_subsection.refinement_messages)
    messages.append(HumanMessage(content=feedback_prompt))

    # invoke LLM to refine content
    logger.info("Refining content with LLM...")
    llm = get_text_llm(cfg=cfg)
    ai_response = await llm.ainvoke(messages)
    refined_content = ai_response.content.strip()
    messages.append(AIMessage(content=refined_content))

    # update subsection
    updated_subsection = current_subsection.model_copy(update={
        "content": refined_content,
        "refinement_messages": messages,
        "revision_count": current_subsection.revision_count + 1
    })

    # update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section

    logger.info("Content refined, going back to content review")
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_CONTENT_REVIEW
        })
    }


def _format_content_issues(content_issues) -> str:
    """Format content review issues into a readable list."""
    if not content_issues:
        return "No specific issues identified."

    lines = []
    for i, issue in enumerate(content_issues, 1):
        lines.append(f"{i}. **{issue.error_type}**: {issue.explanation}")
        lines.append(f"   - Problematic text: \"{issue.reviewed_text}\"")
        lines.append(f"   - Suggestion: {issue.correction_suggestion}")
        lines.append("")

    return "\n".join(lines)


async def process_grounding_feedback(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Process grounding review feedback with RAG tool access.
    Runs an agentic loop where the LLM can search papers to find correct evidence.

    Status transitions:
    - If grounding review passed → COMPLETED
    - If grounding issues found → refine with tool access and go back to READY_FOR_GROUNDING_REVIEW
    """
    cfg = Configuration.from_runnable_config(config)
    progress: RefinementProgress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index

    current_section: Section = state.literature_survey[current_section_idx]
    current_subsection: Subsection = current_section.subsections[current_subsection_idx]

    # get latest review round
    if not current_subsection.review_history:
        logger.warning("No review history found, marking as completed")
        return {
            "refinement_progress": progress.model_copy(update={
                "current_subsection_status": SubsectionStatus.COMPLETED
            })
        }

    latest_review = current_subsection.review_history[-1]

    # Check if grounding review passed
    if latest_review.grounding_review_passed:
        logger.info("Grounding review passed, proceeding to content review")
        return {
            "refinement_progress": progress.model_copy(update={
                "current_subsection_status": SubsectionStatus.READY_FOR_CONTENT_REVIEW
            })
        }

    # grounding review failed
    logger.info("Grounding review failed...")

    # instantiate the tool manually with correct papers
    available_paper_ids = [p.arxiv_id for p in current_subsection.papers]
    embeddings = get_embedding_model(cfg)
    search_tool = create_search_paper_fragments_tool(review_id=state.review_id, available_paper_ids=available_paper_ids, embeddings=embeddings)

    # prepare prompt
    issues_list = _format_grounding_issues(latest_review.grounding_review_results)
    available_papers = _format_available_papers(current_subsection.papers)
    feedback_prompt = cfg.grounding_feedback_prompt.format(issues_list=issues_list,available_papers=available_papers)

    # prepare message thread and LLM with ToolNode for parallel execution
    messages = list(current_subsection.refinement_messages)
    messages.append(HumanMessage(content=feedback_prompt))
    llm = get_text_llm(cfg=cfg)
    llm_with_tools = llm.bind_tools([search_tool])
    tool_node = ToolNode([search_tool])

    max_tool_iterations = 10
    iteration = 0
    while iteration < max_tool_iterations:
        iteration += 1
        logger.info(f"Grounding refinement iteration {iteration}")

        ai_response = await llm_with_tools.ainvoke(messages)
        messages.append(ai_response)

        # check for tool calls
        if not ai_response.tool_calls:
            break

        # execute tool calls in parallel using ToolNode
        logger.info(f"Executing {len(ai_response.tool_calls)} tool calls")
        tool_result = await tool_node.ainvoke({"messages": messages})
        messages.extend(tool_result["messages"])

    # update subsection
    refined_content = ai_response.content.strip()
    updated_subsection = current_subsection.model_copy(update={
        "content": refined_content,
        "refinement_messages": messages,
        "revision_count": current_subsection.revision_count + 1
    })

    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section

    logger.info("Grounding refined, going back to grounding review")
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_GROUNDING_REVIEW
        })
    }


def _format_grounding_issues(grounding_results) -> str:
    """Format grounding review issues into a readable list."""
    if not grounding_results:
        return "No grounding issues identified."

    invalid_issues = [r for r in grounding_results if r.status == "invalid"]
    if not invalid_issues:
        return "No invalid grounding issues found."

    lines = []
    for i, issue in enumerate(invalid_issues, 1):
        lines.append(f"{i}. **{issue.error_type}** for citation {issue.citation}")
        lines.append(f"   - Claim: \"{issue.supported_claim}\"")
        lines.append(f"   - Problem: {issue.explanation}")
        lines.append(f"   - Suggestion: {issue.correction_suggestion}")
        lines.append(f"   - Paper IDs: {', '.join(issue.paper_ids) if issue.paper_ids else 'N/A'}")
        lines.append("")

    return "\n".join(lines)


def _format_available_papers(papers) -> str:
    """Format available papers for the prompt."""
    if not papers:
        return "No papers available."

    lines = []
    for paper in papers:
        lines.append(f"- **{paper.arxiv_id}**: {paper.title}")

    return "\n".join(lines)
