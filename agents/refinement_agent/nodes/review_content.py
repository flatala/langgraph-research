from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    RefinementProgress, SubsectionStatus, ReviewRound,
    ContentReviewFineGrainedResult, ContentReviewOverallAssessment
)
from agents.shared.utils.llm_utils import get_orchestrator_llm

from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import json
import re

load_dotenv(                
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,         
)    

async def review_content(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Perform content quality review.
    Status: READY_FOR_CONTENT_REVIEW → READY_FOR_GROUNDING_REVIEW (if pass) or READY_FOR_FEEDBACK (if fail)
    """
    cfg = Configuration.from_runnable_config(config)
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print("🔍 Reviewing content quality...")
    
    # prepare content review prompt
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    review_prompt = cfg.content_review_prompt.format(
        minimum_score=cfg.minimum_score,
        key_point=current_subsection.key_point_text,
        subsection=current_subsection.content
    )
    
    # create messages for LLM
    system_msg = SystemMessage(content="You are an expert content reviewer for academic literature surveys.")
    user_msg = HumanMessage(content=review_prompt)
    messages = [system_msg, user_msg]
    
    # get LLM and generate content quality review
    llm = get_orchestrator_llm(cfg=cfg)
    print("🤖 Generating content review with LLM...")
    ai_response = await llm.ainvoke(messages)
    review_text = ai_response.content.strip()

    # Add JSON markdown fence stripping logic
    if review_text.startswith("```") and review_text.endswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", review_text)
        if match:
            review_text = match.group(1).strip()
            
    review_data = json.loads(review_text)
    
    # Parse overall assessment
    overall_assessment_data = review_data.get("overall_assessment", {})
    score = overall_assessment_data.get("score", 0)
    meets_minimum = overall_assessment_data.get("meets_minimum", False)
    reasoning = overall_assessment_data.get("reasoning", "")
    
    # Parse fine-grained results
    fine_grained_data = review_data.get("fine_grained_results", [])
    fine_grained_results = []
    for result_data in fine_grained_data:
        result = ContentReviewFineGrainedResult(
            reviewed_text=result_data.get("reviewed_text", ""),
            error_type=result_data.get("error_type", ""),
            explanation=result_data.get("explanation", ""),
            correction_suggestion=result_data.get("correction_suggestion", "")
        )
        fine_grained_results.append(result)
    
    overall_assessment = ContentReviewOverallAssessment(
        score=score,
        meets_minimum=meets_minimum,
        reasoning=reasoning
    )
        
    print(f"📊 Content review score: {score}/10, Passed: {meets_minimum}")
    print(f"📝 Reasoning: {reasoning}")
    if fine_grained_results:
        print(f"🔍 Found {len(fine_grained_results)} fine-grained issues")
            
    
    # create review round with content review results
    review_round = ReviewRound(
        content_review_results=fine_grained_results,
        content_overall_assessment=overall_assessment,
        content_review_passed=meets_minimum
    )
    
    # add review round to subsection
    updated_subsection = current_subsection.model_copy()
    updated_subsection.review_history.append(review_round)
    
    # update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    # # Add AI messages to state for tracking
    # messages_update = list(state.messages) if state.messages else []
    # messages_update.extend([
    #     HumanMessage(content=f"Content review request for subsection {current_subsection_idx+1}"),
    #     AIMessage(content=f"Content review completed: Score {score}/10, {'PASSED' if meets_minimum else 'FAILED'}")
    # ])
    
    # Determine next status
    if meets_minimum:
        next_status = SubsectionStatus.READY_FOR_GROUNDING_REVIEW
        print("✅ Content review passed, ready for grounding review")
    else:
        next_status = SubsectionStatus.READY_FOR_FEEDBACK
        print("❌ Content review failed, ready for feedback processing")
    
    return {
        # TODO: we probably want a seprate message history for each review thread!!!
        # "messages": messages_update,
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": next_status
        })
    }