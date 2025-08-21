from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    RefinementProgress, SubsectionStatus, ReviewRound,
    GroundingIssue, GroundingReviewFineGrainedResult, GroundingReviewOverallAssessment,
    CitationExtraction, CitationClaim
)
from agents.shared.utils.llm_utils import get_orchestrator_llm

from typing import Dict, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import json

load_dotenv(                
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,         
)    

async def review_grounding(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Perform grounding/citation review.
    Status: READY_FOR_GROUNDING_REVIEW → READY_FOR_FEEDBACK
    """    
    cfg = Configuration.from_runnable_config(config)
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index

    print("🔍 Reviewing grounding and citations...")
    
    # get current subsection and prepare citation extraction prompt
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    extract_citations_prompt = cfg.citation_identification_prompt.format(
        paper_segment=current_subsection.content
    )
    user_msg = HumanMessage(content=extract_citations_prompt)
    
    # get LLM and extract all citations from subsection
    llm = get_orchestrator_llm(cfg=cfg).with_config({"response_format": {"type": "json_object"}}) 
    print("🤖 Extracting citations from the subsection...")
    ai_response = await llm.ainvoke([user_msg])
    extraction_response = ai_response.content.strip()
    
    # parse JSON response
    citation_data = json.loads(extraction_response)
    citation_extraction = CitationExtraction.from_json(citation_data)
    print(f"📊 Extracted {citation_extraction.total_citations} citations from subsection")

    # aggregate all citation claims by paper arxiv ids (if is one of sources, gets adde to list)  
    available_arxiv_ids = {paper.arxiv_id for paper in current_subsection.papers}
    claims_by_arxiv_id: Dict[str, List[CitationClaim]] = {}
    for claim in citation_extraction.citation_claims:
        for cited_paper_id in claim.cited_papers:
            if cited_paper_id in available_arxiv_ids:
                if cited_paper_id not in claims_by_arxiv_id:
                    claims_by_arxiv_id[cited_paper_id] = []
                claims_by_arxiv_id[cited_paper_id].append(claim)
            else:
                # throw exception for hallucinated / mismatched citations 
                # (papers taht are not in the subsection's sources)
                raise ValueError(
                    f"Hallucinated citation detected: ArXiv ID '{cited_paper_id}' not found in subsection papers. "
                    f"Available papers: {list(available_arxiv_ids)}. "
                    f"Citation claim: '{claim.citation}' in context: '{claim.full_sentence}'"
                )

    pprint(citation_extraction.citation_claims)


    groudedness_reviews = []
    for arxiv_id, claims in claims_by_arxiv_id.items():
        # TODO: add some check at earlier stage (when .papers is populated) to ensure that there is no 
        # dulicate papers in that list, maybe use a dictionary to construct it in the first place so
        # duplications wont be even possible there?
        full_paper = next(p for p in current_subsection.papers if p.arxiv_id == arxiv_id)
        
        # format claims for the grounding review prompt
        groudedness_review_context = ""
        for i, claim in enumerate(claims, 1):
            groudedness_review_context += f"\n**Claim {i}:**\n"
            groudedness_review_context += f"   Citation: {claim.citation}\n"
            groudedness_review_context += f"   Supported claim: {claim.supported_claim}\n"
            groudedness_review_context += f"   Full sentence: {claim.full_sentence}\n"
            groudedness_review_context += f"   Context: {claim.surrounding_context}\n\n"
        
        # prepare grounding review prompt
        grounding_review_prompt = cfg.review_grounding_prompt.format(
            citation_claims=groudedness_review_context,
            full_paper_content=full_paper.content.page_content
        )
        
        # create message and invoke LLM
        grounding_user_msg = HumanMessage(content=grounding_review_prompt)
        print(f"🤖 Performing grounding review for paper {arxiv_id}...")
        grounding_response = await llm.ainvoke([grounding_user_msg])
        grounding_result = grounding_response.content.strip()
        
        # parse JSON response
        grounding_data = json.loads(grounding_result)
        
        # Parse fine-grained results
        fine_grained_data = grounding_data.get("fine_grained_results", [])
        parsed_results = []
        for result_data in fine_grained_data:
            issues_data = result_data.get("issues_found", [])
            issues = []
            for issue_data in issues_data:
                issue = GroundingIssue(
                    severity=issue_data.get("severity", ""),
                    issue_type=issue_data.get("issue_type", ""),
                    problematic_text=issue_data.get("problematic_text", ""),
                    explanation=issue_data.get("explanation", ""),
                    source_evidence=issue_data.get("source_evidence", ""),
                    recommendation=issue_data.get("reccomendation", "")  # Note: typo in prompt
                )
                issues.append(issue)
            
            grounding_result_obj = GroundingReviewFineGrainedResult(
                citation=result_data.get("citation", ""),
                supported_claim=result_data.get("supported_claim", ""),
                verification_status=result_data.get("verification_status", ""),
                accuracy_score=result_data.get("accuracy_score", 0),
                issues_found=issues,
                source_location=result_data.get("source_location", ""),
                confidence_level=result_data.get("confidence_level", "")
            )
            parsed_results.append(grounding_result_obj)
        
        # store parsed response
        groudedness_reviews.extend(parsed_results)
        
        print(f"✅ Grounding review completed for {full_paper.title}: {len(parsed_results)} claims analyzed")

    print(f"📊 Completed grounding reviews for {len(groudedness_reviews)} total claims")

    # Parse overall assessment from the last paper's review (they should all have similar structure)
    if groudedness_reviews and grounding_data:
        overall_data = grounding_data.get("overall_assessment", {})
        grounding_overall = GroundingReviewOverallAssessment(
            total_claims_verified=overall_data.get("total_claims_verified", 0),
            fully_supported_claims=overall_data.get("fully_supported_claims", 0),
            problematic_claims=overall_data.get("problematic_claims", 0)
        )
    else:
        grounding_overall = None

    # Create or update review round with grounding results
    # Check if there's already a review round from content review
    if current_subsection.review_history:
        # Update the most recent review round
        latest_review_round = current_subsection.review_history[-1]
        updated_review_round = latest_review_round.model_copy(update={
            "grounding_review_results": groudedness_reviews,
            "grounding_overall_assessment": grounding_overall,
            "grounding_review_passed": True  # Always passes per requirements
        })
        # Replace the last review round
        feedback_history = current_subsection.review_history[:-1] + [updated_review_round]
    else:
        # Create new review round with only grounding results
        review_round = ReviewRound(
            grounding_review_results=groudedness_reviews,
            grounding_overall_assessment=grounding_overall,
            grounding_review_passed=True  # Always passes per requirements
        )
        feedback_history = [review_round]
    
    # Add feedback and citations to subsection
    updated_subsection = current_subsection.model_copy(update={
        "feedback_history": feedback_history,
        "citations": citation_extraction.citation_claims
    })
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_FEEDBACK
        })
    }