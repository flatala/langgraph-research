from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    SubsectionStatus, Subsection, ReviewRound,
    GroundingReviewFineGrainedResult, GroundingReviewOverallAssessment,
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
    current_subsection: Subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]

    print("🔍 Reviewing grounding and citations...")

    # extract citations from the subsection
    claims_by_arxiv_id, citations = await _extract_citations(cfg, current_subsection)

    # prepare per-papare groudedness checks
    groudedness_reviews = []
    grounding_overall = None
    for arxiv_id, claims in claims_by_arxiv_id.items():
        results, overall = await _review_grounding_for_paper(cfg, arxiv_id, claims, current_subsection)
        groudedness_reviews.extend(results)
        grounding_overall = overall  

    # create or update review round with grounding results
    # check if there's already a review round from content review
    if current_subsection.review_history:
        latest_review_round = current_subsection.review_history[-1]
        updated_review_round = latest_review_round.model_copy(update={
            "grounding_review_results": groudedness_reviews,
            "grounding_overall_assessment": grounding_overall,
            "grounding_review_passed": _has_no_grounding_issues(groudedness_reviews)
        })
        feedback_history = current_subsection.review_history[:-1] + [updated_review_round]
    else:
        review_round = ReviewRound(
            grounding_review_results=groudedness_reviews,
            grounding_overall_assessment=grounding_overall,
            grounding_review_passed=_has_no_grounding_issues(groudedness_reviews)
        )
        feedback_history = [review_round]
    
    # add feedback and citations to subsection
    updated_subsection = current_subsection.model_copy(update={
        "feedback_history": feedback_history,
        "citations": citations
    })
    
    # update literature survey
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


async def _extract_citations(
    cfg: Configuration, 
    current_subsection: Subsection
) -> tuple[Dict[str, List[CitationClaim]], List[CitationClaim]]:
    
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

    # aggregate all citation claims by paper arxiv ids (if is one of sources, gets add to list)  
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
            
    return claims_by_arxiv_id, citation_extraction.citation_claims


async def _review_grounding_for_paper(
    cfg: Configuration,
    arxiv_id: str,
    claims: List[CitationClaim],
    current_subsection: Subsection,
) -> tuple[List[GroundingReviewFineGrainedResult], Optional[GroundingReviewOverallAssessment]]:
    """
    Perform grounding review for a single paper (one arXiv id).

    Returns:
        (fine_grained_results, overall_assessment_or_none)
    """

    llm = get_orchestrator_llm(cfg=cfg).with_config({"response_format": {"type": "json_object"}})
    full_paper = next((p for p in current_subsection.papers if p.arxiv_id == arxiv_id), None)
    if full_paper is None:
        raise ValueError(f"Paper with arXiv id {arxiv_id} not found in subsection.papers")

    # format claims into context string
    groundedness_review_context = ""
    for i, claim in enumerate(claims, 1):
        groundedness_review_context += (
            f"\n**Claim {i}:**\n"
            f"   Citation: {claim.citation}\n"
            f"   Supported claim: {claim.supported_claim}\n"
            f"   Full sentence: {claim.full_sentence}\n"
            f"   Context: {claim.surrounding_context}\n\n"
        )

    # prepare prompt
    grounding_review_prompt = cfg.review_grounding_prompt.format(
        citation_claims=groundedness_review_context,
        full_paper_content=full_paper.content.page_content,
    )

    # LLM call
    print(f"🤖 Performing grounding review for paper {arxiv_id}...")
    grounding_response = await llm.ainvoke([HumanMessage(content=grounding_review_prompt)])
    grounding_result_raw = grounding_response.content.strip()
    grounding_data = json.loads(grounding_result_raw)

    # fine-grained result parsing - each result is now an individual issue
    fine_results: List[GroundingReviewFineGrainedResult] = []
    for result_data in grounding_data.get("fine_grained_results", []):
        fine_results.append(
            GroundingReviewFineGrainedResult(
                paper_id=arxiv_id,  # Add paper ID for traceability
                severity=result_data.get("severity", ""),
                issue_type=result_data.get("issue_type", ""),
                citation=result_data.get("citation", ""),
                supported_claim=result_data.get("supported_claim", ""),
                verification_status=result_data.get("verification_status", ""),
                accuracy_score=result_data.get("accuracy_score", 0),
                problematic_text=result_data.get("problematic_text", ""),
                explanation=result_data.get("explanation", ""),
                source_evidence=result_data.get("source_evidence", ""),
                recommendation=result_data.get("recommendation", "") or result_data.get("reccomendation", ""),
                source_location=result_data.get("source_location", ""),
                confidence_level=result_data.get("confidence_level", ""),
            )
        )

    overall_data = grounding_data.get("overall_assessment", {})
    overall_assessment = GroundingReviewOverallAssessment(
        total_claims_verified=overall_data.get("total_claims_verified", 0),
        fully_supported_claims=overall_data.get("fully_supported_claims", 0),
        problematic_claims=overall_data.get("problematic_claims", 0),
    )

    print(f"✅ Grounding review completed for {full_paper.title}: {len(fine_results)} claims analyzed")
    return fine_results, overall_assessment


def _has_no_grounding_issues(grounding_reviews: List[GroundingReviewFineGrainedResult]) -> bool:
    """Check if grounding review passed by counting total issues across all papers."""
    # Since each result is now an individual issue, the count is simply the list length
    # But we only count results that actually represent issues (not fully supported claims)
    issues = [result for result in grounding_reviews 
             if result.verification_status in ["partially_supported", "unsupported", "misrepresented", "contradicted"]]
    return len(issues) == 0
