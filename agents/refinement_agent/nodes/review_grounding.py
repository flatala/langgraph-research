from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    SubsectionStatus, Subsection, ReviewRound,
    GroundingCheckResult, CitationExtraction, CitationClaim,
    PaperWithSegements
)
from agents.shared.utils.llm_utils import get_orchestrator_llm

from typing import Dict, Optional, List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import json
import asyncio

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

    # Prepare verification tasks
    verification_tasks = []
    for arxiv_id, claims in claims_by_arxiv_id.items():
        # Get paper content once
        paper = next((p for p in current_subsection.papers if p.arxiv_id == arxiv_id), None)
        if not paper: 
            print(f"⚠️ Paper {arxiv_id} not found in sources, skipping verification for its claims.")
            continue
        
        for claim in claims:
            verification_tasks.append(_verify_single_claim(cfg, claim, paper))

    # Execute all verifications in parallel
    print(f"🤖 Verifying {len(verification_tasks)} claims in parallel...")
    verification_results = await asyncio.gather(*verification_tasks)
    
    # create or update review round with grounding results
    # check if there's already a review round from content review
    if current_subsection.review_history:
        latest_review_round = current_subsection.review_history[-1]
        updated_review_round = latest_review_round.model_copy(update={
            "grounding_review_results": verification_results,
            "grounding_review_passed": _has_no_grounding_issues(verification_results)
        })
        feedback_history = current_subsection.review_history[:-1] + [updated_review_round]
    else:
        review_round = ReviewRound(
            grounding_review_results=verification_results,
            grounding_review_passed=_has_no_grounding_issues(verification_results)
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
) -> Tuple[Dict[str, List[CitationClaim]], List[CitationClaim]]:
    
    extract_citations_prompt = cfg.citation_identification_prompt.format(
        paper_segment=current_subsection.content
    )
    user_msg = HumanMessage(content=extract_citations_prompt)
    
    # get LLM and extract all citations from subsection
    llm = get_orchestrator_llm(cfg=cfg).with_config({"response_format": {"type": "json_object"}}) 
    print("🤖 Extracting citations from the subsection...")
    ai_response = await llm.ainvoke([user_msg])
    extraction_response = ai_response.content.strip()
    
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
                # Warning instead of crash, as parallel execution can handle issues
                print(f"⚠️ Hallucinated citation detected: ArXiv ID '{cited_paper_id}' not found in subsection papers. Claim: {claim.citation}")
                
    return claims_by_arxiv_id, citation_extraction.citation_claims


async def _verify_single_claim(
    cfg: Configuration,
    claim: CitationClaim,
    paper: PaperWithSegements,
) -> GroundingCheckResult:
    """
    Verify a single claim against the paper content using LLM.
    """
    llm = get_orchestrator_llm(cfg=cfg).with_config({"response_format": {"type": "json_object"}})
    
    verify_prompt = cfg.verify_claim_prompt.format(
        citation=claim.citation,
        claim=claim.supported_claim,
        paper_content=paper.content.page_content
    )
    
    response = await llm.ainvoke([HumanMessage(content=verify_prompt)])
    result_data = json.loads(response.content.strip())
    
    return GroundingCheckResult(
        paper_id=paper.arxiv_id,
        citation=result_data.get("citation", claim.citation),
        supported_claim=result_data.get("supported_claim", claim.supported_claim),
        status=result_data.get("status", "invalid"),
        error_type=result_data.get("error_type"),
        explanation=result_data.get("explanation"),
        correction_suggestion=result_data.get("correction_suggestion")
    )



def _has_no_grounding_issues(grounding_reviews: List[GroundingCheckResult]) -> bool:
    """Check if grounding review passed by counting total invalid claims."""
    issues = [result for result in grounding_reviews if result.status == "invalid"]
    if issues:
        print(f"❌ Found {len(issues)} grounding issues")
    else:
        print("✅ No grounding issues found")
    return len(issues) == 0