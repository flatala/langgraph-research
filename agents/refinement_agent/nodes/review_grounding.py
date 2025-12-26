from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    SubsectionStatus, Subsection, ReviewRound,
    GroundingCheckResult, CitationExtraction, CitationClaim,
    PaperWithSegements, RefinementProgress
)
from agents.shared.utils.llm_utils import get_orchestrator_llm
from agents.shared.utils.json_utils import clean_and_parse_json

from typing import Dict, Optional, List, Tuple
from pathlib import Path
from dotenv import load_dotenv
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

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
    progress: RefinementProgress = state.refinement_progress
    current_section_idx: int = progress.current_section_index
    current_subsection_idx: int = progress.current_subsection_index    
    current_subsection: Subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]

    logger.info("Reviewing grounding and citations...")

    # extract citations from the subsection
    citations: List[CitationClaim] = await _extract_citations(cfg, current_subsection)

    # prepare verification tasks
    verification_tasks = []
    for citation_claim in citations:
        # fetch papers cited in the claim
        papers: List[PaperWithSegements] = []
        for paper_id in citation_claim.cited_papers:
            paper = next((p for p in current_subsection.papers if p.arxiv_id == paper_id), None)
            papers.append(paper)
        verification_tasks.append(_verify_single_claim(cfg, citation_claim, papers))

    # execute all verifications in parallel
    logger.info(f"Verifying {len(verification_tasks)} claims in parallel...")
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
        "review_history": feedback_history,
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
    
    # prepare prompt
    extract_citations_prompt = cfg.citation_identification_prompt.format(paper_segment=current_subsection.content)
    user_msg = HumanMessage(content=extract_citations_prompt)
    
    # get LLM and extract all citations from subsection
    logger.info("Extracting citations from the subsection...")
    llm = get_orchestrator_llm(cfg=cfg).with_config({"response_format": {"type": "json_object"}}) 
    ai_response = await llm.ainvoke([user_msg])

    # parse response into the model
    extraction_response = ai_response.content.strip()
    citation_data = clean_and_parse_json(extraction_response)
    citation_extraction: CitationExtraction = CitationExtraction.from_json(citation_data)
    logger.info(f"Extracted {citation_extraction.total_citations} citations from subsection")

    # ensure that no papers were hallucinated
    available_arxiv_ids = {paper.arxiv_id for paper in current_subsection.papers}
    for claim in citation_extraction.citation_claims:
        cited_paper_ids = claim.cited_papers
        for cited_id in cited_paper_ids:
            if cited_id not in available_arxiv_ids:
                logger.warning(f"Hallucinated citation detected: ArXiv ID '{cited_id}' not found in subsection papers. Claim: {claim.citation}")
                
    return citation_extraction.citation_claims


async def _verify_single_claim(
    cfg: Configuration,
    claim: CitationClaim,
    papers: List[PaperWithSegements],
) -> GroundingCheckResult:
    """
    Verify a single claim against the paper content using LLM.
    """
    # format supporting texts
    supporting_content = "\n\n".join(
        f"# Supporting content {i}\n"
        f"- **Title**: {p.title}\n"
        f"- **arXiv ID**: {p.arxiv_id}\n"
        f"- **Content**:\n{(p.content.page_content if p.content else '').strip()}"
        for i, p in enumerate(papers, 1)
    )

    # prepare promopt
    verify_prompt = cfg.verify_claim_prompt.format(
        citation=claim.citation,
        claim=claim.supported_claim,
        supporting_content=supporting_content
    )
    
    # invoke llm to verify the claim
    llm = get_orchestrator_llm(cfg=cfg).with_config({"response_format": {"type": "json_object"}})
    response = await llm.ainvoke([HumanMessage(content=verify_prompt)])
    result_data = clean_and_parse_json(response.content.strip())
    
    return GroundingCheckResult(
        paper_ids=[p.arxiv_id for p in papers],
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
        logger.info(f"Found {len(issues)} grounding issues")
    else:
        logger.info("No grounding issues found")
    return len(issues) == 0