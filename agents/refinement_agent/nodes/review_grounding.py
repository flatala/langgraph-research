from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.prebuilt import ToolNode

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.refinement_agent.tools import create_search_paper_fragments_tool
from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import (
    SubsectionStatus, Subsection, ReviewRound,
    GroundingCheckResult, CitationExtraction, CitationClaim,
    PaperWithSegements, RefinementProgress
)
from agents.shared.utils.llm_utils import get_orchestrator_llm, get_embedding_model, invoke_llm_with_json_retry
from data.vector_store.manager import VectorStoreManager

from typing import Dict, Optional, List, Tuple
from agents.shared.state.refinement_components import Section
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
    current_section: Section = state.literature_survey[current_section_idx]
    current_subsection: Subsection = current_section.subsections[current_subsection_idx]

    logger.info("Reviewing grounding and citations...")

    # create shared instances once for all verifications
    vector_manager = VectorStoreManager()
    embeddings = get_embedding_model(cfg)

    # extract citations from the subsection
    citations: List[CitationClaim] = await _extract_citations(cfg, current_subsection)

    # get review context (current section content including current subsection)
    review_context = _get_review_context(current_section, current_subsection_idx)

    # prepare verification tasks
    verification_tasks = []
    for citation_claim in citations:
        # fetch papers cited in the claim
        papers: List[PaperWithSegements] = []
        for paper_id in citation_claim.cited_papers:
            paper = next((p for p in current_subsection.papers if p.arxiv_id == paper_id), None)
            papers.append(paper)
        verification_tasks.append(_verify_single_claim(
            cfg, citation_claim, papers, state.review_id, review_context,
            vector_manager=vector_manager, embeddings=embeddings
        ))

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
    
    # Always go to process_grounding_feedback which handles both pass/fail
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_GROUNDING_REVISION
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
    citation_data = await invoke_llm_with_json_retry(llm, [user_msg], max_retries=cfg.llm_max_retries)
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


def _get_review_context(section: Section, current_subsection_idx: int) -> str:
    """Get content from current section up to and including current subsection."""
    parts = [f"## Section: {section.section_title}"]

    # Include all subsections up to and including the current one
    for i, subsection in enumerate(section.subsections[:current_subsection_idx + 1]):
        if subsection.content:
            prefix = "(CURRENT) " if i == current_subsection_idx else ""
            parts.append(f"### {prefix}{subsection.key_point_text}\n{subsection.content}")

    return "\n\n".join(parts)


async def _verify_single_claim(
    cfg: Configuration,
    claim: CitationClaim,
    papers: List[PaperWithSegements],
    review_id: str,
    review_context: str,
    vector_manager: VectorStoreManager,
    embeddings,
) -> GroundingCheckResult:
    """
    Verify a single claim against paper content using LLM with optional tool use.
    Uses RAG to retrieve relevant fragments as seed evidence, then allows
    the LLM to search for more evidence if needed.
    """

    # create search tool with available papers
    available_paper_ids = [p.arxiv_id for p in papers if p is not None]
    search_tool = create_search_paper_fragments_tool(
        review_id=review_id,
        available_paper_ids=available_paper_ids,
        embeddings=embeddings
    )

    # build initial seed evidence using RAG for each paper
    supporting_parts = []
    for i, p in enumerate(papers, 1):
        if p is None:
            continue

        # load relevant segments from vector store
        vectorstore = vector_manager.load_collection(review_id, p.arxiv_id, embeddings)
        relevant_docs = vectorstore.similarity_search(claim.supported_claim, k=5)
        fragments = "\n".join(f"  - {doc.page_content}" for doc in relevant_docs)

        supporting_parts.append(
            f"# Paper {i}: {p.title}\n"
            f"- **arXiv ID**: {p.arxiv_id}\n"
            f"- **Relevant fragments**:\n{fragments}"
        )

    supporting_content = "\n\n".join(supporting_parts)

    # format available papers for prompt
    available_papers_str = "\n".join(
        f"- **{p.arxiv_id}**: {p.title}"
        for p in papers if p is not None
    )

    # prepare prompt with review context and tool info
    verify_prompt = cfg.verify_claim_prompt.format(
        citation=claim.citation,
        claim=claim.supported_claim,
        supporting_content=supporting_content,
        review_context=review_context,
        available_papers=available_papers_str
    )

    # set up LLM with tool binding and ToolNode for parallel execution
    llm = get_orchestrator_llm(cfg=cfg)
    llm_with_tools = llm.bind_tools([search_tool])
    tool_node = ToolNode([search_tool])

    # initialize message thread
    messages = [HumanMessage(content=verify_prompt)]

    # agentic loop with iteration limit
    max_iterations = 5
    iteration = 0
    final_response = None
    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"Claim verification iteration {iteration} for: {claim.citation}")

        ai_response = await llm_with_tools.ainvoke(messages)
        messages.append(ai_response)

        # check for tool calls
        if not ai_response.tool_calls:
            final_response = ai_response.content.strip()
            break

        # execute tool calls in parallel using ToolNode
        logger.debug(f"Executing {len(ai_response.tool_calls)} tool calls for claim verification")
        tool_result = await tool_node.ainvoke({"messages": messages})
        messages.extend(tool_result["messages"])

    # parse JSON response
    result_data = await _parse_verification_response(final_response, cfg)

    return GroundingCheckResult(
        paper_ids=[p.arxiv_id for p in papers if p is not None],
        citation=result_data.get("citation", claim.citation),
        supported_claim=result_data.get("supported_claim", claim.supported_claim),
        status=result_data.get("status", "invalid"),
        error_type=result_data.get("error_type"),
        explanation=result_data.get("explanation"),
        correction_suggestion=result_data.get("correction_suggestion")
    )


async def _parse_verification_response(response: str, cfg: Configuration) -> dict:
    """Parse the verification response, handling potential JSON extraction needs."""
    import re

    try:
        # try direct JSON parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # if parsing fails, try to extract JSON from mixed content
        logger.warning("Direct JSON parsing failed, attempting extraction")

        # look for JSON block in the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # fall back to using LLM to extract JSON
        logger.warning("JSON extraction failed, using LLM retry")
        llm = get_orchestrator_llm(cfg=cfg).with_config(
            {"response_format": {"type": "json_object"}}
        )
        retry_result = await invoke_llm_with_json_retry(
            llm,
            [HumanMessage(content=f"Extract and return only the JSON verification result from this response:\n\n{response}")],
            max_retries=cfg.llm_max_retries
        )
        return retry_result


def _has_no_grounding_issues(grounding_reviews: List[GroundingCheckResult]) -> bool:
    """Check if grounding review passed by counting total invalid claims."""
    issues = [result for result in grounding_reviews if result.status == "invalid"]
    if issues:
        logger.info(f"Found {len(issues)} grounding issues")
    else:
        logger.info("No grounding issues found")
    return len(issues) == 0