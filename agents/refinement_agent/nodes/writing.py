from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode
from typing import Dict, Optional, List
from pathlib import Path
from dotenv import load_dotenv

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.refinement_agent.tools import create_search_paper_fragments_tool
from agents.shared.state.main_state import AgentState
from agents.shared.state.planning_components import PaperRef, Plan, SectionPlan, KeyPoint
from agents.shared.utils.llm_utils import get_text_llm, get_embedding_model
from agents.shared.state.refinement_components import (
    RefinementProgress, Section, Subsection, PaperWithSegements,
    SectionStatus, SubsectionStatus
)

import asyncio
import re
import logging

logger = logging.getLogger(__name__)
load_dotenv(
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,
)    

async def prepare_subsection_context(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Prepare RAG context for current subsection by retrieving relevant paper segments.
    
    Status update: READY_FOR_CONTEXT_PREP → READY_FOR_WRITING
    """
    progress: RefinementProgress = state.refinement_progress
    plan: Plan = state.plan
    current_section_idx: int = progress.current_section_index
    current_subsection_idx: int = progress.current_subsection_index
    logger.info(f"Preparing context for Section {current_section_idx+1}, Subsection {current_subsection_idx+1}")

    section_plan: SectionPlan = plan.plan[current_section_idx]
    key_point: KeyPoint = section_plan.key_points[current_subsection_idx]

    # process papers in parallel
    tasks = [
        _build_rag_and_retrieve_segments_async(
            paper_ref, key_point,
            review_id=state.review_id,
            section_index=current_section_idx,
            subsection_index=current_subsection_idx,
            config=config
        )
        for paper_ref in key_point.papers
    ]
    papers_with_segments = await asyncio.gather(*tasks)
    
    # create subsection with all context
    subsection = Subsection(
        subsection_index=current_subsection_idx,
        subsection_title=key_point.text,  # Use key point text as subsection title, TODO: fix
        papers=papers_with_segments,
        key_point_text=key_point.text,
        content="",
        revision_count=0,
        review_history=[],
    )
    
    # ensure section exists and add subsection
    literature_survey = list(state.literature_survey)
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
    
    current_section = literature_survey[current_section_idx]
    updated_section = current_section.model_copy()

    # ensure that the subsection list has enough indices
    while len(updated_section.subsections) <= current_subsection_idx:
        updated_section.subsections.append(None)
    
    updated_section.subsections[current_subsection_idx] = subsection
    literature_survey[current_section_idx] = updated_section
    
    logger.info(f"Context prepared: {len(papers_with_segments)} paper segments")
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_section_status": SectionStatus.IN_PROGRESS,
            "current_subsection_status": SubsectionStatus.READY_FOR_WRITING
        })
    }


async def _build_rag_and_retrieve_segments_async(
    paper_ref: PaperRef,
    key_point: KeyPoint,
    review_id: str,
    section_index: int,
    subsection_index: int,
    *,
    config: Optional[RunnableConfig] = None
) -> PaperWithSegements:
    """Async wrapper for parallel paper processing using thread pool."""
    return await asyncio.to_thread(
        _build_rag_and_retrieve_segments,
        paper_ref, key_point, review_id, section_index, subsection_index,
        config=config
    )


def _build_rag_and_retrieve_segments(
    paper_ref: PaperRef,
    key_point: KeyPoint,
    review_id: str,
    section_index: int,
    subsection_index: int,
    *,
    config: Optional[RunnableConfig] = None
) -> PaperWithSegements:
    """Download paper, chunk content, and retrieve top 5 relevant segments using vector similarity.

    Uses persistent ChromaDB for caching embeddings and SQLite DB for tracking papers.
    Also uses temporary paper cache to avoid re-downloading during a single review.
    """
    from data.database.crud import ReviewDB
    from data.vector_store.manager import VectorStoreManager
    from data.temp_cache.paper_cache import PaperCache

    cfg = Configuration.from_runnable_config(config)
    db = ReviewDB()
    vector_manager = VectorStoreManager()
    paper_cache = PaperCache(review_id)

    logger.info(f"Processing paper: {paper_ref.title}")
    arxiv_id = _extract_arxiv_id(paper_ref.url)

    # check temporary paper cache first (avoids re-downloading during this review)
    doc = paper_cache.get(arxiv_id)
    if doc:
        logger.info(f"Using cached paper document")
    else:
        # download paper
        logger.info(f"Downloading paper document")
        loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_full_text=True)
        docs = loader.load()
        doc = docs[0]
        paper_cache.save(arxiv_id, doc)

    # extract metadata from document
    authors = doc.metadata.get('Authors', '').split(', ') if doc.metadata.get('Authors') else ["Unknown"]
    year_str = doc.metadata.get('Published', '').split('-')[0] if doc.metadata.get('Published') else None
    year = int(year_str) if year_str else paper_ref.year
    summary = doc.metadata.get('Summary', '')

    # check if we already have embeddings for this paper
    vector_collection = db.get_vector_collection(review_id, arxiv_id)
    embeddings = get_embedding_model(cfg)

    if vector_collection:
        # load existing embeddings
        logger.info(f"Using cached embeddings ({vector_collection.total_chunks} chunks)")
        vectorstore = vector_manager.load_collection(review_id, arxiv_id, embeddings)
    else:
        # create new embeddings
        logger.info(f"Creating embeddings")

        # save paper to database
        db.get_or_create_paper(
            arxiv_id=arxiv_id,
            title=paper_ref.title,
            authors=authors,
            url=paper_ref.url,
            year=year,
            summary=summary
        )

        # chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents([doc])

        # create and persist vector store
        vectorstore = vector_manager.create_collection(
            review_id=review_id,
            paper_id=arxiv_id,
            documents=chunks,
            embedding_function=embeddings
        )

        # register in database
        db.register_vector_collection(
            review_id=review_id,
            paper_id=arxiv_id,
            collection_name=f"{review_id}_{arxiv_id}",
            embedding_model=cfg.embedding_model,
            chunk_size=800,
            chunk_overlap=200,
            total_chunks=len(chunks)
        )

    # query vector store for relevant segments
    # NOTE: maybe worth adding rag access as a tool for the writing agent?
    query = key_point.text
    relevant_docs = vectorstore.similarity_search_with_score(query, k=5)
    relevant_docs.sort(key=lambda x: x[1])

    relevant_segments = []
    for i, (doc_chunk, score) in enumerate(relevant_docs, 1):
        logger.debug(f"{i}. [Score: {score:.3f}] \n {doc_chunk.page_content[:100]}...")
        relevant_segments.append(f"[Score: {score:.3f}] {doc_chunk.page_content}")

    # link paper to this review/section/subsection
    db.link_paper_to_review(
        review_id=review_id,
        paper_id=arxiv_id,
        section_index=section_index,
        subsection_index=subsection_index,
        citation=f"({paper_ref.title}, {year})",
        relevance_score=relevant_docs[0][1] if relevant_docs else 0.0
    )

    return PaperWithSegements(
        title=paper_ref.title,
        authors=authors,
        arxiv_id=arxiv_id,
        arxiv_url=paper_ref.url,
        citation=f"({paper_ref.title}, {year})",
        content=doc,
        relevant_segments=relevant_segments
    )


def _extract_arxiv_id(url: str) -> str:
    """Extract ArXiv paper ID from URL, removing version suffixes."""
    if url and "arxiv.org" in url:
        url = re.sub(r'v\d+$', '', url)
    arxiv_id = url.split("/")[-1] if url else ""
    return arxiv_id


async def write_subsection(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Generate subsection content using LLM with paper segments, tool access, and completed sections as context.

    Status update: READY_FOR_WRITING → READY_FOR_CONTENT_REVIEW
    """
    cfg = Configuration.from_runnable_config(config)
    progress = state.refinement_progress
    plan = state.plan
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index

    logger.info(f"Writing subsection {current_subsection_idx+1} of section {current_section_idx+1}")

    current_section: Section = state.literature_survey[current_section_idx]
    current_subsection: Subsection = current_section.subsections[current_subsection_idx]
    section_plan: SectionPlan = plan.plan[current_section_idx]

    # prepare context
    papers_context = _prepare_paper_segments_string(current_subsection.papers)
    sections_context = _prepare_completed_content_string(state.literature_survey)
    available_papers = _format_available_papers(current_subsection.papers)

    # create search tool for the available papers
    available_paper_ids = [p.arxiv_id for p in current_subsection.papers]
    embeddings = get_embedding_model(cfg)
    search_tool = create_search_paper_fragments_tool(
        review_id=state.review_id,
        available_paper_ids=available_paper_ids,
        embeddings=embeddings
    )

    # compile the writing prompt
    writing_prompt = cfg.write_subsection_prompt.format(
        preceeding_sections=sections_context,
        key_point_text=current_subsection.key_point_text,
        section_title=section_plan.title,
        section_outline=section_plan.outline,
        subsection_index=current_subsection_idx + 1,
        total_subsections=len(section_plan.key_points),
        paper_segments=papers_context.strip(),
        available_papers=available_papers
    )

    # set up LLM with tool binding and ToolNode for parallel execution
    logger.info("Generating subsection content with LLM (with tool access)...")
    system_msg = SystemMessage(content=cfg.system_prompt)
    user_msg = HumanMessage(content=writing_prompt)
    messages = [system_msg, user_msg]

    llm = get_text_llm(cfg=cfg)
    llm_with_tools = llm.bind_tools([search_tool])
    tool_node = ToolNode([search_tool])

    # allow LLM to search for more evidence if needed
    max_tool_iterations = 5
    iteration = 0
    ai_response = None
    while iteration < max_tool_iterations:
        iteration += 1
        logger.debug(f"Writing iteration {iteration}")

        ai_response = await llm_with_tools.ainvoke(messages)
        messages.append(ai_response)

        # check for tool calls
        if not ai_response.tool_calls:
            break

        # execute tool calls in parallel using ToolNode
        logger.info(f"Executing {len(ai_response.tool_calls)} tool calls for evidence gathering")
        tool_result = await tool_node.ainvoke({"messages": messages})
        messages.extend(tool_result["messages"])

    generated_content = ai_response.content.strip()

    # store the message thread for continuous refinement
    refinement_messages = messages

    # update subsection with generated content and message thread
    updated_subsection = current_subsection.model_copy(update={
        "content": generated_content,
        "refinement_messages": refinement_messages
    })
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()

    # ensure that the subsection list has enough indices
    while len(updated_section.subsections) <= current_subsection_idx:
        updated_section.subsections.append(None)

    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section

    logger.info("Content written, ready for content review")
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_CONTENT_REVIEW
        })
    }


def _format_available_papers(papers: List[PaperWithSegements]) -> str:
    """Format available papers as a list of citable arxiv IDs."""
    if not papers:
        return "No papers available for this subsection."

    lines = []
    for paper in papers:
        lines.append(f"- **{paper.arxiv_id}**: {paper.title}")

    return "\n".join(lines)


def _prepare_paper_segments_string(papers: List[PaperWithSegements]) -> str:
    """Format paper segments into structured text for LLM prompt context."""
    paper_segments_text = ""
    for i, paper in enumerate(papers, 1):
        author_names = []
        for author in paper.authors:
            if author and author != "Unknown":
                if ',' in author:
                    last_name = author.split(',')[0].strip()
                else:
                    last_name = author.split()[-1] if author.split() else author
                author_names.append(last_name)
        authors_str = ", ".join(author_names) if author_names else "Unknown"
        
        paper_segments_text += f"\n**Paper {i}: {paper.title}**\n"
        paper_segments_text += f"**Authors**: {authors_str}\n"
        paper_segments_text += f"**ArXiv ID**: {paper.arxiv_id}\n"
        paper_segments_text += f"**Relevant Segments**:\n"
        
        for j, segment in enumerate(paper.relevant_segments, 1):
            paper_segments_text += f"  - Fragment {j}: {segment}\n"
        
        paper_segments_text += "\n"

    return paper_segments_text


def _prepare_completed_content_string(sections: List[Section]) -> str:
    """Format already completed sections and subsections into context string for LLM."""
    section_context = ""
    for i, section in enumerate(sections, 1):
        section_context += f"\n**Section {i}: {section.section_title}**\n"
        section_context += section.section_introduction
        section_context += "\n"
        for j, subsection in enumerate(section.subsections, 1):
            section_context += f"\n**Subsection {i}.{j}: {subsection.subsection_title}**\n"
            section_context += subsection.content
            section_context += "\n"
        section_context += "\n"
    return section_context