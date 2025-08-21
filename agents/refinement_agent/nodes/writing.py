from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.planning_components import PaperRef, Plan, SectionPlan, KeyPoint
from agents.shared.utils.llm_utils import get_text_llm
from agents.shared.state.refinement_components import (
    RefinementProgress, Section, Subsection, PaperWithSegements,
    SectionStatus, SubsectionStatus
)

from typing import Dict, Optional, List
from pathlib import Path
from dotenv import load_dotenv
import re

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
    print(f"\nPreparing context for Section {current_section_idx+1}, Subsection {current_subsection_idx+1}\n")

    section_plan: SectionPlan = plan.plan[current_section_idx]
    key_point: KeyPoint = section_plan.key_points[current_subsection_idx]

    papers_with_segments = []
    for paper_ref in key_point.papers:
        paper_with_segments = _build_rag_and_retrieve_segments(paper_ref, key_point)
        papers_with_segments.append(paper_with_segments)
    
    # Create subsection with all context
    subsection = Subsection(
        subsection_index=current_subsection_idx,
        subsection_title=key_point.text,  # Use key point text as subsection title, TODO: fix
        papers=papers_with_segments,
        key_point_text=key_point.text,
        content="",
        revision_count=0,
        review_history=[],
    )
    
    # Ensure section exists and add subsection
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
    
    print(f"Context prepared: {len(papers_with_segments)} paper segments")
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_section_status": SectionStatus.IN_PROGRESS,
            "current_subsection_status": SubsectionStatus.READY_FOR_WRITING
        })
    }


def _build_rag_and_retrieve_segments(paper_ref: PaperRef, key_point: KeyPoint) -> PaperWithSegements:
    """Download paper, chunk content, and retrieve top 5 relevant segments using vector similarity."""

    print(f"Processing paper: {paper_ref.title}\n")

    arxiv_id = _extract_arxiv_id(paper_ref.url)
    loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_full_text=True)
    docs = loader.load()
    doc = docs[0]
    authors = doc.metadata.get('Authors', '').split(', ') if doc.metadata.get('Authors') else ["Unknown"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    query = key_point.text
    relevant_docs = vectorstore.similarity_search_with_score(query, k=5)
    relevant_docs.sort(key=lambda x: x[1])
    
    relevant_segments = []
    for i, (doc_chunk, score) in enumerate(relevant_docs, 1):
        print(f"  {i}. [Score: {score:.3f}] \n {doc_chunk.page_content} \n")
        relevant_segments.append(f"[Score: {score:.3f}] {doc_chunk.page_content}")

    return PaperWithSegements(
        title=paper_ref.title,
        authors=authors,
        arxiv_id=arxiv_id,
        arxiv_url=paper_ref.url,
        citation=f"({paper_ref.title}, {paper_ref.year})",
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
    """Generate subsection content using LLM with paper segments and completed sections as context.
    
    Status update: READY_FOR_WRITING → READY_FOR_CONTENT_REVIEW
    """

    cfg = Configuration.from_runnable_config(config)
    progress = state.refinement_progress
    plan = state.plan
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print(f"✍️  Writing subsection {current_subsection_idx+1} of section {current_section_idx+1}")
    
    current_section: Section = state.literature_survey[current_section_idx]
    current_subsection: Subsection = current_section.subsections[current_subsection_idx]
    section_plan: SectionPlan = plan.plan[current_section_idx]
    
    # format paper segments for the subsection writing prompt
    papers_context = _prepare_paper_segments_string(current_subsection.papers)

    # prepare and format the already completed content to use as context
    sections_context = _prepare_completed_content_string(state.literature_survey)
    
    # compile the writing prompt
    writing_prompt = cfg.write_subsection_prompt.format(
        preceeding_sections=sections_context,
        key_point_text=current_subsection.key_point_text,
        section_title=section_plan.title,
        section_outline=section_plan.outline,
        subsection_index=current_subsection_idx + 1,
        total_subsections=len(section_plan.key_points),
        paper_segments=papers_context.strip()
    )
    
    # get LLM and generate subsection content
    print("Generating subsection content with LLM...")
    system_msg = SystemMessage(content=cfg.system_prompt)
    user_msg = HumanMessage(content=writing_prompt)
    messages = [system_msg, user_msg]
    llm = get_text_llm(cfg=cfg)
    ai_response = await llm.ainvoke(messages)
    generated_content = ai_response.content.strip()
    
    # update subsection with generated content
    updated_subsection = current_subsection.model_copy(update={
        "content": generated_content
    })
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()

    # ensure that the subsection list has enough indices
    while len(updated_section.subsections) <= current_subsection_idx:
        updated_section.subsections.append(None)

    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    print("Content written, ready for content review")
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_CONTENT_REVIEW
        })
    }

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