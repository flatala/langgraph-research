from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.planning_components import Plan, SectionPlan, KeyPoint
from agents.shared.utils.llm_utils import get_text_llm
from agents.shared.state.refinement_components import (
    RefinementProgress, Section, Subsection, PaperWithSegements,
    SectionStatus, SubsectionStatus
)

from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import re

load_dotenv(                
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,         
)    

async def prepare_subsection_context(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Comprehensive preparation: RAG + context + subsection setup.
    Status: READY_FOR_CONTEXT_PREP → READY_FOR_WRITING
    """
    progress: RefinementProgress = state.refinement_progress
    plan: Plan = state.plan
    current_section_idx: int = progress.current_section_index
    current_subsection_idx: int = progress.current_subsection_index
    section_plan: SectionPlan = plan.plan[current_section_idx]
    key_point: KeyPoint = section_plan.key_points[current_subsection_idx]
    print(f"\n📝 Preparing context for Section {current_section_idx+1}, Subsection {current_subsection_idx+1}\n")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    papers_with_segments = []
    
    for i, paper_ref in enumerate(key_point.papers):
        print(f"Processing paper {i+1}/{len(key_point.papers)}: {paper_ref.title}\n")
        
        url = paper_ref.url
        if url and "arxiv.org" in url:
            url = re.sub(r'v\d+$', '', url)
        
        arxiv_id = url.split("/")[-1] if url else "unknown"
        
        try:
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
        
            
            print(f"Creating FAISS index for {len(chunks)} chunks...\n")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            query = key_point.text
            print(f"🔍 Query: '{query}'")
        
            relevant_docs = vectorstore.similarity_search_with_score(query, k=5)
            relevant_docs.sort(key=lambda x: x[1])
            
            relevant_segments = []
            print(f"📄 Retrieved segments from '{paper_ref.title}': \n")
            for i, (doc_chunk, score) in enumerate(relevant_docs, 1):
                print(f"  {i}. [Score: {score:.3f}] \n {doc_chunk.page_content} \n")
                relevant_segments.append(f"[Score: {score:.3f}] {doc_chunk.page_content}")
            
            if not relevant_segments:
                relevant_segments = ["No relevant segments found"]
                print("  ❌ No relevant segments found")
            
            paper_with_segments = PaperWithSegements(
                title=paper_ref.title,
                authors=authors,
                arxiv_id=arxiv_id,
                arxiv_url=paper_ref.url,
                citation=f"({paper_ref.title}, {paper_ref.year})",
                content=doc,
                relevant_segments=relevant_segments
            )
            
            print(f"✅ Found {len(relevant_segments)} relevant segments for {paper_ref.title}")
            
        except Exception as e:
            print(f"Error processing paper {arxiv_id}: {e}")
        
        papers_with_segments.append(paper_with_segments)
    
    # Create subsection with all context
    subsection = Subsection(
        subsection_index=current_subsection_idx,
        subsection_title=key_point.text,  # Use key point text as subsection title
        papers=papers_with_segments,
        key_point_text=key_point.text,
        content="",
        revision_count=0,
        feedback_history=[],
        citations=[]
    )
    
    # Ensure section exists and add subsection
    literature_survey = list(state.literature_survey)
    
    # Create section if it doesn't exist
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
    
    # Add/update subsection
    current_section = literature_survey[current_section_idx]
    updated_section = current_section.model_copy()
    
    # Ensure subsections list is long enough
    while len(updated_section.subsections) <= current_subsection_idx:
        updated_section.subsections.append(None)
    
    updated_section.subsections[current_subsection_idx] = subsection
    literature_survey[current_section_idx] = updated_section
    
    print(f"✅ Context prepared: {len(papers_with_segments)} papers with segments")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_section_status": SectionStatus.IN_PROGRESS,
            "current_subsection_status": SubsectionStatus.READY_FOR_WRITING
        })
    }

async def write_subsection(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Write content for current subsection.
    Status: READY_FOR_WRITING → READY_FOR_CONTENT_REVIEW
    """
    cfg = Configuration.from_runnable_config(config)
    progress = state.refinement_progress
    plan = state.plan
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print(f"✍️  Writing subsection {current_subsection_idx+1} of section {current_section_idx+1}")
    
    current_section = state.literature_survey[current_section_idx]
    current_subsection = current_section.subsections[current_subsection_idx]
    section_plan = plan.plan[current_section_idx]
    
    # Format paper segments for the subsection writing prompt
    paper_segments_text = ""
    for i, paper in enumerate(current_subsection.papers, 1):
        # Extract author last names for citation format
        author_names = []
        for author in paper.authors:
            if author and author != "Unknown":
                # Take last name (assuming format like "First Last" or "Last, First")
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

    # prepare and format the already completed content to use as context
    section_context = ""
    for i, section in enumerate(state.literature_survey, 1):
        section_context += f"\n**Section {i}: {section.section_title}**\n"
        section_context += section.section_introduction
        section_context += "\n"
        for j, subsection in enumerate(section.subsections, 1):
            section_context += f"\n**Subsection {i}.{j}: {subsection.subsection_title}**\n"
            section_context += subsection.content
            section_context += "\n"
        section_context += "\n"
    
    # compile the writing prompt
    writing_prompt = cfg.write_subsection_prompt.format(
        preceeding_sections=section_context,
        key_point_text=current_subsection.key_point_text,
        section_title=section_plan.title,
        section_outline=section_plan.outline,
        subsection_index=current_subsection_idx + 1,
        total_subsections=len(section_plan.key_points),
        paper_segments=paper_segments_text.strip()
    )
    
    # create messages for LLM
    system_msg = SystemMessage(content=cfg.system_prompt)
    user_msg = HumanMessage(content=writing_prompt)
    messages = [system_msg, user_msg]
    
    # get LLM and generate subsection content
    llm = get_text_llm(cfg=cfg)
    print("🤖 Generating subsection content with LLM...")
    ai_response = await llm.ainvoke(messages)
    generated_content = ai_response.content.strip()
    print(f"✅ Generated {len(generated_content)} characters of content")
    
    # update subsection with generated content
    updated_subsection = current_subsection.model_copy(update={
        "content": generated_content
    })
    
    # update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    
    # extend list if needed
    while len(updated_section.subsections) <= current_subsection_idx:
        updated_section.subsections.append(None)

    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    print("✅ Content written, ready for content review")
    
    # Print current state for debugging
    print("\n" + "="*80)
    print("🔍 CURRENT LITERATURE SURVEY STATE")
    print("="*80)
    current_section = literature_survey[current_section_idx]
    current_section.print_section(include_segments=True)
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_CONTENT_REVIEW
        })
    }