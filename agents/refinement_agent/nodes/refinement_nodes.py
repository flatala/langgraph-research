from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.planning_components import KeyPoint, Plan, SectionPlan
from agents.shared.state.refinement_components import RefinementProgress
from agents.shared.state.refinement_components import (
    RefinementProgress, Section, Subsection, PaperWithSegements,
    SectionStatus, SubsectionStatus, ReviewType, ReviewFeedback
)

from typing import Dict, Optional, List
from pathlib import Path
from collections import defaultdict
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
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
        
            
            print(f"Creating FAISS index for {len(chunks)} chunks...\n")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            query = key_point.text
            print(f"🔍 Query: '{query}'")
        
            relevant_docs = vectorstore.similarity_search_with_score(query, k=15)
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
    
    # Extend subsections list if needed
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
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print(f"✍️  Writing subsection {current_subsection_idx+1} of section {current_section_idx+1}")
    
    # Get current subsection
    current_section = state.literature_survey[current_section_idx]
    current_subsection = current_section.subsections[current_subsection_idx]
    
    # TODO: Implement actual writing logic with LLM
    placeholder_content = f"""
### {current_subsection.key_point_text}

[TODO: Generate actual content using LLM with:]
- Key point: {current_subsection.key_point_text}
- Papers: {len(current_subsection.papers)} papers with segments
- Previous context for flow

This subsection discusses {current_subsection.key_point_text} based on the retrieved paper segments.
""".strip()
    
    # Update subsection with content
    updated_subsection = current_subsection.model_copy(update={
        "content": placeholder_content
    })
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    print("✅ Content written, ready for content review")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_CONTENT_REVIEW
        })
    }


async def review_content(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Perform content quality review.
    Status: READY_FOR_CONTENT_REVIEW → READY_FOR_GROUNDING_REVIEW (if pass) or READY_FOR_FEEDBACK (if fail)
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print("🔍 Reviewing content quality...")
    
    # Get current subsection
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    
    # TODO: Implement actual content review with LLM
    # For now, simple logic for testing
    review_passed = (current_subsection.revision_count % 2 == 0)
    
    feedback = ReviewFeedback(
        review_type=ReviewType.CONTENT,
        passed=review_passed,
        feedback=f"TODO: Implement actual content review. Current: {'PASS' if review_passed else 'FAIL'}",
        suggestions=["Improve clarity", "Add more examples"] if not review_passed else None
    )
    
    # Add feedback to subsection
    updated_subsection = current_subsection.model_copy()
    updated_subsection.feedback_history.append(feedback)
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    # Determine next status
    if review_passed:
        next_status = SubsectionStatus.READY_FOR_GROUNDING_REVIEW
        print("✅ Content review passed, ready for grounding review")
    else:
        next_status = SubsectionStatus.READY_FOR_FEEDBACK
        print("❌ Content review failed, ready for feedback processing")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": next_status
        })
    }


async def review_grounding(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Perform grounding/citation review.
    Status: READY_FOR_GROUNDING_REVIEW → READY_FOR_FEEDBACK
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print("🔍 Reviewing grounding and citations...")
    
    # Get current subsection
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    
    # TODO: Implement actual grounding review with LLM
    # For now, simple logic for testing
    review_passed = (current_subsection.revision_count % 3 != 1)  # Different pattern than content
    
    feedback = ReviewFeedback(
        review_type=ReviewType.GROUNDING,
        passed=review_passed,
        feedback=f"TODO: Implement actual grounding review. Current: {'PASS' if review_passed else 'FAIL'}",
        suggestions=["Fix citations", "Verify claims"] if not review_passed else None
    )
    
    # Add feedback to subsection
    updated_subsection = current_subsection.model_copy()
    updated_subsection.feedback_history.append(feedback)
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    if review_passed:
        print("✅ Grounding review passed, ready for feedback processing")
    else:
        print("❌ Grounding review failed, ready for feedback processing")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_FEEDBACK
        })
    }


def process_feedback(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Process all feedback and determine if subsection is approved or needs revision.
    Status: READY_FOR_FEEDBACK → COMPLETED (if approved) or READY_FOR_REVISION (if not)
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    print("📊 Processing feedback...")
    
    # Get current subsection feedback
    current_subsection = state.literature_survey[current_section_idx].subsections[current_subsection_idx]
    feedback_history = current_subsection.feedback_history
    
    # Check if both content and grounding passed
    content_passed = any(f.review_type == ReviewType.CONTENT and f.passed for f in feedback_history)
    grounding_passed = any(f.review_type == ReviewType.GROUNDING and f.passed for f in feedback_history)
    
    if content_passed and grounding_passed:
        print("✅ All reviews passed - subsection approved!")
        next_status = SubsectionStatus.COMPLETED
    else:
        print("❌ Reviews failed - subsection needs revision")
        next_status = SubsectionStatus.READY_FOR_REVISION
    
    return {
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": next_status
        })
    }


def start_revision(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Start revision process: increment revision count and go back to writing.
    Status: READY_FOR_REVISION → READY_FOR_WRITING
    """
    progress = state.refinement_progress
    current_section_idx = progress.current_section_index
    current_subsection_idx = progress.current_subsection_index
    
    # Update revision count
    current_section = state.literature_survey[current_section_idx]
    current_subsection = current_section.subsections[current_subsection_idx]
    
    updated_subsection = current_subsection.model_copy(update={
        "revision_count": current_subsection.revision_count + 1
    })
    
    # Update literature survey
    literature_survey = list(state.literature_survey)
    updated_section = literature_survey[current_section_idx].model_copy()
    updated_section.subsections[current_subsection_idx] = updated_subsection
    literature_survey[current_section_idx] = updated_section
    
    print(f"🔄 Starting revision #{updated_subsection.revision_count} for subsection {current_subsection_idx+1}")
    
    return {
        "literature_survey": literature_survey,
        "refinement_progress": progress.model_copy(update={
            "current_subsection_status": SubsectionStatus.READY_FOR_WRITING
        })
    }





