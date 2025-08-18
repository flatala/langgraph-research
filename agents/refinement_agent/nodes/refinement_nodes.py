from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agents.refinement_agent.agent_config import RefinementAgentConfiguration as Configuration
from agents.shared.state.main_state import AgentState
from agents.shared.state.planning_components import KeyPoint, Plan, SectionPlan
from agents.shared.state.refinement_components import RefinementProgress
from agents.shared.state.refinement_components import (
    RefinementProgress, Section, Subsection, PaperWithSegements,
    SectionStatus, SubsectionStatus, ReviewType, ReviewFeedback,
    CitationExtraction, CitationClaim
)
from agents.shared.utils.llm_utils import get_text_llm, get_orchestrator_llm

from typing import Dict, Optional, List
from pprint import pprint
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import re
import json

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
    review_data = json.loads(review_text)
    score = review_data.get("score", 0)
    meets_minimum = review_data.get("meets_minimum", False)
    feedback_text = review_data.get("feedback", "")
        
    print(f"📊 Content review score: {score}/10, Passed: {meets_minimum}")
    if feedback_text:
        print(f"📝 Feedback: {feedback_text}")
            
    
    # create feedback object
    feedback = ReviewFeedback(
        review_type=ReviewType.CONTENT,
        passed=meets_minimum,
        feedback=feedback_text,
        suggestions=None
    )
    
    # add feedback to subsection
    updated_subsection = current_subsection.model_copy()
    updated_subsection.feedback_history.append(feedback)
    
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
        
        # store raw response
        groudedness_reviews.append({
            "arxiv_id": arxiv_id,
            "paper_title": full_paper.title,
            "claims_count": len(claims),
            "raw_response": grounding_result
        })

        pprint(grounding_result)
        
        print(f"✅ Grounding review completed for {full_paper.title}")

    print(f"📊 Completed grounding reviews for {len(groudedness_reviews)} papers")


    # Perform grounding review based on extracted citations
    feedback = None
    
    # Add feedback and citations to subsection
    updated_subsection = current_subsection.model_copy()
    updated_subsection.feedback_history.append(feedback)
    updated_subsection.citations = citation_extraction.citation_claims
    
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





