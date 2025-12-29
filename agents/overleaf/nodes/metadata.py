"""Metadata generation nodes for Overleaf export."""

import logging
from typing import Dict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from agents.shared.state.main_state import AgentState
from agents.shared.state.overleaf_components import OverleafProgress
from agents.shared.utils.llm_utils import get_text_llm
from agents.overleaf.agent_config import OverleafAgentConfiguration as Configuration

logger = logging.getLogger(__name__)


async def collect_papers(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Collect all unique papers from the literature survey."""
    logger.info("Collecting papers from literature survey...")

    papers_map = {}
    for section in state.literature_survey:
        for subsection in section.subsections:
            if subsection:
                for paper in subsection.papers:
                    if paper.arxiv_id not in papers_map:
                        papers_map[paper.arxiv_id] = paper

    logger.info(f"Collected {len(papers_map)} unique papers")

    progress = state.overleaf_progress or OverleafProgress()
    return {
        "overleaf_progress": progress.model_copy(update={"papers_map": papers_map})
    }


async def generate_title(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Generate a proper academic title for the literature survey."""
    logger.info("Generating survey title...")

    cfg = Configuration.from_runnable_config(config)
    llm = get_text_llm(cfg)

    section_titles = [s.section_title for s in state.literature_survey]
    section_titles_str = "\n".join(f"- {t}" for t in section_titles)

    prompt = cfg.generate_title_prompt.format(
        topic=state.topic,
        section_titles=section_titles_str
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        title = response.content.strip().strip('"').strip("'")
        logger.info(f"Generated title: {title}")
    except Exception as e:
        logger.warning(f"Title generation failed: {e}, using original topic")
        title = state.topic

    progress = state.overleaf_progress or OverleafProgress()
    return {
        "overleaf_progress": progress.model_copy(update={"generated_title": title})
    }


async def generate_section_intros(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Generate introductions for each section."""
    logger.info("Generating section introductions...")

    cfg = Configuration.from_runnable_config(config)
    llm = get_text_llm(cfg)

    section_intros = {}

    for section in state.literature_survey:
        # Build context
        subsection_summaries = []
        for sub in section.subsections:
            if sub and sub.content:
                summary = sub.content[:200] + "..." if len(sub.content) > 200 else sub.content
                subsection_summaries.append(f"- {sub.key_point_text}: {summary}")

        prompt = cfg.generate_section_intro_prompt.format(
            section_title=section.section_title,
            section_outline=section.section_outline,
            subsection_summaries="\n".join(subsection_summaries[:5])
        )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            section_intros[section.section_index] = response.content.strip()
            logger.info(f"Generated intro for section {section.section_index}")
        except Exception as e:
            logger.warning(f"Section intro generation failed for section {section.section_index}: {e}")

    progress = state.overleaf_progress
    return {
        "overleaf_progress": progress.model_copy(update={"section_intros": section_intros})
    }


async def generate_subsection_titles(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Generate improved academic titles for subsections."""
    logger.info("Generating subsection titles...")

    cfg = Configuration.from_runnable_config(config)
    llm = get_text_llm(cfg)

    subsection_titles = {}

    for section in state.literature_survey:
        for sub in section.subsections:
            if sub and sub.key_point_text:
                prompt = cfg.generate_subsection_title_prompt.format(
                    key_point_text=sub.key_point_text
                )

                try:
                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    title = response.content.strip().strip('"').strip("'")
                    key = f"{section.section_index},{sub.subsection_index}"
                    subsection_titles[key] = title
                except Exception as e:
                    logger.warning(f"Subsection title generation failed: {e}")

    logger.info(f"Generated {len(subsection_titles)} subsection titles")

    progress = state.overleaf_progress
    return {
        "overleaf_progress": progress.model_copy(update={"subsection_titles": subsection_titles})
    }
