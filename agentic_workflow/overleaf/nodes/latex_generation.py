"""LaTeX content generation nodes for Overleaf export."""

import re
import logging
from typing import Dict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_workflow.shared.state.main_state import AgentState
from agentic_workflow.shared.state.refinement_components import Section, Subsection
from agentic_workflow.shared.utils.llm_utils import get_text_llm
from agentic_workflow.overleaf.agent_config import OverleafAgentConfiguration as Configuration

logger = logging.getLogger(__name__)


async def generate_latex_content(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Generate the main LaTeX document content."""
    logger.info("Generating LaTeX content...")

    cfg = Configuration.from_runnable_config(config)
    llm = get_text_llm(cfg)
    progress = state.overleaf_progress

    # Document preamble
    latex = r"""\documentclass[12pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black]{hyperref}
\usepackage[backend=biber,style=numeric,sorting=none]{biblatex}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

% Bibliography
\addbibresource{references.bib}

% Page setup
\geometry{margin=1in}
\onehalfspacing

% Document info
"""
    # Title
    title = progress.generated_title or state.topic
    escaped_title = _escape_latex(title)
    latex += f"\\title{{{escaped_title}}}\n"
    latex += "\\author{Generated Literature Survey}\n"
    latex += "\\date{\\today}\n\n"

    latex += r"""\begin{document}

\maketitle
\tableofcontents
\newpage

"""

    # Sections
    for section in state.literature_survey:
        section_content = await _format_section(section, progress, llm, cfg)
        latex += section_content

    latex += r"""
\newpage
\printbibliography

\end{document}
"""

    return {
        "overleaf_progress": progress.model_copy(update={"latex_content": latex})
    }


async def generate_bibtex_content(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Generate the BibTeX references file."""
    logger.info("Generating BibTeX content...")

    progress = state.overleaf_progress
    entries = []

    for arxiv_id, paper in progress.papers_map.items():
        # Create key from arxiv_id
        base_id = re.sub(r'v\d+$', '', paper.arxiv_id)
        cite_key = f"arxiv_{base_id.replace('.', '_')}"

        # Format authors
        authors = " and ".join(paper.authors) if paper.authors else "Unknown"

        # Extract year
        try:
            year_prefix = paper.arxiv_id[:2]
            year = f"20{year_prefix}" if int(year_prefix) < 50 else f"19{year_prefix}"
        except (ValueError, IndexError):
            year = "2023"

        # Escape title
        title = _escape_bibtex(paper.title)

        entry = f"""@article{{{cite_key},
    title = {{{title}}},
    author = {{{authors}}},
    year = {{{year}}},
    eprint = {{{paper.arxiv_id}}},
    archivePrefix = {{arXiv}},
    primaryClass = {{cs}},
    url = {{{paper.arxiv_url}}}
}}"""
        entries.append(entry)

    bibtex = "\n\n".join(entries)

    return {
        "overleaf_progress": progress.model_copy(update={"bibtex_content": bibtex})
    }


async def _format_section(section: Section, progress, llm, cfg) -> str:
    """Format a single section with subsections."""
    content = ""

    # Section title (keep original from plan)
    escaped_title = _escape_latex(section.section_title)
    content += f"\\section{{{escaped_title}}}\n\n"

    # Section introduction
    intro = progress.section_intros.get(section.section_index, section.section_introduction)
    if intro:
        formatted_intro = await _format_math_for_latex(intro, llm, cfg)
        escaped_intro = _escape_latex_preserve_math(formatted_intro)
        content += f"{escaped_intro}\n\n"

    # Subsections
    for subsection in section.subsections:
        if subsection:
            subsection_content = await _format_subsection(subsection, section.section_index, progress, llm, cfg)
            content += subsection_content

    return content


async def _format_subsection(subsection: Subsection, section_index: int, progress, llm, cfg) -> str:
    """Format a single subsection."""
    content = ""

    # Use generated title if available
    key = f"{section_index},{subsection.subsection_index}"
    title = progress.subsection_titles.get(key, subsection.subsection_title or subsection.key_point_text)
    escaped_title = _escape_latex(title)
    content += f"\\subsection{{{escaped_title}}}\n\n"

    if subsection.content:
        # Convert citations
        converted_content = _convert_citations(subsection.content)
        # Format math
        formatted_content = await _format_math_for_latex(converted_content, llm, cfg)
        # Escape special chars
        escaped_content = _escape_latex_preserve_commands(formatted_content)
        content += f"{escaped_content}\n\n"

    return content


def _convert_citations(text: str) -> str:
    """Convert [Author_YEAR(ArxivID)] citations to \\cite{key} format."""
    def replace_citation(match):
        citation_block = match.group(1)
        parts = citation_block.split(';')
        cite_keys = []

        for part in parts:
            arxiv_match = re.search(r'\(([0-9]+\.[0-9]+(?:v\d+)?)\)', part.strip())
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                base_id = re.sub(r'v\d+$', '', arxiv_id)
                cite_key = f"arxiv_{base_id.replace('.', '_')}"
                cite_keys.append(cite_key)

        if cite_keys:
            return f"\\cite{{{','.join(cite_keys)}}}"
        return match.group(0)

    pattern = r'\[([^\]]+\([0-9]+\.[0-9]+(?:v\d+)?\)[^\]]*)\]'
    return re.sub(pattern, replace_citation, text)


async def _format_math_for_latex(content: str, llm, cfg) -> str:
    """Use LLM to convert mathematical expressions to proper LaTeX format."""
    if not content or not content.strip():
        return content

    user_prompt = f"Format the mathematical expressions in this text for LaTeX:\n\n{content}"

    try:
        response = await llm.ainvoke([
            SystemMessage(content=cfg.format_math_prompt),
            HumanMessage(content=user_prompt)
        ])
        formatted = response.content.strip()

        # Sanity check
        if len(formatted) < len(content) * 0.5 or len(formatted) > len(content) * 2:
            logger.warning("LLM math formatting produced suspicious output, using original")
            return content

        return formatted
    except Exception as e:
        logger.warning(f"LLM math formatting failed: {e}, using original content")
        return content


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not text:
        return ""

    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def _escape_latex_preserve_math(text: str) -> str:
    """Escape LaTeX but preserve math mode."""
    if not text:
        return ""

    math_patterns = [
        (r'\$[^$]+\$', '__INLINE_MATH_'),
        (r'\\\[.*?\\\]', '__DISPLAY_MATH_'),
        (r'\\begin\{equation\}.*?\\end\{equation\}', '__EQ_MATH_'),
    ]

    preserved = []
    result = text

    for pattern, placeholder in math_patterns:
        matches = re.findall(pattern, result, re.DOTALL)
        for match in matches:
            key = f"{placeholder}{len(preserved)}__"
            preserved.append((key, match))
            result = result.replace(match, key, 1)

    result = result.replace('&', r'\&')
    result = result.replace('%', r'\%')
    result = result.replace('#', r'\#')

    for key, match in preserved:
        result = result.replace(key, match)

    return result


def _escape_latex_preserve_commands(text: str) -> str:
    """Escape LaTeX but preserve \\cite commands and math mode."""
    if not text:
        return ""

    patterns_to_preserve = [
        r'\\cite\{[^}]+\}',
        r'\$[^$]+\$',
        r'\\\[.*?\\\]',
        r'\\[a-zA-Z]+\{[^}]*\}',
        r'\\[a-zA-Z]+',
    ]

    preserved = []
    result = text

    for pattern in patterns_to_preserve:
        matches = re.findall(pattern, result, re.DOTALL)
        for match in matches:
            key = f"__PRESERVE_{len(preserved)}__"
            preserved.append((key, match))
            result = result.replace(match, key, 1)

    result = result.replace('&', r'\&')
    result = result.replace('%', r'\%')
    result = result.replace('#', r'\#')

    for key, match in preserved:
        result = result.replace(key, match)

    return result


def _escape_bibtex(text: str) -> str:
    """Escape special characters for BibTeX."""
    if not text:
        return ""

    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')

    return text
