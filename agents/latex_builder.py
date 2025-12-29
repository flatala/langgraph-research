import re
import zipfile
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from agents.shared.state.main_state import AgentState
from agents.shared.state.refinement_components import Section, Subsection, PaperWithSegements
from agents.shared.state.planning_components import Plan
from agents.shared.main_config import MainConfiguration
from agents.shared.utils.llm_utils import get_text_llm

logger = logging.getLogger(__name__)


class LatexBuilder:
    """Convert literature survey to LaTeX + BibTeX format."""

    def __init__(
        self,
        literature_survey: List[Section],
        plan: Plan,
        topic: str,
        review_id: str,
        config: Optional[RunnableConfig] = None,
    ):
        self.literature_survey = literature_survey
        self.plan = plan
        self.topic = topic
        self.review_id = review_id
        self.config = config
        self.papers_map: Dict[str, PaperWithSegements] = {}  # arxiv_id -> paper

        cfg = MainConfiguration.from_runnable_config(config)
        self.llm = get_text_llm(cfg)

    async def build(self, output_dir: str = "exports") -> str:
        """
        Generate ZIP with main.tex + references.bib.
        Returns path to the created ZIP file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # collect all papers from subsections
        self._collect_papers()

        # generate LaTeX and BibTeX content
        latex_content = await self._generate_latex()
        bibtex_content = self._generate_bibtex()

        # create zip
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"review_{self.review_id[:8]}_{timestamp}.zip"
        zip_path = output_path / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("main.tex", latex_content)
            zf.writestr("references.bib", bibtex_content)

        logger.info(f"LaTeX export saved to: {zip_path}")
        return str(zip_path)

    def _collect_papers(self) -> None:
        """Collect all unique papers from subsections."""
        for section in self.literature_survey:
            for subsection in section.subsections:
                for paper in subsection.papers:
                    if paper.arxiv_id not in self.papers_map:
                        self.papers_map[paper.arxiv_id] = paper

    async def _generate_latex(self) -> str:
        """Generate the main.tex content."""
        latex = r"""\documentclass[12pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
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
        escaped_topic = self._escape_latex(self.topic)
        latex += f"\\title{{{escaped_topic}}}\n"
        latex += "\\author{Generated Literature Review}\n"
        latex += "\\date{\\today}\n\n"
        latex += r"""\begin{document}

\maketitle
\tableofcontents
\newpage

"""

        for section in self.literature_survey:
            section_content = await self._format_section(section)
            latex += section_content

        latex += r"""

\newpage
\printbibliography

\end{document}
"""
        return latex

    async def _format_section(self, section: Section) -> str:
        """Format a single section with subsections."""
        content = ""

        # title
        escaped_title = self._escape_latex(section.section_title)
        content += f"\\section{{{escaped_title}}}\n\n"

        # section intro
        if section.section_introduction:
            formatted_intro = await self._format_math_for_latex(section.section_introduction)
            escaped_intro = self._escape_latex_preserve_math(formatted_intro)
            content += f"{escaped_intro}\n\n"

        # subsections
        for subsection in section.subsections:
            subsection_content = await self._format_subsection(subsection)
            content += subsection_content

        return content

    async def _format_subsection(self, subsection: Subsection) -> str:
        """Format a single subsection."""
        content = ""
        escaped_title = self._escape_latex(subsection.subsection_title or subsection.key_point_text)
        content += f"\\subsection{{{escaped_title}}}\n\n"
        if subsection.content:
            converted_content = self._convert_citations(subsection.content)
            formatted_content = await self._format_math_for_latex(converted_content)
            escaped_content = self._escape_latex_preserve_commands(formatted_content)
            content += f"{escaped_content}\n\n"

        return content

    def _convert_citations(self, text: str) -> str:
        """
        Convert [Author_YEAR(ArxivID)] citations to \\cite{key} format.

        Examples:
        - [Smith_2023(2301.12345)] -> \\cite{arxiv_2301_12345}
        - [Smith_2023(2301.12345); Jones_2022(2201.54321)] -> \\cite{arxiv_2301_12345,arxiv_2201_54321}
        """
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
            return match.group(0)  # Return original if no match

        # match: [anything(arxiv_id)] or [a(id); b(id); ...]
        pattern = r'\[([^\]]+\([0-9]+\.[0-9]+(?:v\d+)?\)[^\]]*)\]'
        return re.sub(pattern, replace_citation, text)

    async def _format_math_for_latex(self, content: str) -> str:
        """
        Use LLM to convert mathematical expressions to proper LaTeX format.
        """
        if not content or not content.strip():
            return content

        system_prompt = """You are a LaTeX formatting assistant. Your task is to identify mathematical expressions in the text and convert them to proper LaTeX math mode.

Rules:
1. Wrap inline math expressions in $...$ (e.g., variables like x, y, simple expressions like x^2)
2. Wrap display equations (standalone formulas) in \\[...\\]
3. Convert Greek letters to LaTeX commands (e.g., alpha -> \\alpha, beta -> \\beta)
4. Fix subscripts and superscripts (e.g., x_1 -> $x_1$, x^2 -> $x^2$)
5. Convert common math notation (e.g., sqrt -> \\sqrt, sum -> \\sum, integral -> \\int)
6. DO NOT modify \\cite{...} commands - leave them exactly as they are
7. DO NOT change any non-mathematical text
8. DO NOT add any explanations - return ONLY the formatted text

Return the text with mathematical expressions properly formatted for LaTeX compilation."""

        user_prompt = f"""Format the mathematical expressions in this text for LaTeX:

{content}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            formatted = response.content.strip()
            if len(formatted) < len(content) * 0.5 or len(formatted) > len(content) * 2:
                logger.warning("LLM math formatting produced suspicious output, using original")
                return content

            return formatted
        except Exception as e:
            logger.warning(f"LLM math formatting failed: {e}, using original content")
            return content

    def _generate_bibtex(self) -> str:
        """Generate references.bib content."""
        entries = []
        for arxiv_id, paper in self.papers_map.items():
            entry = self._format_bibtex_entry(paper)
            entries.append(entry)

        return "\n\n".join(entries)

    def _format_bibtex_entry(self, paper: PaperWithSegements) -> str:
        """Format a single BibTeX entry."""
        base_id = re.sub(r'v\d+$', '', paper.arxiv_id)
        cite_key = f"arxiv_{base_id.replace('.', '_')}"
        authors = " and ".join(paper.authors) if paper.authors else "Unknown"
        year_prefix = paper.arxiv_id[:2]
        year = f"20{year_prefix}" if int(year_prefix) < 50 else f"19{year_prefix}"
        title = self._escape_bibtex(paper.title)
        entry = f"""@article{{{cite_key},
    title = {{{title}}},
    author = {{{authors}}},
    year = {{{year}}},
    eprint = {{{paper.arxiv_id}}},
    archivePrefix = {{arXiv}},
    primaryClass = {{cs}},
    url = {{{paper.arxiv_url}}}
}}"""
        return entry

    def _escape_latex(self, text: str) -> str:
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

    def _escape_latex_preserve_math(self, text: str) -> str:
        """Escape LaTeX but preserve math mode ($...$, \\[...\\])."""
        if not text:
            return ""

        # find and temporarily replace math expressions
        math_patterns = [
            (r'\$[^$]+\$', '__INLINE_MATH_'),
            (r'\\\[.*?\\\]', '__DISPLAY_MATH_'),
            (r'\\begin\{equation\}.*?\\end\{equation\}', '__EQ_MATH_'),
        ]

        preserved = []
        result = text

        for pattern, placeholder in math_patterns:
            matches = re.findall(pattern, result, re.DOTALL)
            for i, match in enumerate(matches):
                key = f"{placeholder}{len(preserved)}__"
                preserved.append((key, match))
                result = result.replace(match, key, 1)

        # escape remaining text
        result = result.replace('&', r'\&')
        result = result.replace('%', r'\%')
        result = result.replace('#', r'\#')

        # restore math expressions
        for key, match in preserved:
            result = result.replace(key, match)

        return result

    def _escape_latex_preserve_commands(self, text: str) -> str:
        """Escape LaTeX but preserve \\cite commands and math mode."""
        if not text:
            return ""

        # patterns to preserve
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

        # escape remaining special characters
        result = result.replace('&', r'\&')
        result = result.replace('%', r'\%')
        result = result.replace('#', r'\#')

        # restore preserved content
        for key, match in preserved:
            result = result.replace(key, match)

        return result

    def _escape_bibtex(self, text: str) -> str:
        """Escape special characters for BibTeX."""
        if not text:
            return ""

        # escape BibTeX special characters
        text = text.replace('&', r'\&')
        text = text.replace('%', r'\%')
        text = text.replace('$', r'\$')
        text = text.replace('#', r'\#')
        text = text.replace('_', r'\_')

        return text


async def build_latex(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Graph node that builds LaTeX export from the completed literature survey.
    """
    logger.info("Building LaTeX export...")

    if not state.literature_survey:
        logger.warning("No literature survey found, skipping LaTeX export")
        return {"latex_export_path": None}

    builder = LatexBuilder(
        literature_survey=state.literature_survey,
        plan=state.plan,
        topic=state.topic,
        review_id=state.review_id,
        config=config,
    )

    zip_path = await builder.build()

    logger.info(f"LaTeX export completed: {zip_path}")
    return {"latex_export_path": zip_path}
