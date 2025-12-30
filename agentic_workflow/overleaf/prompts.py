"""Prompts for Overleaf/LaTeX export generation."""

GENERATE_TITLE_PROMPT = """Generate a concise, academic title for a literature survey paper.

Research Topic: {topic}

Sections covered:
{section_titles}

Requirements:
- Should be a proper academic paper title (typically 8-15 words)
- Should capture the main theme and scope
- Should sound professional and scholarly
- Do NOT include "A Survey of" or "A Review of" - just the descriptive title
- Return ONLY the title, nothing else"""


GENERATE_SECTION_INTRO_PROMPT = """Write a 2-3 sentence introduction for this section of a literature survey.

Section Title: {section_title}

Section Purpose (from research plan):
{section_outline}

Key topics covered in this section:
{subsection_summaries}

Write an introduction that:
1. Reflects the section's purpose as described above
2. Provides context for what this section covers
3. Flows naturally into the subsections

Return ONLY the introduction paragraph, nothing else."""


GENERATE_SUBSECTION_TITLE_PROMPT = """Convert this key point into a concise academic subsection title (3-8 words):

Key point: {key_point_text}

Return ONLY the title, nothing else."""


FORMAT_MATH_PROMPT = """You are a LaTeX formatting assistant. Your task is to identify mathematical expressions in the text and convert them to proper LaTeX math mode.

Rules:
1. Wrap inline math expressions in $...$ (e.g., variables like x, y, simple expressions like x^2)
2. Wrap display equations (standalone formulas) in \\[...\\]
3. Convert Greek letters to LaTeX commands (e.g., alpha -> \\alpha, beta -> \\beta)
4. Fix subscripts and superscripts (e.g., x_1 -> $x_1$, x^2 -> $x^2$)
5. Convert common math notation (e.g., sqrt -> \\sqrt, sum -> \\sum, integral -> \\int)
6. DO NOT modify \\cite{{...}} commands - leave them exactly as they are
7. DO NOT change any non-mathematical text
8. DO NOT add any explanations - return ONLY the formatted text

Return the text with mathematical expressions properly formatted for LaTeX compilation."""
