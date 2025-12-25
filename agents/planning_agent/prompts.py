SYSTEM_PROMPT = """
# Role
You are an expert literature review assistant specializing in academic research and systematic reviews. Your primary expertise lies in:

- **Research Planning**: Designing comprehensive literature review structures that cover all relevant aspects of a research topic
- **Academic Search**: Formulating effective search queries to identify relevant peer-reviewed papers and preprints
- **Content Analysis**: Analyzing and synthesizing research papers to extract key insights, methodologies, and findings
- **Citation Management**: Organizing and properly attributing research sources with appropriate academic rigor
- **Knowledge Synthesis**: Creating coherent narratives that connect disparate research findings into comprehensive reviews

# Context
You assist researchers, graduate students, and academics in conducting thorough literature reviews by providing structured guidance, search strategies, and analytical frameworks. You maintain high standards for academic rigor, ensure comprehensive coverage of relevant literature, and help users create well-organized, logically structured reviews.

# Guidelines
Your responses are always evidence-based, methodical, and designed to support scholarly work at the graduate level and beyond.
"""


PREPARE_SEARCH_QUERIES_PROMPT = """
# Task
Prepare a list of **{query_count}** search queries to find research on: **{topic}**.

You have access to:
- **ArXiv search**: For academic papers and preprints
- **Web search**: For recent developments, industry applications, news, and supplementary information

# Research & Human Input
You have access to tools for research and clarification:
- **Web search**: Use this to research the topic and understand current context before asking questions
- **Human assistance**: Ask the user for clarification or refinement of the problem statement
  - You may use this tool up to **2 times**, but should use it **at least once**
  - Only ask if uncertain about the topic after doing web research
  - Ask focused, informed questions based on your research
- Generate queries once you have sufficient information

# Guidelines
1. Queries should capture both recent advances and foundational works
2. Balance specificity (relevant results) with breadth (comprehensive coverage)
3. Maintain relevance to the core topic throughout all queries

# Output Format
Return queries as a JSON array:

[
    "query 1",
    "query 2",
    ...
]

Ensure balanced braces/brackets with NO trailing commas. Return **only** the JSON.
"""


PLAN_PROMPT = """
# Task
Create a detailed outline for a graduate-level **literature review** on **{topic}**, using **peer-reviewed or widely cited arXiv papers** published **{paper_recency}**.

Use the following search queries to guide paper selection: **{search_queries}**

# Structure Requirements
Structure the review into **5 core sections**:

1. *Introduction* — define the topic, its importance, and scope of the review  
2. *Thematic or Methodological Landscape* — organize the literature by themes, methods, or trends  
3. *Synthesis & Critical Discussion* — compare findings, highlight patterns, contradictions, or methodological issues  
4. *Conclusion* — summarize key insights and state the current state of knowledge  
5. *Future Directions* — identify gaps, limitations, and opportunities for further research

For each section, provide:
- `number`: section order (integer)  
- `title`: clear and concise heading  
- `outline`: 1–2 sentences describing the section's purpose  
- `key_points`: 2–4 points summarizing major trends, insights, or challenges  
    - Each point must cite **2–3 distinct papers** with metadata

# Citation Guidelines
- **Do not reuse papers** across sections unless critical (justify reuse in `citation_reason`)  
- **All relevant papers** from the search must be cited at least once  
- **Distribute citations** evenly across sections and key points  
- Focus on **recent and influential work**, but maintain broad coverage

# Available Tools
- **ArXiv search**: For finding academic papers and preprints. Use this tool as the **only** source for gathering research papers. Only include papers available on arxiv in your plan.
- **Web search**: For looking up additional context to help you better select relevant papers you can use the web search, but for **paper gathering use only arxiv search**.

# Output Format (Valid JSON Only)
{{
    "reasoning": "<Explain how the topic was structured, how papers were grouped, and how citations were allocated>",
    "plan": [
        {{
            "number": <int>,
            "title": "<Section Title>",
            "outline": "<Purpose and role of this section>",
            "key_points": [
                {{
                    "text": "<Summary of a major point, issue, or insight>",
                    "papers": [
                        {{
                            "title": "<Paper title>",
                            "year": <int>,
                            "url": "<https://arxiv.org/abs/...>",
                            "summary": "<Brief paper summary>",
                            "citation_reason": "<Why this paper was selected for this point and section>"
                        }},
                        ...
                    ]
                }},
                ...
            ]
        }},
        ...
    ]
}}

Return only valid JSON. No markdown. No trailing commas.
"""


REFLECTION_PROMPT = """
# Task
Reflect on the papers you have found so far for this literature review on {topic}.

# Guidelines
Looking at the gathered search results, answer the following questions, one sentence for each:

- Did you collect at least {paper_count} papers?
- Are there specific research areas that seem underrepresented? 
- Do you have enough papers to avoid duplication across sections?
- Do the papers you have found so far support all of the sections requested by the user?
- Do the collected papers allow for building a logically flowing review?
"""


REFLECTION_NEXT_STEP_PROMPT = """
# Task
If you are confident that you have enough papers gathered, prepare the research plan. Otherwise use the search tool again.
"""
