SYSTEM_PROMPT = """You are an expert literature review assistant specializing in academic research and systematic reviews. Your primary expertise lies in:

• **Research Planning**: Designing comprehensive literature review structures that cover all relevant aspects of a research topic
• **Academic Search**: Formulating effective search queries to identify relevant peer-reviewed papers and preprints
• **Content Analysis**: Analyzing and synthesizing research papers to extract key insights, methodologies, and findings
• **Citation Management**: Organizing and properly attributing research sources with appropriate academic rigor
• **Knowledge Synthesis**: Creating coherent narratives that connect disparate research findings into comprehensive reviews

You assist researchers, graduate students, and academics in conducting thorough literature reviews by providing structured guidance, search strategies, and analytical frameworks. You maintain high standards for academic rigor, ensure comprehensive coverage of relevant literature, and help users create well-organized, logically structured reviews.

Your responses are always evidence-based, methodical, and designed to support scholarly work at the graduate level and beyond."""

PREPARE_SEARCH_QUERIES_PROMPT = '''───────────────
Task
───────────────
Prepare a list of **{query_count}** search queries to be used on ArXiv for papers on: **{topic}**.

───────────────
Human Input
───────────────
You have access to a tool to ask the user for clarification or refinement of the problem statement.  
- You may use this tool up to **2 times**, but must use it **at least once**
- Use it if uncertain about the topic or if clarification would improve query effectiveness
- Generate queries once you have sufficient information

───────────────
Guidelines
───────────────
1. Queries should capture both recent advances and foundational works
2. Balance specificity (relevant results) with breadth (comprehensive coverage)
3. Maintain relevance to the core topic throughout all queries

───────────────
Output Format
───────────────
Return queries as a JSON array:

[
    "query 1",
    "query 2",
    ...
]

Ensure balanced braces/brackets with NO trailing commas. Return **only** the JSON.
'''


PLAN_PROMPT = """───────────────
Task
───────────────
Draft a comprehensive outline for a graduate‑level **literature review** on **{topic}**, using peer‑reviewed or widely‑cited arXiv papers **{paper_recency}**. 

Available search queries: **{search_queries}**

───────────────
Structure Requirements
───────────────
Create **4–6 major sections** in logical order:
1. *Background/Motivation* – problem definition and significance
2. *Taxonomy/Categorisation* – organize existing work meaningfully  
3. *Methods/Approaches* – compare techniques, algorithms, frameworks
4. *Evaluation & Benchmarks* – datasets, metrics, experimental results
5. *Applications/Mitigation* – how research addresses the problem
6. *Open Challenges & Future Directions* – gaps, limitations, opportunities

**For each section provide:**
- `number`: section order (1, 2, ...)
- `title`: concise heading  
- `outline`: 1–2 sentences on section purpose and fit within review
- `key_points`: 2–4 bullet points, each citing 2–3 primary papers

───────────────
Citation Guidelines
───────────────
• **Avoid paper reuse** across sections unless absolutely central (justify in comments)
• **Cite every relevant paper** from search results at least once
• **Distribute citations evenly** across sections and key points
• **Prioritize recent and highly-cited works** while maintaining broad coverage

───────────────
Output Format (Valid JSON Only)
───────────────
{{
    "reasoning": "<Step-by-step explanation of structure selection, paper distribution, and citation strategy>",
    "plan": [
        {{
            "number": <int>,
            "title": "<section title>",
            "outline": "<Section role and purpose in the review>",
            "key_points": [
                {{
                    "text": "<Point description and relevance>",
                    "papers": [
                        {{ 
                            "title": "<paper title>",
                            "year": <int>,
                            "url": "<https://arxiv.org/abs/...>",
                            "summary": "<Brief paper summary>",
                            "citation_reason": "<Why this paper for this point/section>"
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

Return balanced JSON with NO trailing commas. JSON only.
"""


REFLECTION_PROMPT = """

Reflect on the papers you have found so far for this literature review on {topic}.

Looking at the gathered search results, answer the following questions, one sentence for each:

- Did you collect at least {paper_count} papers?
- Are there specific research areas that seem underrepresented? 
- Do you have enough papers to avoid duplication across sections?
- Do the papers you have found so far support all of the sections requested by the user?
- Do the collected papers allow for building a logically flowing review?

"""


REFLECTION_NEXT_STEP_PROMPT = """

If you are confident that you have neough papers gathered, prepare the research plan. Otherwise use the search tool again.

"""