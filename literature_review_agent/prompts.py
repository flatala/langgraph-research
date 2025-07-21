# PREPARE_SEARCH_QUERIES_PROMPT = '''

# You are an expert research assistant.

# ───────────────
# Task
# ───────────────
# Your task is to prepare a list of **{query_count}** search queries to be used on ArXiv to look for papers on the following topic: **{topic}**.

# ───────────────
# Guidlines
# ───────────────
# Please follow these guidliness:

# 1. The prepared querries should be effective in searching a broad range of papers related to the topic, including both recent and foundational works.
# 2. Ensure that the queries are specific enough to yield relevant results, but broad enough to capture a wide range of literature.
# 3. Ensure that the queries stay relevant and related to the topic.

# ───────────────
# Output Format
# ───────────────
# Please return the queries in a JSON array format, with each query as a string. The output should look like this:

# [
#     "query 1",
#     "query 2",
#     ...
# ]

# Make sure all braces/brackets are balanced and NO trailing commas appear.
# Return **only** the JSON – nothing else.
# '''

PREPARE_SEARCH_QUERIES_PROMPT = '''

You are an expert research assistant.

───────────────
Task
───────────────
Your task is to prepare a list of **{query_count}** search queries to be used on ArXiv to look for papers on the following topic: **{topic}**.

───────────────
Human Input
───────────────
You have access to a tool that lets you ask the user (a human) for clarification or to refine the problem statement and the exact area of the literature survey.  
- You may use this tool up to **2 times**, you have to use it **at least** one time.
- Use it if you are uncertain about the topic or if clarification would help produce more effective queries.
- Stop and generate the queries as soon as you have enough information.

───────────────
Guidelines
───────────────
1. The prepared queries should be effective in searching a broad range of papers related to the topic, including both recent and foundational works.
2. Ensure that the queries are specific enough to yield relevant results, but broad enough to capture a wide range of literature.
3. Ensure that the queries stay relevant and related to the topic.

───────────────
Output Format
───────────────
Please return the queries in a JSON array format, with each query as a string. The output should look like this:

[
    "query 1",
    "query 2",
    ...
]

Make sure all braces/brackets are balanced and NO trailing commas appear.
Return **only** the JSON – nothing else.
'''



PLAN_PROMPT = """
You are an expert research assistant.

───────────────
Task
───────────────
Draft an outline (plan) for a graduate‑level **literature review** on **{topic}**,
using peer‑reviewed or widely‑cited arXiv papers **{paper_recency}**.  
The refined search queries you can use are: **{search_queries}**.
Ensure that the plan is comprehensive, well‑structured, and each section follows logically from the previous sections.

───────────────
Structure guidelines
───────────────
• The review should have **4 – 6 major sections**, ordered logically:
  1. *Background / Motivation* – define the problem and explain why it matters.  
  2. *Taxonomy / Categorisation* – organise existing work into meaningful buckets.  
  3. *Methods / Approaches* – compare key techniques, algorithms or frameworks.  
  4. *Evaluation & Benchmarks* – datasets, metrics, experimental results.  
  5. *Mitigation / Applications* – how current research addresses the problem.  
  6. *Open Challenges & Future Directions* – gaps, limitations, promising ideas.  
  (Merge or split as needed, but keep 4 – 6 total.)

• **For each section** provide:
  – `number`     the section’s order (1, 2, …).  
  – `title`     a concise heading.  
  – `outline`    1‑‑2 sentences explaining why this section is included and how it fits into the entire review.  
  – `key_points`  2 – 4 bullet points, each citing 2 – 3 primary papers.

───────────────
Critical citation & coverage guidelines
───────────────
• For **each section**, you **must cite papers that have NOT been used in previous sections** whenever possible.
• **Do NOT reuse the same paper in multiple sections** unless it is *absolutely central* to both; if you do reuse, briefly justify why in the "comment" field.
• Ensure that **every relevant paper from your arXiv search is cited at least once** somewhere in the review (unless clearly irrelevant).
• **Distribute paper citations as evenly as possible across all sections and key points**, to maximize diversity and coverage.
• If there are more papers than needed, prioritize the most recent and highly-cited works for key sections, but strive for broad representation.

───────────────
Output format  (MUST be valid JSON – no Markdown, no comments)
───────────────
{{
    "reasoning": "<Step-by-step explanation of how you selected the structure, papers, and citations for the plan.>",
    "plan": [
        {{
            "number": <int>,
            "title": "<section title>",
            "outline": "<A couple sentence explanation of the section's role in the review, what does it present, and why.>",
            "key_points": [
                {{
                    "text": "<A couple of senetcnes describing the meaning and relevance of the point that is beging made.>",
                    "papers": [
                        {{ 
                            "title": "<paper title>",
                            "year": <int>,
                            "url":   "<https://arxiv.org/abs/…>",
                            "summary": "<A brief summary of the paper.>" 
                            "citation_reason": "<A short explanation why this exact paper was selected for the respective point and section.>" 
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

Ensure all braces/brackets are balanced and **NO trailing commas** appear.  
Return **only** the JSON – nothing else.
"""

