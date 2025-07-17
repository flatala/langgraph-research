PREPARE_SEARCH_QUERIES_PROMPT = '''

You are an expert research assistant.

───────────────
Task
───────────────
Your task is to prepare a list of **{query_count}** search queries to be used on ArXiv to look for papers on the following topic: **{topic}**.

───────────────
Guidlines
───────────────
Please follow these guidliness:

1. The prepared querries should be effective in searching a broad range of papers related to the topic, including both recent and foundational works.
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
Important extra guidlines
───────────────
• !!! Cite only peer‑reviewed arXiv papers; AVOID duplicates unless central. !!! 
• !!! If you are struggling to find wnough entries to avoid duplicates, fell free to prepare additional search queries beond the provided ones. !!!

───────────────
Output format  (MUST be valid JSON – no Markdown, no comments)
───────────────
[
    {{
        "number": <int>,
        "title": "<section title>",
        "outline": "<1‑2 sentence explanation of its role in the review>",
        "key_points": [
            {{
                "text": "<concise sentence ideally ≤ 25 words>",
                "papers": [
                    {{ "title": "<paper title>",
                       "url":   "<https://arxiv.org/abs/…>",
                       "comment": "<one‑line note on why the paper is cited>" }},
                    ...
                ]
            }},
            ...
        ]
    }},
    ...
]

Ensure all braces/brackets are balanced and **NO trailing commas** appear.  
Return **only** the JSON – nothing else.
"""

