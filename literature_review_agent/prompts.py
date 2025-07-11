PLAN_PROMPT = """
You are an expert research assistant.

Task ▸ Draft an outline (plan) for a graduate-level **literature review** on **{topic}**,
using peer-reviewed or widely‐cited arXiv papers **{paper_recency}**.

───────────────
Structure guidelines
───────────────
• The review should have 4 – 6 major sections, ordered logically:
  1. *Background / Motivation* – define the problem and explain why it matters.  
  2. *Taxonomy / Categorisation* – organise existing work into meaningful buckets.  
  3. *Methods / Approaches* – compare key techniques, algorithms or frameworks.  
  4. *Evaluation & Benchmarks* – datasets, metrics, experimental results.  
  5. *Mitigation / Applications* – how current research addresses the problem.  
  6. *Open Challenges & Future Directions* – gaps, limitations, promising ideas.  
  (Merge or split these as needed, but keep 4–6 total.)

• **Each section** must list **2 – 4 key points** that a reader should remember.

• Cite **2 – 3 primary papers** per key point (more only if essential).

───────────────
Paper selection rules
───────────────
• Prefer papers with high citation count, strong experimental setup, or novel insight.  
• Cover a mix of survey/overview papers and specialised studies.  
• Avoid duplicate papers across points unless absolutely central.

───────────────
Output format  (MUST be valid JSON, no Markdown, no comments)
───────────────
[
    {{
        "title": "<section title>",
        "key_points": [
            {{
                "text": "<concise sentence ideally ≤ 25 words>",
                "papers": [
                    {{ "title": "<paper title>",
                       "url":   "<https://arxiv.org/abs/…>",
                       "comment": "<one-line note on why the paper is cited>" }},
                    ...
                ]
            }},
            ...
        ]
    }},
    ...
]

Make sure all braces/brackets are balanced and NO trailing commas appear.
Return **only** the JSON – nothing else.
"""
