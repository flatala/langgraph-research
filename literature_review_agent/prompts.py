PLAN_PROMPT = """You are an expert research assistant.

Generate a literature-review plan on {topic} using papers {paper_recency}.

Return **only** JSON in exactly this shape:
[
    {{
        "title": "<section title>",
        "key_points": [
            {{
                "text": "<point>",
                "papers": ["<url1>", "<url2>", "..."]
            }},
            ...
        ]
    }},
    ...
]
"""