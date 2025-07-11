from literature_review_graph.graph import graph

if __name__ == "__main__":
    init_state = {
        "topic": "LLM hallucination mitigation",
        "paper_recency": "after 2023",
        "plan": "",
        "documents": [],
        "draft_sections": [],
        "verified_sections": [],
    }

    result = graph.invoke(init_state)
    print(result["verified_sections"][0])