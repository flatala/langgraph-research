from literature_review_agent.graph import graph
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint

if __name__ == "__main__":

    load_dotenv(                
        Path(__file__).resolve().parent.parent / ".env",
        override=False,         
    )    

    init_state = {
        "topic": "LLM hallucination mitigation",
        "paper_recency": "after 2023",
        "plan": [],
        "info": [],
        "draft_sections": [],
        "verified_sections": [],
    }

    result = graph.invoke(init_state)
    pprint(result)