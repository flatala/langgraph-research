from literature_review_agent.graph import graph
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import asyncio

if __name__ == "__main__":

    load_dotenv(                
        Path(__file__).resolve().parent.parent / ".env",
        override=False,         
    )    

    init_state = {
        "messages": [],
        "topic": "Zero and Few-Shot based autoamtic propaganda detection",
        "paper_recency": "after 2023",
        "plan": [],
        "info": [],
        "draft_sections": [],
        "verified_sections": [],
    }

    result = asyncio.run(graph.ainvoke(init_state))
    pprint(result['plan'])