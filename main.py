from literature_review_agent.graph import graph
from literature_review_agent.state import LitState
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import asyncio

if __name__ == "__main__":

    load_dotenv(                
        Path(__file__).resolve().parent.parent / ".env",
        override=False,         
    )    

    init_state = LitState(               
        messages=[],
        info=None,                       
        topic="Personalisation and conditional alignment of LLMs.",
        paper_recency="after 2023",
        search_queries=[],
        plan=[],
        draft_sections=[],
        verified_sections=[],
    )

    result_dict = asyncio.run(graph.ainvoke(init_state))
    final_state = LitState(**result_dict)


    final_state.print_messages()
    final_state.print_plan()