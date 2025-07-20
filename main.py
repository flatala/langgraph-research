from literature_review_agent.graph import graph
from langchain_core.runnables import RunnableConfig
from literature_review_agent.state import LitState, CachingOptions
from literature_review_agent.utils import print_plan
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
        caching_options={
            "cached_plan_id": '0988bc8d-095c95bb-dae967dd-7afcb319',
            "cached_section_ids": None
        },          
        messages=[],
        documents=None,
        retriever=None,                     
        topic="Personalisation and conditional alignment of LLMs.",
        paper_recency="after 2023",
        search_queries=[],
        plan=None,
        draft_sections=[],
        verified_sections=[],
    )

    config = RunnableConfig(recursion_limit=50)
    result_dict = asyncio.run(graph.ainvoke(init_state, config))

    final_state = LitState(**result_dict)
    print_plan(final_state["plan"])