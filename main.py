from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from literature_review_agent.state import LitState, CachingOptions
from literature_review_agent.utils import print_plan
from literature_review_agent.graph import graph

from IPython.display import Image, display
from dotenv import load_dotenv
from pathlib import Path

import asyncio
import uuid

from literature_review_agent.state import LitState
import asyncio


async def run_workflow_async(init_state, graph, cfg):
    current_input = init_state
    while True:
        result = await graph.ainvoke(current_input, cfg)

        # if workflow is interrupted and expects human input, provide it
        if "__interrupt__" in result:
            print(result["__interrupt__"][0].value["query"] + '\n')
            answer = await asyncio.get_event_loop().run_in_executor(None, input, "Please provide the answer:")
            current_input = Command(resume={"data": answer})
            continue

        # if the workflow is completed, return the result
        return result  


if __name__ == "__main__":

    load_dotenv(                
        Path(__file__).resolve().parent.parent / ".env",
        override=False,         
    )    

    TOPIC = 'Personalisation and conditional alignment of LLMs.'
    PAPER_RECENCY = 'after 2023'

    init_state = LitState( 
        caching_options={
            "cached_plan_id": '0988bc8d-095c95bb-dae967dd-7afcb319',
            "cached_section_ids": None
        },          
        # caching_options=None,
        messages=[],
        documents=None,
        retriever=None,                     
        topic=TOPIC,
        paper_recency=PAPER_RECENCY,
        search_queries=[],
        plan=None,
        draft_sections=[],
        verified_sections=[],
        completed=False
    )

    thread_id = str(uuid.uuid4())
    config = RunnableConfig(
        recursion_limit=50,           
        configurable={"thread_id": thread_id}
    )

    img_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img_bytes)

    final_state = asyncio.run(run_workflow_async(init_state, graph, config))
    if final_state["plan"] is None:
        print("No plan generated.")
        latest_msg = final_state["messages"][-1]
        print(latest_msg)
    else:
        print_plan(final_state["plan"])