from literature_review_agent.graph import graph
from langchain_core.runnables import RunnableConfig
from literature_review_agent.state import LitState, CachingOptions
from literature_review_agent.utils import print_plan
from langchain_core.messages import ToolMessage
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import asyncio


from literature_review_agent.state import LitState
import asyncio

def run_workflow(init_state, config, graph):
    state = init_state
    while True:
        result_dict = asyncio.run(graph.ainvoke(state, config))
        state = LitState(**result_dict)

        # 1. Check if we're waiting for human input (your tool uses `interrupt`)
        # Typically, you'd look for a specific field, but since your human tool just returns the answer,
        # you can check the latest message content for a prompt/request.
        latest_message = state["messages"][-1]
        # Adjust this condition if your interrupt result is different!
        if hasattr(latest_message, "content") and state["completed"] is True:
            print(latest_message.content)
            user_input = input("Human input required: ")
            
            # The tool_call_id should match what was requested (you can extract it from the interrupt/tool call)
            tool_call_id = latest_message.tool_calls[0]['id']
            tool_message = ToolMessage(
                name="human_assistance",
                tool_call_id=tool_call_id,
                content=user_input
            )
            # Add this response to the messages, then resume
            state["messages"].append(tool_message)
            continue  # Resume workflow loop
        else:
            break  # Workflow finished, no more human input needed

    return state



if __name__ == "__main__":

    load_dotenv(                
        Path(__file__).resolve().parent.parent / ".env",
        override=False,         
    )    

    init_state = LitState( 
        # caching_options={
        #     "cached_plan_id": '0988bc8d-095c95bb-dae967dd-7afcb319',
        #     "cached_section_ids": None
        # },          
        caching_options=None,
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

    final_state = run_workflow(init_state, config, graph)
    if final_state["plan"] is None:
        print("No plan generated.")
        latest_msg = final_state["messages"][-1]
        print(latest_msg)
    else:
        print_plan(final_state["plan"])

# if __name__ == "__main__":

#     load_dotenv(                
#         Path(__file__).resolve().parent.parent / ".env",
#         override=False,         
#     )    

#     init_state = LitState( 
#         # caching_options={
#         #     "cached_plan_id": '0988bc8d-095c95bb-dae967dd-7afcb319',
#         #     "cached_section_ids": None
#         # },          
#         caching_options=None,
#         messages=[],
#         documents=None,
#         retriever=None,                     
#         topic="Personalisation and conditional alignment of LLMs.",
#         paper_recency="after 2023",
#         search_queries=[],
#         plan=None,
#         draft_sections=[],
#         verified_sections=[],
#     )

#     config = RunnableConfig(recursion_limit=50)
#     result_dict = asyncio.run(graph.ainvoke(init_state, config))

#     final_state = LitState(**result_dict)
#     print_plan(final_state["plan"])