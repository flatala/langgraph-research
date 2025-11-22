from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agents.shared.state.main_state import AgentState
from agents.planning_agent.graph import planning_graph
from agents.refinement_agent.graph import refinement_graph
from agents.graph import graph
from data.database.crud import ReviewDB

from dotenv import load_dotenv
from pathlib import Path
import os

import asyncio
import uuid


async def run_workflow_async(init_state, graph, cfg):
    current_input = init_state
    while True:
        result = await graph.ainvoke(current_input, cfg)

        # if workflow is interrupted and expects human input, provide it
        if "__interrupt__" in result:
            print(result["__interrupt__"][0].value["query"] + '\n')
            answer = await asyncio.get_event_loop().run_in_executor(None, input, "Please provide the answer: ")
            current_input = Command(resume={"data": answer})
            continue

        # if the workflow is completed, return the result
        return result  


if __name__ == "__main__":

    load_dotenv(
        Path(__file__).resolve().parent / ".env",
        override=False,
    )

    TOPIC = 'Personalisation and conditional alignment of LLMs.'
    PAPER_RECENCY = 'after 2023'

    # Initialize database and create new review
    db = ReviewDB()
    review = db.create_review(
        topic=TOPIC,
        paper_recency=PAPER_RECENCY,
        orchestrator_model=os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-4o"),
        text_model=os.getenv("TEXT_MODEL", "openai/gpt-4-turbo"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    )

    print(f"\n{'='*60}")
    print(f"Starting new literature review")
    print(f"Review ID: {review.id}")
    print(f"Topic: {TOPIC}")
    print(f"{'='*60}\n")

    init_state = AgentState(
        review_id=review.id,
        topic=TOPIC,
        paper_recency=PAPER_RECENCY,
        completed=False,
        messages=[],
        search_queries=[],
        plan=None
    )

    thread_id = str(uuid.uuid4())
    graph_config = RunnableConfig(
        recursion_limit=50,           
        configurable={"thread_id": thread_id}
    )

    # Generate Mermaid diagrams (viewable in VS Code with Mermaid extensions or on GitHub)
    mermaid_syntax = graph.get_graph().draw_mermaid()
    with open("graph_diagrams/main_graph.mmd", "w") as f:
        f.write(mermaid_syntax)

    mermaid_syntax = planning_graph.get_graph().draw_mermaid()
    with open("graph_diagrams/planning_graph.mmd", "w") as f:
        f.write(mermaid_syntax)

    mermaid_syntax = refinement_graph.get_graph().draw_mermaid()
    with open("graph_diagrams/refinement_graph.mmd", "w") as f:
        f.write(mermaid_syntax)

    final_state_dict = asyncio.run(run_workflow_async(init_state, graph, graph_config))
    final_state = AgentState(**final_state_dict)

    if final_state.plan is None:
        print("No plan generated.")
        latest_msg = final_state.messages[-1]
        print(latest_msg)
        db.update_review_status(review.id, 'failed')
    else:
        final_state.plan.print_plan()

        # Update review status
        if final_state.completed:
            db.update_review_status(review.id, 'completed')
            db.update_review_metrics(
                review.id,
                total_sections=len(final_state.literature_survey),
                total_papers_used=len(db.get_papers_for_review(review.id))
            )
            print(f"\n{'='*60}")
            print(f"✓ Review completed and saved to database")
            print(f"Review ID: {review.id}")
            print(f"{'='*60}\n")