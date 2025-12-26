from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.shared.state.main_state import AgentState
from agents.graph import graph
from data.database.crud import ReviewDB
from agents.shared.utils.logging_utils import setup_logging
from agents.shared.utils.callbacks import RichProgressCallbackHandler

from dotenv import load_dotenv
from pathlib import Path
import os
import asyncio
import uuid
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

async def run_workflow_async(init_state, graph_instance, config, console: Console):
    current_input = init_state
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Agent is running...", total=None)
        
        # Setup callback handler
        callback_handler = RichProgressCallbackHandler(progress, task_id)
        config['callbacks'] = [callback_handler]

        while True:
            result = await graph_instance.ainvoke(current_input, config)

            logger.debug(f"Graph returned. Keys in result: {list(result.keys())}")

            if "__interrupt__" in result:
                progress.stop()
                query = result["__interrupt__"][0].value["query"]
                console.print(Panel(query, title="[bold yellow]Human Input Required[/bold yellow]"))
                answer = await asyncio.get_event_loop().run_in_executor(None, console.input, "Please provide the answer: ")
                current_input = Command(resume={"data": answer})
                progress.start()
                continue

            return result

def main():
    load_dotenv(
        Path(__file__).resolve().parent / ".env",
        override=False,
    )

    console = Console()
    console.print(Panel("Literature Review Agent", style="bold blue", expand=False))

    topic = console.input("Enter the research topic: ")
    paper_recency = console.input("Enter paper recency (e.g., 'after 2023', 'last 2 years'): ")

    # Initialize database and create new review
    db = ReviewDB()
    review = db.create_review(
        topic=topic,
        paper_recency=paper_recency,
        orchestrator_model=os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-4o"),
        text_model=os.getenv("TEXT_MODEL", "openai/gpt-4-turbo"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    )
    
    console.print(Panel(
        f"Review ID: {review.id}\nTopic: {topic}",
        title="[bold green]Starting New Literature Review[/bold green]",
        expand=False
    ))

    init_state = AgentState(
        review_id=review.id,
        topic=topic,
        paper_recency=paper_recency,
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

    final_state_dict = asyncio.run(run_workflow_async(init_state, graph, graph_config, console))
    final_state = AgentState(**final_state_dict)

    if final_state.plan is None:
        console.print(Panel("No plan was generated. The review process failed.", style="bold red"))
        latest_msg = final_state.messages[-1] if final_state.messages else "No messages in final state."
        logger.error(f"Review failed. Last message: {latest_msg}")
        db.update_review_status(review.id, 'failed')
    else:
        logger.info("Final plan generated:\n%s", final_state.plan.print_plan())

        if final_state.completed:
            db.update_review_status(review.id, 'completed')
            db.update_review_metrics(
                review.id,
                total_sections=len(final_state.literature_survey),
                total_papers_used=len(db.get_papers_for_review(review.id))
            )
            console.print(Panel(
                f"Review ID: {review.id}",
                title="[bold green]✓ Review Completed and Saved[/bold green]",
                expand=False
            ))

if __name__ == "__main__":
    main()