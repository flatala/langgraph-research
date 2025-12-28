from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict

try:
    from rich.progress import Progress
except ImportError:
    Progress = None

class RichProgressCallbackHandler(BaseCallbackHandler):
    """A callback handler that uses rich.progress to display the status of the agent."""

    STAGE_NAMES = {
        "planning": "Planning",
        "refinement": "Refinement",
        "append_system_prompt": "Initializing",
        "refine_problem_statement": "Refining problem",
        "parse_queries_add_plan_prompt": "Parsing queries",
        "plan_literature_review": "Planning review",
        "reflect_on_papers": "Reflecting on papers",
        "parse_plan": "Parsing plan",
        "tools_1": "Using tools",
        "tools_2": "Searching papers",
        "initialise_refinement_progress": "Initializing",
        "prepare_subsection_context": "Preparing context",
        "write_subsection": "Writing subsection",
        "review_content": "Reviewing content",
        "process_content_feedback": "Fixing content",
        "review_grounding": "Checking citations",
        "process_grounding_feedback": "Fixing citations",
        "advance_to_next": "Advancing",
        "complete_refinement": "Completing",
        "cleanup_temp_cache": "Cleaning up",
    }

    def __init__(self, progress: "Progress", task_id):
        if Progress is None:
            raise ImportError("rich is not installed. Please install it with 'pip install rich'")
        self.progress = progress
        self.task_id = task_id
        self.current_stage = "Starting"

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run when a chain starts."""
        chain_name = kwargs.get("name")
        
        if chain_name and chain_name in self.STAGE_NAMES:
            self.current_stage = self.STAGE_NAMES[chain_name]
            self.progress.update(self.task_id, description=f"{self.current_stage}...")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> Any:
        """Run when a tool starts."""
        tool_name = serialized.get("name", "Unnamed Tool")
        self.progress.update(self.task_id, description=f"{self.current_stage}: {tool_name}")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> Any:
        """Run when an LLM starts."""
        self.progress.update(self.task_id, description=f"{self.current_stage}...")
