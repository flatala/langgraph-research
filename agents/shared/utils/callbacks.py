from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict

# Try to import from rich, but don't fail if it's not installed
try:
    from rich.progress import Progress
except ImportError:
    Progress = None

class RichProgressCallbackHandler(BaseCallbackHandler):
    """A callback handler that uses rich.progress to display the status of the agent."""

    def __init__(self, progress: "Progress", task_id):
        if Progress is None:
            raise ImportError("rich is not installed. Please install it with 'pip install rich'")
        self.progress = progress
        self.task_id = task_id

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run when a chain starts."""
        if serialized is None:
            return

        chain_name = "Unknown"
        if "id" in serialized and serialized["id"] is not None:
            name_parts = serialized["id"]
            if name_parts and isinstance(name_parts, list) and len(name_parts) > 0:
                chain_name = name_parts[-1]

        self.progress.update(self.task_id, description=f"Running: {chain_name}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> Any:
        """Run when a tool starts."""
        tool_name = serialized.get("name", "Unnamed Tool")
        self.progress.update(self.task_id, description=f"Using Tool: {tool_name}")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> Any:
        """Run when an LLM starts."""
        self.progress.update(self.task_id, description="Thinking...")
