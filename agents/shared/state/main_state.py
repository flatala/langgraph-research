from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional, List
from textwrap import indent

from agents.shared.state.planning_components import Plan
from agents.shared.state.refinement_components import RefinementProgress, Section

import json

class AgentState(BaseModel):
    # initial params
    topic: str
    paper_recency: str
    completed: bool
    review_id: str  # Database review ID

    # history of messages
    messages: list = Field(default_factory=list)

    # arxiv search queries and survey plan
    search_queries: Optional[List[str]] = None
    plan: Optional[Plan] = None

    # survey refinement
    refinement_progress: Optional[RefinementProgress] = None
    literature_survey: List[Section] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def print_messages(self, max_chars: int | None = None, prettify_json: bool = True) -> str:
        """
        Pretty‑print the full message history.

        • Shows       USER / ASSISTANT text
        • Shows → TOOL <name> ARGS          (arguments the assistant sent)
        • Shows TOOL[<name>] OUT            (tool output)
        """

        
        def _fmt(txt: str) -> str:
            """Optionally pretty‑print JSON and truncate."""
            if prettify_json:
                try:
                    txt = json.dumps(json.loads(txt), indent=2)
                except Exception:
                    pass
            if max_chars and len(txt) > max_chars:
                txt = txt[:max_chars] + " …[truncated]"
            return txt

        lines: List[str] = ["\nConversation log\n" + "─" * 60]
        for i, msg in enumerate(self.messages, 1):

            if isinstance(msg, HumanMessage):
                lines.append(f"{i:02d}  [USER]\n{indent(_fmt(msg.content), '   ')}")

            elif isinstance(msg, AIMessage):
                if msg.content:
                    lines.append(f"{i:02d}  [ASSISTANT]\n{indent(_fmt(msg.content), '   ')}")
                for tc in msg.tool_calls or []:
                    args = _fmt(json.dumps(tc["args"]))
                    lines.append(
                        f"{i:02d}  [ASSISTANT → TOOL {tc['name']} ARGS]\n{indent(args, '   ')}"
                    )

            elif isinstance(msg, ToolMessage):
                out = _fmt(str(msg.content))
                lines.append(
                    f"{i:02d}  [TOOL {msg.name}] OUT\n{indent(out, '   ')}"
                )

            else:
                lines.append(f"{i:02d}  [{msg.__class__.__name__.upper()}]\n{indent(_fmt(str(msg.content)), '   ')}")

        log = "\n".join(lines)
        print(log)
        return log

