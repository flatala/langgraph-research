from typing_extensions import TypedDict, Annotated, Optional, Any, List
from textwrap import indent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dataclasses import dataclass, field
from literature_review_agent.utils import reduce_docs
import json

class PaperRef(TypedDict):
    title: str     
    year: int
    url: str  
    summary: str
    citation_reason: str

class KeyPoint(TypedDict):
    text: str
    papers: List[PaperRef]  

class Section(TypedDict):
    number: int
    title: str
    outline: str
    key_points: List[KeyPoint]

class Plan(TypedDict):
    plan: List[Section]
    reasoning: str

@dataclass(kw_only=True)
class LitState:
    messages: list
    info: Optional[dict[str, Any]] = field(default=None)
    # documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)

    topic: str
    paper_recency: str   

    search_queries: List[str]    
    plan: Plan    

    draft_sections: List[str]          
    verified_sections: List[str]

    def print_plan(self, *, include_papers: bool = True) -> str:
        """
        Nicely format the reasoning and the plan for quick inspection.
        """
        print("\n\nResearch Plan & Reasoning:\n")

        if not self.plan or not self.plan.get("plan"):
            msg = "⚠️  Plan is empty."
            print(msg)
            return msg

        lines: List[str] = []

        # Print reasoning
        reasoning = self.plan.get("reasoning", "").strip()
        if reasoning:
            lines.append("Reasoning:\n" + indent(reasoning, "  "))
            lines.append("-" * 60)

        # Print plan sections
        for section in self.plan["plan"]:
            lines.append(f"Section {section['number']}: {section['title']}")
            lines.append(indent(section['outline'], "  "))
            for kp in section['key_points']:
                lines.append(indent(f"• {kp['text']}", "  "))
                if include_papers:
                    for p in kp["papers"]:
                        # Show all fields, safely handling missing ones
                        paper_str = f"- {p['title']} ({p['year']}) <{p['url']}>"
                        # Add citation_reason and summary if available
                        citation_reason = p.get("citation_reason") or p.get("comment") or ""
                        summary = p.get("summary") or ""
                        if citation_reason:
                            paper_str += f"\n      • Citation reason: {citation_reason}"
                        if summary:
                            paper_str += f"\n      • Summary: {summary}"
                        lines.append(indent(paper_str, "      "))
            lines.append("")  # blank line between sections

        formatted = "\n".join(lines)
        print(formatted)
        return formatted

    def print_messages(
        self,
        *,
        max_chars: int | None = None,
        prettify_json: bool = True,
    ) -> str:
        """
        Pretty‑print the full `self.messages` history.

        • Shows       USER / ASSISTANT text
        • Shows → TOOL <name> ARGS          (arguments the assistant sent)
        • Shows TOOL[<name>] OUT            (tool output)
        """
        from textwrap import indent
        import json, re
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

        if not self.messages:
            print("⚠️  No messages recorded.")
            return ""

        lines: list[str] = ["\nConversation log\n" + "─" * 60]
        for i, msg in enumerate(self.messages, 1):

            # ------------- human -----------------
            if isinstance(msg, HumanMessage):
                lines.append(f"{i:02d}  [USER]\n{indent(_fmt(msg.content), '   ')}")

            # ------------- assistant -------------
            elif isinstance(msg, AIMessage):
                if msg.content:
                    lines.append(f"{i:02d}  [ASSISTANT]\n{indent(_fmt(msg.content), '   ')}")
                # any tool calls?
                for tc in msg.tool_calls or []:
                    args = _fmt(json.dumps(tc["args"]))
                    lines.append(
                        f"{i:02d}  [ASSISTANT → TOOL {tc['name']} ARGS]\n{indent(args, '   ')}"
                    )

            # ------------- tool output -----------
            elif isinstance(msg, ToolMessage):
                out = _fmt(str(msg.content))
                lines.append(
                    f"{i:02d}  [TOOL {msg.name}] OUT\n{indent(out, '   ')}"
                )

            # ------------- other -----------------
            else:
                lines.append(f"{i:02d}  [{msg.__class__.__name__.upper()}]\n{indent(_fmt(str(msg.content)), '   ')}")

        log = "\n".join(lines)
        print(log)
        return log
