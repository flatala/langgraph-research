from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.shared.state import Plan
from textwrap import indent
from typing import List

def print_plan(plan: Plan, include_papers: bool = True) -> str:
    """
    Nicely format the reasoning and the plan for quick inspection.
    """
    print("\n\nResearch Plan & Reasoning:\n")
    lines: List[str] = []

    # Print reasoning
    reasoning = plan.get("reasoning", "").strip()
    if reasoning:
        lines.append("Reasoning:\n" + indent(reasoning, "  "))
        lines.append("-" * 60)

    # Print plan sections
    for section in plan["plan"]:
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

def print_messages(messages: List, max_chars: int | None = None, prettify_json: bool = True,) -> str:
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

    lines: list[str] = ["\nConversation log\n" + "─" * 60]
    for i, msg in enumerate(messages, 1):

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