from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from literature_review_agent.configuration import Configuration
from literature_review_agent.state import Plan

from typing import Any, Union, Optional, Literal, List
from dotenv import load_dotenv
from pathlib import Path
from textwrap import indent

import hashlib
import uuid


load_dotenv(                
    Path(__file__).resolve().parent.parent / ".env",
    override=False,         
)    

def get_orchestrator_llm(cfg: Configuration) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.orchestrator_model,
        api_key=cfg.openai_api_key,
        streaming=True,
    )

def get_text_llm(cfg: Configuration) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.text_model,
        api_key=cfg.openai_api_key,
        streaming=True,
    )

def _generate_uuid(page_content: str) -> str:
    """Generate a UUID for a document based on page content."""
    md5_hash = hashlib.md5(page_content.encode()).hexdigest()
    return str(uuid.UUID(md5_hash))

def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": _generate_uuid(new)})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid") or _generate_uuid(
                    item.get("page_content", "")
                )

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**{**item, "metadata": {**metadata, "uuid": item_id}})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid", "")
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list


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