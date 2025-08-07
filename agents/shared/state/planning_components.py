from pydantic import BaseModel, Field
from typing import List
from textwrap import indent

class PaperRef(BaseModel):
    title: str     
    year: int
    url: str  
    summary: str
    citation_reason: str

class KeyPoint(BaseModel):
    text: str
    papers: List[PaperRef] = Field(default_factory=list)

class SectionPlan(BaseModel):
    number: int
    title: str
    outline: str
    key_points: List[KeyPoint] = Field(default_factory=list)

class Plan(BaseModel):
    plan: List[SectionPlan] = Field(default_factory=list)
    reasoning: str = ""
    
    def print_plan(self, include_papers: bool = True) -> str:
        """
        Nicely format the reasoning and the plan for quick inspection.
        """
        print("\n\nResearch Plan & Reasoning:\n")
        lines: List[str] = []

        # Print reasoning
        reasoning = self.reasoning.strip()
        if reasoning:
            lines.append("Reasoning:\n" + indent(reasoning, "  "))
            lines.append("-" * 60)

        # Print plan sections
        for section in self.plan:
            lines.append(f"Section {section.number}: {section.title}")
            lines.append(indent(section.outline, "  "))
            for kp in section.key_points:
                lines.append(indent(f"• {kp.text}", "  "))
                if include_papers:
                    for p in kp.papers:
                        # Show all fields
                        paper_str = f"- {p.title} ({p.year}) <{p.url}>"
                        # Add citation_reason and summary
                        citation_reason = p.citation_reason or ""
                        summary = p.summary or ""
                        if citation_reason:
                            paper_str += f"\n      • Citation reason: {citation_reason}"
                        if summary:
                            paper_str += f"\n      • Summary: {summary}"
                        lines.append(indent(paper_str, "      "))
            lines.append("")  # blank line between sections

        formatted = "\n".join(lines)
        print(formatted)
        return formatted