#!/usr/bin/env python3
"""Generate graph diagrams for all LangGraph workflows."""

import sys
import base64
import httpx
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_workflow.graph import graph as main_graph
from agentic_workflow.planning.graph import planning_graph
from agentic_workflow.refinement.graph import refinement_graph
from agentic_workflow.overleaf.graph import overleaf_graph


def generate_diagrams(output_dir: Path = None):
    """Generate Mermaid (.mmd) and PNG diagrams for all graphs."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    # (graph, xray, layout) - LR=horizontal, TD=vertical
    graphs = {
        "main_graph": (main_graph, False, "TD"),
        "planning_graph": (planning_graph, True, "TD"),
        "refinement_graph": (refinement_graph, True, "TD"),
        "overleaf_graph": (overleaf_graph, True, "TD"),
    }

    for name, (graph, xray, layout) in graphs.items():
        print(f"Generating {name} (xray={xray}, layout={layout})...")

        mermaid = graph.get_graph(xray=xray).draw_mermaid()

        # Apply layout
        if layout == "LR":
            mermaid = mermaid.replace("graph TD;", "graph LR;")

        # Replace config with compact settings
        old_config = """---
config:
  flowchart:
    curve: linear
---"""
        new_config = f"""---
config:
  flowchart:
    curve: linear
    nodeSpacing: 15
    rankSpacing: 25
---"""
        mermaid = mermaid.replace(old_config, new_config)

        mmd_path = output_dir / f"{name}.mmd"
        mmd_path.write_text(mermaid)
        print(f"  Saved: {mmd_path}")

        # Generate PNG with LR layout using mermaid.ink API
        mermaid_encoded = base64.b64encode(mermaid.encode("utf8")).decode("ascii")
        png_bytes = httpx.get(f"https://mermaid.ink/img/{mermaid_encoded}").content

        png_path = output_dir / f"{name}.png"
        png_path.write_bytes(png_bytes)
        print(f"  Saved: {png_path}")


    print("\nDone!")


if __name__ == "__main__":
    generate_diagrams()
