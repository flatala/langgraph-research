#!/usr/bin/env python3
"""Generate graph diagrams for all LangGraph workflows."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.graph import graph as main_graph
from agents.planning_agent.graph import planning_graph
from agents.refinement_agent.graph import refinement_graph
from agents.overleaf.graph import overleaf_graph


def generate_diagrams(output_dir: Path = None):
    """Generate Mermaid (.mmd) and PNG diagrams for all graphs."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    graphs = {
        "main_graph": (main_graph, False),
        "planning_graph": (planning_graph, True),
        "refinement_graph": (refinement_graph, True),
        "overleaf_graph": (overleaf_graph, True),
    }

    for name, (graph, xray) in graphs.items():
        print(f"Generating {name} (xray={xray})...")

        mermaid = graph.get_graph(xray=xray).draw_mermaid()
        mmd_path = output_dir / f"{name}.mmd"
        mmd_path.write_text(mermaid)
        print(f"  Saved: {mmd_path}")

        png_bytes = graph.get_graph(xray=xray).draw_mermaid_png()
        png_path = output_dir / f"{name}.png"
        png_path.write_bytes(png_bytes)
        print(f"  Saved: {png_path}")


    print("\nDone!")


if __name__ == "__main__":
    generate_diagrams()
