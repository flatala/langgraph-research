"""Export node for creating the final ZIP file."""

import zipfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from langchain_core.runnables import RunnableConfig

from agents.shared.state.main_state import AgentState

logger = logging.getLogger(__name__)


async def create_zip_export(state: AgentState, *, config: Optional[RunnableConfig] = None) -> Dict:
    """Create ZIP file with main.tex and references.bib."""
    logger.info("Creating ZIP export...")

    progress = state.overleaf_progress
    output_dir = "exports"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"review_{state.review_id[:8]}_{timestamp}.zip"
    zip_path = output_path / zip_filename

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", progress.latex_content or "")
        zf.writestr("references.bib", progress.bibtex_content or "")

    logger.info(f"LaTeX export saved to: {zip_path}")

    return {"latex_export_path": str(zip_path)}
