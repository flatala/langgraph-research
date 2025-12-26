"""Temporary paper document cache for a single review session."""
from pathlib import Path
from typing import Optional
import pickle
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class PaperCache:
    """Cache for downloaded paper documents during a review session."""

    def __init__(self, review_id: str):
        self.review_id = review_id
        self.cache_dir = Path(__file__).parent / review_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, arxiv_id: str) -> Optional[Document]:
        """Get cached paper document if it exists."""
        cache_file = self.cache_dir / f"{arxiv_id}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def save(self, arxiv_id: str, document: Document) -> None:
        """Save paper document to cache."""
        cache_file = self.cache_dir / f"{arxiv_id}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(document, f)

    def cleanup(self) -> None:
        """Delete all cached papers for this review."""
        if self.cache_dir.exists():
            for file in self.cache_dir.iterdir():
                file.unlink()
            self.cache_dir.rmdir()
            logger.info(f"Cleaned up temporary paper cache for review {self.review_id}")
