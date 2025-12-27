"""ChromaDB vector store manager for paper embeddings."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manage ChromaDB vector collections for paper embeddings."""

    def __init__(self, base_path: str = "data/vector_store/collections"):
        """Initialize vector store manager.

        Args:
            base_path: Base directory for storing vector collections
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def get_collection_path(self, review_id: str, paper_id: str) -> str:
        """Get filesystem path for a specific paper's vector collection.

        Args:
            review_id: Review UUID
            paper_id: Paper arxiv_id

        Returns:
            Path to collection directory
        """
        return os.path.join(self.base_path, f"review_{review_id}", f"paper_{paper_id}")

    def collection_exists(self, review_id: str, paper_id: str) -> bool:
        """Check if a vector collection exists on disk.

        Args:
            review_id: Review UUID
            paper_id: Paper arxiv_id

        Returns:
            True if collection exists and has data
        """
        path = self.get_collection_path(review_id, paper_id)
        return os.path.exists(path) and len(os.listdir(path)) > 0

    def create_collection(self, review_id: str, paper_id: str,
                         documents: List[Document], embedding_function) -> Chroma:
        """Create and persist a new vector collection.

        Args:
            review_id: Review UUID
            paper_id: Paper arxiv_id
            documents: List of document chunks to embed
            embedding_function: Embedding model instance

        Returns:
            Chroma vectorstore instance
        """
        persist_directory = self.get_collection_path(review_id, paper_id)
        os.makedirs(persist_directory, exist_ok=True)

        collection_name = f"review_{review_id}_paper_{paper_id}"

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name
        )

        logger.info(f"Created vector collection: {len(documents)} chunks")
        return vectorstore

    def load_collection(self, review_id: str, paper_id: str,
                       embedding_function) -> Optional[Chroma]:
        """Load an existing vector collection from disk.

        Args:
            review_id: Review UUID
            paper_id: Paper arxiv_id
            embedding_function: Embedding model instance

        Returns:
            Chroma vectorstore instance or None if doesn't exist
        """
        if not self.collection_exists(review_id, paper_id):
            return None

        persist_directory = self.get_collection_path(review_id, paper_id)
        collection_name = f"review_{review_id}_paper_{paper_id}"

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )

        return vectorstore

    def get_or_create_collection(self, review_id: str, paper_id: str,
                                 documents: Optional[List[Document]],
                                 embedding_function) -> Chroma:
        """Get existing collection or create new one if it doesn't exist.

        Args:
            review_id: Review UUID
            paper_id: Paper arxiv_id
            documents: Documents to embed (required if creating new)
            embedding_function: Embedding model instance

        Returns:
            Chroma vectorstore instance

        Raises:
            ValueError: If collection doesn't exist and documents not provided
        """
        existing = self.load_collection(review_id, paper_id, embedding_function)

        if existing:
            return existing

        if documents is None:
            raise ValueError(
                f"Documents must be provided to create new collection for "
                f"review {review_id}, paper {paper_id}"
            )

        return self.create_collection(review_id, paper_id, documents, embedding_function)

    def delete_collection(self, review_id: str, paper_id: str):
        """Delete a vector collection from disk.

        Args:
            review_id: Review UUID
            paper_id: Paper arxiv_id
        """
        import shutil
        path = self.get_collection_path(review_id, paper_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"Deleted vector collection at {path}")

    def delete_review_collections(self, review_id: str):
        """Delete all vector collections for a review.

        Args:
            review_id: Review UUID
        """
        import shutil
        review_path = os.path.join(self.base_path, f"review_{review_id}")
        if os.path.exists(review_path):
            shutil.rmtree(review_path)
            logger.info(f"Deleted all collections for review {review_id}")
