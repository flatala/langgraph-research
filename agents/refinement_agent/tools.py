"""Tools for refinement agent, particularly for grounding revision with RAG."""
from langchain_core.tools import tool
from data.vector_store.manager import VectorStoreManager
from typing import List, Callable
import logging

logger = logging.getLogger(__name__)


def create_search_paper_fragments_tool(
    review_id: str,
    available_paper_ids: List[str],
    embeddings
) -> Callable:
    """Factory function to create a search_paper_fragments tool with bound context.

    Args:
        review_id: The review UUID for accessing vector collections
        available_paper_ids: List of arXiv IDs that are available for this subsection
        embeddings: The embedding model instance

    Returns:
        A tool function that can search paper fragments
    """
    vector_manager = VectorStoreManager()

    @tool
    def search_paper_fragments(paper_id: str, query: str) -> str:
        """Search for relevant fragments in a paper's vector store.

        Use this tool to find supporting evidence from papers when fixing grounding issues.

        Args:
            paper_id: The arXiv ID of the paper to search (e.g., "2401.12345")
            query: The search query to find relevant content

        Returns:
            Relevant text fragments from the paper that match the query
        """
        # Validate paper_id is available for this subsection
        if paper_id not in available_paper_ids:
            available = ", ".join(available_paper_ids) if available_paper_ids else "none"
            return f"Error: Paper '{paper_id}' is not available for this subsection. Available papers: {available}"

        # Load vector store for the paper
        vectorstore = vector_manager.load_collection(review_id, paper_id, embeddings)

        if vectorstore is None:
            return f"Error: Could not load vector store for paper '{paper_id}'"

        # Search for relevant fragments
        try:
            docs = vectorstore.similarity_search(query, k=5)
            if not docs:
                return f"No relevant fragments found for query: '{query}'"

            fragments = []
            for i, doc in enumerate(docs, 1):
                fragments.append(f"Fragment {i}:\n{doc.page_content}")

            logger.info(f"Found {len(docs)} fragments for query '{query[:50]}...' in paper {paper_id}")
            return "\n\n".join(fragments)

        except Exception as e:
            logger.error(f"Error searching paper {paper_id}: {e}")
            return f"Error searching paper: {str(e)}"

    return search_paper_fragments
