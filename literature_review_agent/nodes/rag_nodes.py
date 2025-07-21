from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from literature_review_agent.state import LitState, Plan
from literature_review_agent.utils import reduce_docs

import re


async def prepare_rag_knowledge_base(state: LitState, *, config=None) -> dict:
    print("Preparing the RAG setup...")

    # 1. Collect all unique arXiv URLs from the plan
    arxiv_urls = set()
    plan_dict: Plan = state.get("plan", {})
    for section in plan_dict.get("plan", []): 
        for kp in section["key_points"]:
            for paper in kp["papers"]:
                url = paper.get("url")
                if url and "arxiv.org" in url:
                    # Removing version suffix (e.g., v1, v2) for clean querying
                    url = re.sub(r'v\d+$', '', url)
                    arxiv_urls.add(url)

    # 2. Download all full-text PDFs
    docs = []
    for url in arxiv_urls:
        arxiv_id = url.split("/")[-1]
        try:
            loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_full_text=True)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    # 3. Chunk the docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # 4. Embed and index (batching because of the total size)
    embeddings = OpenAIEmbeddings()
    max_chunks_per_batch = 250
    db = None
    for i in range(0, len(split_docs), max_chunks_per_batch):
        batch = split_docs[i:i + max_chunks_per_batch]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    print("Rag setup ready...\n")


    # TODO: move the rag to the refinement subgraph, do not use memory for teh subgraph
    # figure out how to handle the rag better
    return {
        # "retriever": db.as_retriever(),
        "documents": reduce_docs(state.get("documents"), split_docs)
    }

# NOTE: for later
# def get_context_for_section(section, retriever, k=8):
#     query = section['outline'] + "\n" + " ".join([kp["text"] for kp in section["key_points"]])
#     docs = retriever.get_relevant_documents(query)  # or .similarity_search(query, k=k)
#     return "\n".join([doc.page_content for doc in docs])