from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi


class CustomBM25Retriever(BaseRetriever):
    """BM25 retriever using rank-bm25 (LangChain compatible)"""

    documents: list[Document]
    bm25: BM25Okapi
    corpus: list[list[str]]
    k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, documents: list[Document], k: int = 3):
        """Initialize BM25 retriever"""
        # Tokenize documents
        corpus = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(corpus)

        super().__init__(documents=documents, bm25=bm25, corpus=corpus, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> list[Document]:
        """Retrieve documents matching a query"""

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.k
        ]

        # Return top documents
        results = [
            self.documents[i]
            for i in top_indices
            if scores[i] > 0  # Only include non-zero scores
        ]

        return results
