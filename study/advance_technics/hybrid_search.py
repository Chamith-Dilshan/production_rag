"""
Need to focus on only get what matters for the asked question.
Quality over Quantity.

Why vector search failed in production ->
1. Exact Identifiers Without Semantic Content
Product codes, SKUs, and IDs like SKU-9847-XL have no semantic meaning to embedding models.
They're just character sequences. When you search for a specific product ID, the embedding captures
the character patterns, but semantically identical products with different IDs won't match.
Vector search is similarity-based, not identity-based.

Real impact: You search for SKU-001 but the document uses product_id: 001. They're the same thing,
but vector search treats them as different.

2. Acronyms and Abbreviations
Acronyms like WCAG, GDPR, HIPAA, or API often appear without explanation in documents.
Embedding models see them as rare tokens with limited training context. When you search for
"WCAG compliance," the model understands "compliance" semantically but treats WCAG as
just a sequence of letters.

Real impact: A document says, "WCAG 2.1 requirements" but your search for "Web Content
Accessibility Guidelines" finds nothing because the model never learned
that WCAG = Web Content Accessibility Guidelines.

3. Proper Nouns and Specific Names like "John Smith," "MySQL," or "AWS Lambda" are proper nouns
that need exact matching, not semantic similarity. Vector search will return documents mentioning
people or systems with similar-sounding names, not the exact person or product you need.

Real impact: You're looking for "John Smith's Q3 report," but vector search returns documents
about "Jane Smith's Q3 report" or "John Doe's accounting files" because it matches on semantic
similarity rather than exact identity.

4. Project-Specific Error Codes like E_DD_APIFAILED, ERR_500_DB, or TIMEOUT_001 are internal,
non-standard strings. Embedding models have no training data on your company's error code system.
They can't understand what these codes mean.

Real impact: When a customer reports "Error E_DD_APIFAILED," you search your knowledge base for
that code, but vector search returns irrelevant documents because the model sees it as meaningless
characters, not as representing "API failure in the database module."

5. Numbers and Versions
Version numbers like 3.2.1, v2.0, or dates like 2024-09-10 are often treated as noise by embedding models.
Semantic similarity won't distinguish between v1.5 and v2.0 effectively.

Real impact: You need documentation for Python 3.9, but vector search returns results for Python 3.8 or
Python 3.10 because numerical differences are semantically "close" but functionally incompatible.

# Vector Search vs. BM25 Search

## Vector Search (Semantic)

**How it works:** Converts text to embeddings and finds similar vectors. Understands meaning.

**Strengths:**
- Typos & variations: ✅ Works (`complianc` finds `compliance`)
- Natural language queries: ✅ Works ("Tell me about access rules" finds WCAG docs)
- Synonyms: ✅ Works (`car` finds `automobile`)

**Weaknesses:**
- Exact codes: ❌ Fails (`HGC-34dsAS23` is just noise)
- Acronyms: ❌ Fails (`WCAG` treated as random letters)
- Error codes: ❌ Fails (`E_DD_APIFAILED` is meaningless)

**Cost:** Higher (embeddings needed)
**Speed:** Slower (embedding computation)

---

## BM25 (Keyword Search)

**How it works:** Ranks documents based on keyword frequency and relevance. Exact term matching.

**Strengths:**
- Exact codes: ✅ Works (`HGC-34dsAS23` matches immediately)
- Acronyms: ✅ Works (matches `WCAG` exactly)
- Error codes: ✅ Works (matches exactly)

**Weaknesses:**
- Typos & variations: ❌ Fails (typo = no match)
- Natural language queries: ❌ Mediocre (needs exact keywords)
- Synonyms: ❌ Fails (no synonym understanding)

**Cost:** Lower (lightweight)
**Speed:** Fast (simple term matching)

---

### Use hybrid when:
- Enterprice data with codes, IDs, acronyms, error codes
- Technical documentation
- Legal documents
- Mixed query types
- Accuracy is critical

### Skip hybrid when:
- Simple Q&A chatbot
- Creative writting assistant
- Quick prototypes
- Latency critical (add ~20-50ms)

### Use Both (Hybrid) ← Best for production:

Combine BM25 and Vector search results using Reciprocal Rank Fusion (RRF).
    RRF formula: score = 1 / (k + rank)

BM25 doesn't support incremental updates, so we need to rebuild it
when adding new docs.

K value -> retrieve more values and let RFF sort it.(K= 4 or higher recommend)
"""

import tempfile

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever

# from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from study.fixses.custom_bm25_retriever import CustomBM25Retriever

embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# Test documents for hybrid search patterns
documents = [
    Document(
        id="DOC-001",
        page_content="WCAG 2.1 Level AA Compliance Guidelines: Web Content Accessibility Guidelines (WCAG) 2.1 provides the technical standards for making web content accessible to people with disabilities. Level AA conformance requires meeting all Level A criteria plus additional Level AA criteria.",
        metadata={
            "id": "DOC-001",
            "title": "WCAG Accessibility Standards",
            "section": "Compliance",
            "version": "2.1",
            "source": "wcag_guidelines.pdf",
            "page": 1,
            "type": "compliance",
        },
    ),
    Document(
        id="DOC-002",
        page_content="Error Code E_DD_APIFAILED: The API request failed due to network timeout. Error Code E_DD_INVALID_TOKEN: The authentication token is invalid or expired. Error Code E_DD_RATELIMIT: Rate limit exceeded. Please retry after 60 seconds.",
        metadata={
            "id": "DOC-002",
            "title": "API Error Codes Reference",
            "section": "Error Handling",
            "error_codes": ["E_DD_APIFAILED", "E_DD_INVALID_TOKEN", "E_DD_RATELIMIT"],
            "source": "api_errors.pdf",
            "page": 5,
            "type": "config",
        },
    ),
    Document(
        id="DOC-003",
        page_content="Product SKU-HGC-34dsAS23 is a high-capacity storage device with 2TB capacity. Related SKUs: SKU-HGC-16dsAS23 (1TB), SKU-HGC-8dsAS23 (512GB). All models support USB 3.0 and come with warranty coverage.",
        metadata={
            "id": "DOC-003",
            "title": "Hardware Product Catalog",
            "section": "Storage Devices",
            "skus": ["SKU-HGC-34dsAS23", "SKU-HGC-16dsAS23", "SKU-HGC-8dsAS23"],
            "source": "product_catalog.pdf",
            "page": 12,
            "type": "hardware",
        },
    ),
    Document(
        id="DOC-004",
        page_content="How to ensure data security and compliance with modern standards. Implementing best practices for user authentication, encryption protocols, and regular security audits. Organizations must maintain compiance with industry regulations and conduct penetration testing quarterly.",
        metadata={
            "id": "DOC-004",
            "title": "Data Security Best Practices",
            "section": "Security",
            "keywords": ["security", "encryption", "authentication"],
            "source": "security_guide.pdf",
            "page": 8,
            "type": "compliance",
        },
    ),
    Document(
        id="DOC-005",
        page_content="John Smith, the lead architect at TechCorp, designed the microservices infrastructure. Other key team members include Jane Doe (DevOps), Michael Johnson (Backend Lead), and Sarah Williams (Frontend Lead). The team follows agile methodology with 2-week sprints.",
        metadata={
            "id": "DOC-005",
            "title": "Team Structure and Roles",
            "section": "Organization",
            "team_members": [
                "John Smith",
                "Jane Doe",
                "Michael Johnson",
                "Sarah Williams",
            ],
            "company": "TechCorp",
            "source": "team_handbook.pdf",
            "page": 3,
            "type": "project",
        },
    ),
    Document(
        id="DOC-006",
        page_content="Router configuration: 10.0.0.1/24, 192.168.0.1/24, 172.16.0.1/24. Switch configuration: 10.0.0.2, 192.168.0.2, 172.16.0.2.",
        metadata={
            "id": "DOC-006",
            "title": "Company Annual Report",
            "section": "Financials",
            "year": 2024,
            "source": "annual_report.pdf",
            "page": 1,
            "type": "config",
        },
    ),
]


# def create_retrievers():
#     # create the vector store
#     vector_store = Chroma.from_documents(
#         documents=documents,
#         collection_name="example_collection",
#         embedding=embedding_model,
#         persist_directory=tempfile.mkdtemp(),  # Where to save data locally, remove if not necessary
#     )
#
#     # create the vector retriever
#     vector_retriever = vector_store.as_retriever(
#         search_type="similarity", search_kwargs={"k": 3}
#     )
#
#     # create a BN25 retriever
#     # bm25_retriever = BM25Retriever.from_documents(documents=documents, k=3)
#
#     # Initialize retriever
#     bm25_retriever = CustomBM25Retriever(documents)
#
#     # combined with EsembleRetriever
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
#     )
#
#     return vector_retriever, bm25_retriever, ensemble_retriever


# def test_query(query, name, retriever):
#     results = retriever.invoke(query)
#     print(f"\nQuery: {query}\n{name} Retriever Results:\n")
#     for i, doc in enumerate(results):
#         print(f"{i + 1}. {doc.page_content[:80]}...")
#
#     return results


def create_retrievers(documents, embedding_model):
    """Create vector, BM25, and ensemble retrievers"""

    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=documents,
        collection_name="example_collection",
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )

    print("Creating vector retriever...")
    vector_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    print("Creating BM25 retriever...")
    bm25_retriever = CustomBM25Retriever(documents, k=4)

    print("Creating ensemble retriever...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    return vector_retriever, bm25_retriever, hybrid_retriever


def test_query(query: str, name: str, retriever):
    """Test a retriever with a query"""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{name} Retriever Results:")
    print(f"{'=' * 60}")

    results = retriever.invoke(query)

    if not results:
        print("No results found.")
        return results

    for i, doc in enumerate(results, 1):
        doc_id = doc.metadata.get("id", "N/A")
        content = doc.page_content[:100].replace("\n", " ")
        print(f"\n{i}. [ID: {doc_id}]")
        print(f"   {content}...")

    return results


if __name__ == "__main__":
    # Create retrievers
    vector_retriever, bm25_retriever, hybrid_retriever = create_retrievers(
        documents, embedding_model
    )

    # Test queries
    test_queries = [
        "Fix router configuration",
        "E_DD_APIFAILED error",
        "API security best practices",
    ]

    for query in test_queries:
        test_query(query, "Vector", vector_retriever)
        test_query(query, "BM25", bm25_retriever)
        test_query(query, "Hybrid", hybrid_retriever)
