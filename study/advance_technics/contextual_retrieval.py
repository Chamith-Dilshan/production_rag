"""
Contextual Retrieval (Anthropic's Technique)

Problem: When you chunk documents, chunks lose their context.
"The company was founded in 1994" - Which company? What document is this from?

Solution: Use an LLM to prepend context to each chunk BEFORE embedding.
Result: 67% fewer retrieval failures (Anthropic research).

This technique was introduced by Anthropic in late 2024 and has become
a standard practice for production RAG systems.
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# THE PROBLEM: CHUNKS LOSE CONTEXT
# ============================================================


def demonstrate_context_loss():
    """
    Show how chunking causes context loss.
    """

    print("=" * 60)
    print("THE PROBLEM: Chunks Lose Context")
    print("=" * 60)

    # Original document
    full_document = """
    ACME Corporation Annual Report 2025

    Company Overview:
    ACME Corporation was founded in 1994 in San Francisco. The company
    specializes in manufacturing industrial equipment for the mining sector.

    Financial Highlights:
    Revenue for fiscal year 2025 reached $4.2 billion, representing a 15%
    increase from the previous year. The company's profit margin improved
    to 18%, up from 14% in 2024.

    Future Outlook:
    The company plans to expand into renewable energy equipment in 2026.
    A new manufacturing facility will open in Austin, Texas. The company
    expects revenue growth of 20% in the coming fiscal year.
    """

    # Simulate chunking
    chunks = [
        "The company specializes in manufacturing industrial equipment for the mining sector.",
        "Revenue for fiscal year 2025 reached $4.2 billion, representing a 15% increase from the previous year.",
        "The company plans to expand into renewable energy equipment in 2026.",
    ]

    print("\nOriginal Document: ACME Corporation Annual Report 2025")
    print("\nAfter chunking, we get isolated pieces like:")
    for i, chunk in enumerate(chunks, 1):
        print(f'\n  Chunk {i}: "{chunk}"')

    print("\n" + "-" * 60)
    print("THE PROBLEM:")
    print("-" * 60)
    print(
        """
    - Chunk 1: "The company" - Which company?
    - Chunk 2: "fiscal year 2025" - For what company?
    - Chunk 3: "The company plans" - Plans of what company?

    When a user asks "What is ACME's revenue?", semantic search might
    not match Chunk 2 because "ACME" doesn't appear in the chunk!
    """
    )


# ============================================================
# THE SOLUTION: CONTEXTUAL RETRIEVAL
# ============================================================


def add_contextual_prefix(
    chunk: str, full_document: str, document_title: str, llm: ChatGroq
) -> str:
    """
    Use LLM to generate a contextual prefix for a chunk.

    This is the core of Contextual Retrieval:
    - Give the LLM the full document and the specific chunk
    - Ask it to write a brief context that situates the chunk
    - Prepend this context to the chunk before embedding
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at adding context to document chunks.

Given a document and a specific chunk from that document, write a SHORT
context (1-2 sentences) that helps situate the chunk within the document.

The context should include:
- The document title/source if relevant
- Key entities mentioned earlier in the document
- Any important context that helps understand this chunk

Keep it brief - just enough to disambiguate the chunk.
Output ONLY the contextual prefix, nothing else.""",
            ),
            (
                "human",
                """Document Title: {title}

Full Document:
{document}

Chunk to contextualize:
{chunk}

Write a brief contextual prefix for this chunk:""",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {"title": document_title, "document": full_document, "chunk": chunk}
    )

    return response.content


def demonstrate_contextual_retrieval():
    """
    Show how Contextual Retrieval solves the context loss problem.
    """

    print("\n" + "=" * 60)
    print("THE SOLUTION: Contextual Retrieval")
    print("=" * 60)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
    )

    # Full document for context
    full_document = """
    ACME Corporation Annual Report 2025

    Company Overview:
    ACME Corporation was founded in 1994 in San Francisco. The company
    specializes in manufacturing industrial equipment for the mining sector.

    Financial Highlights:
    Revenue for fiscal year 2025 reached $4.2 billion, representing a 15%
    increase from the previous year. The company's profit margin improved
    to 18%, up from 14% in 2024.

    Future Outlook:
    The company plans to expand into renewable energy equipment in 2026.
    A new manufacturing facility will open in Austin, Texas. The company
    expects revenue growth of 20% in the coming fiscal year.
    """

    document_title = "ACME Corporation Annual Report 2025"

    # Original chunks (without context)
    original_chunks = [
        "The company specializes in manufacturing industrial equipment for the mining sector.",
        "Revenue for fiscal year 2025 reached $4.2 billion, representing a 15% increase from the previous year.",
        "The company plans to expand into renewable energy equipment in 2026.",
    ]

    print("\nAdding contextual prefixes to each chunk...")
    print("-" * 60)

    contextualized_chunks = []
    for i, chunk in enumerate(original_chunks, 1):
        print(f"\nProcessing chunk {i}...")

        # Generate contextual prefix
        context_prefix = add_contextual_prefix(
            chunk=chunk,
            full_document=full_document,
            document_title=document_title,
            llm=llm,
        )

        # Combine prefix + original chunk
        contextualized = f"{context_prefix} {chunk}"
        contextualized_chunks.append(contextualized)

        print(f'  Original: "{chunk[:50]}..."')
        print(f'  Context:  "{context_prefix}"')
        print(f'  Combined: "{contextualized[:80]}..."')

    return original_chunks, contextualized_chunks


# ============================================================
# RETRIEVAL COMPARISON
# ============================================================


def compare_retrieval(original_chunks: list, contextualized_chunks: list):
    """
    Compare retrieval quality between original and contextualized chunks.
    """

    print("\n" + "=" * 60)
    print("RETRIEVAL COMPARISON")
    print("=" * 60)

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # Create vector stores
    original_docs = [
        Document(page_content=c, metadata={"type": "original"}) for c in original_chunks
    ]
    context_docs = [
        Document(page_content=c, metadata={"type": "contextual"})
        for c in contextualized_chunks
    ]

    vs_original = Chroma.from_documents(
        documents=original_docs, embedding=embeddings, collection_name="original_chunks"
    )

    vs_contextual = Chroma.from_documents(
        documents=context_docs,
        embedding=embeddings,
        collection_name="contextual_chunks",
    )

    # Test queries
    test_queries = [
        "What is ACME's revenue?",
        "What does ACME Corporation manufacture?",
        "What are ACME's expansion plans?",
    ]

    print("\nTesting retrieval with different queries:")
    print("-" * 60)

    for query in test_queries:
        print(f'\nQuery: "{query}"')

        # Search original
        orig_results = vs_original.similarity_search_with_score(query, k=1)
        orig_doc, orig_score = orig_results[0]

        # Search contextual
        ctx_results = vs_contextual.similarity_search_with_score(query, k=1)
        ctx_doc, ctx_score = ctx_results[0]

        print(f"\n  Original chunks (score {orig_score:.3f}):")
        print(f'    "{orig_doc.page_content[:60]}..."')

        print(f"\n  Contextual chunks (score {ctx_score:.3f}):")
        print(f'    "{ctx_doc.page_content[:60]}..."')

        # Lower score = better match in Chroma
        if ctx_score < orig_score:
            improvement = ((orig_score - ctx_score) / orig_score) * 100
            print(f"\n  -> Contextual retrieval {improvement:.1f}% better match!")

    # Cleanup
    vs_original.delete_collection()
    vs_contextual.delete_collection()


# ============================================================
# PRODUCTION IMPLEMENTATION
# ============================================================


def create_contextual_chunks(
    documents: list[Document],
    llm: ChatGroq,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Production-ready function to create contextualized chunks.

    This is the function you'd use in a real RAG pipeline.
    Note: This adds latency and cost during indexing, but it's a one-time cost.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    contextualized_docs = []

    for doc in documents:
        # Get full document content and title
        full_content = doc.page_content
        doc_title = doc.metadata.get(
            "title", doc.metadata.get("source", "Unknown Document")
        )

        # Split into chunks
        chunks = text_splitter.split_text(full_content)

        # Contextualize each chunk
        for i, chunk in enumerate(chunks):
            # Generate context prefix
            context_prefix = add_contextual_prefix(
                chunk=chunk,
                full_document=full_content,
                document_title=doc_title,
                llm=llm,
            )

            # Create new document with context
            contextualized_content = f"{context_prefix} {chunk}"

            contextualized_docs.append(
                Document(
                    page_content=contextualized_content,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "original_chunk": chunk,
                        "context_prefix": context_prefix,
                    },
                )
            )

    return contextualized_docs


def demo_production_pipeline():
    """
    Demonstrate a production-ready contextual retrieval pipeline.
    """

    print("\n" + "=" * 60)
    print("PRODUCTION PIPELINE DEMO")
    print("=" * 60)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
    )
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # Sample documents
    documents = [
        Document(
            page_content="""
            TechStartup Inc. Series B Funding Announcement

            TechStartup Inc., a leading AI infrastructure company based in Seattle,
            today announced the closing of its Series B funding round. The round
            raised $45 million, led by Sequoia Capital with participation from
            Andreessen Horowitz.

            The company plans to use the funds to expand its engineering team and
            accelerate product development. CEO Jane Smith stated that the company
            expects to double its headcount by end of 2026.

            TechStartup's flagship product, AIFlow, helps enterprises deploy and
            manage large language models in production. The platform currently
            serves over 200 enterprise customers.
            """,
            metadata={
                "title": "TechStartup Inc. Series B Announcement",
                "source": "press_release.pdf",
            },
        )
    ]

    print("\nStep 1: Creating contextualized chunks...")
    contextualized_docs = create_contextual_chunks(
        documents=documents, llm=llm, chunk_size=200, chunk_overlap=50
    )

    print(f"Created {len(contextualized_docs)} contextualized chunks")

    print("\nStep 2: Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=contextualized_docs,
        embedding=embeddings,
        collection_name="contextual_demo",
    )

    print("\nStep 3: Testing retrieval...")
    query = "How much funding did the Seattle AI company raise?"

    results = vectorstore.similarity_search(query, k=2)

    print(f'\nQuery: "{query}"')
    print("\nTop results:")
    for i, doc in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Content: {doc.page_content[:100]}...")
        print(
            f"    Context prefix: {doc.metadata.get('context_prefix', 'N/A')[:50]}..."
        )

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "-" * 60)
    print("PRODUCTION CONSIDERATIONS:")
    print("-" * 60)
    print(
        """
    1. COST: ~$0.01-0.05 per document for context generation
       - One-time cost at indexing time
       - Much cheaper than retrieval failures

    2. LATENCY: Adds 1-2s per chunk during indexing
       - Batch process documents offline
       - No impact on query latency

    3. STORAGE: Chunks are ~20-30% larger
       - Minimal impact on vector DB costs

    4. WHEN TO USE:
       - Documents have important context in headers/titles
       - Entities are referenced with pronouns ("the company", "they")
       - Documents come from multiple sources
    """
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.2: CONTEXTUAL RETRIEVAL")
    print("=" * 60)
    print("\nAnthropic's technique to reduce retrieval failures by 67%")
    print("=" * 60)

    # 1. Show the problem
    demonstrate_context_loss()

    # 2. Show the solution
    original, contextualized = demonstrate_contextual_retrieval()

    # 3. Compare retrieval
    compare_retrieval(original, contextualized)

    # 4. Production demo
    demo_production_pipeline()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
    1. Chunks lose context when split from documents
    2. Contextual Retrieval adds context BEFORE embedding
    3. Use LLM to generate 1-2 sentence context prefix
    4. One-time cost at indexing, permanent retrieval improvement
    5. Anthropic reports 67% fewer retrieval failures
    6. Combine with BM25 hybrid search for best results
    """
    )
