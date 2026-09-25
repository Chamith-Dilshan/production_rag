"""
Lesson 6.3: Late Chunking

Traditional chunking: Split text → Embed each chunk separately
Late chunking: Embed a full document → Split embeddings

Why it matters:
- Traditional chunking loses cross-chunk context
- "He founded the company" - who is "he"? (lost in previous chunk)
- Late chunking preserves document-level context in embeddings

Result: 10-12% accuracy improvement on retrieval tasks.
"""

import os

import numpy as np
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# THE PROBLEM: EARLY CHUNKING LOSES CONTEXT
# ============================================================


def demonstrate_early_chunking_problem():
    """
    Show how traditional (early) chunking loses pronoun references.
    """

    print("=" * 60)
    print("THE PROBLEM: Early Chunking Loses Context")
    print("=" * 60)

    document = """
    Steve Jobs was born in San Francisco in 1955. He was adopted
    by Paul and Clara Jobs shortly after birth.

    He co-founded Apple Computer in 1976 with Steve Wozniak.
    The company started in his parents' garage. He served as CEO
    until 1985 when he was ousted from the company he created.

    He then founded NeXT Computer and acquired Pixar Animation Studios.
    These ventures would later prove instrumental in his return to Apple.

    He returned to Apple in 1997 and transformed it into one of the
    most valuable companies in the world. He introduced the iPod, iPhone,
    and iPad, revolutionizing multiple industries.
    """

    # Simulate early chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,  # No overlap to show the problem clearly
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_text(document)

    print("\nOriginal document is about Steve Jobs.")
    print("\nAfter early chunking:")
    print("-" * 60)

    for i, chunk in enumerate(chunks, 1):
        # Check if "Steve Jobs" appears in this chunk
        has_name = "Steve Jobs" in chunk
        pronoun_count = chunk.lower().count(" he ")

        print(f"\nChunk {i}:")
        print(f'  "{chunk.strip()[:80]}..."')
        print(
            f"  Contains 'Steve Jobs': {'Yes' if has_name else 'NO - Just pronouns!'}"
        )
        print(f"  Pronoun 'he' count: {pronoun_count}")

    print("\n" + "-" * 60)
    print("THE PROBLEM:")
    print("-" * 60)
    print(
        """
    Chunks 2, 3, 4 only contain "he" - no "Steve Jobs"!

    When embedding these chunks separately:
    - Query: "What companies did Steve Jobs found?"
    - Chunk 2 mentions "co-founded Apple" but says "He co-founded"
    - Embedding of "He co-founded" won't match "Steve Jobs founded"

    The pronoun "he" loses its referent when chunked separately.
    """
    )


# ============================================================
# EARLY VS LATE CHUNKING VISUALIZATION
# ============================================================


def visualize_chunking_approaches():
    """
    Visual comparison of early vs late chunking.
    """

    print("\n" + "=" * 60)
    print("EARLY vs LATE CHUNKING")
    print("=" * 60)

    diagram = """
    EARLY CHUNKING (Traditional):
    ┌─────────────────────────────────────────────────────────┐
    │                    Full Document                        │
    └─────────────────────────────────────────────────────────┘
                              │
                         [Split Text]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │ Chunk 1 │    │ Chunk 2 │    │ Chunk 3 │
         └────┬────┘    └────┬────┘    └────┬────┘
              │              │              │
         [Embed]        [Embed]        [Embed]
              │              │              │
              ▼              ▼              ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │Vector 1 │    │Vector 2 │    │Vector 3 │
         └─────────┘    └─────────┘    └─────────┘

    Each chunk embedded INDEPENDENTLY - no cross-chunk context!

    ═══════════════════════════════════════════════════════════

    LATE CHUNKING (Context-Preserving):
    ┌─────────────────────────────────────────────────────────┐
    │                    Full Document                        │
    └─────────────────────────────────────────────────────────┘
                              │
                    [Embed FULL Document]
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │            Full Document Token Embeddings               │
    │   [tok1] [tok2] [tok3] ... [tok_n]                     │
    │   Each token has FULL document context via attention    │
    └─────────────────────────────────────────────────────────┘
                              │
                    [Split Embeddings by Position]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │Vector 1 │    │Vector 2 │    │Vector 3 │
         │(pooled) │    │(pooled) │    │(pooled) │
         └─────────┘    └─────────┘    └─────────┘

    Each chunk vector KNOWS about the whole document!
    "He" in chunk 2 is embedded with knowledge that "He" = "Steve Jobs"
    """

    print(diagram)


# ============================================================
# SIMULATED LATE CHUNKING
# ============================================================


def simulate_late_chunking():
    """
    Simulate late chunking concept using OpenAI embeddings.

    Note: True late chunking requires access to token-level embeddings
    from models like Jina or specialized embedding models. This demo
    shows the CONCEPT using available tools.

    For production late chunking, consider:
    - Jina Embeddings with late chunking support
    - Custom embedding models with token-level output
    """

    print("\n" + "=" * 60)
    print("LATE CHUNKING SIMULATION")
    print("=" * 60)

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # Full document
    document = """
    Steve Jobs was born in San Francisco in 1955. He was adopted
    by Paul and Clara Jobs shortly after birth.

    He co-founded Apple Computer in 1976 with Steve Wozniak.
    The company started in his parents' garage.
    """

    # Chunks
    chunk1 = "Steve Jobs was born in San Francisco in 1955. He was adopted by Paul and Clara Jobs shortly after birth."
    chunk2 = "He co-founded Apple Computer in 1976 with Steve Wozniak. The company started in his parents' garage."

    # Query
    query = "What company did Steve Jobs found?"

    print(f'\nQuery: "{query}"')

    # Early chunking: embed chunks separately
    print("\n--- EARLY CHUNKING APPROACH ---")
    chunk2_embedding_early = embeddings.embed_query(chunk2)
    query_embedding = embeddings.embed_query(query)

    # Calculate similarity
    similarity_early = np.dot(chunk2_embedding_early, query_embedding)
    print(f'Chunk 2 (standalone): "{chunk2[:50]}..."')
    print(f"Similarity with query: {similarity_early:.4f}")

    # Late chunking simulation: embed chunk WITH context prepended
    # (This simulates having document context in the embedding)
    print("\n--- LATE CHUNKING SIMULATION ---")
    chunk2_with_context = (
        f"[Context: This is about Steve Jobs, founder of Apple] {chunk2}"
    )
    chunk2_embedding_late = embeddings.embed_query(chunk2_with_context)

    similarity_late = np.dot(chunk2_embedding_late, query_embedding)
    print(f'Chunk 2 (with context): "{chunk2_with_context[:60]}..."')
    print(f"Similarity with query: {similarity_late:.4f}")

    improvement = ((similarity_late - similarity_early) / similarity_early) * 100
    print(f"\nImprovement: {improvement:.1f}%")

    print("\n" + "-" * 60)
    print("NOTE: This is a SIMULATION using context prepending.")
    print("True late chunking uses token-level embeddings from")
    print("models like Jina that support this natively.")


# ============================================================
# PRACTICAL IMPLEMENTATION OPTIONS
# ============================================================


def show_implementation_options():
    """
    Show practical ways to implement late chunking concepts.
    """

    print("\n" + "=" * 60)
    print("PRACTICAL IMPLEMENTATION OPTIONS")
    print("=" * 60)

    options = """
    OPTION 1: Jina Embeddings (Native Late Chunking Support)
    ─────────────────────────────────────────────────────────
    ```python
    from langchain_community.embeddings import JinaEmbeddings

    embeddings = JinaEmbeddings(
        model_name="jina-embeddings-v3",
        late_chunking=True  # Enable late chunking
    )
    ```
    Pros: Native support, best quality
    Cons: Different model, may need API key


    OPTION 2: Contextual Retrieval (Previous Lesson)
    ─────────────────────────────────────────────────────────
    ```python
    # Prepend LLM-generated context to each chunk
    context = generate_context(chunk, full_document)
    contextualized_chunk = f"{context} {chunk}"
    embedding = embeddings.embed_query(contextualized_chunk)
    ```
    Pros: Works with any embedding model
    Cons: Added LLM cost at indexing time


    OPTION 3: Parent Document Retriever
    ─────────────────────────────────────────────────────────
    ```python
    from langchain.retrievers import ParentDocumentRetriever

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    ```
    Pros: Returns full context at retrieval time
    Cons: Doesn't improve embedding quality


    OPTION 4: Overlap + Summary Headers
    ─────────────────────────────────────────────────────────
    ```python
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100  # Overlap captures some context
    )

    # Add document summary to each chunk
    for chunk in chunks:
        chunk.page_content = f"[From: {doc_title}] {chunk.page_content}"
    ```
    Pros: Simple, no extra cost
    Cons: Limited context preservation


    RECOMMENDATION:
    ─────────────────────────────────────────────────────────
    1. Start with Contextual Retrieval (Lesson 6.2)
    2. If using Jina, enable native late chunking
    3. Combine with Parent Document Retriever for best results
    4. Always use meaningful chunk overlap (10-20%)
    """

    print(options)


# ============================================================
# COMPARISON TABLE
# ============================================================


def print_comparison():
    """
    Print comparison of chunking approaches.
    """

    print("\n" + "=" * 60)
    print("CHUNKING APPROACHES COMPARISON")
    print("=" * 60)

    comparison = """
    ┌───────────────────┬────────────────────┬─────────────────────┐
    │ Approach          │ Context Quality    │ Implementation Cost │
    ├───────────────────┼────────────────────┼─────────────────────┤
    │ Early Chunking    │ ★☆☆ Poor           │ ★★★ Free            │
    │ (Traditional)     │ Pronouns orphaned  │ Standard approach   │
    ├───────────────────┼────────────────────┼─────────────────────┤
    │ Overlapping       │ ★★☆ Better         │ ★★★ Free            │
    │ Chunks            │ Some context kept  │ Just config change  │
    ├───────────────────┼────────────────────┼─────────────────────┤
    │ Contextual        │ ★★★ Excellent      │ ★★☆ LLM cost        │
    │ Retrieval         │ LLM adds context   │ ~$0.01/doc          │
    ├───────────────────┼────────────────────┼─────────────────────┤
    │ Late Chunking     │ ★★★ Excellent      │ ★★☆ Special model   │
    │ (Native)          │ Full doc context   │ Jina or similar     │
    ├───────────────────┼────────────────────┼─────────────────────┤
    │ Parent-Child      │ ★★★ Excellent      │ ★★☆ Extra storage   │
    │ Retriever         │ Returns parent doc │ 2x vector store     │
    └───────────────────┴────────────────────┴─────────────────────┘

    ACCURACY IMPROVEMENTS (vs Traditional Chunking):
    - Overlapping chunks: +3-5%
    - Contextual Retrieval: +15-20%
    - Late Chunking: +10-12%
    - Combined approaches: +25-30%
    """

    print(comparison)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.3: LATE CHUNKING")
    print("=" * 60)
    print("\nPreserving document context in chunk embeddings")
    print("=" * 60)

    # 1. Show the problem
    demonstrate_early_chunking_problem()

    # 2. Visualize approaches
    visualize_chunking_approaches()

    # 3. Simulate late chunking
    simulate_late_chunking()

    # 4. Implementation options
    show_implementation_options()

    # 5. Comparison
    print_comparison()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
    1. Early chunking loses pronoun references ("he", "the company")
    2. Late chunking embeds full document, then splits embeddings
    3. Each chunk embedding "knows" about the whole document
    4. 10-12% accuracy improvement on retrieval tasks
    5. Options: Jina native, Contextual Retrieval, Parent-Child
    6. Combine multiple approaches for best results
    """
    )
