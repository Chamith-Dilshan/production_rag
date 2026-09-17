from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def demo_parent_document_retriever():
    """Parent Document Retriever: small chunks for search, large for context.
        with regular retrival, the context that retrieve my not complete chunks.
        in that cases, llm have to work with incomplete chunks. but with ParentDocumentRetriever
        it will retrieve large chunk that cover the focus area. this way llm have a complete
        chunk to work on and avoid hallucination or guessing
    """

    print("=" * 60)
    print("PARENT DOCUMENT RETRIEVER")
    print("Small chunks for precise search, large chunks for context")
    print("=" * 60)
    # Long document to demonstrate parent/child splitting
    long_doc = Document(
        page_content="""
# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

Several frameworks exist for building AI agents:

LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
        """,
        metadata={"source": "ai_agents_guide.md"},
    )

    # Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Storage
    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
    )
    store = InMemoryStore()

    # Create retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Add document
    retriever.add_documents([long_doc])

    # Search
    query = "What is LangGraph used for?"

    print(f"\nQuery: {query}")

    # Regular retrieval (would get small chunks)
    child_docs = vectorstore.similarity_search(query, k=1)
    print("\n--- Vector store similarity search (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(f"Content: {child_docs[0].page_content}")

    # Parent retrieval (gets full context)
    parent_docs = retriever.invoke(query)
    print("\n--- Parent-Child Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content preview: {parent_docs[0].page_content[:300]}...")


if __name__ == "__main__":
    demo_parent_document_retriever()
