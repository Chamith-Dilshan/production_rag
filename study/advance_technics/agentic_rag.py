"""
Lesson 6.4: Agentic RAG with LangGraph

Traditional RAG: Query → Retrieve → Generate (one-shot)
Agentic RAG: Query → Retrieve → Evaluate → [Retry if needed] → Generate

This is THE pattern for 2026 - RAG systems that can:
- Evaluate if retrieved documents are relevant
- Reformulate queries and retry
- Use multiple retrieval strategies
- Self-correct and iterate

LangGraph 1.x is the production standard for building these workflows.
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import END, StateGraph

load_dotenv()

# ============================================================
# STATE DEFINITION
# ============================================================


class RAGState(TypedDict):
    """
    State schema for our agentic RAG workflow.

    LangGraph uses TypedDict for state (not Pydantic in 1.x).
    The Annotated[list, add] tells LangGraph to merge lists.
    """

    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int


# ============================================================
# SETUP: Vector Store with Sample Data
# ============================================================


def create_sample_vectorstore():
    """Create a vector store with sample documents for the demo."""

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    documents = [
        Document(
            page_content="""
            LangGraph is a library for building stateful, multi-actor applications
            with LLMs. It extends LangChain with the ability to create cyclic graphs,
            enabling more complex agent workflows. LangGraph provides built-in support
            for persistence, streaming, and human-in-the-loop workflows.
            """,
            metadata={"source": "langgraph_docs.md", "topic": "langgraph"},
        ),
        Document(
            page_content="""
            To install LangGraph, use pip: pip install langgraph>=1.1.0
            LangGraph requires Python 3.10 or higher. It works seamlessly with
            LangChain 1.x and supports both sync and async execution.
            """,
            metadata={"source": "langgraph_install.md", "topic": "installation"},
        ),
        Document(
            page_content="""
            StateGraph is the core abstraction in LangGraph. You define nodes
            (functions that process state) and edges (transitions between nodes).
            Conditional edges allow branching based on state values. The graph
            is compiled into an executable workflow.
            """,
            metadata={"source": "langgraph_concepts.md", "topic": "stategraph"},
        ),
        Document(
            page_content="""
            The weather in Seattle is typically rainy in winter and mild in summer.
            Average temperatures range from 40°F in January to 75°F in July.
            The city receives about 37 inches of rain annually.
            """,
            metadata={"source": "weather.md", "topic": "weather"},
        ),
        Document(
            page_content="""
            Python was created by Guido van Rossum and first released in 1991.
            It emphasizes code readability and simplicity. Python is widely used
            in web development, data science, AI, and automation.
            """,
            metadata={"source": "python_history.md", "topic": "python"},
        ),
    ]

    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embeddings, collection_name="agentic_rag_demo"
    )

    return vectorstore


# ============================================================
# NODE FUNCTIONS
# ============================================================


def retrieve_documents(state: RAGState) -> dict:
    """
    Retrieve documents based on the query.
    Uses rewritten_query if available, otherwise original query.
    """
    query = state.get("rewritten_query") or state["query"]

    print(f"\n[RETRIEVE] Searching for: '{query}'")

    vectorstore = state.get("_vectorstore")  # Injected at runtime
    if not vectorstore:
        # Fallback - create new (in production, pass via config)
        vectorstore = create_sample_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(query)

    print(f"[RETRIEVE] Found {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(
            f"  {i}. {doc.metadata.get('source', 'unknown')}: {doc.page_content[:50]}..."
        )

    return {"documents": documents}


def grade_documents(state: RAGState) -> dict:
    """
    Grade retrieved documents for relevance to the query.
    This is the KEY difference from traditional RAG - we evaluate before generating.
    """
    query = state["query"]
    documents = state["documents"]

    print(f"\n[GRADE] Evaluating {len(documents)} documents for relevance...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
    )

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a relevance grader. Given a user query and a document,
determine if the document contains information relevant to answering the query.

Output ONLY a number between 0 and 1:
- 1.0 = Highly relevant, directly answers the query
- 0.7 = Somewhat relevant, contains related information
- 0.3 = Marginally relevant, tangentially related
- 0.0 = Not relevant at all

Output ONLY the number, nothing else.""",
            ),
            (
                "human",
                """Query: {query}

Document: {document}

Relevance score (0-1):""",
            ),
        ]
    )

    # Grade each document and calculate average
    scores = []
    relevant_docs = []

    for doc in documents:
        chain = grading_prompt | llm
        result = chain.invoke({"query": query, "document": doc.page_content})

        try:
            score = float(result.content.strip())
        except ValueError:
            score = 0.5  # Default if parsing fails

        scores.append(score)
        print(f"  - {doc.metadata.get('source', 'unknown')}: {score:.2f}")

        if score >= 0.5:  # Keep documents with score >= 0.5
            relevant_docs.append(doc)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"[GRADE] Average relevance: {avg_score:.2f}")
    print(f"[GRADE] Keeping {len(relevant_docs)}/{len(documents)} documents")

    return {"documents": relevant_docs, "relevance_score": avg_score}


def rewrite_query(state: RAGState) -> dict:
    """
    Rewrite the query to improve retrieval.
    Called when initial retrieval doesn't find relevant documents.
    """
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[REWRITE] Attempt {retry_count + 1}: Improving query...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
    )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a query rewriter for a RAG system.
The original query didn't retrieve relevant documents.

Rewrite the query to be more specific and likely to match relevant documents.
Consider:
- Adding synonyms or related terms
- Being more specific about what information is needed
- Rephrasing to match how documentation is typically written

Output ONLY the rewritten query, nothing else.""",
            ),
            (
                "human",
                """Original query: {query}

Rewritten query:""",
            ),
        ]
    )

    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    rewritten = result.content.strip()

    print(f"[REWRITE] Original: '{query}'")
    print(f"[REWRITE] Rewritten: '{rewritten}'")

    return {"rewritten_query": rewritten, "retry_count": retry_count + 1}


def generate_answer(state: RAGState) -> dict:
    """
    Generate the final answer using retrieved documents.
    """
    query = state["query"]
    documents = state["documents"]

    print(f"\n[GENERATE] Creating answer from {len(documents)} documents...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
    )

    # Format documents
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in documents
        ]
    )

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant answering questions based on provided context.

Use ONLY the information in the context to answer. If the context doesn't contain
enough information, say so clearly.

Always cite your sources by mentioning which document the information came from.""",
            ),
            (
                "human",
                """Context:
{context}

Question: {query}

Answer:""",
            ),
        ]
    )

    chain = generate_prompt | llm
    result = chain.invoke({"context": context, "query": query})

    print("[GENERATE] Answer generated")

    return {"generation": result.content}


def generate_fallback(state: RAGState) -> dict:
    """
    Generate a fallback response when retrieval fails after all retries.
    """
    query = state["query"]

    print(f"\n[FALLBACK] Retrieval failed after {state.get('retry_count', 0)} attempts")

    fallback_message = f"""I couldn't find relevant information to answer your question: "{query}"

This could mean:
1. The information isn't in my knowledge base
2. Try rephrasing your question with different terms
3. The topic might not be covered in the available documents

Would you like to try a different question?"""

    return {"generation": fallback_message}


# ============================================================
# ROUTING FUNCTIONS
# ============================================================


def should_retry_or_generate(
    state: RAGState,
) -> Literal["rewrite", "generate", "fallback"]:
    """
    Decide whether to retry retrieval or proceed to generation.

    This is the BRAIN of agentic RAG - making decisions based on retrieval quality.
    """
    relevance_score = state.get("relevance_score", 0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    documents = state.get("documents", [])

    print(
        f"\n[ROUTER] Evaluating: score={relevance_score:.2f}, retries={retry_count}/{max_retries}, docs={len(documents)}"
    )

    # If we have relevant documents, generate
    if relevance_score >= 0.5 and len(documents) > 0:
        print("[ROUTER] -> GENERATE (good relevance)")
        return "generate"

    # If we can retry, rewrite query
    if retry_count < max_retries:
        print("[ROUTER] -> REWRITE (low relevance, retrying)")
        return "rewrite"

    # Out of retries
    if len(documents) > 0:
        print("[ROUTER] -> GENERATE (out of retries, using available docs)")
        return "generate"
    else:
        print("[ROUTER] -> FALLBACK (no relevant documents)")
        return "fallback"


# ============================================================
# BUILD THE GRAPH
# ============================================================


def build_agentic_rag_graph():
    """
    Build the LangGraph workflow for agentic RAG.

    Flow:
    1. retrieve -> grade -> [decision]
    2. If low relevance and retries left: rewrite -> retrieve (loop)
    3. If good relevance or out of retries: generate
    4. If no documents at all: fallback
    """

    # Create the graph with our state schema
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("fallback", generate_fallback)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges
    workflow.add_edge("retrieve", "grade")

    # Conditional edge from grade
    workflow.add_conditional_edges(
        "grade",
        should_retry_or_generate,
        {"rewrite": "rewrite", "generate": "generate", "fallback": "fallback"},
    )

    # After rewrite, go back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Terminal nodes
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # Compile the graph
    app = workflow.compile()

    return app


# ============================================================
# DEMO
# ============================================================


def run_demo():
    """Run the agentic RAG demo."""

    print("=" * 60)
    print("AGENTIC RAG DEMO")
    print("=" * 60)

    # Create vector store
    print("\nSetting up vector store...")
    vectorstore = create_sample_vectorstore()

    # Build the graph
    print("Building agentic RAG graph...")
    app = build_agentic_rag_graph()

    # Test queries
    test_queries = [
        "How do I install LangGraph?",  # Should find relevant docs
        "What is StateGraph in LangGraph?",  # Should find relevant docs
        "How do I make pizza?",  # Should NOT find relevant docs (tests fallback)
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        # Run the graph
        initial_state = {
            "query": query,
            "rewritten_query": "",
            "documents": [],
            "generation": "",
            "relevance_score": 0.0,
            "retry_count": 0,
            "max_retries": 2,
            "_vectorstore": vectorstore,  # Pass vectorstore via state
        }

        result = app.invoke(initial_state)

        print("\n" + "-" * 60)
        print("FINAL ANSWER:")
        print("-" * 60)
        print(result["generation"])

    # Cleanup
    vectorstore.delete_collection()


# ============================================================
# VISUALIZE THE GRAPH
# ============================================================


def print_graph_structure():
    """Print the graph structure for understanding."""

    print("\n" + "=" * 60)
    print("AGENTIC RAG GRAPH STRUCTURE")
    print("=" * 60)

    structure = """
    ┌─────────────────────────────────────────────────────────────┐
    │                      START                                  │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    RETRIEVE     │
                    │  (get docs from │
                    │   vectorstore)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     GRADE       │
                    │ (LLM evaluates  │
                    │   relevance)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     (low relevance,   (good relevance)  (no docs,
      can retry)                         out of retries)
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   REWRITE   │  │  GENERATE   │  │  FALLBACK   │
    │ (improve    │  │  (create    │  │  (graceful  │
    │   query)    │  │   answer)   │  │   failure)  │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           │                ▼                ▼
           │         ┌─────────────────────────┐
           │         │          END            │
           │         └─────────────────────────┘
           │
           └──────────► (back to RETRIEVE)

    KEY INSIGHT: The loop between REWRITE → RETRIEVE → GRADE
    allows the system to self-correct and improve retrieval!
    """

    print(structure)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.4: AGENTIC RAG WITH LANGGRAPH")
    print("=" * 60)
    print("\nRAG that thinks, evaluates, and retries!")
    print("=" * 60)

    # Show graph structure
    print_graph_structure()

    # Run demo
    run_demo()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
    1. Traditional RAG is one-shot: retrieve → generate
    2. Agentic RAG adds evaluation and retry loops
    3. LangGraph StateGraph enables cyclic workflows
    4. Key pattern: retrieve → grade → [rewrite if needed] → generate
    5. Graceful fallback when retrieval fails completely
    6. This is THE production RAG pattern for 2026

    WHEN TO USE AGENTIC RAG:
    - Complex queries that might need reformulation
    - High-stakes applications where answer quality matters
    - When you have diverse document types
    - User-facing applications (vs batch processing)
    """
    )
