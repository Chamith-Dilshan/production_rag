import os
import tempfile
from typing import Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_vector_store(
    documents: list[Document],
    embedding_model,
    strategy: Literal["semantic", "recursive"] = "semantic",
) -> Chroma:
    """Build a vector store using semantic chunking by default, with recursive fallback."""
    persist_dir = tempfile.mkdtemp()

    try:
        if strategy == "semantic":
            print("Applying Semantic Chunking...")
            # SemanticChunker expects a list of text strings or Documents
            texts = [doc.page_content for doc in documents]
            semantic_chunker = SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
            chunks = semantic_chunker.create_documents(texts=texts)
            # Reattach metadata to the created chunks
            for i, chunk in enumerate(chunks):
                if i < len(documents):
                    chunk.metadata = documents[i].metadata
        else:
            raise Exception("Forcing recursive splitter")

    except Exception as e:
        print(
            f"Semantic chunking failed ({e}), falling back to RecursiveCharacterTextSplitter."
        )
        recursive_chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = recursive_chunker.split_documents(documents=documents)

    print(f"Total chunks created ({strategy}): {len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        collection_name=f"collection_{strategy}",
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return vector_store


def run_comparison(documents: list[Document], embedding_model, questions: list[str]):
    """Run side-by-side retrieval and generation comparison for both strategies."""
    strategies: list[Literal["semantic", "recursive"]] = ["semantic", "recursive"]
    stores = {}

    for strategy in strategies:
        print(f"\n--- Initializing Vector Store for: {strategy} ---")
        stores[strategy] = get_vector_store(
            documents, embedding_model, strategy=strategy
        )

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context: {context}
        Question: {question}
        Answer:
        Make sure to answer in the same language as the question, and if 
        you don't know the answer, just say that you don't know, don't try to make up an answer.
        """)

    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    for q in questions:
        print("\n==============================")
        print(f"Question: {q}")
        print("==============================")

        for strategy in strategies:
            retriever = stores[strategy].as_retriever(
                search_type="similarity", search_kwargs={"k": 2}
            )
            retrieved_docs = retriever.invoke(q)

            print(f"\n[{strategy.upper()} RETRIEVED DOCS]:")
            for idx, d in enumerate(retrieved_docs):
                print(f"  {idx + 1}. {d.page_content[:120]}...")

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke(q)
            print(f"\n[{strategy.upper()} ANSWER]:\n{answer}")


if __name__ == "__main__":
    embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b", dimensions=512)

    sample_documents = [
        Document(
            id="DOC-001",
            page_content="WCAG 2.1 Level AA Compliance Guidelines: Web Content Accessibility Guidelines (WCAG) 2.1 provides the technical standards for making web content accessible to people with disabilities. Level AA conformance requires meeting all Level A criteria plus additional Level AA criteria.",
            metadata={"title": "WCAG Accessibility Standards", "type": "compliance"},
        ),
        Document(
            id="DOC-002",
            page_content="Error Code E_DD_APIFAILED: The API request failed due to network timeout. Error Code E_DD_INVALID_TOKEN: The authentication token is invalid or expired. Error Code E_DD_RATELIMIT: Rate limit exceeded. Please retry after 60 seconds.",
            metadata={"title": "API Error Codes Reference", "type": "config"},
        ),
        Document(
            id="DOC-003",
            page_content="Product SKU-HGC-34dsAS23 is a high-capacity storage device with 2TB capacity. Related SKUs: SKU-HGC-16dsAS23 (1TB), SKU-HGC-8dsAS23 (512GB). All models support USB 3.0 and come with warranty coverage.",
            metadata={"title": "Hardware Product Catalog", "type": "hardware"},
        ),
        Document(
            id="DOC-004",
            page_content="How to ensure data security and compliance with modern standards. Implementing best practices for user authentication, encryption protocols, and regular security audits. Organizations must maintain compiance with industry regulations and conduct penetration testing quarterly.",
            metadata={"title": "Data Security Best Practices", "type": "compliance"},
        ),
        Document(
            id="DOC-005",
            page_content="John Smith, the lead architect at TechCorp, designed the microservices infrastructure. Other key team members include Jane Doe (DevOps), Michael Johnson (Backend Lead), and Sarah Williams (Frontend Lead). The team follows agile methodology with 2-week sprints.",
            metadata={"title": "Team Structure and Roles", "type": "project"},
        ),
        Document(
            id="DOC-006",
            page_content="Router configuration: 10.0.0.1/24, 192.168.0.1/24, 172.16.0.1/24. Switch configuration: 10.0.0.2, 192.168.0.2, 172.16.0.2.",
            metadata={"title": "Company Annual Report", "type": "config"},
        ),
    ]

    test_questions = [
        "What are the error codes for API failures and rate limits?",
        "Who is the lead architect at TechCorp and what is their team structure?",
    ]

    run_comparison(sample_documents, embedding_model, test_questions)
