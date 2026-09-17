import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.ContextualCompressionRetriever").setLevel(logging.INFO)

INFO_BURIED = [
    Document(
        page_content="""ACME AI SOLUTIONS - COMPANY HISTORY AND TECHNOLOGY STACK

Founded in 2018 by three Stanford graduates, ACME AI Solutions began as a
small consulting firm helping enterprises adopt machine learning. Our first
office was a converted garage in Palo Alto, and we had just two laptops and
a dream. The early days were challenging - we survived on instant ramen and
the occasional pizza from the client meetings.

In 2019, we secured our first major contract with a Fortune 500 retailer,
helping them build a recommendation engine. This led to rapid growth and we
moved to a proper office space in San Francisco. By 2020, we had grown to
50 employees and opened offices in Austin and Seattle.

Our current technology stack has evolved significantly over the years. For
backend services, we use Python and FastAPI. Our data pipeline runs on
Apache Spark and Airflow. For frontend, we've standardized on React and
TypeScript.

LangChain is a framework for building LLM applications. It provides tools
for prompts, chains, agents, and memory. LangChain supports multiple LLM
providers including OpenAI, Anthropic, and local models like Llama.

The company culture at ACME emphasizes work-life balance. We offer unlimited
PTO, which most employees use for an average of 25 days per year. Our
engineering teams follow agile methodology with two-week sprints.

Our revenue has grown consistently, from $2M in 2019 to $45M in 2023. We
project $70M for 2024, driven by our new enterprise AI platform. The company
went through Series B funding in 2022, raising $80M at a $500M valuation.

Employee benefits include comprehensive health insurance through Aetna, a
401(k) with 4% matching, and a generous equity package.""",
        metadata={"source": "acme_company_overview.pdf"},
    ),
    Document(
        page_content="""ACME AI PLATFORM - TECHNICAL DOCUMENTATION v2.4

Chapter 1: System Architecture Overview

The ACME AI Platform is built on a microservices architecture deployed on
AWS EKS (Elastic Kubernetes Service). Each microservice is containerized
using Docker and orchestrated by Kubernetes. We use Istio as our service
mesh for traffic management and observability.

Our database layer consists of PostgreSQL for transactional data, Redis
for caching, and Pinecone for vector storage. All databases are deployed
in high-availability configurations with automatic failover.

Chapter 2: Authentication and Authorization

User authentication is handled through Auth0, supporting both SSO via SAML
2.0 and OAuth 2.0 flows. We implement role-based access control (RBAC) with
four default roles: Admin, Developer, Analyst, and Viewer.

Chapter 3: AI Framework Integration

LangGraph is a library for building stateful, multi-actor applications with
LLMs. Key features include state management, cycles and loops, human-in-the-
loop workflows, and persistence. LangGraph extends LangChain for complex
agent architectures.

Chapter 4: Monitoring and Logging

We use DataDog for application performance monitoring (APM) and log
aggregation. All services emit structured JSON logs that are collected and
indexed for searching. Alert thresholds are configured for latency (p99 >
500ms), error rates (> 1%), and resource utilization (CPU > 80%).

Chapter 5: Disaster Recovery

Our disaster recovery plan includes daily database backups stored in S3
with cross-region replication. RTO is 4 hours, and RPO is 1 hour.""",
        metadata={"source": "technical_docs_v2.4.pdf"},
    ),
]


def create_base_vectorstore():
    """Create a basic vector store for demos."""
    return Chroma.from_documents(
        documents=INFO_BURIED,
        embedding=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
    )


def demo_contextual_compression():
    """Contextual Compression extracts only relevant parts. In this method
    There will be a more llm calls happens in the background than normal retrival.
    but in a larger unstructured data, this method give a dramatic context reduction.
    This also help to reduce the noise of the retrival data.
    """

    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("Extracts only query-relevant content from documents")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    query = "What frameworks exist for building LLM applications?"

    print(f"\nQuery: {query}")

    # Without compression
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print("\n--- WITHOUT Compression (full chunks) ---")
    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")

    # With compression
    compressed_docs = compression_retriever.invoke(query)
    print("\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content}\n")


if __name__ == "__main__":
    demo_contextual_compression()
