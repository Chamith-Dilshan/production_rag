import os
import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")


def create_knowledge_base():
    """Create a vector knowledge-base"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc = Document(
        id="id1",
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-dogs-doc"},
    )
    chunks = splitter.split_documents(documents=[doc])
    print(f"Number of chunks: {len(chunks)}, chunk size: {len(chunks[0].page_content)}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        collection_name="example_collection",
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),  # Where to save data locally, remove if not necessary
    )

    return vector_store


def basic_rag_pipeline():
    vector_store = create_knowledge_base()
    retriver = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # llm = ChatOllama(model="qwen3.5:2b", temperature=0.4, max_tokens=1024)
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0.7,
        reasoning_effort="medium",
    )

    prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context: {context}
        Question: {question}
        Answer:
        Make sure to answer in the same language as the question, and if 
        you don't know the answer, just say that you don't know, don't try to make up an answer.
        """)

    # Format retrieved docs
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    # Rag chain
    chain = (
        {"context": retriver | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = [
        "Why are dogs great companions?",
        "What are the best features of dogs?",
        "What is langchain?",
    ]

    for q in question:
        print(f"Question: {q}")
        print(chain.invoke(q))


if __name__ == "__main__":
    basic_rag_pipeline()
