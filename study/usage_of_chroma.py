import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def use_of_pure_chroma():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="project_plans")

    collection.upsert(
        ids=["id1", "id2"],
        documents=[
            "This is a document about pineapple",
            "This is a document about oranges",
        ],
    )

    results = collection.query(
        query_texts=[
            "This is a query document about hawaii"
        ],  # Chroma will embed this for you
        n_results=2,  # how many results to return
    )
    print(results)


def use_of_chroma_with_langchain():
    """
    metadata can capture the source of the document, its relationship to other documents,
    and other information. An individual Document often represents a chunk of a larger document.
    The score represents the distance between the query and the document.
    If you want to know the similarity score, you can use,
    similarity = 1/ (1 + distance)
     or
    similarity = 1 - (distance / max_distance)
    """
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    documents = [
        Document(
            id="id1",
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            id="id2",
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    vector_store.add_documents(documents=documents)

    results = vector_store.similarity_search_with_score(
        query="why dogs are great companions?"
    )

    question_embedding = embeddings.embed_query("why dogs are great companions?")
    result2 = vector_store.similarity_search_by_vector(question_embedding)

    doc, score = results[0]
    print(f"Result-1 Score: {score}")
    print(doc)

    doc = result2[0]
    print("\nResult-2")
    print(doc)


if __name__ == "__main__":
    # use_of_pure_chroma()
    use_of_chroma_with_langchain()
