"""First up and running a pgvector store using docker ->
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e
POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
"""

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# See the docker command above to launch a postgres instance with pgvector enabled.
# langchain-postgres works only with psycopg3
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "my_docs"

docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]

vector_store = PGVector(
    embeddings=embedding_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

# similarity_search
results = vector_store.similarity_search(
    "kitty", k=10, filter={"id": {"$in": [1, 5, 2, 9]}}
)

# similarity_search_with_score
results_with_score = vector_store.similarity_search_with_score(query="cats", k=1)

# query by turning into retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})


if __name__ == "__main__":
    print("similarity_search")
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")

    print("\nsimilarity_search_with_score")
    for doc, score in results_with_score:
        print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

    print("\nAs a retriever")
    print(retriever.invoke("kitty"))
