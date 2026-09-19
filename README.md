### Per-Required

setup langfuse and pgvector database.
you can quickly set them using docker.
You also need an API key for LLM model use,
or you can use Ollama.
Or you can use Groq since they provide a generous
amount of free tier.

uv init
uv venv
.venv/Scripts/activate

uv add langchain langchain-core langchain-community langchain-groq langgraph python-dotenv pymupdf

EMBEDDING_MODEL = "qwen3-embedding"

In a RAG / retrieval context, it is important to make sure
that the chunker and tokenizer are using the same embedding model.
Make sure that indexing and querying are using the same embedding model.
When you're deciding what embedding model is going to use, you need to take account of its dimensions as well.
For most cases 768 to 1536 dimensions are good.

Chunk size is matter it shouldn't be too small or too large. The sweet spot is between 200-1000 tokens.
Next is overlap between chunks. It helps to preserve context.

Split boundaries ->

1. Fixed Chunking is bad because it can chunk the information of the doc fixed often incomplete chunks.

2. Recursive chunk is more of cutting at the paragraph / sentence level. (This is the default chunking strategy
   langchain
   uses).
   It has a decision tree guiding how the chunking is done.

3. We can use Semantic chunk to cut at meaningful boundaries.it first embeds each sentence and compares
   adjacent sentence embeddings to decide the chunking. Then splits when the embedding similarity is low.

Late chunking is about embedding the full document first then token embeddings. After that we do the chunking.
This is not like traditional chunking where we do the chunking first, then embedding. In this way the chunk embeddings
have full context where traditional chunks don't have an idea what other chunks contain.
This can help use to get 10-12% accuracy improvement.
We can use some special embedding models like jina-embedding-v2 (please check the latest info before use).

Context Type ->
code, legal, Markdown, etc.
each context type has different treatment.

Keep in mind ->

1. Use the same embedding model for indexing and querying.
2. Use the same tokenizer for indexing and querying.
3. Embedding quality is more important than quality, so focus on getting quality vectors over
   massive, noise datasets.
4. Test retrieval seperatly from generation.

Why Most of RAG failed in Production?

1. Bad Chunking.
2. Embedding quality and mismatch.
3. Retrival quality and noise.
4. Context overflow (exceed the context window)
5. Hallucination.

Why LLM debugging is hard

1. Non-deterministic -> the same input can produce different output.
2. Cascading failures -> If a bad search result came it will cause a bad analysis which leads to bad results.
3. Silent failures -> No Crashes it will give a very confident wrong answer.
4. Cost surprises -> 10 iterations instead of 2.

Observability ->

Traces -

* Agent Flow
* Inputs/Outputs
* Tool calls
* Decisions made

Metrics ->

* Token count
* Latency per node
* Cost per run
* Error rates

Evaluation ->

* Correctness
* Relevance
* Human feedback
* Regression detection

Vector Index Tuning ->

Index types ->

* HNSW
* IVFFlat

### HNSW

An HNSW index creates a multilayer graph. It has better query performance than IVFFlat (in terms of speed-recall
tradeoff),
but has slower build times and uses more memory. Also, an index can be created without any data in the table since there
isn’t a training step like IVFFlat.

Specify HNSW parameters ->

* m - the max number of connections per layer (16 by default)
* ef_construction - the size of the dynamic candidate list for constructing the graph (64 by default)

A higher value of ef_construction provides better recall at the cost of index build time / insert speed.

### IVFFlat

An IVFFlat index divides vectors into lists, and then searches a subset of those lists that are closest to the query
vector. It has faster build times and uses less memory than HNSW, but has lower query performance (in terms of
speed-recall tradeoff).

Three keys to achieving good recall are:

* Create the index after the table has some data
* Choose an appropriate number of lists - a good place to start is rows / 1000 for up to 1M rows and sqrt (rows) for
  over 1M rows
* When querying, specify an appropriate number of probes (higher is better for recall, lower is better for speed) - a
  good place to start is sqrt (lists)

### How to create an HNSW index

#### pgvector

types ->

* vector - up to 2,000 dimensions
* halfvec - up to 4,000 dimensions
* bit - up to 64,000 dimensions
* sparsevec - up to 1,000 non-zero elements

Distance functions ->

* L2 distance
* Inner product
* Cosine distance
* L1 distance
* Hamming distance
* Jaccard distance

based on that the implementation will be different.
this is example for Cosine distance.

````pgvector
CREATE INDEX ON documents
	USING hnsw (embedding vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);
````

````pgvector
# At query time, set ef_search
SET hnsw.ef_search = 100; # Higher = more accurate and slower
````

#### Chroma

````chroma
collection = client.create_collection(
	name="my_collection"
	metadata={
		'hnsw:M': 16
		'hnsw:construction_ef': 100
		'hnsw:search_ef': 50
	}
)
````

### Cost Optimization Strategies

#### Reduce Dimension

	Most of the time you may use 1536 dimmetions for the models.
	you can reduce it to around 512 to save the cost.

```
embedding_model = OllamaEmbeddings(
	model="qwen3-embedding:0.6b", 
	dimensions=512
)
```

#### Quantization

	It is about converting float32 to int8 or binary, this will reduce
    bytes per dimension.

#### Batch Queries

	The idea is insted of doing individual 10K queries, you can do batch queries.
	Fewer round trips means you can save, but the provider and the model it self should
    should support it.

````python
# Bad: Individual API calls
for query in queries:
	results = index.query(query)

# Good: Batch API calls
results = index.query(
	queries,
	batch=True
)
````

#### Caching

	Observe the frequent queries and cache them.
	important part is that you need to embedd them first and then cash it then use similarity search 
	or similar method to retrieve them.

#### Right Size

	Always start small and continue monitoring. Observe the cost and adjust the parameters accordingly.
	if the need arrive you can then scale up as needed.



Stop (pause, keep container):

docker stop pgvector-container
Start (resume stopped container):

docker start pgvector-container
Reset (delete container and data, recreate fresh):

docker stop pgvector-container docker rm pgvector-container docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
Remove image (after stopping/removing container):

docker rmi pgvector/pgvector:pg16
The container is now stopped. Use docker start pgvector-container to bring it back without recreating it.
