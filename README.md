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

2. Recursive chunk is more of cutting at the paragraph / sentence level.(This is the default chunking strategy langchain
   uses).
   It has a decision tree guiding how the chunking is done.

3. We can use Semantic chunk to cut at meaningful boundaries.it first embeds each sentence and compares
   adjacent sentence embeddings to decide the chunking. Then splits when the embedding similarity is low.

Late chunking is about embedding the full document first then token embeddings. After that we do the chunking.
This is not like traditional chunking where we do the chunking first, then embedding. In this way the chunk embeddings
have full context where traditional chunks don't have an idea what other chunks contain.
This can help use to get 10-12% accuracy improvement.
We can use some special embedding models like jina-embedding-v2(please check the latest info before use).

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
