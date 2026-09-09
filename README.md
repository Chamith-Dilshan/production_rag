uv init
uv venv
.venv/Scripts/activate

uv add langchain langchain-core langchain-community langchain-groq langgraph python-dotenv pymupdf

EMBEDDING_MODEL = "qwen3-embedding"
