import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("RATE_LIMIT", "1000/minute")
os.environ.setdefault("CACHE_TTL_SECONDS", "300")
os.environ.setdefault("MAX_RETRIES", "3")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PRIMARY_MODEL", "openai/gpt-oss-20b")
os.environ.setdefault("FALLBACK_MODEL", "openai/gpt-oss-20b")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "test-secret")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "test-public")
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
