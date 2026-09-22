from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Application config
    APP_ENV: str
    LOG_LEVEL: str
    RATE_LIMIT: str
    CACHE_TTL_SECONDS: int
    MAX_RETRIES: int

    # LLM config
    GROQ_API_KEY: str
    PRIMARY_MODEL: str = "qwen/qwen3.8-27b"
    FALLBACK_MODEL: str = "openai/gpt-oss-20b"

    # Langfuse config
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        # env_file=".env",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "prod"


@lru_cache
def get_settings() -> Settings:
    """Cached settings - loaded once and reused everywhere"""
    return Settings()


settings = get_settings()
