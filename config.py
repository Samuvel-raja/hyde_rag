from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )
    
    openai_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "hyde_pdf_docs"
    hypo_docs_count: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    qdrant_api_key: Optional[str] = None

settings = Config()