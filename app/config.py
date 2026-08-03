
#Central configuration module


from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int

    # Redis
    redis_host: str
    redis_port: int

    # Qdrant
    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str

    # MinIO
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_raw_pdfs: str
    minio_bucket_parsed: str
    minio_bucket_figures: str

    # LLM
    llm_base_url: str
    llm_model_name: str

    # App
    tau_high: float
    tau_low: float
    log_level: str
    
    # Caching
    redis_cache_ttl: int = 3600
    
    # Conversation
    chat_history_limit: int = 6

    @property
    def postgres_url(self) -> str:
        """Builds the SQLAlchemy connection string from individual parts."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()