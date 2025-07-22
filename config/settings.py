import os
from typing import Optional, List, Dict, Any
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings
from pathlib import Path


class ComponentsConfig(BaseSettings):
    """Configuration settings for the Components library."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Allow extra fields from .env
    }

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    llama_cloud_api_key: Optional[str] = Field(default=None, env="LLAMA_CLOUD_API_KEY")

    # Model Settings
    llm_model: str = Field(default="gpt-4o", env="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")

    # Neo4j Database Settings
    neo4j_url: str = Field(default="bolt://localhost:7687", validation_alias=AliasChoices("NEO4J_URL", "neo4j_url"))
    neo4j_username: str = Field(default="neo4j", validation_alias=AliasChoices("NEO4J_USERNAME", "neo4j_db_user", "neo4j_username"))
    neo4j_password: str = Field(default="llamaindex", validation_alias=AliasChoices("NEO4J_PASSWORD", "neo4j_db_password", "neo4j_password"))
    neo4j_database: str = Field(default="neo4j", validation_alias=AliasChoices("NEO4J_DATABASE", "neo4j_database"))

    # Knowledge Graph Settings
    kg_extractors: List[str] = Field(
        default=["implicit", "llm"],
        env="KG_EXTRACTORS"
    )
    max_paths_per_chunk: int = Field(default=10, env="MAX_PATHS_PER_CHUNK")
    num_workers: int = Field(default=4, env="NUM_WORKERS")

    # Retrieval Settings
    similarity_top_k: int = Field(default=2, env="SIMILARITY_TOP_K")
    path_depth: int = Field(default=1, env="PATH_DEPTH")
    include_text: bool = Field(default=True, env="INCLUDE_TEXT")

    # Document Processing Settings
    chunk_size: int = Field(default=1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=20, env="CHUNK_OVERLAP")

    # LlamaParse Settings
    result_type: str = Field(default="text", env="LLAMAPARSE_RESULT_TYPE")
    verbose: bool = Field(default=True, env="VERBOSE")
    show_progress: bool = Field(default=True, env="SHOW_PROGRESS")

    # Storage Settings
    storage_dir: Path = Field(default=Path("./storage"), env="STORAGE_DIR")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")

    # Agent Settings
    agent_memory_enabled: bool = Field(default=True, env="AGENT_MEMORY_ENABLED")
    allow_parallel_tool_calls: bool = Field(default=False, env="ALLOW_PARALLEL_TOOL_CALLS")

    def get_neo4j_settings(self) -> Dict[str, Any]:
        """Get Neo4j-specific settings."""
        return {
            "url": self.neo4j_url,
            "username": self.neo4j_username,
            "password": self.neo4j_password,
            "database": self.neo4j_database,
        }

_config: Optional[ComponentsConfig] = None

def get_config() -> ComponentsConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ComponentsConfig()
    return _config
