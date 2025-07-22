import os
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.schema import TextNode, Document

from config.settings import get_config, ComponentsConfig

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def get_embedding_model(self, **kwargs) -> BaseEmbedding:
        """Get the embedding model instance."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def get_embedding_model(self, **kwargs) -> OpenAIEmbedding:
        """Get OpenAI embedding model."""
        return OpenAIEmbedding(
            model=self.model_name,
            api_key=self.api_key,
            **kwargs
        )

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

class EmbeddingManager:
    """
    Manager for handling embeddings across different providers and models.

    This class provides a unified interface for generating embeddings,
    managing embedding models, and handling embedding-related operations.
    """
    def __init__(self, config: Optional[ComponentsConfig] = None):
        """
        Initialize the EmbeddingManager.

        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self._embedding_model = None
        self._provider = None

    @property
    def embedding_model(self) -> BaseEmbedding:
        """Get or create embedding model instance."""
        if self._embedding_model is None:
            self._embedding_model = self._create_embedding_model()
        return self._embedding_model

    def _create_embedding_model(self) -> BaseEmbedding:
        """Create embedding model based on configuration."""
        if not self.config.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        provider = OpenAIEmbeddingProvider(
            model_name=self.config.embedding_model,
            api_key=self.config.openai_api_key
        )

        self._provider = provider
        model = provider.get_embedding_model()

        print(f"Created embedding model: {provider.get_model_name()}")
        return model
