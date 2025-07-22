from config.settings import get_config, ComponentsConfig
from typing import Optional
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import Document
from typing import List
from llama_index.core import PropertyGraphIndex
from llama_index.llms.openai import OpenAI
from core.embeddings import EmbeddingManager
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)

class KnowledgeGraphBuilder:
    """
    Builder for constructing and managing knowledge graphs.

    This class provides methods for creating property graphs from documents,
    managing graph stores, and performing knowledge graph operations.
    """

    def __init__(
        self,
        config: Optional[ComponentsConfig] = None,
        graph_store: Optional[Neo4jPGStore] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None
    ):
        """
        Initialize the KnowledgeGraphBuilder.

        Args:
            config: Configuration instance. If None, uses global config.
            graph_store: Neo4j graph store instance. If None, creates new one.
            llm: LLM instance for extraction. If None, creates from config.
            embed_model: Embedding model. If None, creates from config.
        """
        self.config = config or get_config()
        self._graph_store = graph_store
        self._llm = llm
        self._embed_model = embed_model
        self._index = None
        self._extractors = None

    @property
    def graph_store(self) -> Neo4jPGStore:
        """Get or create Neo4j graph store."""
        if self._graph_store is None:
            self._graph_store = self._create_graph_store()
        return self._graph_store

    @property
    def llm(self) -> LLM:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    @property
    def embed_model(self) -> BaseEmbedding:
        """Get or create embedding model."""
        if self._embed_model is None:
            embedding_manager = EmbeddingManager(self.config)
            self._embed_model = embedding_manager.embedding_model
        return self._embed_model

    @property
    def extractors(self) -> List:
        """Get or create knowledge graph extractors."""
        if self._extractors is None:
            self._extractors = self._create_extractors()
        return self._extractors

    def _create_extractors(self) -> List:
        """Create knowledge graph extractors based on configuration."""
        extractors = []

        for extractor_type in self.config.kg_extractors:
            if extractor_type == "implicit":
                extractor = ImplicitPathExtractor()
                extractors.append(extractor)
                print("Added ImplicitPathExtractor")

            elif extractor_type == "llm":
                extractor = SimpleLLMPathExtractor(
                    llm=self.llm,
                    num_workers=self.config.num_workers,
                    max_paths_per_chunk=self.config.max_paths_per_chunk,
                )
                extractors.append(extractor)
                print("Added SimpleLLMPathExtractor")

            elif extractor_type == "schema":
                # Schema-based extraction would require predefined schema
                print("Schema-based extraction not implemented yet")
                continue

            else:
                print(f"Unknown extractor type: {extractor_type}")
                continue

        print(f"Created {len(extractors)} extractors")
        return extractors

    def _create_graph_store(self) -> Neo4jPGStore:
        """Create Neo4j graph store from configuration."""
        neo4j_settings = self.config.get_neo4j_settings()

        try:
            graph_store = Neo4jPGStore(
                username=neo4j_settings["username"],
                password=neo4j_settings["password"],
                url=neo4j_settings["url"],
                database=neo4j_settings.get("database", "neo4j"),
            )

            print(f"Connected to Neo4j at {neo4j_settings['url']}")
            return graph_store
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def _create_llm(self) -> LLM:
        """Create LLM instance from configuration."""
        if not self.config.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        llm = OpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            api_key=self.config.openai_api_key,
        )

        print(f"Created LLM: {self.config.llm_model}")
        return llm

    def build_graph_from_documents(
        self,
        documents: List[Document],
        show_progress: Optional[bool] = None
    ) -> PropertyGraphIndex:
        """
        Build knowledge graph from documents.

        Args:
            documents: List of documents to process
            show_progress: Whether to show progress. If None, uses config.

        Returns:
            PropertyGraphIndex instance
        """
        if show_progress is None:
            show_progress = self.config.show_progress

        try:
            self._index = PropertyGraphIndex.from_documents(
                documents,
                embed_model=self._embed_model,
                kg_extractors=self._extractors,
                property_graph_store=self.graph_store,
                show_progress=show_progress,
            )

            print("Successfully built knowledge graph")
            return self._index

        except Exception as e:
            print(f"Failed to build knowledge graph: {str(e)}")
            raise
