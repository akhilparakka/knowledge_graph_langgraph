from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import  PropertyGraphIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.property_graph import VectorContextRetriever
from abc import ABC, abstractmethod
from typing import List

class RetrieverStrategy(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes for the given query."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the retrieval strategy."""
        pass

class KnowledgeGraphRetrieverStrategy(RetrieverStrategy):
    """Knowledge graph-based retrieval strategy."""

    def __init__(
        self,
        kg_index: PropertyGraphIndex,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 2,
        path_depth: int = 1,
        include_text: bool = True
    ):
        self.kg_index = kg_index
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        self.path_depth = path_depth
        self.include_text = include_text
        self._retriever = None

    @property
    def retriever(self):
        """Get or create knowledge graph retriever."""
        if self._retriever is None:
            self._retriever = VectorContextRetriever(
                self.kg_index.property_graph_store,
                embed_model=self.embed_model,
                similarity_top_k=self.similarity_top_k,
                path_depth=self.path_depth,
                include_text=self.include_text,
            )
        return self._retriever

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using knowledge graph traversal."""
        try:
            nodes = self.retriever.retrieve(query_bundle)
            print(f"Knowledge graph retrieval returned {len(nodes)} nodes")
            return nodes
        except Exception as e:
            print(f"Knowledge graph retrieval failed: {str(e)}")
            return []

    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "knowledge_graph"
