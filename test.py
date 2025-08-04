from pathlib import Path
from core.document_processor import DocumentProcessor
from core.retriever import KnowledgeGraphRetrieverStrategy
from config.settings import get_config
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.schema import QueryBundle
from llama_index.core.indices.property_graph import ImplicitPathExtractor, SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import argparse
import os

PERSIST_DIR = "./storage"
INPUT_DIR = "input"
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.doc'}

def setup_models(config):
    """Initialize and return LLM and embedding models."""
    llm = OpenAI(
        model=config.llm_model,
        temperature=config.llm_temperature,
        api_key=config.openai_api_key
    )
    embed_model = OpenAIEmbedding(
        model=config.embedding_model,
        api_key=config.openai_api_key
    )
    return llm, embed_model

def get_graph_store(config):
    """Initialize and return Neo4j graph store."""
    return Neo4jPGStore(
        username=config.neo4j_username,
        password=config.neo4j_password,
        url=config.neo4j_url,
    )

def build_graph():
    """Build and persist the knowledge graph from documents."""
    print("Building knowledge graph...")
    config = get_config()

    # Initialize components
    processor = DocumentProcessor(config)
    graph_store = get_graph_store(config)
    llm, embed_model = setup_models(config)

    input_files = [
        file for file in Path(INPUT_DIR).iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not input_files:
        print(f"No supported files found in '{INPUT_DIR}'")
        return

    print(f"Processing {len(input_files)} files...")
    all_docs = []
    for file_path in input_files:
        try:
            docs = processor.load_documents_from_file(file_path, use_llama_parse=True)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

    sub_docs = processor.split_documents_into_pages(all_docs)
    print(f"Total pages after splitting: {len(sub_docs)}")

    index = PropertyGraphIndex.from_documents(
        sub_docs,
        embed_model=embed_model,
        kg_extractors=[
            ImplicitPathExtractor(),
            SimpleLLMPathExtractor(
                llm=llm,
                num_workers=config.num_workers,
                max_paths_per_chunk=config.max_paths_per_chunk,
            ),
        ],
        property_graph_store=graph_store,
        show_progress=config.show_progress,
    )

    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    print(f"✓ Knowledge graph built and persisted to {PERSIST_DIR}")
    print(f"✓ Neo4j available at {config.neo4j_url}")

def query_graph(query_text: str):
    """Query the existing knowledge graph."""
    print(f"Querying: '{query_text}'")
    config = get_config()

    try:
        # Initialize components
        graph_store = get_graph_store(config)
        llm, embed_model = setup_models(config)

        # Load storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR,
            graph_store=graph_store
        )

        # Load index without re-embedding
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            storage_context=storage_context,
            embed_model=embed_model,
            llm=llm,
            include_embeddings=False
        )

        # Setup retriever
        retriever = KnowledgeGraphRetrieverStrategy(
            kg_index=index,
            embed_model=embed_model,
            similarity_top_k=5,
            path_depth=2,
            include_text=True
        )

        # Execute query
        results = retriever.retrieve(QueryBundle(query_str=query_text))

        if not results:
            print("No results found")
            return

        print(f"\nFound {len(results)} results:")
        for i, node in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {node.score:.3f}) ---")
            print(node.node.get_content()[:500] + "...")
            print("-" * 60)

    except Exception as e:
        print(f"Query failed: {str(e)}")
        print("Ensure:")
        print("1. Neo4j is running (docker compose up)")
        print("2. You've built the graph first (uv run main.py --build)")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Builder and Query Tool")
    parser.add_argument("--build", action="store_true", help="Build the knowledge graph")
    parser.add_argument("--query", type=str, help="Query to execute against the graph")

    args = parser.parse_args()

    if args.build:
        build_graph()
    elif args.query:
        query_graph(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
