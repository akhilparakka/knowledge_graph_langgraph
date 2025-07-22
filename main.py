#!/usr/bin/env python3
"""
Main script to demonstrate DocumentProcessor functionality.
Loads documents from the input folder and processes them.
"""

from pathlib import Path
from core.document_processor import DocumentProcessor
from config.settings import get_config
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import ImplicitPathExtractor, SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def main():
    """Main function to load and process documents from input folder."""
    print("Starting GraphMemory Document Processing...")

    config = get_config()

    processor = DocumentProcessor(config)

    input_folder = Path("input")
    supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
    input_files = [
        file for file in input_folder.iterdir()
        if file.is_file() and file.suffix.lower() in supported_extensions
    ]

    if not input_files:
        print(f"No supported files found in '{input_folder}' folder.")
        print(f"Supported extensions: {', '.join(supported_extensions)}")
        return

    print(f"Found {len(input_files)} files to process:")
    for file in input_files:
        print(f"  - {file.name}")

    print("\nProcessing documents...")

    all_documents = []
    for file_path in input_files:
        try:
            print(f"\nProcessing: {file_path.name}")

            documents = processor.load_documents_from_file(
                file_path=file_path,
                use_llama_parse=True
            )

            all_documents.extend(documents)

        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {str(e)}")

    sub_docs = processor.split_documents_into_pages(all_documents)

    print(f"\n{'='*50}")
    print(f"Total pages After Splitting: {len(sub_docs)}")

    doc_stats = processor.get_document_stats(sub_docs)
    print(doc_stats)

    if all_documents:
        total_chars = sum(len(doc.text) for doc in all_documents)
        print(f"Total text content: {total_chars} characters")
        print(f"Average document size: {total_chars // len(all_documents)} characters")

    # Build Knowledge Graph
    if sub_docs:
        build_knowledge_graph(sub_docs, config)



def build_knowledge_graph(documents, config):
    """Build a simple knowledge graph from documents."""
    print(f"\n{'='*50}")
    print("Building Knowledge Graph...")

    # Debug: Check API key loading
    print(f"Debug: OpenAI API Key loaded: {'Yes' if config.openai_api_key else 'No'}")
    if config.openai_api_key:
        print(f"Debug: API key starts with: {config.openai_api_key[:10]}...")
    else:
        print("Debug: OpenAI API key not found in config!")

    try:
        # Setup Neo4j graph store
        graph_store = Neo4jPGStore(
            username=config.neo4j_username,
            password=config.neo4j_password,
            url=config.neo4j_url,
        )

        # Setup LLM and embedding models
        llm = OpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=config.openai_api_key
        )
        embed_model = OpenAIEmbedding(
            model=config.embedding_model,
            api_key=config.openai_api_key
        )

        print("Creating PropertyGraph Index...")

        # Build knowledge graph index
        index = PropertyGraphIndex.from_documents(
            documents,
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

        print("✓ Knowledge Graph created successfully!")
        print(f"✓ Graph stored in Neo4j at {config.neo4j_url}")
        print(f"✓ Access Neo4j browser at http://localhost:7474")

    except Exception as e:
        print(f"✗ Error building knowledge graph: {str(e)}")
        print("Make sure Neo4j is running: docker compose up")


if __name__ == "__main__":
    main()
