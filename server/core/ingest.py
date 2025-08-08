from llama_index.core.indices.property_graph import ImplicitPathExtractor, SimpleLLMPathExtractor
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPGStore
from core.document_processor import DocumentProcessor
from llama_index.llms.openai import OpenAI
from config.settings import get_config
from pathlib import Path
import tempfile
import os
import asyncio
import concurrent.futures
from server.minio_client.client import MinioClient

PERSIST_DIR = "./storage"
async def ingest_file(file):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, file.filename)

            with open(temp_file_path, 'wb') as temp_file:
                content = await file.read()
                temp_file.write(content)
                # build_knowledge_graph
                await build_knowledge_graph(file, config=get_config())

    except ValueError as e:
        raise e

    return {"message": "File ingested successfully"}

def get_graph_store(config):
    """Initialize and return Neo4j graph store."""
    return Neo4jPGStore(
        username=config.neo4j_username,
        password=config.neo4j_password,
        url=config.neo4j_url,
    )

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

async def build_knowledge_graph(file, config):
    print("Building knowledge graph...")

    llm, embed_model = setup_models(config)
    graph_store = get_graph_store(config)
    processor = DocumentProcessor(config)

    file_extension = Path(file.filename).suffix.lower()
    supported_extensions = ['.pdf', '.doc', '.docx', '.txt', '.md']
    if file_extension in supported_extensions:
        await file.seek(0)
        all_docs = await processor.load_documents_from_file(file, use_llama_parse=True)
        print(f"Loaded {len(all_docs)} documents from {file.filename}")
        sub_docs = processor.split_documents_into_pages(all_docs)
        print(f"Total pages after splitting: {len(sub_docs)}")
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            index = await loop.run_in_executor(
                executor,
                lambda: PropertyGraphIndex.from_documents(
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
            )
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

        print(f"✓ Knowledge graph built and persisted to {PERSIST_DIR}")
        print(f"✓ Neo4j available at {config.neo4j_url}")
    else:
        print(f"Unsupported file type: {file_extension}")
