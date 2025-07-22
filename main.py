#!/usr/bin/env python3
"""
Main script to demonstrate DocumentProcessor functionality.
Loads documents from the input folder and processes them.
"""

import os
from pathlib import Path
from core.document_processor import DocumentProcessor
from config.settings import get_config


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



if __name__ == "__main__":
    main()
