from llama_index.core.node_parser import SentenceSplitter
from config.settings import get_config, ComponentsConfig
from llama_parse import LlamaParse
from typing import Optional
from typing import List, Dict, Any
from llama_index.core import Document
from pathlib import Path
from copy import deepcopy
import tempfile
import os

class DocumentProcessor:
    def __init__(self, config: Optional[ComponentsConfig] = None):
        self.config = config or get_config()
        self._node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._llama_parse = None

    @property
    def llama_parse(self) -> LlamaParse:
        """Get or create LlamaParse instance."""
        if self._llama_parse is None:
            if not self.config.llama_cloud_api_key:
                raise ValueError(
                    "LlamaCloud API key is required. Set LLAMA_CLOUD_API_KEY environment variable."
                )

            self._llama_parse = LlamaParse(
                result_type=self.config.result_type,
                verbose=self.config.verbose,
                api_key=self.config.llama_cloud_api_key,
            )
        return self._llama_parse

    async def load_documents_from_file(
        self,
        file_object,
        use_llama_parse: bool = True
    ) -> List[Document]:
        """
        Load documents from an uploaded file object.

        Args:
            file_object: Uploaded file object with read() method
            use_llama_parse: Whether to use LlamaParse for parsing

        Returns:
            List of Document objects
        """

        filename = getattr(file_object, 'filename', 'unknown')
        file_extension = Path(filename).suffix.lower()

        if use_llama_parse and file_extension == '.pdf':
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                await file_object.seek(0)
                content = await file_object.read()
                if isinstance(content, str):
                    content = content.encode('utf-8')
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                result = self._load_with_llama_parse(Path(tmp_file_path))
                return result
            finally:
                os.unlink(tmp_file_path)
        else:
            await file_object.seek(0)
            content = await file_object.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            return self._load_with_simple_loader_from_content(content, filename)

    def _load_with_simple_loader_from_content(self, content: str, filename: str) -> List[Document]:
        """Load documents from content string."""
        from llama_index.core import Document
        return [Document(text=content, metadata={"filename": filename})]

    def _load_with_llama_parse(self, file_path: Path) -> List[Document]:
        """Load documents using LlamaParse."""
        try:
            documents = self.llama_parse.load_data(str(file_path))
            return documents
        except Exception as e:
            print(f"LlamaParse failed for {file_path}: {str(e)}")
            return self._load_with_simple_loader(file_path)

    def _load_with_simple_loader(self, file_path: Path) -> List[Document]:
        """Load documents using simple text loader."""
        try:
            # Handle PDF files differently when LlamaParse is not available
            if file_path.suffix.lower() == '.pdf':
                print(f"Warning: PDF file {file_path.name} requires LlamaParse or a PDF library for proper text extraction.")
                print("Returning empty document with metadata only.")
                document = Document(
                    text=f"PDF file: {file_path.name} - Content extraction requires LlamaParse API key or PDF processing library.",
                    metadata={
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "file_size": file_path.stat().st_size,
                        "processing_note": "PDF content not extracted - requires LlamaParse or PDF library"
                    }
                )
                return [document]

            # For text files, try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise ValueError(f"Could not decode file {file_path} with any supported encoding")

            document = Document(
                text=content,
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "file_size": file_path.stat().st_size,
                }
            )

            return [document]

        except Exception as e:
            print(f"Simple loader failed for {file_path}: {str(e)}")
            raise

    def split_documents_into_pages(self, documents: List[Document]) -> List[Document]:
                """
                Split documents into pages based on LlamaParse page separators.

                Args:
                    documents: List of documents to split

                Returns:
                    List of split documents
                """
                sub_docs = []

                for doc in documents:
                    doc_chunks = doc.text.split("\n---\n")

                    for i, doc_chunk in enumerate(doc_chunks):
                        if doc_chunk.strip():
                            sub_doc = Document(
                                text=doc_chunk.strip(),
                                metadata={
                                    **deepcopy(doc.metadata),
                                    "page_number": i + 1,
                                    "total_pages": len(doc_chunks),
                                    "is_sub_document": True,
                                }
                            )
                            sub_docs.append(sub_doc)

                print(f"Split {len(documents)} documents into {len(sub_docs)} pages")
                return sub_docs

    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the loaded documents.

        Args:
            documents: List of documents

        Returns:
            Dictionary with document statistics
        """
        total_docs = len(documents)
        total_chars = sum(len(doc.text) for doc in documents)
        total_words = sum(len(doc.text.split()) for doc in documents)

        file_types = {}
        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1

        avg_chars = total_chars / total_docs if total_docs > 0 else 0
        avg_words = total_words / total_docs if total_docs > 0 else 0

        return {
            "total_documents": total_docs,
            "total_characters": total_chars,
            "total_words": total_words,
            "average_characters_per_doc": avg_chars,
            "average_words_per_doc": avg_words,
            "file_types": file_types,
        }
