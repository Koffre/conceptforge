# src/rag_engine.py
"""
RAG Engine for document processing and semantic search.

This module handles:
- Loading PDF documents
- Splitting documents into chunks
- Creating embeddings and vector stores
- Performing semantic searches
"""

import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGEngine:
    """
    A RAG (Retrieval-Augmented Generation) engine for document processing.

    This class provides methods to load documents, create embeddings,
    and perform semantic searches.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the RAG engine.

        Args:
            chunk_size: Size of each text chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store: Optional[InMemoryVectorStore] = None
        self.documents: List[Document] = []
        self.embeddings = None

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """
        Initialize the Google Generative AI embeddings.
        Uses the gemini-embedding-001 model (available in all regions).
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )

        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
            print("✅ Embeddings model loaded: gemini-embedding-001")
        except Exception as e:
            raise ValueError(f"Failed to initialize embeddings: {e}")

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF document from the given file path.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {file_path}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        print(f"✅ Loaded {len(documents)} pages from: {os.path.basename(file_path)}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of Document objects to split

        Returns:
            List of Document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)

        print(f"✅ Split into {len(chunks)} chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        return chunks

    def create_index(self, documents: List[Document]) -> InMemoryVectorStore:
        """
        Create a vector store index from the documents.

        Args:
            documents: List of Document objects to index

        Returns:
            InMemoryVectorStore instance
        """
        if not documents:
            raise ValueError("No documents to index")

        if not self.embeddings:
            self._init_embeddings()

        # Create the vector store
        vector_store = InMemoryVectorStore(self.embeddings)

        # Add documents to the store
        vector_store.add_documents(documents)

        # Store for later use
        self.vector_store = vector_store
        self.documents = documents

        print(f"✅ Index created with {len(documents)} chunks")
        return vector_store

    def load_pdf_and_index(self, file_path: str) -> InMemoryVectorStore:
        """
        Complete pipeline: load a PDF, split it, and create an index.

        Args:
            file_path: Path to the PDF file

        Returns:
            InMemoryVectorStore instance
        """
        print(f"📄 Processing: {file_path}")
        print("-" * 40)

        # 1. Load the PDF
        documents = self.load_pdf(file_path)

        # 2. Split into chunks
        chunks = self.split_documents(documents)

        # 3. Create index
        vector_store = self.create_index(chunks)

        print("-" * 40)
        print("✅ RAG engine ready!")
        return vector_store

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform a semantic search on the indexed documents.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of Document objects (the most relevant chunks)
        """
        if not self.vector_store:
            raise ValueError("No index found. Please load a document first.")

        results = self.vector_store.similarity_search(query, k=k)

        print(f"🔍 Found {len(results)} relevant chunks for: '{query[:50]}...'")
        return results

    def get_document_summary(self, k: int = 5) -> str:
        """
        Get a summary of the indexed document by combining the first few chunks.

        Args:
            k: Number of chunks to include in the summary

        Returns:
            Combined text from the first k chunks
        """
        if not self.documents:
            raise ValueError("No documents loaded.")

        # Take the first k chunks as a "summary"
        summary_chunks = self.documents[:k]
        summary_text = "\n\n".join([chunk.page_content for chunk in summary_chunks])

        return summary_text[:2000]  # Limit to 2000 characters for readability

    def clear(self) -> None:
        """Clear the current index and documents."""
        self.vector_store = None
        self.documents = []
        print("🗑️  RAG engine cleared.")


# --- Helper function for quick use ---

def create_rag_engine(file_path: str, chunk_size: int = 1000) -> RAGEngine:
    """
    Convenience function to create a RAG engine from a PDF file.

    Args:
        file_path: Path to the PDF file
        chunk_size: Size of each text chunk

    Returns:
        Configured RAGEngine instance
    """
    engine = RAGEngine(chunk_size=chunk_size)
    engine.load_pdf_and_index(file_path)
    return engine