"""ConceptForge RAG engine with persistent multi-document library."""

import hashlib
import io
import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


# Windows console compatibility
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
    )
except AttributeError:
    pass


class RAGEngine:
    """
    Persistent RAG engine with a multi-document library.

    All indexed documents remain stored in ChromaDB.
    Only one document can be active at a time.

    Document identity is based on the PDF content rather than its
    filesystem path, so the same PDF remains the same document even
    if its path representation changes.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.vector_store: Optional[Chroma] = None
        self.documents: List[Document] = []
        self.embeddings = None

        # Persistent storage
        self.persist_directory = "./chroma_db"
        self.collection_name = "conceptforge_docs"
        self.active_document_file = "./active_document.json"

        # Active document
        self.active_document_id: Optional[str] = None
        self.active_document_name: Optional[str] = None

        # Initialize
        self._init_embeddings()
        self._load_existing_index()
        self._load_active_document()
        self._load_active_document_chunks()

    # ============================================================
    # EMBEDDINGS
    # ============================================================

    def _init_embeddings(self) -> None:
        """Initialize the Google Gemini embedding model."""

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )

        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )

            print("Embeddings model loaded: gemini-embedding-001")

        except Exception as e:
            raise ValueError(
                f"Failed to initialize embeddings: {e}"
            )

    # ============================================================
    # CHROMADB
    # ============================================================

    def _load_existing_index(self) -> None:
        """Load the persistent ChromaDB collection if it exists."""

        if not os.path.exists(self.persist_directory):
            return

        try:
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            count = store._collection.count()

            self.vector_store = store

            if count > 0:
                print(
                    f"Loaded existing ChromaDB index from "
                    f"{self.persist_directory} ({count} chunks)"
                )
            else:
                print("Empty ChromaDB index found.")

        except Exception as e:
            print(
                f"Could not load existing ChromaDB index: {e}"
            )
            self.vector_store = None

    def _ensure_vector_store(self) -> Chroma:
        """Create the ChromaDB store if it does not already exist."""

        if self.vector_store is None:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

        return self.vector_store

    # ============================================================
    # DOCUMENT IDENTITY
    # ============================================================

    def _calculate_document_id(self, file_path: str) -> str:
        """
        Calculate a stable document ID from the PDF content.

        The filesystem path is deliberately NOT included.

        Therefore:

            same PDF + different path = same document_id
        """

        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                hasher.update(chunk)

        return hasher.hexdigest()

    def _normalize_document_name(self, file_path: str) -> str:
        """
        Return only the filename.

        This prevents Windows path separators from becoming part of
        the document identity or display name.
        """

        return os.path.basename(
            os.path.normpath(file_path)
        )

    def _get_chunk_id(
        self,
        document_id: str,
        document: Document,
        index: int,
    ) -> str:
        """
        Generate a deterministic ID for a document chunk.
        """

        page = document.metadata.get("page", 0)

        start_index = document.metadata.get(
            "start_index",
            index,
        )

        raw_id = (
            f"{document_id}:"
            f"{page}:"
            f"{start_index}:"
            f"{document.page_content}"
        )

        return hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

    # ============================================================
    # ACTIVE DOCUMENT STATE
    # ============================================================

    def _load_active_document(self) -> None:
        """Restore the previously selected active document."""

        if not os.path.exists(
            self.active_document_file
        ):
            return

        try:
            with open(
                self.active_document_file,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            self.active_document_id = data.get(
                "document_id"
            )

            self.active_document_name = data.get(
                "document_name"
            )

            if self.active_document_id:
                print(
                    f"Active document restored: "
                    f"{self.active_document_name}"
                )

        except Exception as e:
            print(
                f"Could not load active document state: {e}"
            )

            self.active_document_id = None
            self.active_document_name = None

    def _save_active_document(self) -> None:
        """Persist the active document selection."""

        if not self.active_document_id:
            self._clear_active_document_file()
            return

        data = {
            "document_id": self.active_document_id,
            "document_name": self.active_document_name,
        }

        with open(
            self.active_document_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _clear_active_document_file(self) -> None:
        """Delete the active document state file."""

        try:
            if os.path.exists(
                self.active_document_file
            ):
                os.remove(
                    self.active_document_file
                )

        except OSError as e:
            print(
                f"Could not clear active document state: {e}"
            )

    def _load_active_document_chunks(self) -> None:
        """
        Restore the chunks belonging to the active document.
        """

        self.documents = []

        if (
            not self.vector_store
            or not self.active_document_id
        ):
            return

        try:
            result = self.vector_store.get(
                where={
                    "document_id": self.active_document_id
                },
                include=[
                    "documents",
                    "metadatas",
                ],
            )

            texts = result.get("documents") or []
            metadatas = result.get("metadatas") or []

            self.documents = [
                Document(
                    page_content=text,
                    metadata=metadata or {},
                )
                for text, metadata in zip(
                    texts,
                    metadatas,
                )
            ]

            if self.documents:
                print(
                    f"Restored {len(self.documents)} "
                    f"chunks for active document: "
                    f"{self.active_document_name}"
                )

            else:
                print(
                    "Active document was not found "
                    "in the ChromaDB index."
                )

                self.active_document_id = None
                self.active_document_name = None

                self._clear_active_document_file()

        except Exception as e:
            print(
                f"Could not restore active document chunks: {e}"
            )

            self.documents = []

    # ============================================================
    # DOCUMENT LIBRARY
    # ============================================================

    def get_indexed_documents(self) -> List[str]:
        """
        Return all unique indexed document names.
        """

        if not self.vector_store:
            return []

        try:
            result = self.vector_store.get(
                include=["metadatas"]
            )

            names = set()

            for metadata in (
                result.get("metadatas") or []
            ):
                if not metadata:
                    continue

                source_name = metadata.get(
                    "source_name"
                )

                if source_name:
                    names.add(source_name)
                    continue

                # Compatibility with older indexes
                source = metadata.get("source")

                if source:
                    names.add(
                        os.path.basename(
                            source.replace("\\", "/")
                        )
                    )

            return sorted(names)

        except Exception as e:
            print(
                f"Could not retrieve indexed documents: {e}"
            )

            return []

    def get_indexed_chunk_count(self) -> int:
        """Return the total number of chunks in ChromaDB."""

        if not self.vector_store:
            return 0

        try:
            return self.vector_store._collection.count()

        except Exception:
            return 0

    def _get_document_id_by_name(
        self,
        document_name: str,
    ) -> Optional[str]:
        """
        Resolve a document name to its document ID.

        Uses source_name as the primary lookup field.
        """

        if not self.vector_store:
            return None

        normalized_name = os.path.basename(
            document_name.replace("\\", "/")
        )

        try:
            # Preferred method for indexes created by this version
            result = self.vector_store.get(
                where={
                    "source_name": normalized_name
                },
                limit=1,
                include=["metadatas"],
            )

            metadatas = (
                result.get("metadatas") or []
            )

            if metadatas:
                return metadatas[0].get(
                    "document_id"
                )

            # Compatibility with older indexes
            result = self.vector_store.get(
                include=["metadatas"]
            )

            for metadata in (
                result.get("metadatas") or []
            ):
                if not metadata:
                    continue

                source = metadata.get("source")

                if not source:
                    continue

                stored_name = os.path.basename(
                    source.replace("\\", "/")
                )

                if stored_name == normalized_name:
                    return metadata.get(
                        "document_id"
                    )

        except Exception as e:
            print(
                f"Could not resolve document "
                f"'{document_name}': {e}"
            )

        return None

    # ============================================================
    # ACTIVE DOCUMENT API
    # ============================================================

    def get_active_document(self) -> Optional[str]:
        """Return the active document name."""

        return self.active_document_name

    def get_active_document_id(self) -> Optional[str]:
        """Return the active document ID."""

        return self.active_document_id

    def set_active_document(
        self,
        document_name: str,
    ) -> bool:
        """
        Select an already-indexed document as active.
        """

        document_id = self._get_document_id_by_name(
            document_name
        )

        if not document_id:
            print(
                f"Document not found: {document_name}"
            )
            return False

        self.active_document_id = document_id
        self.active_document_name = (
            self._normalize_document_name(
                document_name
            )
        )

        self._save_active_document()
        self._load_active_document_chunks()

        print(
            f"Active document set to: "
            f"{self.active_document_name}"
        )

        return True

    def clear_active_document(self) -> None:
        """
        Clear the active document selection.

        This DOES NOT delete documents from ChromaDB.
        """

        self.active_document_id = None
        self.active_document_name = None
        self.documents = []

        self._clear_active_document_file()

        print("Active document cleared.")

    # ============================================================
    # PDF PROCESSING
    # ============================================================

    def load_pdf(
        self,
        file_path: str,
    ) -> List[Document]:
        """Load a PDF document."""

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(
                f"File must be a PDF: {file_path}"
            )

        documents = PyPDFLoader(
            file_path
        ).load()

        print(
            f"Loaded {len(documents)} pages from: "
            f"{self._normalize_document_name(file_path)}"
        )

        return documents

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents into retrieval chunks."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(
            documents
        )

        print(
            f"Split into {len(chunks)} chunks "
            f"(size: {self.chunk_size}, "
            f"overlap: {self.chunk_overlap})"
        )

        return chunks

    # ============================================================
    # INDEXING
    # ============================================================

    def create_index(
        self,
        documents: List[Document],
    ) -> Chroma:
        """
        Add a document to the persistent library.

        If the exact same PDF is already indexed,
        no duplicate chunks are added.

        The newly processed document becomes active.
        """

        if not documents:
            raise ValueError(
                "No documents to index"
            )

        if not self.embeddings:
            self._init_embeddings()

        store = self._ensure_vector_store()

        # Original PDF path stored by PyPDFLoader
        source = documents[0].metadata.get(
            "source",
            "",
        )

        if not source:
            raise ValueError(
                "Document source information is missing."
            )

        source_name = self._normalize_document_name(
            source
        )

        # IMPORTANT:
        # Document ID depends ONLY on PDF content.
        document_id = self._calculate_document_id(
            source
        )

        prepared_documents = []
        chunk_ids = []

        for index, document in enumerate(
            documents
        ):
            metadata = dict(
                document.metadata
            )

            metadata["document_id"] = (
                document_id
            )

            metadata["source_name"] = (
                source_name
            )

            document.metadata = metadata

            chunk_id = self._get_chunk_id(
                document_id,
                document,
                index,
            )

            prepared_documents.append(
                document
            )

            chunk_ids.append(chunk_id)

        # Check whether these exact chunks already exist
        existing_ids = set()

        try:
            existing = store.get(
                ids=chunk_ids,
                include=[],
            )

            existing_ids = set(
                existing.get("ids") or []
            )

        except Exception:
            pass

        new_documents = []
        new_ids = []

        for document, chunk_id in zip(
            prepared_documents,
            chunk_ids,
        ):
            if chunk_id not in existing_ids:
                new_documents.append(
                    document
                )
                new_ids.append(chunk_id)

        if new_documents:
            store.add_documents(
                new_documents,
                ids=new_ids,
            )

            print(
                f"Added {len(new_documents)} "
                f"new chunks to ChromaDB."
            )

        else:
            print(
                "Document chunks already exist "
                "in ChromaDB."
            )

        print(
            f"ChromaDB collection contains "
            f"{self.get_indexed_chunk_count()} chunks."
        )

        # Make this document active
        self.active_document_id = document_id
        self.active_document_name = (
            source_name
        )

        self.documents = (
            prepared_documents
        )

        self._save_active_document()

        return store

    def load_pdf_and_index(
        self,
        file_path: str,
    ) -> Chroma:
        """
        Complete pipeline:
        load PDF → split → index → activate.
        """

        print(
            f"Processing: {file_path}"
        )

        print("-" * 40)

        documents = self.load_pdf(
            file_path
        )

        chunks = self.split_documents(
            documents
        )

        store = self.create_index(
            chunks
        )

        print("-" * 40)

        print(
            "RAG engine ready with persistent storage."
        )

        return store

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> List[Document]:
        """
        Semantic search restricted to the active document.
        """

        if not self.vector_store:
            self._load_existing_index()

        if not self.vector_store:
            raise ValueError(
                "No indexed documents found."
            )

        if not self.active_document_id:
            raise ValueError(
                "No active document selected. "
                "Please select a document first."
            )

        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter={
                "document_id":
                self.active_document_id
            },
        )

        print(
            f"Found {len(results)} relevant chunks "
            f"for: '{query[:50]}...'"
        )

        return results

    # ============================================================
    # ACTIVE DOCUMENT CONTENT
    # ============================================================

    def get_document_chunks(
        self,
        k: int = 5,
    ) -> List[Document]:
        """
        Return chunks from the active document only.
        """

        if not self.vector_store:
            self._load_existing_index()

        if (
            not self.vector_store
            or not self.active_document_id
        ):
            return []

        try:
            result = self.vector_store.get(
                where={
                    "document_id":
                    self.active_document_id
                },
                limit=k,
                include=[
                    "documents",
                    "metadatas",
                ],
            )

            return [
                Document(
                    page_content=text,
                    metadata=metadata or {},
                )
                for text, metadata in zip(
                    result.get("documents") or [],
                    result.get("metadatas") or [],
                )
            ]

        except Exception as e:
            print(
                f"Could not retrieve document chunks: {e}"
            )

            return []

    def get_document_summary(
        self,
        k: int = 5,
    ) -> str:
        """
        Return source text from the active document.

        The MCP/LLM layer is responsible for generating
        the actual natural-language summary.
        """

        if not self.active_document_id:
            raise ValueError(
                "No active document selected. "
                "Please select a document first."
            )

        chunks = self.get_document_chunks(k)

        if not chunks:
            raise ValueError(
                "No chunks found for the active document."
            )

        return "\n\n".join(
            chunk.page_content
            for chunk in chunks
        )[:2000]

    # ============================================================
    # CLEAR / STATUS
    # ============================================================

    def clear(self) -> None:
        """
        Clear only the active document selection.

        Nothing is deleted from ChromaDB.
        """

        self.clear_active_document()

        print(
            "RAG engine active document cleared."
        )

    def has_index(self) -> bool:
        """Return True if ChromaDB contains indexed chunks."""

        if self.vector_store:
            try:
                return (
                    self.vector_store
                    ._collection
                    .count()
                    > 0
                )
            except Exception:
                return False

        if not os.path.exists(
            self.persist_directory
        ):
            return False

        try:
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            if (
                store._collection.count()
                > 0
            ):
                self.vector_store = store
                return True

        except Exception:
            pass

        return False


# ================================================================
# HELPER
# ================================================================

def create_rag_engine(
    file_path: str,
    chunk_size: int = 1000,
) -> RAGEngine:
    """
    Convenience function for creating and indexing a RAG engine.
    """

    engine = RAGEngine(
        chunk_size=chunk_size
    )

    engine.load_pdf_and_index(
        file_path
    )

    return engine