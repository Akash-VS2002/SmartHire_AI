"""
rag/vectorstore.py
------------------
ChromaDB-based vector store for resume and JD chunks.

Architecture:
  - Each analysis session gets a unique in-memory ChromaDB collection.
  - Chunks are embedded via SentenceTransformer and stored with metadata.
  - Exposes similarity_search() for RAG context retrieval.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings

from rag.embeddings import get_embeddings, get_single_embedding
from utils.text_cleaner import chunk_text

logger = logging.getLogger(__name__)


class ResumeVectorStore:
    """
    Manages ChromaDB collections for resume and JD text chunks.
    Uses an ephemeral (in-memory) client so no disk setup is required.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialise an ephemeral ChromaDB client.

        Args:
            session_id: Unique ID for this analysis session.
                        If None, a UUID is generated automatically.
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        # In-memory client — no persistence required for this pipeline.
        # Switch to chromadb.PersistentClient(path="./chroma_db") for disk storage.
        self.client = chromadb.EphemeralClient()
        self._resume_collection: Optional[chromadb.Collection] = None
        self._jd_collection: Optional[chromadb.Collection] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get existing or create new ChromaDB collection."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},  # use cosine distance
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_resume(
        self,
        resume_text: str,
        candidate_name: str = "candidate",
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        """
        Chunk and index a resume into the resume collection.

        Args:
            resume_text:    Full cleaned resume text.
            candidate_name: Used as metadata for retrieval filtering.
            chunk_size:     Characters per chunk.
            overlap:        Overlapping characters between chunks.

        Returns:
            Number of chunks stored.
        """
        collection_name = f"resumes_{self.session_id}"
        self._resume_collection = self._get_or_create_collection(collection_name)

        chunks = chunk_text(resume_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            logger.warning(f"No chunks generated for resume: {candidate_name}")
            return 0

        embeddings = get_embeddings(chunks)
        ids = [f"{candidate_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"candidate": candidate_name, "chunk_index": i} for i in range(len(chunks))]

        self._resume_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(f"Stored {len(chunks)} resume chunks for '{candidate_name}'.")
        return len(chunks)

    def add_job_description(
        self,
        jd_text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        """
        Chunk and index the job description into the JD collection.

        Args:
            jd_text:    Full cleaned JD text.
            chunk_size: Characters per chunk.
            overlap:    Overlapping characters.

        Returns:
            Number of chunks stored.
        """
        collection_name = f"jd_{self.session_id}"
        self._jd_collection = self._get_or_create_collection(collection_name)

        chunks = chunk_text(jd_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            logger.warning("No chunks generated for job description.")
            return 0

        embeddings = get_embeddings(chunks)
        ids = [f"jd_{i}" for i in range(len(chunks))]
        metadatas = [{"source": "job_description", "chunk_index": i} for i in range(len(chunks))]

        self._jd_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(f"Stored {len(chunks)} JD chunks.")
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        collection: str = "resume",
        candidate_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[str]:
        """
        Retrieve the most semantically similar chunks to the query.

        Args:
            query:          Query string for similarity search.
            collection:     "resume" or "jd".
            candidate_name: If provided, filter resume results by this candidate.
            top_k:          Number of top results to return.

        Returns:
            List of matching text chunks.
        """
        try:
            if collection == "resume" and self._resume_collection:
                coll = self._resume_collection
            elif collection == "jd" and self._jd_collection:
                coll = self._jd_collection
            else:
                logger.warning(f"Collection '{collection}' not initialised.")
                return []

            query_embedding = get_single_embedding(query)

            where_filter: Optional[Dict] = None
            if candidate_name and collection == "resume":
                where_filter = {"candidate": candidate_name}

            results = coll.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, coll.count()),
                where=where_filter,
                include=["documents"],
            )

            docs: List[str] = []
            if results and results.get("documents"):
                for doc_list in results["documents"]:
                    docs.extend(doc_list)
            return docs

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_resume_context(self, query: str, candidate_name: str, top_k: int = 4) -> str:
        """
        Retrieve relevant resume context for LLM prompts.

        Returns:
            Joined context string ready for inclusion in a prompt.
        """
        chunks = self.similarity_search(
            query=query,
            collection="resume",
            candidate_name=candidate_name,
            top_k=top_k,
        )
        return "\n\n".join(chunks) if chunks else ""

    def get_jd_context(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve relevant JD context for LLM prompts.

        Returns:
            Joined context string ready for inclusion in a prompt.
        """
        chunks = self.similarity_search(query=query, collection="jd", top_k=top_k)
        return "\n\n".join(chunks) if chunks else ""

    def reset(self) -> None:
        """Delete all collections in this session (cleanup)."""
        try:
            for name in [f"resumes_{self.session_id}", f"jd_{self.session_id}"]:
                self.client.delete_collection(name)
            self._resume_collection = None
            self._jd_collection = None
            logger.info("VectorStore reset successfully.")
        except Exception as e:
            logger.warning(f"Reset encountered an error: {e}")
