"""
NutriNexa Persistent Vector Store Module.
Wrapper ChromaDB PersistentClient untuk penyimpanan vektor dokumen SOP gizi
dan pedoman food waste dengan metrik cosine distance dan operasi idempotent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.knowledge_base.schemas import (
    DocumentChunk,
    VectorStoreError,
)

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Penyimpanan vektor persisten lokal berbasis ChromaDB PersistentClient.
    Dikonfigurasi dengan metrik cosine distance dan dukungan operasi batching.
    """

    DEFAULT_PERSIST_DIR: str = "storage/chroma_db"
    DEFAULT_COLLECTION_NAME: str = "nutrinexa_knowledge_base"
    DEFAULT_EXPECTED_DIM: int = 384

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        expected_dim: int = DEFAULT_EXPECTED_DIM,
        client: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.expected_dim = expected_dim

        # Injeksi client eksternal (misal mock client untuk unit testing)
        self._client = client
        self._collection: Any | None = None

    @property
    def client(self) -> Any:
        """Lazy initialization ChromaDB PersistentClient."""
        if self._client is None:
            self.init_client()
        return self._client

    @property
    def collection(self) -> Any:
        """Mendapatkan atau membuat ChromaDB collection dengan metrik cosine distance."""
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection

    def init_client(self) -> None:
        """Inisialisasi ChromaDB PersistentClient lokal."""
        if self._client is not None:
            return

        try:
            import chromadb

            # Buat direktori penyimpanan jika belum ada
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info("Menginisialisasi ChromaDB PersistentClient pada direktori: %s", self.persist_directory)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory.resolve()))
        except Exception as e:
            raise VectorStoreError(f"Gagal menginisialisasi ChromaDB PersistentClient: {e}") from e

    def _get_or_create_collection(self) -> Any:
        """Membuat atau membuka collection yang sudah ada dengan konfigurasi cosine space."""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(
                f"Gagal membuka atau membuat koleksi ChromaDB '{self.collection_name}': {e}"
            ) from e

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Menyimpan sekumpulan DocumentChunk beserta vektor embedding-nya secara batching
        menggunakan metode upsert deterministik untuk mencegah duplikasi data.
        """
        if not isinstance(chunks, list) or len(chunks) == 0:
            raise ValueError("Daftar DocumentChunk tidak boleh kosong.")
        if not isinstance(embeddings, list) or len(embeddings) == 0:
            raise ValueError("Daftar embeddings tidak boleh kosong.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Jumlah chunk ({len(chunks)}) tidak sepadan dengan jumlah embedding ({len(embeddings)})."
            )

        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for chunk, emb in zip(batch_chunks, batch_embeddings, strict=True):
                # Validasi dimensi vektor embedding
                if len(emb) != self.expected_dim:
                    raise VectorStoreError(
                        f"Dimensi vektor ({len(emb)}) pada chunk '{chunk.chunk_id}' "
                        f"tidak sesuai dengan dimensi koleksi ({self.expected_dim})."
                    )

                ids.append(chunk.chunk_id)
                documents.append(chunk.text)
                metadatas.append(dict(chunk.metadata))

            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=batch_embeddings,
                    metadatas=metadatas,
                )
            except Exception as e:
                raise VectorStoreError(
                    f"Gagal melakukan upsert batch pada koleksi '{self.collection_name}': {e}"
                ) from e

        return total_chunks

    def delete_by_source(self, source_id: str) -> None:
        """
        Menghapus seluruh chunk dokumen yang memiliki `source_id` bersangkutan dari ChromaDB.
        """
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Parameter `source_id` wajib berupa string non-kosong.")

        try:
            self.collection.delete(where={"source_id": source_id.strip()})
            logger.info("Berhasil menghapus seluruh chunk untuk source_id: %s", source_id)
        except Exception as e:
            raise VectorStoreError(
                f"Gagal menghapus dokumen dengan source_id '{source_id}': {e}"
            ) from e

    def replace_source_documents(
        self,
        source_id: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Operasi atomik untuk memperbarui revisi dokumen:
        Menghapus seluruh versi chunk lama berdasarkan `source_id`, lalu memasukkan chunk baru.
        """
        self.delete_by_source(source_id)
        return self.add_chunks(chunks, embeddings, batch_size=batch_size)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Melakukan pencarian kemiripan kosinus terhadap koleksi ChromaDB.
        """
        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            raise ValueError("Query embedding tidak boleh kosong.")
        if len(query_embedding) != self.expected_dim:
            raise VectorStoreError(
                f"Dimensi query embedding ({len(query_embedding)}) tidak sesuai "
                f"dengan dimensi koleksi ({self.expected_dim})."
            )
        if top_k < 1:
            raise ValueError("Nilai `top_k` minimal bernilai 1.")

        include_fields = include if include is not None else ["documents", "metadatas", "distances"]
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": include_fields,
        }
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
            return results
        except Exception as e:
            raise VectorStoreError(f"Kegagalan eksekusi pencarian vektor pada ChromaDB: {e}") from e

    def count(self) -> int:
        """Mengembalikan total jumlah chunk dokumen yang tersimpan dalam koleksi."""
        try:
            return self.collection.count()
        except Exception as e:
            raise VectorStoreError(f"Gagal menghitung jumlah record koleksi: {e}") from e

