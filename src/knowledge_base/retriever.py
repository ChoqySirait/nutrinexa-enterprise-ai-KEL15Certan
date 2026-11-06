"""
NutriNexa Semantic Retriever Module.
Mengkoordinasikan EmbeddingEngine dan ChromaVectorStore untuk mengeksekusi
pencarian semantik SOP gizi dan pedoman food waste berbasis kemiripan kosinus.
"""

from __future__ import annotations

import logging
from typing import Any

from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.schemas import (
    RetrievalResult,
    RetrieverError,
)
from src.knowledge_base.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Mesin pencarian semantik tingkat tinggi untuk subsistem Knowledge Base NutriNexa.
    Menerima kueri mentah, melakukan encoding asimetris, dan mengembalikan RetrievalResult terstruktur.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: ChromaVectorStore,
        default_top_k: int = 5,
    ) -> None:
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Mengeksekusi pencarian semantik untuk sebuah kueri teks pengguna.
        Menghasilkan daftar RetrievalResult berurutan berdasarkan relevansi kosinus tertinggi.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Kueri pencarian tidak boleh kosong atau hanya berupa whitespace.")

        effective_top_k = top_k if top_k is not None else self.default_top_k
        if effective_top_k < 1:
            raise ValueError("Nilai `top_k` pencarian minimal bernilai 1.")

        # 1. Hasilkan embedding query asimetris (prefix otomatis 'query: ')
        try:
            query_res = self.embedding_engine.embed_query(query)
            query_vec = query_res.embedding
        except Exception as e:
            raise RetrieverError(f"Kegagalan inferensi embedding untuk kueri pencarian: {e}") from e

        # 2. Eksekusi pencarian vektor pada ChromaDB
        try:
            raw_results = self.vector_store.search(
                query_embedding=query_vec,
                top_k=effective_top_k,
                where=where,
            )
        except Exception as e:
            raise RetrieverError(f"Kegagalan pencarian vektor pada Vector Store: {e}") from e

        # 3. Parsing dan strukturisasi hasil pencarian
        ids_list = raw_results.get("ids", [[]])
        docs_list = raw_results.get("documents", [[]])
        meta_list = raw_results.get("metadatas", [[]])
        dist_list = raw_results.get("distances", [[]])

        # Pastikan list tidak kosong
        ids = ids_list[0] if ids_list else []
        documents = docs_list[0] if docs_list else []
        metadatas = meta_list[0] if meta_list else []
        distances = dist_list[0] if dist_list else []

        structured_results: list[RetrievalResult] = []
        for rank_idx, (chunk_id, doc_text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True),
            start=1,
        ):
            # Rumus similarity kosinus: similarity = 1.0 - distance
            cosine_similarity = 1.0 - float(dist)

            structured_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=doc_text,
                    metadata=dict(meta) if meta else {},
                    rank=rank_idx,
                    distance=float(dist),
                    similarity=float(cosine_similarity),
                )
            )

        return structured_results

