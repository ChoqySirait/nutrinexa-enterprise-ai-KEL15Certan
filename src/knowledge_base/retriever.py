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

import numpy as np

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Mesin pencarian semantik tingkat tinggi untuk subsistem Knowledge Base NutriNexa.
    Menerima kueri mentah, melakukan encoding asimetris, dan mengembalikan RetrievalResult terstruktur.
    Mendukung penyaringan ambang batas kemiripan (min_similarity) dan diversifikasi hasil (MMR).
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

    @staticmethod
    def _compute_mmr(
        candidate_embeddings: list[list[float]],
        candidate_similarities: list[float],
        top_k: int,
        mmr_lambda: float,
    ) -> list[int]:
        """
        Menghitung urutan indeks kandidat menggunakan Maximal Marginal Relevance (MMR).
        MMR = argmax_{d_i in R \\ S} [ lambda * Sim(d_i, q) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j) ]
        """
        if not candidate_embeddings or top_k <= 0:
            return []

        cand_embs = np.array(candidate_embeddings, dtype=np.float32)
        norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cand_embs = cand_embs / norms

        selected_indices: list[int] = []
        unselected_indices: list[int] = list(range(len(cand_embs)))

        # 1. Pilih dokumen pertama dengan kemiripan tertinggi terhadap kueri
        first_selected = max(unselected_indices, key=lambda idx: candidate_similarities[idx])
        selected_indices.append(first_selected)
        unselected_indices.remove(first_selected)

        # 2. Pilih dokumen berikutnya secara iteratif dengan memaksimalkan skor MMR
        while len(selected_indices) < top_k and unselected_indices:
            best_score = -float("inf")
            best_cand_idx: int | None = None

            selected_matrix = cand_embs[selected_indices]  # Shape: (len(selected), dim)

            for cand_idx in unselected_indices:
                cand_vec = cand_embs[cand_idx]
                sim_to_query = candidate_similarities[cand_idx]

                # Kemiripan kosinus maksimum terhadap dokumen yang sudah terpilih
                sim_to_selected = float(np.max(np.dot(selected_matrix, cand_vec)))

                mmr_score = mmr_lambda * sim_to_query - (1.0 - mmr_lambda) * sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_cand_idx = cand_idx

            if best_cand_idx is not None:
                selected_indices.append(best_cand_idx)
                unselected_indices.remove(best_cand_idx)
            else:
                break

        return selected_indices

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        min_similarity: float | None = None,
        enable_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ) -> list[RetrievalResult]:
        """
        Mengeksekusi pencarian semantik untuk sebuah kueri teks pengguna.
        Menghasilkan daftar RetrievalResult berurutan berdasarkan relevansi kosinus tertinggi.

        Parameters
        ----------
        query : str
            Kueri pencarian teks pengguna.
        top_k : int | None
            Jumlah dokumen teratas yang ingin dikembalikan (default: default_top_k).
        where : dict[str, Any] | None
            Filter metadata opsional untuk ChromaDB.
        min_similarity : float | None
            Ambang batas kemiripan kosinus minimum [0.0, 1.0]. Dokumen dengan similarity
            di bawah ambang batas ini akan disaring keluar (Out-of-Distribution filtering).
        enable_mmr : bool
            Jika True, menerapkan Maximal Marginal Relevance (MMR) re-ranking untuk
            mencegah redundansi hasil dan meningkatkan keragaman dokumen (diversity).
        mmr_lambda : float
            Bobot trade-off MMR [0.0, 1.0]. Nilai 1.0 murni relevansi kueri, 0.0 murni keberagaman.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Kueri pencarian tidak boleh kosong atau hanya berupa whitespace.")

        effective_top_k = top_k if top_k is not None else self.default_top_k
        if effective_top_k < 1:
            raise ValueError("Nilai `top_k` pencarian minimal bernilai 1.")

        if min_similarity is not None and not (0.0 <= min_similarity <= 1.0):
            raise ValueError("Nilai `min_similarity` harus berada dalam rentang [0.0, 1.0].")

        if not (0.0 <= mmr_lambda <= 1.0):
            raise ValueError("Nilai `mmr_lambda` harus berada dalam rentang [0.0, 1.0].")

        # 1. Hasilkan embedding query asimetris (prefix otomatis 'query: ')
        try:
            query_res = self.embedding_engine.embed_query(query)
            query_vec = query_res.embedding
        except Exception as e:
            raise RetrieverError(f"Kegagalan inferensi embedding untuk kueri pencarian: {e}") from e

        # 2. Tentukan ukuran kandidat dan field include
        include_fields = ["documents", "metadatas", "distances"]
        if enable_mmr:
            include_fields.append("embeddings")
            candidate_fetch_k = max(effective_top_k * 3, 10)
        elif min_similarity is not None:
            candidate_fetch_k = max(effective_top_k * 2, effective_top_k)
        else:
            candidate_fetch_k = effective_top_k

        # 3. Eksekusi pencarian vektor pada ChromaDB
        try:
            raw_results = self.vector_store.search(
                query_embedding=query_vec,
                top_k=candidate_fetch_k,
                where=where,
                include=include_fields,
            )
        except Exception as e:
            raise RetrieverError(f"Kegagalan pencarian vektor pada Vector Store: {e}") from e

        # 4. Parsing dan ekstraksi hasil mentah
        ids_list = raw_results.get("ids", [[]])
        docs_list = raw_results.get("documents", [[]])
        meta_list = raw_results.get("metadatas", [[]])
        dist_list = raw_results.get("distances", [[]])
        embs_list = raw_results.get("embeddings", [[]]) if enable_mmr else []

        ids = ids_list[0] if ids_list else []
        documents = docs_list[0] if docs_list else []
        metadatas = meta_list[0] if meta_list else []
        distances = dist_list[0] if dist_list else []
        has_embeddings = embs_list is not None and len(embs_list) > 0 and embs_list[0] is not None and len(embs_list[0]) > 0
        embeddings = embs_list[0] if has_embeddings else []

        # 5. Filter berdasarkan ambang batas kemiripan (min_similarity)
        valid_indices: list[int] = []
        similarities: list[float] = []
        for idx, dist in enumerate(distances):
            sim = 1.0 - float(dist)
            if min_similarity is not None and sim < min_similarity:
                continue
            valid_indices.append(idx)
            similarities.append(sim)

        if not valid_indices:
            return []

        # 6. Re-ranking: MMR atau potong top_k langsung
        if enable_mmr and has_embeddings and len(valid_indices) > 1:
            filtered_embs = [embeddings[i] for i in valid_indices]
            mmr_order = self._compute_mmr(
                candidate_embeddings=filtered_embs,
                candidate_similarities=similarities,
                top_k=effective_top_k,
                mmr_lambda=mmr_lambda,
            )
            final_indices = [valid_indices[i] for i in mmr_order]
        else:
            final_indices = valid_indices[:effective_top_k]

        # 7. Konstruksi daftar RetrievalResult terstruktur
        structured_results: list[RetrievalResult] = []
        for rank_idx, orig_idx in enumerate(final_indices, start=1):
            dist = float(distances[orig_idx])
            sim = 1.0 - dist
            structured_results.append(
                RetrievalResult(
                    chunk_id=ids[orig_idx],
                    text=documents[orig_idx],
                    metadata=dict(metadatas[orig_idx]) if metadatas[orig_idx] else {},
                    rank=rank_idx,
                    distance=dist,
                    similarity=sim,
                )
            )

        return structured_results

