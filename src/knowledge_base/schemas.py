"""
NutriNexa Knowledge Base Schemas & Data Contracts.
Mendefinisikan entitas data terstruktur untuk modul Ingestion, Embedding,
ChromaDB Vector Store, dan Semantic Retrieval (Milestone T03).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Primitive scalar types yang didukung secara resmi oleh ChromaDB metadata
ChromaScalar = str | int | float | bool

SUPPORTED_METADATA_KEYS: frozenset[str] = frozenset({
    "source_id",
    "chunk_index",
    "source_title",
    "issuer",
    "year",
    "document_type",
    "category",
    "language",
    "page_start",
    "page_end",
    "section_title",
    "source_path",
    "source_url",
    "file_hash",
    "pipeline_version",
})


class KnowledgeBaseError(Exception):
    """Base exception untuk seluruh subsistem Knowledge Base NutriNexa."""


class SchemaValidationError(KnowledgeBaseError, ValueError):
    """Dilemparkan ketika data input atau metadata melanggar kontrak skema."""


class TokenLimitExceededError(KnowledgeBaseError, ValueError):
    """Dilemparkan ketika panjang token teks melebihi kapasitas konteks model (512 token)."""


class EmbeddingEngineError(KnowledgeBaseError):
    """Dilemparkan ketika terjadi kegagalan pada proses inferensi embedding."""


class VectorStoreError(KnowledgeBaseError):
    """Dilemparkan ketika terjadi kegagalan operasi pada ChromaDB."""


class RetrieverError(KnowledgeBaseError):
    """Dilemparkan ketika terjadi kegagalan pada eksekusi semantic retrieval."""


@dataclass(frozen=True)
class DocumentChunk:
    """
    Representasi segmen teks atomik dari dokumen SOP gizi enterprise atau panduan food waste.
    """
    chunk_id: str
    text: str
    metadata: dict[str, ChromaScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Memvalidasi integritas data chunk dan kompatibilitas metadata ChromaDB."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise SchemaValidationError("`chunk_id` wajib berupa string non-kosong.")

        if not isinstance(self.text, str) or not self.text.strip():
            raise SchemaValidationError("`text` wajib berupa string non-kosong.")

        if not isinstance(self.metadata, dict):
            raise SchemaValidationError("`metadata` wajib berupa dictionary.")

        # Validasi metadata wajib: source_id dan chunk_index
        if "source_id" not in self.metadata:
            raise SchemaValidationError("Metadata wajib menyertakan kunci 'source_id'.")

        source_id = self.metadata["source_id"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise SchemaValidationError("'source_id' dalam metadata wajib berupa string non-kosong.")

        if "chunk_index" not in self.metadata:
            raise SchemaValidationError("Metadata wajib menyertakan kunci 'chunk_index'.")

        chunk_index = self.metadata["chunk_index"]
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise SchemaValidationError("'chunk_index' dalam metadata wajib berupa integer non-negatif.")

        # Validasi tipe nilai metadata (hanya skalar primitif yang didukung ChromaDB)
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise SchemaValidationError(f"Kunci metadata '{key}' tidak valid.")

            if not isinstance(value, (str, int, float, bool)):
                raise SchemaValidationError(
                    f"Nilai metadata untuk kunci '{key}' ({type(value).__name__}) tidak didukung oleh ChromaDB. "
                    "Hanya tipe skalar (str, int, float, bool) yang diizinkan."
                )

    def to_dict(self) -> dict[str, Any]:
        """Konversi objek chunk ke dictionary standar."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentChunk:
        """Instansiasi DocumentChunk dari dictionary."""
        if not isinstance(data, dict):
            raise SchemaValidationError("Input pembuatan chunk harus berupa dictionary.")
        return cls(
            chunk_id=data.get("chunk_id", ""),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Hasil inferensi vektor representasi teks dari EmbeddingEngine.
    """
    text: str
    embedding: list[float]
    token_count: int
    model_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise SchemaValidationError("`text` pada EmbeddingResult wajib berupa string non-kosong.")
        if not isinstance(self.embedding, list) or len(self.embedding) == 0:
            raise SchemaValidationError("`embedding` wajib berupa list float non-kosong.")
        if not isinstance(self.token_count, int) or self.token_count <= 0:
            raise SchemaValidationError("`token_count` wajib berupa integer positif.")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise SchemaValidationError("`model_name` wajib berupa string non-kosong.")


@dataclass(frozen=True)
class RetrievalResult:
    """
    Hasil dokumen chunk yang ditemukan dari pencarian semantik (SemanticRetriever).
    """
    chunk_id: str
    text: str
    metadata: dict[str, ChromaScalar]
    rank: int
    distance: float
    similarity: float

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise SchemaValidationError("`chunk_id` wajib berupa string non-kosong.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise SchemaValidationError("`text` wajib berupa string non-kosong.")
        if not isinstance(self.rank, int) or self.rank < 1:
            raise SchemaValidationError("`rank` pencarian harus berupa integer positif (>= 1).")
        if not isinstance(self.distance, (int, float)):
            raise SchemaValidationError("`distance` harus berupa bilangan numerik.")
        if not isinstance(self.similarity, (int, float)):
            raise SchemaValidationError("`similarity` harus berupa bilangan numerik.")

