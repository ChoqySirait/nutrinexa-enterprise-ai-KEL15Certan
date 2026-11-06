import pytest

from src.knowledge_base.schemas import (
    DocumentChunk,
    EmbeddingResult,
    RetrievalResult,
    SchemaValidationError,
)


def test_document_chunk_valid():
    """Menguji pembuatan DocumentChunk yang valid dengan metadata lengkap."""
    chunk = DocumentChunk(
        chunk_id="SOP-001_0000",
        text="SOP Penanganan Bahan Pangan Mendekati Kadaluarsa.",
        metadata={
            "source_id": "SOP-001",
            "chunk_index": 0,
            "source_title": "SOP Penanganan Bahan",
            "category": "food_waste",
            "year": 2026,
            "is_active": True,
            "waste_reduction_target": 0.85,
        },
    )
    assert chunk.chunk_id == "SOP-001_0000"
    assert "SOP Penanganan" in chunk.text
    assert chunk.metadata["source_id"] == "SOP-001"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["year"] == 2026

    # Test to_dict and from_dict
    chunk_dict = chunk.to_dict()
    assert chunk_dict["chunk_id"] == "SOP-001_0000"
    reconstructed = DocumentChunk.from_dict(chunk_dict)
    assert reconstructed == chunk


def test_document_chunk_missing_required_metadata():
    """Menguji kegagalan validasi jika metadata wajib (source_id / chunk_index) tidak ada."""
    with pytest.raises(SchemaValidationError, match="source_id"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Teks valid",
            metadata={"chunk_index": 0},
        )

    with pytest.raises(SchemaValidationError, match="chunk_index"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Teks valid",
            metadata={"source_id": "SOP-1"},
        )


def test_document_chunk_invalid_empty_fields():
    """Menguji kegagalan validasi jika chunk_id atau text kosong."""
    with pytest.raises(SchemaValidationError, match="`chunk_id`"):
        DocumentChunk(
            chunk_id="",
            text="Teks valid",
            metadata={"source_id": "SOP-1", "chunk_index": 0},
        )

    with pytest.raises(SchemaValidationError, match="`text`"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="   ",
            metadata={"source_id": "SOP-1", "chunk_index": 0},
        )


def test_document_chunk_invalid_metadata_types():
    """Menguji penolakan tipe data non-skalar pada metadata (ChromaDB scalar-only constraint)."""
    with pytest.raises(SchemaValidationError, match="tidak didukung oleh ChromaDB"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Teks valid",
            metadata={
                "source_id": "SOP-1",
                "chunk_index": 0,
                "allergens": ["Telur", "Susu"],  # List tidak diperbolehkan
            },
        )

    with pytest.raises(SchemaValidationError, match="tidak didukung oleh ChromaDB"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Teks valid",
            metadata={
                "source_id": "SOP-1",
                "chunk_index": 0,
                "nested_dict": {"k": "v"},  # Dict tidak diperbolehkan
            },
        )

    with pytest.raises(SchemaValidationError, match="tidak didukung oleh ChromaDB"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Teks valid",
            metadata={
                "source_id": "SOP-1",
                "chunk_index": 0,
                "null_val": None,  # None tidak diperbolehkan
            },
        )


def test_embedding_result_valid():
    """Menguji instansiasi EmbeddingResult yang valid."""
    res = EmbeddingResult(
        text="query: resep hemat",
        embedding=[0.1, 0.2, 0.3],
        token_count=5,
        model_name="intfloat/multilingual-e5-small",
    )
    assert res.text == "query: resep hemat"
    assert len(res.embedding) == 3
    assert res.token_count == 5


def test_embedding_result_invalid():
    """Menguji kegagalan validasi pada EmbeddingResult yang tidak valid."""
    with pytest.raises(SchemaValidationError, match="`embedding`"):
        EmbeddingResult(
            text="resep",
            embedding=[],
            token_count=1,
            model_name="test",
        )

    with pytest.raises(SchemaValidationError, match="`token_count`"):
        EmbeddingResult(
            text="resep",
            embedding=[0.1],
            token_count=0,
            model_name="test",
        )


def test_retrieval_result_valid():
    """Menguji pembuatan RetrievalResult dengan perhitungan similarity yang benar."""
    ret = RetrievalResult(
        chunk_id="c1",
        text="Teks hasil pencarian",
        metadata={"source_id": "SOP-1", "chunk_index": 0},
        rank=1,
        distance=0.25,
        similarity=0.75,
    )
    assert ret.rank == 1
    assert ret.distance == 0.25
    assert ret.similarity == 0.75


def test_retrieval_result_invalid_rank():
    """Menguji validasi rank minimal 1."""
    with pytest.raises(SchemaValidationError, match="`rank`"):
        RetrievalResult(
            chunk_id="c1",
            text="Teks",
            metadata={"source_id": "SOP-1", "chunk_index": 0},
            rank=0,
            distance=0.1,
            similarity=0.9,
        )

