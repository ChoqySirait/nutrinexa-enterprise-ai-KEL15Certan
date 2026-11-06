"""
Unit tests untuk modul WordAwareChunker NutriNexa (src/knowledge_base/chunker.py).
"""

import pytest

from src.knowledge_base.chunker import (
    WordAwareChunker,
    compare_chunking_configurations,
)
from src.knowledge_base.document_loader import LoadedDocument, PageContent


def test_chunker_basic_splitting_and_deterministic_ids():
    """Menguji pemecahan teks dasar dan determinisme format chunk_id."""
    text = (
        "Paragraf pertama membahas protokol rotasi bahan makanan menggunakan prinsip FIFO. "
        "Bahan yang lebih dahulu datang harus digunakan terlebih dahulu untuk menghindari kadaluarsa.\n\n"
        "Paragraf kedua membahas pemanfaatan sisa bahan nabati menjadi kaldu gurih. "
        "Sisa batang pokchoy dan kulit wortel dapat direbus dengan api kecil."
    )

    chunker = WordAwareChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk_text(text, source_id="SOP-WASTE", page_start=1, page_end=1)

    assert len(chunks) >= 2
    # Format deterministic ID: {source_id}_p{page:03d}_c{chunk_index:04d}
    assert chunks[0].chunk_id == "SOP-WASTE_p001_c0000"
    assert chunks[1].chunk_id == "SOP-WASTE_p001_c0001"
    assert chunks[0].metadata["source_id"] == "SOP-WASTE"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1


def test_chunker_sliding_window_overlap():
    """Menguji bahwa overlap kata berhasil disematkan di antara chunk berurutan."""
    words = [f"kata{i}" for i in range(50)]
    text = " ".join(words)

    chunker = WordAwareChunker(chunk_size=25, chunk_overlap=5)
    chunks = chunker.chunk_text(text, source_id="TEST-OVERLAP")

    assert len(chunks) >= 2
    # 5 kata terakhir chunk pertama harus muncul di awal chunk kedua
    chunk1_words = chunks[0].text.split()
    chunk2_words = chunks[1].text.split()

    overlap_words = chunk1_words[-5:]
    assert chunk2_words[:5] == overlap_words


def test_chunker_document_pages_provenance():
    """Menguji bahwa chunk_document mempertahankan page number dari LoadedDocument."""
    doc = LoadedDocument(
        source_id="SOP-DOC-MULTI",
        pages=[
            PageContent(page_number=1, text="Halaman satu memuat panduan sanitasi dapur.", section_title="Sanitasi"),
            PageContent(page_number=2, text="Halaman dua memuat panduan penanganan limbah.", section_title="Limbah"),
        ],
        metadata={"category": "kitchen_management", "year": 2026},
    )

    chunker = WordAwareChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["section_title"] == "Sanitasi"
    assert chunks[0].metadata["category"] == "kitchen_management"
    assert chunks[0].chunk_id == "SOP-DOC-MULTI_p001_c0000"

    assert chunks[1].metadata["page_start"] == 2
    assert chunks[1].metadata["section_title"] == "Limbah"
    assert chunks[1].chunk_id == "SOP-DOC-MULTI_p002_c0001"


def test_chunker_token_limit_compliance():
    """Menguji kepatuhan batas 512 token menggunakan token_counter injeksian."""
    def mock_counter(txt: str) -> int:
        return len(txt.split()) * 2

    # Buat teks panjang
    long_text = " ".join(["nutrisi"] * 300)

    chunker = WordAwareChunker(
        chunk_size=100,
        chunk_overlap=10,
        max_tokens=512,
        token_counter=mock_counter,
    )
    chunks = chunker.chunk_text(long_text, source_id="SOP-TOKEN")

    for c in chunks:
        tok_count = mock_counter(c.text)
        assert tok_count <= 512, f"Chunk {c.chunk_id} melebihi batas token ({tok_count} > 512)"


def test_chunker_invalid_arguments():
    """Menguji penolakan parameter konfigurasi chunker yang tidak valid."""
    with pytest.raises(ValueError, match="chunk_size"):
        WordAwareChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        WordAwareChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="lebih kecil daripada"):
        WordAwareChunker(chunk_size=50, chunk_overlap=50)

    with pytest.raises(ValueError, match="max_tokens"):
        WordAwareChunker(max_tokens=600)


def test_compare_chunking_configurations():
    """Menguji utilitas perbandingan konfigurasi chunking."""
    sample_text = (
        "Pengelolaan limbah pangan di dapur institusi memerlukan kesadaran penuh dari setiap koki. "
        "Pemilahan sampah organik dan anorganik harus dilakukan di stasiun kerja masing-masing. "
        "Sisa sayuran hijau dapat dikeringkan untuk pakan ternak atau diolah menjadi kompos berkualitas tinggi."
    )

    configs = [
        {"chunk_size": 15, "chunk_overlap": 3},
        {"chunk_size": 30, "chunk_overlap": 5},
    ]

    results = compare_chunking_configurations(sample_text, configs)
    assert len(results) == 2
    assert results[0]["chunk_size"] == 15
    assert results[1]["chunk_size"] == 30
    assert results[0]["total_chunks"] >= results[1]["total_chunks"]
    assert "avg_words" in results[0]
    assert "avg_tokens" in results[0]

