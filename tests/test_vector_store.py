import pytest

from src.knowledge_base.schemas import DocumentChunk, VectorStoreError
from src.knowledge_base.vector_store import ChromaVectorStore


def create_sample_chunk(chunk_id: str, text: str, source_id: str = "SOP-01", category: str = "food_waste", idx: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "source_id": source_id,
            "chunk_index": idx,
            "category": category,
            "year": 2026,
        },
    )


def create_deterministic_embedding(val: float, dim: int = 384) -> list[float]:
    vec = [0.0] * dim
    vec[0] = val
    return vec


def test_chroma_vector_store_persistence_and_reopen(tmp_path):
    """Menguji bahwa data tersimpan secara persisten pada disk dan dapat dibuka kembali oleh client baru."""
    persist_dir = tmp_path / "test_chroma_db"

    # Inisialisasi client pertama & simpan data
    store1 = ChromaVectorStore(persist_directory=persist_dir, collection_name="test_persist")
    chunk1 = create_sample_chunk("c1", "Panduan penyimpanan sayur beku", source_id="DOC-1", idx=0)
    emb1 = create_deterministic_embedding(1.0)

    store1.add_chunks([chunk1], [emb1])
    assert store1.count() == 1

    # Inisialisasi client kedua pada direktori yang sama (Simulasi restart aplikasi)
    store2 = ChromaVectorStore(persist_directory=persist_dir, collection_name="test_persist")
    assert store2.count() == 1

    # Cari data dari client kedua
    res = store2.search(query_embedding=emb1, top_k=1)
    assert res["ids"][0][0] == "c1"
    assert "Panduan penyimpanan" in res["documents"][0][0]


def test_chroma_vector_store_upsert_idempotent(tmp_path):
    """Menguji bahwa penambahan ulang chunk dengan ID yang sama tidak menggandakan record (upsert)."""
    store = ChromaVectorStore(persist_directory=tmp_path / "db", collection_name="test_upsert")
    chunk = create_sample_chunk("c1", "Versi Awal", source_id="DOC-1", idx=0)
    emb = create_deterministic_embedding(0.5)

    store.add_chunks([chunk], [emb])
    assert store.count() == 1

    # Upsert ulang ID c1 dengan konten yang diperbarui
    updated_chunk = create_sample_chunk("c1", "Versi Terkini Hasil Revisi", source_id="DOC-1", idx=0)
    store.add_chunks([updated_chunk], [emb])

    assert store.count() == 1
    res = store.search(query_embedding=emb, top_k=1)
    assert res["documents"][0][0] == "Versi Terkini Hasil Revisi"


def test_chroma_vector_store_source_level_replacement_and_delete(tmp_path):
    """Menguji penghapusan dan penggantian atomik seluruh chunk dari source_id tertentu."""
    store = ChromaVectorStore(persist_directory=tmp_path / "db", collection_name="test_source_ops")

    c1 = create_sample_chunk("c1", "Doc 1 Part 1", source_id="SRC-A", idx=0)
    c2 = create_sample_chunk("c2", "Doc 1 Part 2", source_id="SRC-A", idx=1)
    c3 = create_sample_chunk("c3", "Doc 2 Part 1", source_id="SRC-B", idx=0)

    store.add_chunks(
        [c1, c2, c3],
        [create_deterministic_embedding(0.1), create_deterministic_embedding(0.2), create_deterministic_embedding(0.3)],
    )
    assert store.count() == 3

    # Hapus berdasarkan source_id SRC-A
    store.delete_by_source("SRC-A")
    assert store.count() == 1

    # Verifikasi hanya SRC-B yang tersisa
    res = store.search(query_embedding=create_deterministic_embedding(0.3), top_k=5)
    assert len(res["ids"][0]) == 1
    assert res["ids"][0][0] == "c3"

    # Uji replace_source_documents
    c4 = create_sample_chunk("c4", "Doc 2 Part 1 Baru", source_id="SRC-B", idx=0)
    store.replace_source_documents(
        source_id="SRC-B",
        chunks=[c4],
        embeddings=[create_deterministic_embedding(0.9)],
    )
    assert store.count() == 1
    res = store.search(query_embedding=create_deterministic_embedding(0.9), top_k=1)
    assert res["ids"][0][0] == "c4"
    assert res["documents"][0][0] == "Doc 2 Part 1 Baru"


def test_chroma_vector_store_metadata_filtering(tmp_path):
    """Menguji pencarian vektor dengan penyaringan metadata (metadata filtering)."""
    store = ChromaVectorStore(persist_directory=tmp_path / "db", collection_name="test_filter")

    c1 = create_sample_chunk("c1", "SOP Zero Waste Daging", source_id="S1", category="meat_waste")
    c2 = create_sample_chunk("c2", "SOP Gizi Sayuran", source_id="S2", category="vegetable_gizi")

    store.add_chunks(
        [c1, c2],
        [create_deterministic_embedding(0.8), create_deterministic_embedding(0.8)],
    )

    # Filter hanya kategori meat_waste
    filtered_res = store.search(
        query_embedding=create_deterministic_embedding(0.8),
        top_k=5,
        where={"category": "meat_waste"},
    )
    assert len(filtered_res["ids"][0]) == 1
    assert filtered_res["ids"][0][0] == "c1"


def test_chroma_vector_store_input_validation(tmp_path):
    """Menguji penanganan kesalahan pada input yang tidak valid."""
    store = ChromaVectorStore(persist_directory=tmp_path / "db", collection_name="test_val")

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        store.add_chunks([], [])

    chunk = create_sample_chunk("c1", "Teks")
    with pytest.raises(ValueError, match="tidak boleh kosong"):
        store.add_chunks([chunk], [])

    with pytest.raises(ValueError, match="tidak sepadan"):
        store.add_chunks([chunk], [create_deterministic_embedding(0.1), create_deterministic_embedding(0.2)])

    # Dimensi embedding salah (bukan 384)
    with pytest.raises(VectorStoreError, match="Dimensi vektor"):
        store.add_chunks([chunk], [[0.1, 0.2]])

    with pytest.raises(VectorStoreError, match="Dimensi query embedding"):
        store.search(query_embedding=[0.1, 0.2])
