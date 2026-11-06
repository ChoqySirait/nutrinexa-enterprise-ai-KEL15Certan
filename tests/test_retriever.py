import numpy as np
import pytest

from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.retriever import SemanticRetriever
from src.knowledge_base.schemas import DocumentChunk
from src.knowledge_base.vector_store import ChromaVectorStore


class MockDeterministicModel:
    """Mock model embedding untuk menghasilkan embedding deterministik berdasarkan hash teks."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 8,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        device: str = "cpu",
    ) -> np.ndarray:
        vecs = []
        for text in texts:
            # Tetapkan nilai pada dimensi tertentu untuk simulasi jarak kosinus
            vec = np.zeros(self.dim, dtype=np.float32)
            if "ayam" in text.lower():
                vec[0] = 1.0
            elif "ikan" in text.lower():
                vec[1] = 1.0
            elif "sayur" in text.lower():
                vec[2] = 1.0
            else:
                vec[10] = 1.0

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vecs.append(vec)
        return np.array(vecs)


@pytest.fixture
def retriever_setup(tmp_path):
    fake_model = MockDeterministicModel(dim=384)
    engine = EmbeddingEngine(model=fake_model)
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_retriever",
        collection_name="test_retriever",
    )

    chunks = [
        DocumentChunk(
            chunk_id="doc_ayam_1",
            text="SOP Pengolahan Daging Ayam Mendekati Kadaluarsa",
            metadata={"source_id": "SOP-AYAM", "chunk_index": 0, "category": "poultry"},
        ),
        DocumentChunk(
            chunk_id="doc_ikan_1",
            text="Pedoman Pengawetan Ikan Segar Zero Waste",
            metadata={"source_id": "SOP-IKAN", "chunk_index": 0, "category": "seafood"},
        ),
        DocumentChunk(
            chunk_id="doc_sayur_1",
            text="Panduan Reduksi Limbah Sayuran Dapur",
            metadata={"source_id": "SOP-SAYUR", "chunk_index": 0, "category": "produce"},
        ),
    ]

    embeddings = engine.embed_chunks(chunks)
    store.add_chunks(chunks, [e.embedding for e in embeddings])

    retriever = SemanticRetriever(embedding_engine=engine, vector_store=store, default_top_k=2)
    return retriever, store, engine


def test_retriever_query_top_k_ordering(retriever_setup):
    """Menguji bahwa hasil pencarian terurut berdasarkan peringkat (rank) dan relevansi kemiripan."""
    retriever, _, _ = retriever_setup

    results = retriever.retrieve("resep daging ayam", top_k=3)
    assert len(results) == 3

    # Ranking 1 harus dokumen ayam karena kemiripan kosinus tertinggi
    top_result = results[0]
    assert top_result.rank == 1
    assert top_result.chunk_id == "doc_ayam_1"
    assert "Daging Ayam" in top_result.text

    # Verifikasi pengurutan ranking
    assert results[0].rank < results[1].rank < results[2].rank


def test_retriever_cosine_similarity_calculation(retriever_setup):
    """Menguji bahwa similarity dihitung tepat: similarity = 1.0 - distance."""
    retriever, _, _ = retriever_setup

    results = retriever.retrieve("resep daging ayam", top_k=1)
    top = results[0]

    # Karena vektor sama persis, distance ~ 0.0 dan similarity ~ 1.0
    assert np.isclose(top.distance, 0.0, atol=1e-4)
    assert np.isclose(top.similarity, 1.0, atol=1e-4)
    assert np.isclose(top.similarity, 1.0 - top.distance, atol=1e-6)


def test_retriever_metadata_filtering(retriever_setup):
    """Menguji pencarian semantik dengan filter metadata kategori."""
    retriever, _, _ = retriever_setup

    # Cari tanpa kata kunci spesifik tetapi gunakan filter category produce
    results = retriever.retrieve("resep umum", top_k=5, where={"category": "produce"})
    assert len(results) == 1
    assert results[0].chunk_id == "doc_sayur_1"
    assert results[0].metadata["category"] == "produce"


def test_retriever_empty_query_validation(retriever_setup):
    """Menguji penolakan kueri kosong."""
    retriever, _, _ = retriever_setup

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        retriever.retrieve("   ")

    with pytest.raises(ValueError, match="minimal bernilai 1"):
        retriever.retrieve("kueri valid", top_k=0)


def test_retriever_min_similarity_threshold(retriever_setup):
    """Menguji penyaringan hasil kueri dengan ambang batas min_similarity."""
    retriever, _, _ = retriever_setup

    # Kueri 'resep daging ayam' menghasilkan doc_ayam dengan similarity ~1.0, lainnya ~0.0
    # Jika min_similarity = 0.5, hanya dokumen ayam yang harus lolos
    filtered_results = retriever.retrieve("resep daging ayam", top_k=5, min_similarity=0.5)
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk_id == "doc_ayam_1"
    assert filtered_results[0].rank == 1
    assert filtered_results[0].similarity >= 0.5

    # Jika min_similarity = 1.1 (di luar rentang), harus raise ValueError
    with pytest.raises(ValueError, match="min_similarity"):
        retriever.retrieve("resep daging ayam", min_similarity=1.5)

    with pytest.raises(ValueError, match="min_similarity"):
        retriever.retrieve("resep daging ayam", min_similarity=-0.1)

    # Jika ambang batas sangat tinggi dan tidak ada yang lolos
    high_threshold_results = retriever.retrieve("resep daging ayam", min_similarity=0.999999)
    # Jika doc_ayam lolos atau tidak lolos, semua hasil harus >= 0.999999
    for r in high_threshold_results:
        assert r.similarity >= 0.999999


def test_retriever_mmr_diversity_reranking(retriever_setup):
    """Menguji bahwa MMR re-ranking mendiversifikasi dokumen yang dikembalikan."""
    retriever, store, engine = retriever_setup

    # Tambahkan dokumen ayam kedua yang sangat mirip
    extra_chunks = [
        DocumentChunk(
            chunk_id="doc_ayam_2",
            text="SOP Pengolahan Daging Ayam Potong Restoran",
            metadata={"source_id": "SOP-AYAM", "chunk_index": 1, "category": "poultry"},
        )
    ]
    embs = engine.embed_chunks(extra_chunks)
    store.add_chunks(extra_chunks, [e.embedding for e in embs])

    # Tes invalid mmr_lambda
    with pytest.raises(ValueError, match="mmr_lambda"):
        retriever.retrieve("resep ayam", enable_mmr=True, mmr_lambda=1.5)

    # Tanpa MMR: dokumen ayam 1 dan ayam 2 yang mirip akan mendominasi top 2
    standard_res = retriever.retrieve("resep ayam", top_k=2, enable_mmr=False)
    standard_ids = [r.chunk_id for r in standard_res]
    assert "doc_ayam_1" in standard_ids

    # Dengan MMR (lambda rendah untuk diversifikasi maksimal):
    # Dokumen pertama tetap yang paling relevan (ayam), tapi dokumen kedua memilih yang berbeda (ikan/sayur)
    mmr_res = retriever.retrieve("resep ayam", top_k=2, enable_mmr=True, mmr_lambda=0.1)
    assert len(mmr_res) == 2
    assert mmr_res[0].chunk_id.startswith("doc_ayam")
    # Dokumen kedua harus bukan doc_ayam lainnya karena penalti kemiripan antar dokumen terpilih
    assert mmr_res[1].chunk_id in ("doc_ikan_1", "doc_sayur_1")
    assert mmr_res[0].rank == 1
    assert mmr_res[1].rank == 2


