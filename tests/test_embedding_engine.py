import numpy as np
import pytest

from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.schemas import (
    DocumentChunk,
    EmbeddingEngineError,
    TokenLimitExceededError,
)


class FakeTokenizer:
    """Mock tokenizer untuk pengujian unit deterministik dan offline."""

    def __init__(self, token_multiplier: int = 1) -> None:
        self.token_multiplier = token_multiplier

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = []
        for word in text.split():
            tokens.extend([hash(word) % 10000] * self.token_multiplier)
        if add_special_tokens:
            tokens = [101] + tokens + [102]  # [CLS] ... [SEP]
        return tokens


class FakeSentenceTransformer:
    """Mock model SentenceTransformer deterministik tanpa dependensi jaringan."""

    def __init__(self, dim: int = 384, tokenizer: FakeTokenizer | None = None) -> None:
        self.dim = dim
        self.tokenizer = tokenizer or FakeTokenizer()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 8,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        device: str = "cpu",
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 10000
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


@pytest.fixture
def mock_embedding_engine():
    fake_model = FakeSentenceTransformer(dim=384)
    return EmbeddingEngine(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        model=fake_model,
        tokenizer=fake_model.tokenizer,
    )


def test_prefix_formatting_query_and_passage():
    """Menguji penambahan prefix asimetris secara tepat ('passage: ' vs 'query: ')."""
    doc_formatted = EmbeddingEngine.format_text("Standar porsi gizi makan siang", is_query=False)
    assert doc_formatted == "passage: Standar porsi gizi makan siang"

    query_formatted = EmbeddingEngine.format_text("Cari SOP substitusi daging", is_query=True)
    assert query_formatted == "query: Cari SOP substitusi daging"


def test_prefix_no_double_prefixing():
    """Memastikan sistem tidak melakukan double-prefixing jika prefix sudah ada."""
    assert EmbeddingEngine.format_text("passage: SOP Bahan Pangan", is_query=False) == "passage: SOP Bahan Pangan"
    assert EmbeddingEngine.format_text("query: resep hemat", is_query=True) == "query: resep hemat"

    # Pertukaran prefix yang berlawanan
    assert EmbeddingEngine.format_text("passage: teks", is_query=True) == "query: teks"
    assert EmbeddingEngine.format_text("query: teks", is_query=False) == "passage: teks"


def test_empty_input_validation(mock_embedding_engine):
    """Menguji penolakan input kosong pada berbagai fungsi antarmuka engine."""
    with pytest.raises(ValueError, match="tidak boleh kosong"):
        mock_embedding_engine.format_text("")

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        mock_embedding_engine.format_text("   ")

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        mock_embedding_engine.embed_query("   ")

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        mock_embedding_engine.embed_documents([])

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        mock_embedding_engine.embed_documents(["Teks valid", ""])


def test_embedding_dimension_and_normalization(mock_embedding_engine):
    """Menguji output vektor memiliki dimensi tepat 384 dan ternormalisasi L2 (norm ~ 1.0)."""
    result = mock_embedding_engine.embed_query("resep sup wortel zero waste")
    assert len(result.embedding) == 384
    norm = np.linalg.norm(result.embedding)
    assert np.isclose(norm, 1.0, atol=1e-5)

    doc_results = mock_embedding_engine.embed_documents([
        "SOP Penanganan Sayuran",
        "Substitusi Protein Nabati",
    ])
    assert len(doc_results) == 2
    for res in doc_results:
        assert len(res.embedding) == 384
        assert np.isclose(np.linalg.norm(res.embedding), 1.0, atol=1e-5)


def test_embedding_dimension_mismatch():
    """Menguji kegagalan EmbeddingEngineError jika model menghasilkan dimensi abnormal."""
    abnormal_model = FakeSentenceTransformer(dim=512)  # Bukan 384
    engine = EmbeddingEngine(
        model=abnormal_model,
        tokenizer=abnormal_model.tokenizer,
        expected_dim=384,
    )
    with pytest.raises(EmbeddingEngineError, match="Dimensi vektor output"):
        engine.embed_query("kueri")


def test_token_limit_validation():
    """Menguji penolakan teks yang melebihi batas 512 token tanpa pemotongan diam-diam."""
    fake_model = FakeSentenceTransformer(dim=384, tokenizer=FakeTokenizer(token_multiplier=1))
    engine = EmbeddingEngine(model=fake_model, tokenizer=fake_model.tokenizer)

    # Buat teks sepanjang 600 kata (> 512 token)
    long_text = " ".join(["kata"] * 600)
    with pytest.raises(TokenLimitExceededError, match="melebihi batas maksimum 512 token"):
        engine.validate_token_limit(long_text, is_query=False)

    with pytest.raises(TokenLimitExceededError, match="melebihi batas maksimum 512 token"):
        engine.embed_documents([long_text])


def test_embed_chunks(mock_embedding_engine):
    """Menguji integrasi embed_chunks dengan objek DocumentChunk."""
    chunks = [
        DocumentChunk(
            chunk_id="chk_1",
            text="SOP Gizi Seimbang",
            metadata={"source_id": "S1", "chunk_index": 0},
        ),
        DocumentChunk(
            chunk_id="chk_2",
            text="Manajemen Limbah Dapur",
            metadata={"source_id": "S1", "chunk_index": 1},
        ),
    ]
    results = mock_embedding_engine.embed_chunks(chunks)
    assert len(results) == 2
    assert results[0].text == "SOP Gizi Seimbang"
    assert len(results[0].embedding) == 384

