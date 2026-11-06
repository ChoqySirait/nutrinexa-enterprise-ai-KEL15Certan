"""
NutriNexa Embedding Engine Module.
Wrapper modular berbasis Sentence-Transformers untuk model `intfloat/multilingual-e5-small`.
Mengimplementasikan prefixing asimetris, validasi batas 512 token, dan normalisasi L2.
"""

from __future__ import annotations

import logging
from typing import Any

from src.knowledge_base.schemas import (
    DocumentChunk,
    EmbeddingEngineError,
    EmbeddingResult,
    TokenLimitExceededError,
)

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Mesin inferensi dense embedding berbasis Sentence-Transformers CPU-first.
    Menggunakan arsitektur asimetris E5 dengan prefix `passage: ` dan `query: `.
    """

    DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_DEVICE: str = "cpu"
    EXPECTED_DIMENSION: int = 384
    MAX_TOKEN_LIMIT: int = 512

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
        expected_dim: int = EXPECTED_DIMENSION,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.expected_dim = expected_dim

        # Injeksi model / tokenizer untuk testing terisolasi tanpa download eksternal
        self._model = model
        self._tokenizer = tokenizer

    @property
    def model(self) -> Any:
        """Lazy loading model SentenceTransformer."""
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Mengambil tokenizer dari model atau instans yang diinjeksikan."""
        if self._tokenizer is not None:
            return self._tokenizer
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer
        return None

    def load_model(self) -> None:
        """Memuat model Sentence-Transformers ke memori secara eksplisit."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Memuat model embedding '%s' ke device '%s'...", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            raise EmbeddingEngineError(
                f"Gagal memuat model SentenceTransformer '{self.model_name}': {e}"
            ) from e

    @staticmethod
    def format_text(text: str, is_query: bool = False) -> str:
        """
        Menyematkan prefix asimetris sesuai kontrak intfloat/multilingual-e5-small.
        Mencegah double-prefixing jika teks telah memiliki prefix yang bersangkutan.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Teks input tidak boleh kosong atau hanya berupa whitespace.")

        stripped = text.strip()
        target_prefix = "query: " if is_query else "passage: "
        alternate_prefix = "passage: " if is_query else "query: "

        # Jika sudah memiliki target prefix yang benar, jangan lakukan double prefix
        if stripped.startswith(target_prefix):
            return stripped

        # Jika memiliki prefix yang berlawanan, ganti menjadi target prefix
        if stripped.startswith(alternate_prefix):
            return f"{target_prefix}{stripped[len(alternate_prefix):].strip()}"

        return f"{target_prefix}{stripped}"

    def count_tokens(self, text: str, is_query: bool = False) -> int:
        """
        Menghitung jumlah token representasi input setelah diformat dengan prefix dan special tokens.
        Digunakan oleh modul chunker hulu untuk menjamin teks tidak melebihi kuota 512 token.
        """
        formatted = self.format_text(text, is_query=is_query)
        tok = self.tokenizer

        if tok is not None:
            if hasattr(tok, "encode"):
                return len(tok.encode(formatted, add_special_tokens=True))
            if hasattr(tok, "tokenize"):
                return len(tok.tokenize(formatted)) + 2

        # Heuristik fallback jika tokenizer belum/tidak terpasang pada mock sederhana
        return len(formatted.split()) + 2

    def validate_token_limit(self, text: str, is_query: bool = False) -> int:
        """Memvalidasi bahwa panjang token tidak melebihi kapasitas konteks 512 token."""
        token_count = self.count_tokens(text, is_query=is_query)
        if token_count > self.MAX_TOKEN_LIMIT:
            prefix_type = "query" if is_query else "passage"
            raise TokenLimitExceededError(
                f"Input ({prefix_type}) memiliki {token_count} token, melebihi batas maksimum "
                f"{self.MAX_TOKEN_LIMIT} token pada model '{self.model_name}'. "
                "Chunker wajib memangkas teks menjadi lebih ringkas sebelum encoding."
            )
        return token_count

    def embed_texts(self, texts: list[str], is_query: bool = False) -> list[EmbeddingResult]:
        """
        Melakukan embedding sekumpulan teks dengan batching, validasi batas token, dan normalisasi.
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Daftar teks untuk embedding tidak boleh kosong.")

        formatted_texts: list[str] = []
        token_counts: list[int] = []

        for idx, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"Teks pada indeks {idx} tidak boleh kosong atau hanya berupa whitespace.")
            token_count = self.validate_token_limit(t, is_query=is_query)
            formatted_texts.append(self.format_text(t, is_query=is_query))
            token_counts.append(token_count)

        try:
            raw_embeddings = self.model.encode(
                formatted_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                device=self.device,
            )
        except Exception as e:
            if isinstance(e, TokenLimitExceededError):
                raise
            raise EmbeddingEngineError(f"Kegagalan eksekusi encoding model: {e}") from e

        results: list[EmbeddingResult] = []
        for raw_text, emb, tokens in zip(texts, raw_embeddings, token_counts, strict=True):
            vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            if len(vec) != self.expected_dim:
                raise EmbeddingEngineError(
                    f"Dimensi vektor output ({len(vec)}) tidak sesuai dengan dimensi "
                    f"yang diharapkan ({self.expected_dim})."
                )
            results.append(
                EmbeddingResult(
                    text=raw_text,
                    embedding=vec,
                    token_count=tokens,
                    model_name=self.model_name,
                )
            )

        return results

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        """Menghasilkan embedding untuk dokumen/chunk konten SOP (otomatis menerima prefix 'passage: ')."""
        return self.embed_texts(texts, is_query=False)

    def embed_query(self, query: str) -> EmbeddingResult:
        """Menghasilkan embedding untuk kueri pencarian pengguna (otomatis menerima prefix 'query: ')."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Kueri pencarian tidak boleh kosong atau hanya berupa whitespace.")
        results = self.embed_texts([query], is_query=True)
        return results[0]

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddingResult]:
        """Menghasilkan embedding untuk kumpulan objek DocumentChunk."""
        if not isinstance(chunks, list) or len(chunks) == 0:
            raise ValueError("Daftar DocumentChunk tidak boleh kosong.")
        texts = [c.text for c in chunks]
        return self.embed_documents(texts)
    