"""
NutriNexa Word-Aware Semantic Chunker Module.
Memecah dokumen teks SOP menjadi potongan atomik (DocumentChunk) dengan batas kata/paragraf,
overlap adaptif, serta validasi kuota 512 token untuk model intfloat/multilingual-e5-small.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from src.knowledge_base.document_loader import LoadedDocument
from src.knowledge_base.schemas import (
    ChromaScalar,
    DocumentChunk,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)


class WordAwareChunker:
    """
    Chunker hierarkis dan adaptif:
    1. Memecah berbasis paragraf ganda (\\n\\n).
    2. Jika paragraf terlalu panjang, memecah berbasis kalimat.
    3. Jika kalimat masih melebihi batas, memecah berbasis kata.
    4. Menerapkan sliding-window overlap untuk mempertahankan konteks semantik.
    5. Menjamin seluruh chunk aman berada di bawah batas 512 token.
    """

    DEFAULT_CHUNK_SIZE: int = 200  # Dalam hitungan kata
    DEFAULT_CHUNK_OVERLAP: int = 35  # Dalam hitungan kata
    HARD_TOKEN_LIMIT: int = 512  # Kapasitas maksimal konteks e5-small (termasuk prefix & special tokens)
    SAFETY_TOKEN_CEILING: int = 490  # Batas aman sebelum prefix "passage: "

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_tokens: int = HARD_TOKEN_LIMIT,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("`chunk_size` harus berupa integer positif (> 0).")
        if chunk_overlap < 0:
            raise ValueError("`chunk_overlap` tidak boleh negatif.")
        if chunk_overlap >= chunk_size:
            raise ValueError("`chunk_overlap` harus lebih kecil daripada `chunk_size`.")
        if max_tokens <= 0 or max_tokens > self.HARD_TOKEN_LIMIT:
            raise ValueError(f"`max_tokens` harus berada dalam rentang [1, {self.HARD_TOKEN_LIMIT}].")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.token_counter = token_counter

    def count_tokens(self, text: str) -> int:
        """Menghitung estimasi token menggunakan token_counter injeksian atau rasio kata."""
        if self.token_counter is not None:
            return self.token_counter(text)
        # Estimasi konservatif multilingual (rata-rata 1 kata ~ 1.4 token + 2 special tokens + 2 prefix tokens)
        words = text.split()
        return int(len(words) * 1.4) + 4

    def _split_into_sentences(self, text: str) -> list[str]:
        """Memecah teks paragraf menjadi kalimat-kalimat utuh."""
        # Regex memecah tanda baca penutup kalimat (. ! ?) yang diikuti spasi atau baris baru
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences if sentences else [text.strip()]

    def _split_into_units(self, text: str) -> list[str]:
        """
        Memecah teks hierarkis: paragraf -> kalimat -> kata
        hingga setiap unit berada dalam ukuran yang wajar.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: list[str] = []

        for para in paragraphs:
            para_words = para.split()
            if len(para_words) <= self.chunk_size and self.count_tokens(para) <= self.SAFETY_TOKEN_CEILING:
                units.append(para)
            else:
                # Pecah paragraf panjang menjadi kalimat
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    sent_words = sentence.split()
                    if len(sent_words) <= self.chunk_size and self.count_tokens(sentence) <= self.SAFETY_TOKEN_CEILING:
                        units.append(sentence)
                    else:
                        # Pecah kalimat yang sangat panjang berdasarkan kata
                        words = sentence.split()
                        start_w = 0
                        step = max(self.chunk_size - self.chunk_overlap, 1)
                        while start_w < len(words):
                            sub_text = " ".join(words[start_w : start_w + self.chunk_size])
                            units.append(sub_text)
                            start_w += step

        return units

    def chunk_text(
        self,
        text: str,
        source_id: str,
        base_metadata: dict[str, ChromaScalar] | None = None,
        page_start: int = 1,
        page_end: int = 1,
        start_chunk_index: int = 0,
    ) -> list[DocumentChunk]:
        """
        Memecah string teks menjadi list DocumentChunk.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        if not isinstance(source_id, str) or not source_id.strip():
            raise SchemaValidationError("Parameter `source_id` wajib berupa string non-kosong.")

        units = self._split_into_units(text)
        if not units:
            return []

        chunks: list[DocumentChunk] = []
        current_words: list[str] = []
        chunk_index = start_chunk_index

        for unit in units:
            unit_words = unit.split()

            # Periksa apakah menambahkan unit ini akan melebihi chunk_size atau batas token
            tentative_words = current_words + unit_words
            tentative_text = " ".join(tentative_words)
            tentative_tokens = self.count_tokens(tentative_text)

            if len(tentative_words) > self.chunk_size or tentative_tokens > self.SAFETY_TOKEN_CEILING:
                if current_words:
                    chunk_text = " ".join(current_words).strip()
                    if chunk_text:
                        cid = f"{source_id}_p{page_start:03d}_c{chunk_index:04d}"
                        meta: dict[str, ChromaScalar] = {
                            "source_id": source_id,
                            "chunk_index": chunk_index,
                            "page_start": page_start,
                            "page_end": page_end,
                        }
                        if base_metadata:
                            for k, v in base_metadata.items():
                                if k not in {"chunk_index"}:
                                    meta[k] = v

                        chunks.append(DocumentChunk(chunk_id=cid, text=chunk_text, metadata=meta))
                        chunk_index += 1

                    # Siapkan overlap kata untuk sliding window
                    if self.chunk_overlap > 0 and len(current_words) > self.chunk_overlap:
                        current_words = current_words[-self.chunk_overlap :] + unit_words
                    else:
                        current_words = list(unit_words)
                else:
                    current_words = list(unit_words)
            else:
                current_words.extend(unit_words)

        # Simpan sisa kata terakhir jika ada
        if current_words:
            chunk_text = " ".join(current_words).strip()
            if chunk_text:
                cid = f"{source_id}_p{page_start:03d}_c{chunk_index:04d}"
                meta = {
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "page_start": page_start,
                    "page_end": page_end,
                }
                if base_metadata:
                    for k, v in base_metadata.items():
                        if k not in {"chunk_index"}:
                            meta[k] = v

                chunks.append(DocumentChunk(chunk_id=cid, text=chunk_text, metadata=meta))

        return chunks

    def chunk_document(self, document: LoadedDocument) -> list[DocumentChunk]:
        """
        Memecah seluruh halaman dari LoadedDocument menjadi sekumpulan DocumentChunk
        dengan mempertahankan nomor halaman (page provenance) dan metadata induk.
        """
        all_chunks: list[DocumentChunk] = []
        global_chunk_idx = 0

        for page in document.pages:
            if not page.text.strip():
                continue

            page_meta = dict(document.metadata)
            if page.section_title:
                page_meta["section_title"] = page.section_title

            page_chunks = self.chunk_text(
                text=page.text,
                source_id=document.source_id,
                base_metadata=page_meta,
                page_start=page.page_number,
                page_end=page.page_number,
                start_chunk_index=global_chunk_idx,
            )

            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks


def compare_chunking_configurations(
    text: str,
    configurations: list[dict[str, Any]],
    token_counter: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    """
    Utilitas untuk membandingkan performa berbagai konfigurasi chunking
    (chunk_size, chunk_overlap) terhadap satu sampel teks korpus.
    Mengembalikan statistik metrik kuantitatif per konfigurasi.
    """
    comparison_results: list[dict[str, Any]] = []

    for cfg in configurations:
        c_size = cfg.get("chunk_size", WordAwareChunker.DEFAULT_CHUNK_SIZE)
        c_overlap = cfg.get("chunk_overlap", WordAwareChunker.DEFAULT_CHUNK_OVERLAP)

        chunker = WordAwareChunker(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            token_counter=token_counter,
        )

        chunks = chunker.chunk_text(text, source_id="CONFIG_EVAL")
        total_chunks = len(chunks)

        if total_chunks == 0:
            comparison_results.append({
                "chunk_size": c_size,
                "chunk_overlap": c_overlap,
                "total_chunks": 0,
                "avg_words": 0.0,
                "max_words": 0,
                "avg_tokens": 0.0,
                "max_tokens": 0,
            })
            continue

        word_counts = [len(c.text.split()) for c in chunks]
        token_counts = [chunker.count_tokens(c.text) for c in chunks]

        comparison_results.append({
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
            "total_chunks": total_chunks,
            "avg_words": round(sum(word_counts) / total_chunks, 2),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "avg_tokens": round(sum(token_counts) / total_chunks, 2),
            "max_tokens": max(token_counts),
        })

    return comparison_results

