"""
NutriNexa Document Loader Module.
Memuat dokumen SOP gizi enterprise, regulasi, dan pedoman reduksi food waste
dari format Markdown (.md), Plain Text (.txt), dan PDF (.pdf) menggunakan pypdf.
Mengekstrak teks berbasis halaman (provenance), metadata frontmatter, dan checksum SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.knowledge_base.preprocessing import (
    clean_text,
    extract_frontmatter,
    is_valid_document_text,
)
from src.knowledge_base.schemas import ChromaScalar, SchemaValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageContent:
    """Konten teks dari satu halaman dokumen sumber."""
    page_number: int
    text: str
    section_title: str | None = None


@dataclass
class LoadedDocument:
    """Dokumen terstruktur hasil ekstraksi lengkap dari loader."""
    source_id: str
    pages: list[PageContent]
    metadata: dict[str, ChromaScalar] = field(default_factory=dict)
    raw_char_count: int = 0

    @property
    def full_text(self) -> str:
        """Menggabungkan teks seluruh halaman dengan pemisah baris baru ganda."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


class DocumentLoader:
    """
    Loader kanonikal untuk berbagai format dokumen sumber NutriNexa.
    Mendukung .md, .txt, dan .pdf dengan ekstraksi halaman dan metadata provenance.
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".pdf"})

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Menghitung checksum SHA-256 dari berkas sumber untuk pelacakan versi/cache."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_file(
        self,
        file_path: str | Path,
        source_id: str | None = None,
        base_metadata: dict[str, Any] | None = None,
    ) -> LoadedDocument:
        """
        Memuat satu berkas dokumen sumber (.txt, .md, atau .pdf).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Berkas dokumen tidak ditemukan: {path}")
        if not path.is_file():
            raise ValueError(f"Jalur yang diberikan bukan berupa berkas: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Ekstensi berkas '{ext}' tidak didukung. "
                f"Format yang didukung adalah: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if ext in {".txt", ".md"}:
            return self._load_text_or_markdown(path, source_id, base_metadata)
        elif ext == ".pdf":
            return self._load_pdf(path, source_id, base_metadata)
        else:
            raise ValueError(f"Format berkas tidak dikenali: {ext}")

    def _load_text_or_markdown(
        self,
        file_path: Path,
        source_id: str | None = None,
        base_metadata: dict[str, Any] | None = None,
    ) -> LoadedDocument:
        """Memuat berkas Markdown atau teks murni."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                raw_text = f.read()

        file_hash = self.compute_file_hash(file_path)
        frontmatter_meta, body = extract_frontmatter(raw_text)
        cleaned_body = clean_text(body)

        if not is_valid_document_text(cleaned_body, min_chars=10):
            raise ValueError(f"Dokumen '{file_path.name}' kosong atau tidak memiliki teks yang valid.")

        # Tentukan source_id: prioritas argumen > frontmatter > base_metadata > nama file
        effective_source_id = (
            source_id
            or str(frontmatter_meta.get("source_id", ""))
            or (str(base_metadata.get("source_id", "")) if base_metadata else "")
            or file_path.stem.upper()
        )
        if not effective_source_id.strip():
            raise SchemaValidationError(f"Gagal menentukan `source_id` untuk dokumen: {file_path.name}")

        metadata: dict[str, ChromaScalar] = {
            "source_id": effective_source_id.strip(),
            "source_path": str(file_path).replace("\\", "/"),
            "file_hash": file_hash,
            "document_type": "sop" if file_path.suffix == ".md" else "text",
            "language": "id",
        }

        # Gabungkan metadata eksternal (manifest) dan frontmatter
        if base_metadata:
            for k, v in base_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v

        for k, v in frontmatter_meta.items():
            metadata[k] = v

        # Cek jika teks memiliki penanda halaman manual (e.g., <!-- page: 2 -->)
        pages: list[PageContent] = []
        import re
        page_splits = re.split(r"(?:<!--\s*page:\s*(\d+)\s*-->)", cleaned_body)
        if len(page_splits) > 1:
            current_page = 1
            idx = 0
            while idx < len(page_splits):
                segment = page_splits[idx].strip()
                if not segment and idx == 0:
                    idx += 1
                    continue
                if segment.isdigit():
                    current_page = int(segment)
                    idx += 1
                    if idx < len(page_splits):
                        content = clean_text(page_splits[idx])
                        if content:
                            pages.append(PageContent(page_number=current_page, text=content))
                else:
                    if segment:
                        pages.append(PageContent(page_number=current_page, text=segment))
                idx += 1
        else:
            pages.append(PageContent(page_number=1, text=cleaned_body))

        return LoadedDocument(
            source_id=effective_source_id.strip(),
            pages=pages,
            metadata=metadata,
            raw_char_count=len(raw_text),
        )

    def _load_pdf(
        self,
        file_path: Path,
        source_id: str | None = None,
        base_metadata: dict[str, Any] | None = None,
    ) -> LoadedDocument:
        """Memuat berkas PDF menggunakan pustaka pypdf dengan ekstraksi per-halaman."""
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("Pustaka 'pypdf' diperlukan untuk membaca berkas PDF. Instal dengan 'uv add pypdf'.") from e

        file_hash = self.compute_file_hash(file_path)

        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            raise ValueError(f"Gagal membaca berkas PDF '{file_path.name}': {e}") from e

        if len(reader.pages) == 0:
            raise ValueError(f"Berkas PDF '{file_path.name}' tidak memiliki halaman.")

        pages: list[PageContent] = []
        total_raw_chars = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:  # noqa: BLE001
                logger.warning("Gagal mengekstrak teks pada halaman %d berkas %s: %e", page_idx, file_path.name, e)
                page_text = ""

            total_raw_chars += len(page_text)
            cleaned = clean_text(page_text)
            if cleaned:
                pages.append(PageContent(page_number=page_idx, text=cleaned))

        if not pages or sum(len(p.text) for p in pages) < 20:
            raise ValueError(
                f"Dokumen PDF '{file_path.name}' kosong atau berupa hasil scan tanpa OCR layer teks yang terbaca."
            )

        effective_source_id = (
            source_id
            or (str(base_metadata.get("source_id", "")) if base_metadata else "")
            or file_path.stem.upper()
        )

        metadata: dict[str, ChromaScalar] = {
            "source_id": effective_source_id.strip(),
            "source_path": str(file_path).replace("\\", "/"),
            "file_hash": file_hash,
            "document_type": "pdf",
            "language": "id",
            "page_start": 1,
            "page_end": len(reader.pages),
        }

        if base_metadata:
            for k, v in base_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v

        return LoadedDocument(
            source_id=effective_source_id.strip(),
            pages=pages,
            metadata=metadata,
            raw_char_count=total_raw_chars,
        )

    def load_directory(
        self,
        dir_path: str | Path,
        manifest_path: str | Path | None = None,
    ) -> list[LoadedDocument]:
        """
        Memuat seluruh berkas yang didukung dalam sebuah direktori.
        Jika `manifest_path` diberikan, menggunakan metadata dari file manifest.
        """
        directory = Path(dir_path)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Direktori tidak ditemukan: {directory}")

        manifest_by_filename: dict[str, dict[str, Any]] = {}
        manifest_by_source_id: dict[str, dict[str, Any]] = {}

        if manifest_path:
            m_path = Path(manifest_path)
            if m_path.exists():
                with open(m_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    docs_list = data.get("documents", []) if isinstance(data, dict) else data
                    for item in docs_list:
                        if isinstance(item, dict):
                            src_id = item.get("source_id")
                            if src_id:
                                manifest_by_source_id[src_id] = item
                            src_path = item.get("source_path")
                            if src_path:
                                manifest_by_filename[Path(src_path).name] = item

        documents: list[LoadedDocument] = []
        # Cari semua berkas yang didukung
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                meta = manifest_by_filename.get(file_path.name)
                src_id = meta.get("source_id") if meta else None
                try:
                    loaded = self.load_file(file_path, source_id=src_id, base_metadata=meta)
                    documents.append(loaded)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Melewati berkas '%s' karena error: %s", file_path.name, e)

        return documents

