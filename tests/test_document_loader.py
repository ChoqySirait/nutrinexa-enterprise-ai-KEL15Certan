"""
Unit tests untuk modul DocumentLoader NutriNexa (src/knowledge_base/document_loader.py).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.knowledge_base.document_loader import DocumentLoader


def test_load_markdown_file(tmp_path):
    """Menguji pemuatan berkas Markdown dengan frontmatter dan penanda halaman."""
    md_file = tmp_path / "sop_uji.md"
    md_file.write_text(
        """---
source_id: SOP-TEST-001
source_title: Standar Penanganan Daging
year: 2026
category: poultry
---

# Bagian 1
Daging ayam harus disimpan dalam suhu chiller antara 0 hingga 4 derajat Celsius.

<!-- page: 2 -->
# Bagian 2
Pencegahan kontaminasi silang dilakukan dengan memisahkan talenan kuning untuk unggas.
""",
        encoding="utf-8",
    )

    loader = DocumentLoader()
    doc = loader.load_file(md_file)

    assert doc.source_id == "SOP-TEST-001"
    assert doc.metadata["source_id"] == "SOP-TEST-001"
    assert doc.metadata["source_title"] == "Standar Penanganan Daging"
    assert doc.metadata["category"] == "poultry"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "chiller" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "talenan kuning" in doc.pages[1].text
    assert doc.metadata["file_hash"] != ""


def test_load_text_file(tmp_path):
    """Menguji pemuatan berkas Plain Text (.txt)."""
    txt_file = tmp_path / "pedoman_sederhana.txt"
    txt_file.write_text(
        "Pedoman penyimpanan bahan pangan kering pada suhu ruang dengan ventilasi memadai.",
        encoding="utf-8",
    )

    loader = DocumentLoader()
    doc = loader.load_file(txt_file, source_id="TXT-SOP-01")

    assert doc.source_id == "TXT-SOP-01"
    assert doc.metadata["document_type"] == "text"
    assert len(doc.pages) == 1
    assert "ventilasi memadai" in doc.full_text


def test_load_file_not_found():
    """Menguji exception ketika berkas tidak ditemukan."""
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
        loader.load_file("jalur_tidak_ada/file_fiktif.md")


def test_load_unsupported_extension(tmp_path):
    """Menguji penolakan ekstensi berkas yang tidak didukung (.docx, .json, dll)."""
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("col1,col2\n1,2", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(ValueError, match="tidak didukung"):
        loader.load_file(bad_file)


def test_load_empty_document(tmp_path):
    """Menguji penolakan berkas teks yang kosong."""
    empty_file = tmp_path / "kosong.txt"
    empty_file.write_text("   \n\t  ", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(ValueError, match="kosong atau tidak memiliki teks yang valid"):
        loader.load_file(empty_file)


def test_load_pdf_file_mocked(tmp_path):
    """Menguji pemuatan berkas PDF menggunakan mock pypdf."""
    pdf_path = tmp_path / "sop_gizi.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy binary")

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Halaman 1: Standar diet rendah garam maksimal 2000 mg natrium per hari."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Halaman 2: Penggunaan bumbu herbal alami untuk meningkatkan palatabilitas."

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1, mock_page2]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        loader = DocumentLoader()
        doc = loader.load_file(pdf_path, source_id="PDF-GIZI-001")

        assert doc.source_id == "PDF-GIZI-001"
        assert doc.metadata["document_type"] == "pdf"
        assert len(doc.pages) == 2
        assert "rendah garam" in doc.pages[0].text
        assert "bumbu herbal" in doc.pages[1].text
        assert doc.metadata["page_start"] == 1
        assert doc.metadata["page_end"] == 2


def test_load_real_pdf_file_integration():
    """Menguji ekstraksi dokumen PDF nyata secara end-to-end tanpa mock pypdf."""
    from pathlib import Path

    pdf_file = Path("data/fixtures/sample_sop_real.pdf")
    assert pdf_file.exists(), "Berkas fixture PDF riil tidak ditemukan!"

    loader = DocumentLoader()
    doc = loader.load_file(pdf_file, source_id="REAL-PDF-SOP-01")

    assert doc.source_id == "REAL-PDF-SOP-01"
    assert doc.metadata["document_type"] == "pdf"
    assert doc.metadata["page_start"] == 1
    assert doc.metadata["page_end"] == 2
    assert len(doc.pages) == 2

    # Verifikasi provenance halaman 1
    assert doc.pages[0].page_number == 1
    assert "chiller 0 hingga 4 derajat Celsius" in doc.pages[0].text

    # Verifikasi provenance halaman 2
    assert doc.pages[1].page_number == 2
    assert "Kontaminasi Silang Dapur" in doc.pages[1].text
    assert doc.metadata["file_hash"] != ""


def test_load_directory_with_manifest(tmp_path):
    """Menguji pemuatan seluruh direktori dokumen dengan referensi manifest.json."""
    doc1 = tmp_path / "sop_1.md"
    doc1.write_text("# SOP 1\nProsedur pengelolaan limbah organik dapur.", encoding="utf-8")

    doc2 = tmp_path / "sop_2.txt"
    doc2.write_text("Panduan rotasi stok bahan First-Expired-First-Out.", encoding="utf-8")

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "documents": [
            {
                "source_id": "SOP-1",
                "source_path": str(doc1),
                "category": "waste_mitigation",
            },
            {
                "source_id": "SOP-2",
                "source_path": str(doc2),
                "category": "inventory",
            },
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_directory(tmp_path, manifest_path=manifest_file)

    assert len(docs) == 2
    source_ids = {d.source_id for d in docs}
    assert "SOP-1" in source_ids
    assert "SOP-2" in source_ids

