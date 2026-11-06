"""
Unit tests untuk modul preprocessing NutriNexa (src/knowledge_base/preprocessing.py).
"""

from src.knowledge_base.preprocessing import (
    clean_text,
    extract_frontmatter,
    is_valid_document_text,
)


def test_clean_text_whitespace_and_newlines():
    """Menguji normalisasi baris baru Windows/Mac dan spasi berlebih."""
    raw = "Paragraf 1 dengan   spasi   berlebih.\r\n\r\n\r\n\r\nParagraf 2.\r\n"
    cleaned = clean_text(raw)
    # Maksimal 2 baris baru antar paragraf
    assert "\r" not in cleaned
    assert "   " not in cleaned
    assert cleaned == "Paragraf 1 dengan spasi berlebih.\n\nParagraf 2."


def test_clean_text_unicode_normalization():
    """Menguji penggantian karakter tipografi unicode ke format standar ASCII."""
    raw = "\u201cTeks dengan tanda petik ganda\u201d dan tanda \u2018tunggal\u2019 serta strip \u2014 panjang."
    cleaned = clean_text(raw)
    assert '"Teks dengan tanda petik ganda"' in cleaned
    assert "'tunggal'" in cleaned
    assert " - panjang." in cleaned


def test_clean_text_preserves_headings_and_lists():
    """Menguji bahwa heading dan markdown bullet points tetap dipertahankan."""
    raw = """
    # Heading Utama

    ## Subheading

    * Poin A
    * Poin B
    """
    cleaned = clean_text(raw)
    assert "# Heading Utama" in cleaned
    assert "## Subheading" in cleaned
    assert "* Poin A" in cleaned
    assert "* Poin B" in cleaned


def test_is_valid_document_text():
    """Menguji validasi kelayakan teks dokumen."""
    # Teks valid
    assert is_valid_document_text("Ini adalah SOP gizi standar yang memiliki teks panjang dan terbaca.") is True

    # Teks kosong atau whitespace
    assert is_valid_document_text("") is False
    assert is_valid_document_text("   \n\t  ") is False

    # Teks terlalu pendek (< 20 karakter default)
    assert is_valid_document_text("Pendek") is False

    # Teks rusak / kontrol non-printable (simulasi corrupt scanned PDF)
    corrupted = "\x00\x01\x02\x03\x04\x05\x06\x07\x08" * 5 + "teks"
    assert is_valid_document_text(corrupted) is False


def test_extract_frontmatter():
    """Menguji ekstraksi frontmatter YAML sederhana."""
    raw_md = """---
source_id: SOP-TEST-001
source_title: Pedoman Uji
year: 2026
verified: true
score: 9.5
---

# Konten Dokumen
Isi dokumen sesungguhnya dimulai di sini.
"""
    metadata, body = extract_frontmatter(raw_md)
    assert metadata["source_id"] == "SOP-TEST-001"
    assert metadata["source_title"] == "Pedoman Uji"
    assert metadata["year"] == 2026
    assert metadata["verified"] is True
    assert metadata["score"] == 9.5
    assert "# Konten Dokumen" in body
    assert "---" not in body


def test_extract_frontmatter_none():
    """Menguji dokumen tanpa frontmatter."""
    raw_text = "Dokumen tanpa header frontmatter.\nBaris kedua."
    metadata, body = extract_frontmatter(raw_text)
    assert metadata == {}
    assert body == raw_text

