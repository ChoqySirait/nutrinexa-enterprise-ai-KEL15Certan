"""
NutriNexa Document Preprocessing Module.
Menyediakan utilitas pembersihan teks, normalisasi spasi dan paragraf,
validasi konten dokumen, serta ekstraksi frontmatter metadata.
"""

from __future__ import annotations

import re

from src.knowledge_base.schemas import ChromaScalar


def clean_text(text: str) -> str:
    """
    Membersihkan dan menormalisasi teks dokumen:
    1. Mengubah format newline Windows (\\r\\n) atau Mac klasik (\\r) ke standard \\n.
    2. Menormalisasi tanda petik dan strip tipografi unicode ke format standar ASCII.
    3. Menghapus whitespace berlebih pada setiap baris tanpa merusak format heading atau list.
    4. Mengompresi baris kosong beruntun (>= 3) menjadi pemisah paragraf standar (\\n\\n).
    5. Menghilangkan leading dan trailing whitespace.
    """
    if not isinstance(text, str):
        return ""

    if not text.strip():
        return ""

    # 1. Normalisasi newline
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalisasi karakter tipografi unicode
    unicode_replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u2026": "...",
    }
    for orig, repl in unicode_replacements.items():
        normalized = normalized.replace(orig, repl)

    # 3. Normalisasi spasi per baris
    lines = normalized.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        # Ganti tab atau spasi berurutan dengan satu spasi
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned_line)

    normalized = "\n".join(cleaned_lines)

    # 4. Batasi baris kosong berurutan maksimal 2 newline (\n\n)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()


def is_valid_document_text(text: str, min_chars: int = 20) -> bool:
    """
    Memverifikasi apakah teks dokumen memiliki konten alfabetik/numerik yang bermakna.
    Mendeteksi dokumen kosong atau hasil pemindaian (scanned PDF tanpa OCR) yang hanya
    berisi artefak noise atau karakter tidak terbaca.
    """
    if not isinstance(text, str):
        return False

    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return False

    # Hitung rasio karakter alfanumerik terhadap total karakter
    alnum_count = sum(1 for c in cleaned if c.isalnum())
    if alnum_count < min_chars // 2:
        return False

    # Deteksi karakter kontrol non-printable (selain \n, \t, \r)
    control_chars = sum(1 for c in cleaned if ord(c) < 32 and c not in "\n\t\r")
    return (control_chars / len(cleaned)) <= 0.15


def extract_frontmatter(raw_text: str) -> tuple[dict[str, ChromaScalar], str]:
    """
    Mengekstrak frontmatter YAML/metadata sederhana dari awal dokumen Markdown atau teks.
    Format yang didukung:
    ---
    source_id: SOP-001
    source_title: Pedoman Gizi
    year: 2026
    category: waste_mitigation
    ---
    Mengembalikan pasangan (metadata_dict, body_text).
    """
    if not isinstance(raw_text, str) or not raw_text.startswith("---"):
        return {}, raw_text

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw_text, flags=re.DOTALL)
    if not match:
        return {}, raw_text

    frontmatter_content = match.group(1)
    body = raw_text[match.end():]

    metadata: dict[str, ChromaScalar] = {}
    for line in frontmatter_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip("\"'")

            # Parsing tipe skalar dasar
            if val.lower() == "true":
                metadata[key] = True
            elif val.lower() == "false":
                metadata[key] = False
            elif val.isdigit():
                metadata[key] = int(val)
            else:
                try:
                    metadata[key] = float(val)
                except ValueError:
                    metadata[key] = val

    return metadata, body

