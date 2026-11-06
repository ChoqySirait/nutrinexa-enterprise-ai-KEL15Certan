"""
NutriNexa Corpus Ingestion Pipeline Script.
Mengonsumsi dokumen korpus resmi dan manifest terverifikasi, melakukan pembersihan teks,
chunking adaptif berbasis kata, encoding dense vector intfloat/multilingual-e5-small,
dan penyimpanan persisten ke ChromaDB.

Cara Eksekusi:
    uv run python scripts/ingest_corpus.py
    uv run python scripts/ingest_corpus.py --manifest data/knowledge_base/manifest.json --replace
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Pastikan direktori root proyek ada dalam PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.knowledge_base.chunker import WordAwareChunker
from src.knowledge_base.document_loader import DocumentLoader
from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_corpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NutriNexa Enterprise Knowledge Base Ingestion Pipeline")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/knowledge_base/manifest.json",
        help="Jalur berkas manifest metadata korpus (default: data/knowledge_base/manifest.json)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=ChromaVectorStore.DEFAULT_PERSIST_DIR,
        help="Direktori penyimpanan persisten ChromaDB (default: storage/chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=ChromaVectorStore.DEFAULT_COLLECTION_NAME,
        help="Nama koleksi ChromaDB (default: nutrinexa_knowledge_base)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=WordAwareChunker.DEFAULT_CHUNK_SIZE,
        help="Target ukuran potongan teks dalam kata (default: 200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=WordAwareChunker.DEFAULT_CHUNK_OVERLAP,
        help="Overlap antar potongan teks dalam kata (default: 35)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Ganti dokumen yang sudah ada jika source_id sama (replace old version)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya muat dan chunk dokumen tanpa melakukan embedding atau penyimpanan database",
    )
    return parser.parse_args()


def run_ingestion(args: argparse.Namespace) -> None:
    start_time = time.time()
    manifest_path = Path(args.manifest)

    logger.info("=" * 65)
    logger.info("NUTRINEXA KNOWLEDGE BASE INGESTION PIPELINE")
    logger.info("=" * 65)
    logger.info("Manifest berkas: %s", manifest_path)
    logger.info("ChromaDB path  : %s", args.db_dir)
    logger.info("Koleksi target : %s", args.collection)
    logger.info("Chunk config   : size=%d words, overlap=%d words", args.chunk_size, args.chunk_overlap)
    logger.info("Mode replace   : %s", args.replace)
    logger.info("Dry-run        : %s", args.dry_run)

    if not manifest_path.exists():
        logger.error("Berkas manifest '%s' tidak ditemukan!", manifest_path)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    doc_entries = manifest_data.get("documents", [])
    if not doc_entries:
        logger.warning("Manifest tidak memuat daftar dokumen.")
        return

    logger.info("Ditemukan %d entri dokumen dalam manifest.", len(doc_entries))

    # Inisialisasi komponen
    loader = DocumentLoader()
    engine: EmbeddingEngine | None = None
    store: ChromaVectorStore | None = None

    if not args.dry_run:
        logger.info("Memuat model embedding CPU-first (intfloat/multilingual-e5-small)...")
        engine = EmbeddingEngine(
            model_name="intfloat/multilingual-e5-small",
            device="cpu",
            batch_size=8,
        )
        engine.load_model()

        logger.info("Menghubungkan ke Persistent ChromaVectorStore...")
        store = ChromaVectorStore(
            persist_directory=args.db_dir,
            collection_name=args.collection,
        )

    # Inisialisasi Chunker dengan token counter dari engine jika ada
    token_counter = engine.count_tokens if engine is not None else None
    chunker = WordAwareChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        token_counter=token_counter,
    )

    total_loaded_docs = 0
    total_generated_chunks = 0
    total_stored_chunks = 0

    for doc_entry in doc_entries:
        source_id = doc_entry.get("source_id", "UNKNOWN")
        source_path = doc_entry.get("source_path")
        verification = doc_entry.get("verification_status", "unverified")

        if not source_path:
            logger.warning("Entri '%s' tidak memiliki 'source_path', dilewati.", source_id)
            continue

        resolved_path = Path(source_path)
        if not resolved_path.exists():
            logger.warning("Berkas '%s' (source_id: %s) tidak ditemukan pada disk, dilewati.", resolved_path, source_id)
            continue

        logger.info("\n--> Memproses Dokumen: [%s] (%s)", source_id, resolved_path.name)
        logger.info("    Judul: %s", doc_entry.get("source_title", "-"))
        logger.info("    Status Verifikasi: %s", verification)

        try:
            loaded_doc = loader.load_file(resolved_path, source_id=source_id, base_metadata=doc_entry)
            chunks = chunker.chunk_document(loaded_doc)
            total_loaded_docs += 1
            total_generated_chunks += len(chunks)

            logger.info("    Berhasil diekstraksi: %d halaman, %d chunks.", len(loaded_doc.pages), len(chunks))

            if args.dry_run or engine is None or store is None:
                continue

            # Hitung dense embeddings
            logger.info("    Menghasilkan dense embeddings untuk %d chunks...", len(chunks))
            emb_results = engine.embed_chunks(chunks)
            embeddings = [res.embedding for res in emb_results]

            # Simpan ke ChromaDB
            if args.replace:
                logger.info("    Melakukan replacement dokumen (source_id: %s)...", source_id)
                inserted = store.replace_source_documents(source_id, chunks, embeddings)
            else:
                logger.info("    Melakukan upsert dokumen (source_id: %s)...", source_id)
                inserted = store.add_chunks(chunks, embeddings)

            total_stored_chunks += inserted
            logger.info("    [OK] %d chunks tersimpan ke ChromaDB.", inserted)

        except Exception:
            logger.exception("Gagal memproses dokumen '%s'", source_id)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 65)
    logger.info("RINGKASAN EKSEKUSI PIPELINE INGESTION")
    logger.info("=" * 65)
    logger.info("Total Dokumen Diproses : %d", total_loaded_docs)
    logger.info("Total Chunks Dihasilkan: %d", total_generated_chunks)
    logger.info("Total Chunks Tersimpan : %d", total_stored_chunks)
    if store is not None:
        logger.info("Total Record di Koleksi: %d", store.count())
    logger.info("Waktu Eksekusi         : %.2f detik", elapsed)
    logger.info("=" * 65)


if __name__ == "__main__":
    run_ingestion(parse_args())
