"""
NutriNexa Knowledge Base Smoke Test.
Skrip pengujian manual mandiri untuk memverifikasi model intfloat/multilingual-e5-small riil,
ChromaDB PersistentClient, dan SemanticRetriever end-to-end tanpa mock fallback palsu.

Cara Eksekusi:
    uv run python scripts/smoke_test_knowledge_base.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Set offline mode default untuk mencegah DNS retry delays jika model telah ada di cache lokal
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Pastikan direktori root proyek ada dalam PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.retriever import SemanticRetriever
from src.knowledge_base.schemas import DocumentChunk
from src.knowledge_base.vector_store import ChromaVectorStore


def run_smoke_test() -> None:
    print("=" * 70)
    print("NUTRINEXA T03 KNOWLEDGE BASE — SMOKE TEST VERIFICATION (REAL MODEL)")
    print("=" * 70)

    test_db_dir = Path("storage/test_chroma_smoke")
    collection_name = "smoke_test_collection"

    # Bersihkan direktori sisa sebelumnya jika ada
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir, ignore_errors=True)

    try:
        print("\n1. Inisialisasi EmbeddingEngine (intfloat/multilingual-e5-small)...")
        engine = EmbeddingEngine(
            model_name="intfloat/multilingual-e5-small",
            device="cpu",
            batch_size=4,
        )

        print("2. Memuat model intfloat/multilingual-e5-small riil ke CPU...")
        # Wajib menggunakan model riil. Jika gagal, exception langsung dilemparkan (tanpa silent fallback)
        engine.load_model()
        print(f"   [OK] Model berhasil dimuat ke memori (Device: {engine.device}).")

        print("\n3. Pengujian Asymmetric Prefix & Token Counter...")
        sample_query = "Bagaimana cara substitusi daging ayam dengan tempe?"
        sample_doc = "SOP-GIZI: Daging ayam 100g dapat disubstitusi dengan tempe kedelai 150g dengan nilai protein setara."

        query_tokens = engine.count_tokens(sample_query, is_query=True)
        doc_tokens = engine.count_tokens(sample_doc, is_query=False)
        print(f"   - Query token count (with 'query:' prefix): {query_tokens}")
        print(f"   - Doc token count (with 'passage:' prefix): {doc_tokens}")
        assert query_tokens > 0 and query_tokens <= 512, "Query token count di luar batas!"
        assert doc_tokens > 0 and doc_tokens <= 512, "Doc token count di luar batas!"

        print("\n4. Menghasilkan Dense Vector Embedding Riil (384-dimensi, L2-normalized)...")
        emb_res = engine.embed_documents([sample_doc])
        doc_vector = emb_res[0].embedding
        print(f"   [OK] Embedding berhasil dihasilkan: {len(doc_vector)} dimensi.")
        assert len(doc_vector) == 384, f"Ekspektasi 384 dimensi, ditemukan {len(doc_vector)}"

        print("\n5. Inisialisasi Persistent ChromaDB Vector Store...")
        store = ChromaVectorStore(
            persist_directory=test_db_dir,
            collection_name=collection_name,
        )

        chunk = DocumentChunk(
            chunk_id="SOP-SUB-001_p001_c0000",
            text=sample_doc,
            metadata={
                "source_id": "SOP-SUB-001",
                "chunk_index": 0,
                "category": "substitusi_protein",
                "year": 2026,
            },
        )
        inserted_count = store.add_chunks([chunk], [doc_vector])
        print(f"   [OK] Berhasil menyimpan {inserted_count} chunk ke ChromaDB.")
        assert store.count() == 1, "Jumlah chunk di vector store tidak valid!"

        print("\n6. Menjalankan SemanticRetriever dengan Kueri Uji...")
        retriever = SemanticRetriever(embedding_engine=engine, vector_store=store, default_top_k=1)
        results = retriever.retrieve(sample_query, top_k=1)

        assert len(results) == 1, "Retriever tidak mengembalikan hasil!"
        top_res = results[0]
        print(f"   - Rank {top_res.rank}: [{top_res.chunk_id}]")
        print(f"     Cosine Similarity: {top_res.similarity:.4f} (Distance: {top_res.distance:.4f})")
        print(f"     Teks: {top_res.text}")
        print(f"     Metadata: {top_res.metadata}")

        # Verifikasi formula kemiripan
        assert abs(top_res.similarity - (1.0 - top_res.distance)) < 1e-5, "Formula similarity tidak konsisten!"
        print("   [OK] Formula similarity = 1.0 - distance terverifikasi valid.")

        print("\n7. Menguji Persistensi Reopen ChromaDB...")
        store_reopen = ChromaVectorStore(
            persist_directory=test_db_dir,
            collection_name=collection_name,
        )
        assert store_reopen.count() == 1, "Data tidak persisten setelah reopen!"
        print("   [OK] Record tetap utuh setelah client diinisialisasi ulang.")

        # Cleanup direktori pengujian sementara
        shutil.rmtree(test_db_dir, ignore_errors=True)
        print("   [OK] Direktori pengujian sementara berhasil dibersihkan.")

        print("\n" + "=" * 70)
        print("SEMUA VERIFIKASI SMOKE TEST BERHASIL DILALUI DENGAN MODEL RIIL!")
        print("=" * 70)

    except Exception as exc:  # noqa: BLE001
        print(f"\n[FATAL ERROR] Smoke test gagal: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
