import os
from chromadb import PersistentClient

def test_knowledge_base_exists():
    """Menguji apakah folder database ChromaDB sudah terbuat."""
    db_path = os.path.join("data", "chroma_db")
    assert os.path.exists(db_path), "Folder database ChromaDB belum dibuat. Jalankan ingest_kb.py dulu!"

def test_chromadb_retrieval():
    """Menguji apakah pencarian dokumen bekerja dengan benar."""
    db_path = os.path.join("data", "chroma_db")
    client = PersistentClient(path=db_path)
    collection = client.get_collection(name="nutrinexa_kb")
    
    # Pastikan ada data di dalam koleksi
    assert collection.count() > 0, "Koleksi knowledge base kosong!"
    
    # Uji pencarian kata kunci "hipertensi"
    results = collection.query(
        query_texts=["Batas konsumsi garam untuk hipertensi"],
        n_results=1
    )
    
    # Pastikan hasil pencarian mengembalikan dokumen teks
    assert len(results['documents'][0]) > 0, "Pencarian gagal mengembalikan dokumen."
    
    # Pastikan dokumen yang ditemukan relevan (mengandung kata 'hipertensi' atau 'natrium')
    dokumen_ditemukan = results['documents'][0][0].lower()
    assert "hipertensi" in dokumen_ditemukan or "natrium" in dokumen_ditemukan, "Dokumen yang ditemukan tidak relevan dengan pertanyaan."