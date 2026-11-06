import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def ingest_knowledge_base():
    # Tentukan direktori sumber dan penyimpanan database
    kb_dir = os.path.join("data", "knowledge_base")
    db_path = os.path.join("data", "chroma_db")
    
    if not os.path.exists(kb_dir):
        print(f"[Error] Direktori {kb_dir} tidak ditemukan.")
        return

    print("1. Memuat model embedding (sentence-transformers)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("2. Menyiapkan ChromaDB Client (Persistent storage)...")
    client = PersistentClient(path=db_path)
    collection_name = "nutrinexa_kb"
    
    # Ambil atau buat koleksi baru di ChromaDB
    collection = client.get_or_create_collection(name=collection_name)

    print("3. Membaca dokumen dari Knowledge Base...")
    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(kb_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Memecah dokumen berdasarkan paragraf
                chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": filename, "chunk_id": i})
                    ids.append(f"{filename}_chunk_{i}")

    if not documents:
        print("[Warning] Tidak ada teks yang ditemukan.")
        return

    print(f"4. Menghasilkan vektor embedding untuk {len(documents)} bagian teks...")
    embeddings = model.encode(documents).tolist()

    print("5. Menyimpan data ke ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Sukses! {len(documents)} dokumen berhasil disimpan ke '{db_path}'.")

if __name__ == "__main__":
    ingest_knowledge_base()